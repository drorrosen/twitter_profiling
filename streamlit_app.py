import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from datetime import datetime, timedelta
import warnings
import hashlib
import psycopg2
import sys
import subprocess
import time
import threading
import json
from io import StringIO
import importlib.util
import csv
from apify_client import ApifyClient
import concurrent.futures
import traceback
import io

# Set page config FIRST - before any other Streamlit commands
st.set_page_config(
    page_title="Twitter Trader Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Now add custom CSS
st.markdown("""
<style>
    .main-header {
        color: #1DA1F2;
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 1rem;
        text-align: center;
    }
    .card {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-left: 4px solid #1DA1F2;
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 15px;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1DA1F2;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #6c757d;
    }
    .stButton > button {
        background-color: #1DA1F2;
        color: white;
        border: none;
        border-radius: 5px;
        padding: 10px 15px;
        font-weight: 500;
    }
    .stButton > button:hover {
        background-color: #0c85d0;
    }
    .sidebar-header {
        margin-bottom: 20px;
        text-align: center;
    }
    .sidebar-footer {
        position: fixed;
        bottom: 0;
        padding: 10px;
        width: 100%;
        background-color: #f8f9fa;
        border-top: 1px solid #e9ecef;
    }
</style>
""", unsafe_allow_html=True)

def extract_handles_directly():
    """Extract Twitter handles directly from Google Sheet."""
    try:
        # Import directly from google_sheet_connect.py
        from google_sheet_connect import get_sheet_data, extract_username
        
        # Use the correct spreadsheet ID
        SPREADSHEET_ID = '1e7qzNwQv7NCuT9coefCy7v1WtSV5b_FKW6TLy-3b25o'
        RANGE_NAME = 'Sheet1!A1:B'  # Remove the row limit to get all rows
        
        print(f"Attempting to connect to Google Sheet with ID: {SPREADSHEET_ID}")
        
        # Get sheet data
        data = get_sheet_data(SPREADSHEET_ID, RANGE_NAME)
        
        if not data:
            print("❌ Failed to retrieve data from Google Sheet")
            return [], []
        
        print(f"✅ Successfully connected to Google Sheet")
        print(f"Retrieved {len(data)} rows from Google Sheet")
        
        # Debug: Print all rows to verify we're getting the complete data
        print("\nDebug: All rows in sheet:")
        for i, row in enumerate(data):
            print(f"Row {i}: {row}")
        
        # Extract handles
        twitter_handles = []
        twitter_lists = []
        seen_handles = set()  # To prevent duplicates
        
        # Process Twitter Lists (Column A)
        for row in data[1:]:  # Skip header row
            if row and len(row) > 0 and row[0]:  # If there's a value in the first column
                list_id = extract_username(row[0])
                if list_id:
                    twitter_lists.append(list_id)
                    print(f"Added list: {list_id}")
        
        # Process Individual Accounts (Column B)
        for row in data[1:]:  # Skip header row
            if row and len(row) > 1 and row[1]:  # If there's a value in the second column
                username = extract_username(row[1])
                if username and username.lower() not in seen_handles:
                    twitter_handles.append(username)
                    seen_handles.add(username.lower())
                    print(f"Added handle: {username}")
        
        print(f"Extracted {len(twitter_handles)} Twitter handles from column B")
        print(f"Extracted {len(twitter_lists)} Twitter lists from column A")
        
        return twitter_handles, twitter_lists
        
    except Exception as e:
        print(f"❌ Error in extract_handles_directly: {e}")
        print("Please check the Google Sheet connection and make sure the service account has access.")
        return [], []  # Return empty lists instead of raising an exception

# Set environment variables for the prediction process
os.environ['FILTER_BY_TIME_HORIZON'] = 'True'
os.environ['FILTER_BY_TICKER_COUNT'] = 'True'
os.environ['MAX_TICKERS'] = '5'  # Default value, can be overridden by UI

# Import twitter-predictions.py using importlib
try:
    # Use importlib to load the module with a hyphen in the name
    spec = importlib.util.spec_from_file_location("twitter_predictions", "twitter-predictions.py")
    twitter_predictions = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(twitter_predictions)
    
    # Get the required functions
    filter_actionable_tweets = twitter_predictions.filter_actionable_tweets
    add_market_validation_columns = twitter_predictions.add_market_validation_columns
    analyze_user_accuracy = twitter_predictions.analyze_user_accuracy
    upload_to_database = twitter_predictions.upload_to_database
    
    print("Successfully imported twitter-predictions.py module")
except Exception as e:
    print(f"Error importing twitter predictions module: {e}")
    
    # Define placeholder functions
    def filter_actionable_tweets(df):
        st.error(f"Failed to import twitter-predictions.py: {e}")
        return {"actionable_tweets": pd.DataFrame(), "analysis_tweets": pd.DataFrame()}
    
    def add_market_validation_columns(df, all_tweets=None, output_dir=None):
        return pd.DataFrame()
    
    def analyze_user_accuracy(df, min_tweets=5):
        return pd.DataFrame()
    
    def upload_to_database(df):
        return False

# Function to check login credentials
def check_password():
    """Returns `True` if the user had the correct password."""
    
    def password_entered():
        """Checks whether a password entered by the user is correct."""
        # Hardcoded credentials - replace with a more secure method in production
        if st.session_state["username"] == "TwitterProfiling" and st.session_state["password"] == "Twitter135":
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # Don't store password
            del st.session_state["username"]  # Don't store username
        else:
            st.session_state["password_correct"] = False
    
    if "password_correct" not in st.session_state:
        # First run, show inputs for username + password.
        st.markdown("""
        <style>
        .login-container {
            max-width: 400px;
            margin: 0 auto;
            padding: 40px;
            background-color: white;
            border-radius: 10px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
            text-align: center;
        }
        .login-logo {
            width: 60px;
            margin-bottom: 20px;
        }
        .login-title {
            font-size: 24px;
            font-weight: 700;
            color: #1DA1F2;
            margin-bottom: 5px;
        }
        .login-subtitle {
            font-size: 14px;
            color: #657786;
            margin-bottom: 30px;
        }
        .form-label {
            font-size: 14px;
            font-weight: 500;
            color: #14171A;
            text-align: left;
            display: block;
            margin-bottom: 8px;
        }
        .stButton > button {
            background-color: #1DA1F2;
            color: white;
            font-weight: 600;
            border: none;
            padding: 10px 0;
            border-radius: 30px;
            margin-top: 20px;
            width: 100%;
            transition: background-color 0.3s;
        }
        .stButton > button:hover {
            background-color: #1a91da;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Center the login form
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            st.markdown('<div class="login-container">', unsafe_allow_html=True)
            
            # Logo and title
            st.markdown(f'<img src="https://www.iconpacks.net/icons/2/free-twitter-logo-icon-2429-thumb.png" class="login-logo">', unsafe_allow_html=True)
            st.markdown('<h1 class="login-title">Twitter Trader Analysis</h1>', unsafe_allow_html=True)
            st.markdown('<p class="login-subtitle">Enter your credentials to access the dashboard</p>', unsafe_allow_html=True)
            
            # Username field
            st.markdown('<label class="form-label">Username</label>', unsafe_allow_html=True)
            username = st.text_input("Username", key="username", placeholder="Enter username", label_visibility="collapsed")
            
            # Password field
            st.markdown('<label class="form-label">Password</label>', unsafe_allow_html=True)
            password = st.text_input("Password", key="password", type="password", placeholder="Enter password", label_visibility="collapsed")
            
            # Login button
            st.button("Sign In", on_click=password_entered)
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        return False
    
    return st.session_state["password_correct"]

# Function to load data from database
@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_data_from_db():
    """
    Load Twitter data from the database with improved error handling
    """
    try:
        # Create a database connection
        conn = psycopg2.connect(
            host="database-twitter.cdh6c6zr5mbr.us-east-1.rds.amazonaws.com",
            database="postgres",
            user="postgres",
            password="DrorMai531"
        )
        
        # Test if the tweets table exists
        cursor = conn.cursor()
        cursor.execute("SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'tweets')")
        table_exists = cursor.fetchone()[0]
        
        if not table_exists:
            st.error("The 'tweets' table does not exist in the database. Please run the database initialization script.")
            return pd.DataFrame()
        
        # Query to select all tweets
        query = "SELECT * FROM tweets"
        
        # Load the data into a pandas DataFrame
        df = pd.read_sql(query, conn)
        
        # Close the connection
        conn.close()
        
        # Print column info for debugging
        print(f"Columns in database: {df.columns.tolist()}")
        
        # Basic data cleaning and type conversion
        if 'created_at' in df.columns:
            df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')
        
        if 'created_date' in df.columns:
            df['created_date'] = pd.to_datetime(df['created_date'], errors='coerce')
        
        if 'prediction_date' in df.columns:
            df['prediction_date'] = pd.to_datetime(df['prediction_date'], errors='coerce')
        
        # Handle the case where tweet_type might be missing
        if 'tweet_type' not in df.columns:
            # Default all to 'parent' if the column is missing
            df['tweet_type'] = 'parent'
        
        # Handle conversation_id vs tweet_id issues
        if 'conversation_id' not in df.columns and 'tweet_id' in df.columns:
            df['conversation_id'] = df['tweet_id']
            print("Created conversation_id from tweet_id")
        
        # Handle follower columns
        if 'author_followers' not in df.columns or df['author_followers'].isna().all():
            # Try to look for a different column that might have follower info
            for col in df.columns:
                if 'follower' in col.lower() and not df[col].isna().all():
                    df['author_followers'] = df[col]
                    print(f"Using {col} as author_followers")
                    break
        
        # Add a flag column for actionable tweets based on standard criteria
        ticker_col = 'tickers_mentioned' if 'tickers_mentioned' in df.columns else 'tickers'
        
        # Default the column if it doesn't exist
        if ticker_col not in df.columns:
            df[ticker_col] = None
        
        # Fix the sentiment values for filtering
        if 'sentiment' in df.columns:
            # Map variant spellings to standard values
            sentiment_mapping = {
                'bulish': 'bullish',
                'bulllish': 'bullish'
            }
            df['sentiment'] = df['sentiment'].replace(sentiment_mapping)
            
            # Print sentiment value counts for debugging
            print("Sentiment values:", df['sentiment'].value_counts().to_dict())
        
        # Flag actionable tweets - accept any trade_type by removing that filter
        df['is_actionable'] = (
            (df['tweet_type'] == 'parent') &
            df[ticker_col].notna() & 
            (df[ticker_col] != '') &
            df['sentiment'].isin(['bullish', 'bearish'])
            # Removed: & ((df['time_horizon'] != 'unknown') | df['time_horizon'].isna())
            # Removed: & (df['trade_type'] == 'trade_suggestion')
        )
        
        print(f"Loaded {len(df)} tweets from database")
        # Print actionable count
        print(f"Actionable tweets: {df['is_actionable'].sum()}")
        return df
        
    except Exception as e:
        st.error(f"Error loading data from database: {e}")
        import traceback
        st.write(traceback.format_exc())
        # Return empty dataframe with expected columns
        return pd.DataFrame(columns=[
            'tweet_id', 'author', 'text', 'created_at', 'sentiment',
            'tweet_type', 'tickers_mentioned', 'is_actionable'
        ])

# Update the load_data function to filter for actionable tweets
@st.cache_data
def load_data(filepath=None):
    """
    Load data from the database instead of CSV file
    """
    df = load_data_from_db()
    
    # Check if the dataframe is empty
    if df.empty:
        st.error("Failed to load data from database. Please check the database connection and table structure.")
        return None
    
    try:
        # Convert date columns
        date_columns = ['created_at', 'created_date', 'prediction_date', 'start_date', 'end_date']
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        
        # Convert selected columns to numeric
        numeric_columns = [
            'likes', 'retweets', 'replies_count', 'views', 
            'author_followers', 'author_following',
            'price_change_pct', 'actual_return', 'start_price', 'end_price'
        ]
        
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Convert prediction_correct to boolean
        if 'prediction_correct' in df.columns:
            df['prediction_correct'] = df['prediction_correct'].map(
                {True: True, 'True': True, 'true': True, 
                 False: False, 'False': False, 'false': False}
            )
        
        # Set prediction_correct to None for future predictions
        if 'prediction_date' in df.columns and 'prediction_correct' in df.columns:
            df.loc[df['prediction_date'] > pd.Timestamp.today(), 'prediction_correct'] = None
        
        # Add tweet_type if missing
        if 'tweet_type' not in df.columns:
            df['tweet_type'] = 'parent'  # Assume all are parent tweets if not specified
        
        # Add trader column if it doesn't exist
        if 'trader' not in df.columns:
            df['trader'] = df['author']
        
        # Create a new feature for prediction score - safely handling NaN values
        if 'prediction_score' not in df.columns and 'prediction_correct' in df.columns and 'price_change_pct' in df.columns:
            def calculate_prediction_score(row):
                if pd.isna(row['prediction_correct']) or pd.isna(row['price_change_pct']):
                    return None
                
                multiplier = 1 if row['prediction_correct'] else -1
                return abs(float(row['price_change_pct'])) * multiplier
            
            df['prediction_score'] = df.apply(calculate_prediction_score, axis=1)
        
        # FILTER FOR ACTIONABLE TWEETS
        # 1. Create the full dataset (including all tweets for context)
        full_df = df.copy()
        
        # 2. Filter for parent tweets with actionable sentiment and time horizon
        actionable_df = df[
            # Parent tweets only
            (df['tweet_type'] == 'parent') &
            # With bullish or bearish sentiment only (not neutral)
            (df['sentiment'].isin(['bullish', 'bearish'])) &
            # With a valid time horizon (not empty)
            (df['time_horizon'].notna() & (df['time_horizon'] != ''))
        ]
        
        # 3. Get all conversations with actionable parent tweets
        actionable_conv_ids = actionable_df['conversation_id'].unique()
        
        # 4. Filter the full dataset to only include tweets from actionable conversations
        df = full_df[full_df['conversation_id'].isin(actionable_conv_ids)]
        
        # Print stats about filtering
        print(f"Full dataset: {len(full_df)} tweets")
        print(f"Actionable parent tweets: {len(actionable_df)} tweets")
        print(f"Filtered dataset (actionable conversations): {len(df)} tweets")
        
        return df
    
    except Exception as e:
        st.error(f"Error processing data: {e}")
        import traceback
        traceback.print_exc()
        return None

# Function to get unique traders with data cleaning
def get_traders(df):
    """
    Get unique traders from the author column with proper filtering.
    """
    # Check if required columns exist
    if 'author' not in df.columns:
        st.error("Author column missing from data")
        return []
    
    # Get all unique authors (these are our traders)
    all_authors = df['author'].dropna().unique()
    
    # Filter out non-author entries
    valid_traders = []
    for author in all_authors:
        # Skip empty values, numeric values, and ticker symbols
        if not author or not isinstance(author, str):
            continue
            
        # Skip ticker symbols (usually start with $)
        if author.startswith('$'):
            continue
            
        # Skip very short names (likely not real users)
        if len(author) < 2:
            continue
            
        # Skip entries that are just numbers
        if author.isdigit():
            continue
        
        # Add to valid traders list
        valid_traders.append(author)
    
    # Sort and return the list of valid traders
    print(f"Found {len(valid_traders)} valid traders out of {len(all_authors)} unique authors")
    
    return sorted(valid_traders)

# Function to filter data for a specific trader with improved error handling
def filter_trader_data(df, trader_name, show_warnings=True):
    """
    Filter data for a specific trader with improved error handling
    """
    if df is None or df.empty:
        if show_warnings:
            st.warning("No data available.")
        return pd.DataFrame(), pd.DataFrame()
    
    # First ensure we have an 'author' column
    if 'author' not in df.columns:
        if show_warnings:
            st.error("Data missing 'author' column - cannot filter by trader")
        return pd.DataFrame(), pd.DataFrame()
    
    # Filter for the specific trader
    df_user = df[df['author'] == trader_name].copy()
    
    if df_user.empty:
        if show_warnings:
            st.warning(f"No data found for {trader_name}.")
        return pd.DataFrame(), pd.DataFrame()
    
    # Ensure tweet_type column exists
    if 'tweet_type' not in df_user.columns:
        df_user['tweet_type'] = 'parent'  # Default to all parent if missing
    
    # Separate parent tweets
    df_parent = df_user[df_user['tweet_type'] == 'parent'].copy()
    
    # If we have fewer than 3 parent tweets, show warning
    if len(df_parent) < 3 and show_warnings:
        st.warning(f"Insufficient data for reliable analysis. {trader_name} has only {len(df_parent)} parent tweets.")
    
    # Return both the full user dataframe and just the parent tweets
    return df_user, df_parent

# Function to compute trader profile summary
def compute_profile_summary(df_user, df_parent):
    """
    Compute summary metrics for a trader profile with improved error handling and debugging
    """
    # Debug info
    print(f"Computing profile for trader with {len(df_user)} tweets ({len(df_parent)} parent tweets)")
    
    # Check if df_user is empty
    if df_user is None or df_user.empty:
        print("Empty dataframe provided to compute_profile_summary")
        return {
            "total_tweets": 0,
            "total_conversations": 0,
            "prediction_accuracy": None,
            "avg_return": None,
            "followers": "Unknown",
            "following": "Unknown"
        }
    
    # Count total tweets
    total_tweets = len(df_user)
    print(f"Total tweets: {total_tweets}")
    
    # Count unique conversations
    total_conversations = 0
    
    # Check for conversation_id column
    if 'conversation_id' in df_user.columns:
        # Check if conversation_id column has non-null values
        if df_user['conversation_id'].notna().any():
            total_conversations = df_user['conversation_id'].nunique()
            print(f"Found {total_conversations} conversations using conversation_id")
        else:
            print("conversation_id column exists but all values are null")
    else:
        print("No conversation_id column found")
    
    # If conversation_id doesn't work, try tweet_id
    if total_conversations == 0 and 'tweet_id' in df_user.columns:
        # Use tweet_id for parent tweets as fallback
        if df_parent is not None and not df_parent.empty:
            total_conversations = df_parent['tweet_id'].nunique()
            print(f"Using tweet_id as fallback: {total_conversations} unique parent tweets")
        else:
            # Last resort: just count the parent tweets
            total_conversations = len(df_parent) if df_parent is not None else 0
            print(f"Using parent tweet count as fallback: {total_conversations}")
    
    # Get follower information with debugging
    followers = "Unknown"
    following = "Unknown"
    
    # Print available columns
    print(f"Available columns for follower info: {df_user.columns.tolist()}")
    
    # Try different columns for followers
    follower_columns = ['author_followers', 'followers', 'follower_count', 'author_follower_count']
    for col in follower_columns:
        if col in df_user.columns:
            print(f"Checking '{col}' column for follower info")
            # Check if column has any non-null values
            if not df_user[col].isna().all() and df_user[col].notna().sum() > 0:
                followers_val = df_user[col].dropna().iloc[0]
                print(f"Found follower value: {followers_val}")
                try:
                    followers = f"{int(followers_val):,}"
                    print(f"Formatted follower value: {followers}")
                    break
                except:
                    followers = str(followers_val)
                    print(f"Formatted follower value (string): {followers}")
                    break
            else:
                print(f"Column '{col}' exists but all values are null or empty")
    
    # Try different columns for following
    following_columns = ['author_following', 'following', 'following_count', 'author_following_count']
    for col in following_columns:
        if col in df_user.columns:
            print(f"Checking '{col}' column for following info")
            # Check if column has any non-null values
            if not df_user[col].isna().all() and df_user[col].notna().sum() > 0:
                following_val = df_user[col].dropna().iloc[0]
                print(f"Found following value: {following_val}")
                try:
                    following = f"{int(following_val):,}"
                    print(f"Formatted following value: {following}")
                    break
                except:
                    following = str(following_val)
                    print(f"Formatted following value (string): {following}")
                    break
            else:
                print(f"Column '{col}' exists but all values are null or empty")
    
    # Calculate prediction metrics for parent tweets only
    prediction_accuracy = None
    avg_return = None
    
    # Use only parent tweets that have been validated
    if df_parent is not None and not df_parent.empty:
        # Calculate prediction accuracy
        if 'prediction_correct' in df_parent.columns:
            validated_tweets = df_parent[df_parent['prediction_correct'].notna()]
            if len(validated_tweets) > 0:
                # Convert to bool if it's object type to avoid errors
                validated_tweets_copy = validated_tweets.copy()
                if validated_tweets_copy['prediction_correct'].dtype == 'object':
                    validated_tweets_copy['prediction_correct'] = validated_tweets_copy['prediction_correct'].astype(bool)
                prediction_accuracy = validated_tweets_copy['prediction_correct'].mean() * 100
                print(f"Prediction accuracy: {prediction_accuracy:.2f}%")
        
        # Calculate average return
        if 'actual_return' in df_parent.columns:
            return_tweets = df_parent[df_parent['actual_return'].notna()]
            if len(return_tweets) > 0:
                avg_return = return_tweets['actual_return'].mean()
                print(f"Average return: {avg_return:.2f}%")
    
    return {
        "total_tweets": total_tweets,
        "total_conversations": total_conversations,
        "prediction_accuracy": prediction_accuracy,
        "avg_return": avg_return,
        "followers": followers,
        "following": following
    }

# Function to analyze all traders
def analyze_all_traders(df):
    all_traders = get_traders(df)
    trader_metrics = []
    
    for trader in all_traders:
        # Set show_warnings=False to avoid showing warnings for each trader
        df_user, df_parent = filter_trader_data(df, trader, show_warnings=False)
        
        if len(df_parent) < 3:  # Skip traders with too few parent tweets
            continue
            
        # Calculate basic metrics
        accuracy = df_parent['prediction_correct'].mean() * 100 if not df_parent['prediction_correct'].isna().all() else 0
        avg_return = df_parent['actual_return'].mean()
        total_tweets = len(df_user)
        total_convs = df_user['conversation_id'].nunique()
        followers = df_user['author_followers'].max()
        
        # Calculate sentiment distribution
        sentiment_counts = df_parent['sentiment'].value_counts(normalize=True) * 100
        bullish_pct = sentiment_counts.get('bullish', 0)
        bearish_pct = sentiment_counts.get('bearish', 0)
        
        # Calculate most mentioned stocks
        top_stocks = df_parent['validated_ticker'].value_counts().nlargest(3).index.tolist()
        top_stocks_str = ', '.join(top_stocks) if top_stocks else "N/A"
        
        trader_metrics.append({
            'trader': trader,
            'accuracy': accuracy,
            'avg_return': avg_return,
            'total_tweets': total_tweets,
            'total_conversations': total_convs,
            'followers': followers,
            'bullish_pct': bullish_pct,
            'bearish_pct': bearish_pct,
            'top_stocks': top_stocks_str
        })
    
    return pd.DataFrame(trader_metrics)

# Function to create overview dashboard
def create_overview_dashboard(df):
    """
    Create an overview dashboard with improved error handling for figure display
    """
    if df is None or df.empty:
        st.warning("No data available. Please check the database connection.")
        return
    
    st.markdown("<h1 class='main-header'>Dashboard</h1>", unsafe_allow_html=True)
    
    # Print debug info
    print(f"Dashboard loaded with {len(df)} total rows")
    
    # Count actionable tweets
    actionable_count = df['is_actionable'].sum() if 'is_actionable' in df.columns else 0
    print(f"Actionable tweets: {actionable_count}")
    
    # Overview metrics
    st.markdown("<div class='sub-header'>Key Metrics</div>", unsafe_allow_html=True)
    
    # Calculate metrics
    total_traders = df['author'].nunique() if 'author' in df.columns else 0
    total_tweets = len(df)
    
    # Calculate overall accuracy if prediction_correct column exists
    if 'prediction_correct' in df.columns:
        # Handle missing values
        validated_tweets = df[df['prediction_correct'].notna()]
        accuracy = validated_tweets['prediction_correct'].mean() * 100 if len(validated_tweets) > 0 else 0
    else:
        accuracy = 0
    
    # Calculate average return if actual_return column exists
    if 'actual_return' in df.columns:
        return_tweets = df[df['actual_return'].notna()]
        avg_return = return_tweets['actual_return'].mean() if len(return_tweets) > 0 else 0
    else:
        avg_return = 0
    
    # Create a layout with 4 metrics in one row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(label="Traders Analyzed", value=total_traders)
    
    with col2:
        st.metric(label="Total Tweets", value=total_tweets)
    
    with col3:
        st.metric(label="Overall Accuracy", value=f"{accuracy:.1f}%")
    
    with col4:
        st.metric(label="Average Return", value=f"{avg_return:.2f}%")
    
    # Add horizontal separators
    st.markdown("---")
    
    # Performance over time
    st.markdown("<div class='sub-header'>Performance Over Time</div>", unsafe_allow_html=True)
    
    # Check if we have the required date column
    if 'created_at' not in df.columns and 'created_date' not in df.columns:
        st.warning("Date column not found in data. Cannot display performance over time.")
        return
    
    # Use whichever date column is available
    date_col = 'created_at' if 'created_at' in df.columns else 'created_date'
    
    # Create dataframe for time series analysis
    try:
        # Make sure we have date values
        df_dates = df[df[date_col].notna()].copy()
        
        if df_dates.empty:
            st.warning("No valid dates found in data. Cannot display performance over time.")
            return
        
        # Create month column
        df_dates['month'] = df_dates[date_col].dt.to_period('M').astype(str)
        
        # Group by month
        monthly_data = df_dates.groupby('month').agg({
            'tweet_id': 'count',  # Count tweets
            'prediction_correct': lambda x: x.mean() * 100 if len(x.dropna()) > 0 else None,  # Accuracy
            'actual_return': lambda x: x.mean() if len(x.dropna()) > 0 else None  # Return
        }).reset_index()
        
        # Rename columns
        monthly_data.columns = ['month', 'tweet_count', 'accuracy', 'return']
        
        # Sort by month
        monthly_data = monthly_data.sort_values('month')
        
        # Create figure with dual y-axis
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Add bar chart for tweet count
        fig.add_trace(
            go.Bar(
                x=monthly_data['month'],
                y=monthly_data['tweet_count'],
                name="Tweet Count",
                marker_color='lightblue'
            ),
            secondary_y=False
        )
        
        # Add line chart for accuracy
        fig.add_trace(
            go.Scatter(
                x=monthly_data['month'],
                y=monthly_data['accuracy'],
                name="Accuracy (%)",
                marker_color='green',
                mode='lines+markers'
            ),
            secondary_y=True
        )
        
        # Add line chart for return
        fig.add_trace(
            go.Scatter(
                x=monthly_data['month'],
                y=monthly_data['return'],
                name="Return (%)",
                marker_color='red',
                mode='lines+markers'
            ),
            secondary_y=True
        )
        
        # Add a horizontal line at 0% return
        fig.add_shape(
            type='line',
            x0=monthly_data['month'].iloc[0],
            x1=monthly_data['month'].iloc[-1],
            y0=0,
            y1=0,
            line=dict(color='red', width=2, dash='dash'),
            yref='y2'
        )
        
        # Update layout
        fig.update_layout(
            title='Monthly Tweet Count, Accuracy, and Return',
            xaxis_title='Month',
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
            height=500
        )
        
        fig.update_yaxes(title_text='Tweet Count', secondary_y=False)
        fig.update_yaxes(title_text='Percentage (%)', secondary_y=True)
        
        # Display the figure
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error creating performance chart: {e}")
        import traceback
        st.write(traceback.format_exc())
    
    # Try creating the trader comparison table
    try:
        # Analyze all traders
        trader_metrics = analyze_all_traders(df)
        
        if trader_metrics.empty:
            st.warning("No trader metrics available.")
            return
            
        st.markdown('<div class="sub-header">Trader Comparison</div>', unsafe_allow_html=True)
        
        # Format the full table
        display_df = trader_metrics.sort_values('accuracy', ascending=False).copy()
        display_df.columns = [
            'Trader', 'Accuracy (%)', 'Avg Return (%)', 'Total Tweets', 'Total Conversations',
            'Followers', 'Bullish (%)', 'Bearish (%)', 'Top Stocks'
        ]
        display_df['Accuracy (%)'] = display_df['Accuracy (%)'].round(1)
        display_df['Avg Return (%)'] = display_df['Avg Return (%)'].round(2)
        display_df['Bullish (%)'] = display_df['Bullish (%)'].round(1)
        display_df['Bearish (%)'] = display_df['Bearish (%)'].round(1)
        display_df['Followers'] = display_df['Followers'].apply(lambda x: f"{x:,}" if pd.notna(x) else "Unknown")
        
        st.dataframe(display_df, use_container_width=True, height=400)
        
    except Exception as e:
        st.error(f"Error creating trader comparison table: {e}")
        import traceback
        st.write(traceback.format_exc())

# Function to create trader profile dashboard
def create_trader_profile(df, trader_name):
    """
    Create a profile page for a specific trader
    """
    if df.empty:
        st.warning("No data available.")
        return
    
    # Filter for the specific trader
    df_user, df_parent = filter_trader_data(df, trader_name)
    
    if df_user.empty:
        st.warning(f"No data found for {trader_name}.")
        return
        
    # Apply contextual enrichment for traders like JediAnt
    if trader_name == 'JediAnt':
        df_user = enrich_with_context(df_user)
        # Re-filter to get updated parent tweets with enriched tickers
        df_parent = df_user[df_user['tweet_type'] == 'parent'].copy()
        
        # Add an info message about the enrichment
        st.info("📝 Note: For JediAnt, 'China' references are automatically mapped to $KTEC and $FXI tickers.")
    
    # Compute profile summary
    profile = compute_profile_summary(df_user, df_parent)
    
    # Display trader name as page title
    st.header(f"Trader Profile: {trader_name}")
    
    # Check if trader has enough actionable tweets for reliable analysis
    actionable_parent_tweets = df_parent[df_parent['is_actionable'] == True] if 'is_actionable' in df_parent.columns else pd.DataFrame()
    actionable_parent_count = len(actionable_parent_tweets)
    
    if actionable_parent_count < 3:
        st.warning(f"Insufficient data for reliable analysis. {trader_name} has only {actionable_parent_count} actionable parent tweets.")
    
    # Create a layout with 4 metrics in one row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(label="Prediction Accuracy", 
                value=f"{profile['prediction_accuracy']:.1f}%" if profile['prediction_accuracy'] is not None else "Unknown")
    
    with col2:
        st.metric(label="Average Return", 
                value=f"{profile['avg_return']:.2f}%" if profile['avg_return'] is not None else "Unknown")
    
    with col3:
        st.metric(label="Total Conversations", 
                value=profile['total_conversations'])
    
    with col4:
        st.metric(label="Followers", 
                value=profile['followers'])
    
    # Create tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs(["Profile Summary", "Prediction Analysis", "Sentiment Analysis", "Stock Performance"])
    
    # Rest of function continues as before...

# Function to create raw data dashboard
def create_raw_data_dashboard(df):
    """
    Create a dashboard to explore the raw data.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The dataframe containing the data to explore.
    """
    st.markdown("<h1 class='main-header'>Raw Data Explorer</h1>", unsafe_allow_html=True)
    
    # Create tabs for raw data and statistics
    tabs = st.tabs(["Raw Data", "Data Statistics"])
    
    # Raw Data tab
    with tabs[0]:
        st.markdown("<h2 class='sub-header'>Raw Data</h2>", unsafe_allow_html=True)
        
        # Add filters for the raw data
        col1, col2, col3 = st.columns(3)
        
        # Check for author column (traders)
        with col1:
            if 'author' in df.columns:
                authors = ["All"] + sorted(df["author"].unique().tolist())
                selected_author = st.selectbox("Filter by Trader", authors)
            else:
                selected_author = "All"
                st.info("No author column found in the data")
        
        # Check for ticker column and extract first ticker from each entry if comma-separated
        with col2:
            if 'tickers_mentioned' in df.columns:
                # Extract first ticker from comma-separated lists
                first_tickers = df['tickers_mentioned'].apply(lambda x: x.split(',')[0].strip() if isinstance(x, str) and ',' in x else x)
                tickers = ["All"] + sorted(first_tickers.unique().tolist())
                selected_ticker = st.selectbox("Filter by Ticker", tickers)
            elif 'ticker' in df.columns:
                tickers = ["All"] + sorted(df["ticker"].unique().tolist())
                selected_ticker = st.selectbox("Filter by Ticker", tickers)
            else:
                selected_ticker = "All"
                st.info("No ticker column found in the data")
        
        # Date filter
        with col3:
            dates = ["All", "Last 7 Days", "Last 30 Days", "Last 90 Days", "Last 365 Days"]
            selected_date = st.selectbox("Filter by Date", dates, key="date_filter")
        
        # Apply filters
        filtered_df = df.copy()
        
        # Apply author filter
        if selected_author != "All":
            if 'author' in df.columns:
                filtered_df = filtered_df[filtered_df["author"] == selected_author]
        
        # Apply ticker filter - check if ticker is in any position in comma-separated list
        if selected_ticker != "All":
            if 'tickers_mentioned' in df.columns:
                # Filter rows where the selected ticker appears in the comma-separated list
                filtered_df = filtered_df[filtered_df['tickers_mentioned'].apply(
                    lambda x: selected_ticker in [t.strip() for t in x.split(',')] if isinstance(x, str) else False
                )]
            elif 'ticker' in df.columns:
                filtered_df = filtered_df[filtered_df["ticker"] == selected_ticker]
        
        # Apply date filter using the max date in the dataset
        if selected_date != "All":
            # Find date columns
            date_columns = []
            for col in df.columns:
                if 'date' in col.lower() or col.lower() == 'date':
                    date_columns.append(col)
            
            if date_columns:
                # Use the first found date column
                date_col = date_columns[0]
                
                # Get max date from the dataset
                max_date = df[date_col].max()
                
                days = int(selected_date.split()[1])
                cutoff_date = max_date - timedelta(days=days)
                
                filtered_df = filtered_df[filtered_df[date_col] >= cutoff_date]
        
        # Show the filtered dataframe
        st.dataframe(filtered_df, use_container_width=True)
        
        # Download button for the filtered data
        csv = filtered_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Filtered Data as CSV",
            data=csv,
            file_name=f"twitter_trader_data_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
        )
    
    # Data Statistics tab
    with tabs[1]:
        st.markdown("<h2 class='sub-header'>Data Statistics</h2>", unsafe_allow_html=True)
        
        # Create a dashboard of statistics for the data
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("<h3>Numerical Statistics</h3>", unsafe_allow_html=True)
            
            # Select only numerical columns for statistics
            num_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
            if num_cols:
                num_stats = df[num_cols].describe().T
                num_stats = num_stats.reset_index()
                num_stats.columns = ['Column', 'Count', 'Mean', 'Std', 'Min', '25%', '50%', '75%', 'Max']
                st.dataframe(num_stats, use_container_width=True)
            else:
                st.info("No numerical columns found in the dataset.")
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col2:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("<h3>Categorical Statistics</h3>", unsafe_allow_html=True)
            
            # Select categorical columns
            cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            
            if cat_cols:
                selected_cat_col = st.selectbox("Select Categorical Column", cat_cols)
                
                # Calculate value counts
                val_counts = df[selected_cat_col].value_counts().reset_index()
                val_counts.columns = [selected_cat_col, 'Count']
                
                # Create a horizontal bar chart
                fig = px.bar(
                    val_counts.head(20), 
                    x='Count', 
                    y=selected_cat_col,
                    orientation='h',
                    title=f'Top 20 Values in {selected_cat_col}',
                    color='Count',
                    color_continuous_scale='blues'
                )
                
                fig.update_layout(
                    height=600,
                    xaxis_title='Count',
                    yaxis_title=selected_cat_col,
                    yaxis={'categoryorder':'total ascending'}
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No categorical columns found in the dataset.")
                
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Additional statistics
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<h3>Dataset Summary</h3>", unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Records", f"{len(df):,}")
        
        with col2:
            # Check for author column (traders)
            if 'author' in df.columns:
                st.metric("Number of Traders", f"{df['author'].nunique():,}")
            else:
                st.metric("Number of Traders", "N/A")
        
        with col3:
            # Check for ticker column
            if 'ticker' in df.columns:
                st.metric("Number of Tickers", f"{df['ticker'].nunique():,}")
            elif 'tickers_mentioned' in df.columns:
                st.metric("Number of Tickers", f"{df['tickers_mentioned'].nunique():,}")
            else:
                st.metric("Number of Tickers", "N/A")
        
        with col4:
            # Find date columns
            date_columns = []
            for col in df.columns:
                if 'date' in col.lower() or col.lower() == 'date':
                    date_columns.append(col)
            
            if date_columns:
                # Use the first found date column
                date_col = date_columns[0]
                date_range = f"{df[date_col].min().date()} to {df[date_col].max().date()}"
                st.metric("Date Range", date_range)
            else:
                st.metric("Date Range", "N/A")
            
        st.markdown("</div>", unsafe_allow_html=True)

# Function to load the twitter-llm-optimized module dynamically
def load_twitter_llm_module():
    try:
        # Dynamically import the twitter-llms-optimized.py module
        spec = importlib.util.spec_from_file_location("twitter_llm", "twitter-llms-optimized.py")
        twitter_llm = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(twitter_llm)
        return twitter_llm
    except Exception as e:
        st.error(f"Error loading Twitter LLM module: {str(e)}")
        return None

# Add a new function for the data extraction dashboard
def create_data_extraction_dashboard():
    # Initialize session state variables if they don't exist
    if 'twitter_data' not in st.session_state:
        st.session_state.twitter_data = None
    if 'processed_data' not in st.session_state:  # Add this line
        st.session_state.processed_data = None
    
    st.markdown("<h1 class='main-header'>Twitter Data Extraction Dashboard</h1>", unsafe_allow_html=True)
    
    # Create a cleaner layout with two columns
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<h3 style='color: #1DA1F2; margin-bottom: 15px;'>Run Tweet Extraction</h3>", unsafe_allow_html=True)
        
        # Add options for extraction in a more compact layout
        extraction_options_col1, extraction_options_col2 = st.columns(2)
        
        with extraction_options_col1:
            sample_mode = st.checkbox("Sample Mode (100 tweets)", value=True, 
                                     help="Enable to fetch only 100 tweets for testing")
        
        with extraction_options_col2:
            extraction_type = st.radio(
                "Extraction Type",
                [
                    "Daily Only", 
                    "Weekly (Last 7 Days)",  # Add this new option
                    "New Handles Only", 
                    "Complete Extraction"
                ],
                index=0
            )
        
        # Create a container for the button to control its width
        button_col1, button_col2, button_col3 = st.columns([1, 2, 1])
        
        with button_col2:
            # Add a more prominent button to run the extraction with consistent styling
            if st.button("🚀 Start Tweet Extraction", type="primary"):
                # Create a placeholder for the log output
                log_output = st.empty()
                
                # Show a progress message
                st.markdown("""
                <div style="padding: 10px; background-color: #f8f9fa; border-radius: 5px; margin-top: 10px;">
                    <p style="margin: 0; color: #1DA1F2;"><strong>⏳ Initializing extraction process...</strong></p>
                </div>
                """, unsafe_allow_html=True)
                
                # Function to capture and display output
                def run_extraction_with_logs():
                    """Run the tweet extraction process and capture logs"""
                    # Import sys and StringIO directly in the function scope
                    import sys
                    from io import StringIO
                    
                    # Redirect stdout to capture logs
                    old_stdout = sys.stdout
                    mystdout = StringIO()
                    sys.stdout = mystdout
                    
                    try:
                        # Set environment variables based on user selections
                        os.environ["SAMPLE_MODE"] = "True" if sample_mode else "False"
                        print(f"🔧 Sample mode: {'Enabled (100 tweets)' if sample_mode else 'Disabled (100,000 tweets)'}")
                        os.environ["EXTRACTION_TYPE"] = extraction_type
                        
                        # Get handles from Google Sheet directly - no fallbacks
                        twitter_handles, twitter_lists = extract_handles_directly()
                        
                        if not twitter_handles:
                            print("❌ No Twitter handles found. Please check the Google Sheet connection.")
                            return mystdout.getvalue(), 0
                        
                        print(f"✅ Using {len(twitter_handles)} Twitter handles for processing")
                        
                        # Initialize scraper using the existing TwitterScraper class
                        from APIFY_tweet_extraction import TwitterScraper
                        
                        # API token from the existing code
                        api_token = "apify_api_kdevcdwOVQ5K4HugmeLYlaNgaxeOGG3dkcwc"
                        scraper = TwitterScraper(api_token)
                        
                        # Set max_items based on sample mode
                        max_items = 100 if sample_mode else 100000
                        
                        # Initialize scraper using the existing TwitterScraper class
                        from APIFY_tweet_extraction import TwitterScraper
                        
                        # API token from the existing code
                        api_token = "apify_api_kdevcdwOVQ5K4HugmeLYlaNgaxeOGG3dkcwc"
                        scraper = TwitterScraper(api_token)
                        
                        # Update the prepare_input method's default max_items
                        scraper.prepare_input = lambda *args, **kwargs: {
                            "startUrls": kwargs.get('start_urls', []),
                            "searchTerms": kwargs.get('search_terms', []),
                            "twitterHandles": kwargs.get('twitter_handles', []),
                            "conversationIds": kwargs.get('conversation_ids', []),
                            "maxItems": max_items,  # Use our max_items here
                            "sort": "Latest",
                            "tweetLanguage": "en"
                        }
                        
                        # For "New Handles Only" extraction type, we need to get existing authors from the database
                        if extraction_type == "New Handles Only":
                            print("🔍 New Handles Only mode selected - checking database for existing authors...")
                            
                            try:
                                # Get existing authors from the database
                                from APIFY_tweet_extraction import get_existing_authors_from_db
                                existing_authors = get_existing_authors_from_db()
                                
                                # Filter for new handles (not in database)
                                new_handles = [handle for handle in twitter_handles if handle.lower() not in existing_authors]
                                
                                print(f"✅ Found {len(existing_authors)} existing authors in database")
                                print(f"✅ Found {len(twitter_handles)} total handles from Google Sheet")
                                print(f"🆕 Identified {len(new_handles)} new handles for extraction")
                                
                                # Print only the new handles that aren't in the database
                                if new_handles:
                                    print("\n📋 New handles that will be processed:")
                                    for handle in new_handles:
                                        print(f"  • {handle}")
                                    
                                    # Import and use the process_historical_tweets_for_new_handles function
                                    from APIFY_tweet_extraction import process_historical_tweets_for_new_handles, load_processed_handles
                                    
                                    # Load the list of handles that have already been processed historically
                                    processed_handles = load_processed_handles()
                                    print(f"Previously processed {len(processed_handles)} handles historically")
                                    
                                    # Process historical tweets for new handles
                                    new_historical_tweets = process_historical_tweets_for_new_handles(
                                        scraper, new_handles, processed_handles)
                                    print(f"Processed {len(new_historical_tweets)} historical tweets for new handles")
                                    
                                    # Then apply the limit after getting the tweets
                                    if new_historical_tweets:
                                        df = pd.DataFrame(new_historical_tweets)
                                        if len(df) > max_items:
                                            df = df.head(max_items)
                                            print(f"✅ {'Sample' if sample_mode else 'Full'} mode: Limited to {len(df)} tweets")
                                        else:
                                            print(f"✅ Processing {len(df)} tweets")
                                        
                                        # Store in session state
                                        st.session_state.twitter_data = df
                                        tweet_count = len(df)
                                        print(f"✅ Stored {tweet_count} tweets in session state for processing")
                                        return mystdout.getvalue(), tweet_count
                                    else:
                                        print("❌ No new tweets were retrieved")
                                        return mystdout.getvalue(), 0
                                
                            except Exception as e:
                                print(f"❌ Error checking for new handles: {str(e)}")
                                print("⚠️ Unable to process without valid handles")
                                return mystdout.getvalue(), 0
                        
                        elif extraction_type == "Daily Only":
                            os.environ["EXTRACTION_TYPE"] = "daily"
                            period = "daily"
                            print("🔍 DEBUG: Running DAILY ONLY extraction")
                            print(f"🔍 DEBUG: EXTRACTION_TYPE={os.environ['EXTRACTION_TYPE']}, period={period}")
                            
                            # Import and use the process_current_tweets function
                            from APIFY_tweet_extraction import process_current_tweets
                            
                            # Process current tweets (daily) for all handles
                            print("🔍 DEBUG: Calling process_current_tweets with period='daily'")
                            current_tweets = process_current_tweets(
                                scraper, twitter_handles, period=period)
                            print(f"🔍 DEBUG: Returned {len(current_tweets)} tweets from process_current_tweets")
                            
                            # Convert to dataframe
                            if current_tweets:
                                df = pd.DataFrame(current_tweets)
                                
                                # Apply sample mode if enabled
                                if sample_mode and len(df) > 100:
                                    df = df.head(100)
                                    print(f"✅ Sample mode: Limited to {len(df)} tweets")
                                elif not sample_mode and len(df) > 100000:  # Add this condition
                                    df = df.head(100000)
                                    print(f"✅ Full mode: Limited to {len(df)} tweets")
                                else:
                                    print(f"✅ Processing {len(df)} tweets")
                                
                                # Store in session state
                                st.session_state.twitter_data = df
                                tweet_count = len(df)
                                print(f"✅ Stored {tweet_count} tweets in session state for processing")
                                return mystdout.getvalue(), tweet_count
                            else:
                                print("❌ No tweets were retrieved")
                                return mystdout.getvalue(), 0
                        
                        elif extraction_type == "Weekly (Last 7 Days)":  # Handle the new option
                            os.environ["EXTRACTION_TYPE"] = "weekly"
                            period = "weekly"
                            # Import and use the process_current_tweets function
                            from APIFY_tweet_extraction import process_current_tweets
                            
                            # Process current tweets (weekly) for all handles
                            current_tweets = process_current_tweets(
                                scraper, twitter_handles, period=period)
                            print(f"Processed {len(current_tweets)} current tweets")
                            
                            # Convert to dataframe
                            if current_tweets:
                                df = pd.DataFrame(current_tweets)
                                
                                # Apply sample mode if enabled
                                if sample_mode and len(df) > 100:
                                    df = df.head(100)
                                    print(f"✅ Sample mode: Limited to {len(df)} tweets")
                                elif not sample_mode and len(df) > 100000:  # Add this condition
                                    df = df.head(100000)
                                    print(f"✅ Full mode: Limited to {len(df)} tweets")
                                else:
                                    print(f"✅ Processing {len(df)} tweets")
                                
                                # Store in session state
                                st.session_state.twitter_data = df
                                tweet_count = len(df)
                                print(f"✅ Stored {tweet_count} tweets in session state for processing")
                                return mystdout.getvalue(), tweet_count
                            else:
                                print("❌ No tweets were retrieved")
                                return mystdout.getvalue(), 0
                        
                        elif extraction_type == "Complete Extraction":
                            os.environ["EXTRACTION_TYPE"] = "complete"
                            period = "daily"  # Default to daily for complete extraction
                            # Import process_current_tweets and process_historical_tweets
                            from APIFY_tweet_extraction import process_current_tweets, process_historical_tweets
                            
                            # Process both current and historical tweets
                            current_tweets = process_current_tweets(
                                scraper, twitter_handles, period=period)
                            print(f"Processed {len(current_tweets)} current tweets")
                            
                            historical_tweets = process_historical_tweets(
                                scraper, twitter_handles)
                            print(f"Processed {len(historical_tweets)} historical tweets")
                            
                            # Combine all tweets
                            all_tweets = current_tweets + historical_tweets
                            
                            # Convert to dataframe
                            if all_tweets:
                                df = pd.DataFrame(all_tweets)
                                
                                # Apply sample mode if enabled
                                if sample_mode and len(df) > 100:
                                    df = df.head(100)
                                    print(f"✅ Sample mode: Limited to {len(df)} tweets")
                                elif not sample_mode and len(df) > 100000:  # Add this condition
                                    df = df.head(100000)
                                    print(f"✅ Full mode: Limited to {len(df)} tweets")
                                else:
                                    print(f"✅ Processing {len(df)} tweets")
                                
                                # Store in session state
                                st.session_state.twitter_data = df
                                tweet_count = len(df)
                                print(f"✅ Stored {tweet_count} tweets in session state for processing")
                                return mystdout.getvalue(), tweet_count
                            else:
                                print("❌ No tweets were retrieved")
                                return mystdout.getvalue(), 0
                        
                    except Exception as e:
                        print(f"Error during extraction: {str(e)}")
                        import traceback
                        traceback.print_exc(file=mystdout)
                        return mystdout.getvalue(), 0
                    finally:
                        # Restore stdout
                        sys.stdout = old_stdout
                    
                    return mystdout.getvalue(), 0
                
                # Run in a separate thread to avoid blocking the UI
                with st.spinner("Running tweet extraction... This may take a few minutes."):
                    # Execute the extraction
                    logs, tweet_count = run_extraction_with_logs()
                    
                    # Format the logs with syntax highlighting and better styling
                    formatted_logs = logs.replace("\n", "<br>")
                    formatted_logs = formatted_logs.replace("✅", "<span style='color: #17BF63'>✅</span>")
                    formatted_logs = formatted_logs.replace("❌", "<span style='color: #E0245E'>❌</span>")
                    formatted_logs = formatted_logs.replace("⚠️", "<span style='color: #FFAD1F'>⚠️</span>")
                    formatted_logs = formatted_logs.replace("🔄", "<span style='color: #1DA1F2'>🔄</span>")
                    formatted_logs = formatted_logs.replace("🆕", "<span style='color: #17BF63'>🆕</span>")
                    formatted_logs = formatted_logs.replace("📊", "<span style='color: #794BC4'>📊</span>")
                    formatted_logs = formatted_logs.replace("📝", "<span style='color: #1DA1F2'>📝</span>")
                    
                    # Display the logs in a scrollable area with better styling
                    log_output.markdown(f"""
                    <div class='log-container' style='background-color: #1e1e1e; color: #d4d4d4; padding: 15px; border-radius: 8px; font-family: "Consolas", monospace; line-height: 1.5; max-height: 500px; overflow-y: auto;'>
                        {formatted_logs}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Show completion message
                    if tweet_count > 0:
                        st.success(f"Tweet extraction completed successfully! Extracted {tweet_count} tweets.")
                    else:
                        st.warning("Tweet extraction completed but no tweets were found.")
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<h3 style='color: #1DA1F2; margin-bottom: 15px;'>Twitter Handles</h3>", unsafe_allow_html=True)
        
        # Get Twitter handles from Google Sheet
        try:
            twitter_handles, twitter_lists = extract_handles_directly()
            
            # Create a more visually appealing success message
            st.markdown(f"""
            <div style="background-color: #E8F5E9; padding: 10px; border-radius: 5px; margin-bottom: 15px;">
                <p style="margin: 0; color: #2E7D32;"><strong>✅ Found {len(twitter_handles)} Twitter handles in Google Sheet</strong></p>
            </div>
            """, unsafe_allow_html=True)
            
            
        except Exception as e:
            st.error(f"Error fetching Twitter handles: {str(e)}")
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Add a section for data processing with a cleaner design
    st.markdown("<h2 class='sub-header'>Data Processing & Database Upload</h2>", unsafe_allow_html=True)
    
    # Create two columns for processing and upload
    process_col, upload_col = st.columns(2)
    
    with process_col:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<h3 style='color: #1DA1F2; margin-bottom: 15px;'>Process Tweets</h3>", unsafe_allow_html=True)
        
        # Check if we have data in session state
        if st.session_state.twitter_data is not None:
            st.markdown(f"""
            <div style="background-color: #E8F5E9; padding: 10px; border-radius: 5px; margin-bottom: 15px;">
                <p style="margin: 0; color: #2E7D32;"><strong>✅ Tweets available for processing</strong></p>
                <p style="margin-top: 5px; color: #2E7D32;">Ready to process {len(st.session_state.twitter_data)} tweets with LLM analysis</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Add OpenAI API key input
            openai_api_key = st.text_input(
                "OpenAI API Key", 
                type="password",
                help="Enter your OpenAI API key to use for LLM processing",
                placeholder="sk-..."
            )
            
            # Add a slider to select the number of parallel workers
            num_workers = st.slider(
                "Number of parallel workers", 
                min_value=1, 
                max_value=16, 
                value=8,  # Default to 8 workers
                help="Higher values may process faster but use more resources"
            )
            
            # Add a note about worker selection
            st.info(f"""
            💡 **Worker Configuration:**
            - Using {num_workers} parallel workers for processing
            - More workers = faster processing but higher resource usage
            - Recommended: 4-8 workers for most systems
            """)
            
            # Process button
            if st.button("🧠 Process with LLM", use_container_width=True, disabled=not openai_api_key):
                if not openai_api_key:
                    st.error("Please enter your OpenAI API key to continue.")
                else:
                    try:
                        # Check if we have data in session state
                        if 'twitter_data' not in st.session_state or st.session_state.twitter_data is None:
                            st.error("❌ No data in session state! Please run extraction first.")
                            return
                        
                        # Get the data
                        df = st.session_state.twitter_data
                        
                        # Create progress indicators
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        # Setup OpenAI client
                        import openai
                        client = openai.OpenAI(api_key=openai_api_key)
                        
                        # Get user-selected number of workers
                        num_workers = st.session_state.get("num_workers", 8)  # Default to 8 if not set
                        
                        # Define the process_tweet function - using EXACT same prompt as in twitter-llms-optimized.py
                        def process_tweet(tweet_data):
                            try:
                                # Extract tweet text
                                text = tweet_data.get('text', '')
                                if not text:
                                    text = tweet_data.get('fullText', '')
                                
                                # Process with OpenAI - USING THE EXACT SAME PROMPT AND MODEL
                                response = client.chat.completions.create(
                                    model="gpt-4o-mini",  # Use the same model
                                    messages=[
                                        {"role": "system", "content": """
                                        You are a financial market analyst. Analyze the given tweet and return your analysis in JSON format.
                                        Your response should be a valid JSON object with the following structure:
                                        {
                                            "time_horizon": "intraday/daily/weekly/short_term/medium_term/long_term/unknown",
                                            "trade_type": "trade_suggestion/analysis/news/general_discussion/unknown",
                                            "sentiment": "bullish/bearish/neutral"
                                        }

                                        Guidelines for classification:
                                        1. Time horizon:
                                           - intraday: within the same trading day or mentions "today"
                                           - daily: 1-5 trading days, mentions "tomorrow", "next day", or this week
                                           - weekly: 1-4 weeks, mentions "next week" or specific dates within a month
                                           - short_term: 1-3 months, mentions "next month", " next quarter"
                                           - medium_term: 3-6 months, mentions "quarter", or "Q1/Q2/Q3/Q4, "end of year"
                                           - long_term: >6 months, mentions "next year" or longer timeframes
                                           - unknown: if not specified
                                           
                                           IMPORTANT: Pay close attention to time-related words like "today", "tomorrow", "next week", etc.
                                           If the tweet mentions earnings or events happening "tomorrow", classify as "daily".
                                        
                                        2. Type of content:
                                           - trade_suggestion: specific entry/exit points or direct trade recommendations
                                           - analysis: market analysis, chart patterns, fundamentals
                                           - news: market news, company updates, economic data, earnings announcements
                                           - general_discussion: general market talk
                                           - unknown: if unclear
                                        
                                        3. Market sentiment:
                                           - bullish: positive outlook, expecting upward movement
                                           - bearish: negative outlook, expecting downward movement
                                           - neutral: balanced view or no clear direction
                                        """},
                                        {"role": "user", "content": text}
                                    ],
                                    response_format={"type": "json_object"}  # Ensure JSON response
                                )
                                
                                # Parse response
                                llm_response = response.choices[0].message.content
                                
                                # Extract tweet author data if available
                                author_info = tweet_data.get('author', {})
                                author_name = author_info.get('userName', '') if isinstance(author_info, dict) else ''
                                
                                # Create result record
                                result = {
                                    'id': tweet_data.get('id', ''),
                                    'text': text,
                                    'author': author_name,
                                    'created_at': tweet_data.get('createdAt', ''),
                                    'llm_response': llm_response,
                                }
                                
                                # Parse JSON response (should be valid JSON already due to response_format)
                                try:
                                    import json
                                    data = json.loads(llm_response)
                                    
                                    # Add structured fields
                                    result['time_horizon'] = data.get('time_horizon', 'unknown')
                                    result['trade_type'] = data.get('trade_type', 'unknown')
                                    result['sentiment'] = data.get('sentiment', 'neutral')
                                except Exception as json_err:
                                    print(f"Error parsing JSON: {json_err}")
                                
                                return result
                            except Exception as e:
                                return {
                                    'id': tweet_data.get('id', ''),
                                    'text': tweet_data.get('text', ''),
                                    'error': str(e)
                                }
                        
                        # Now process all tweets with parallel execution
                        import concurrent.futures
                        import time
                        
                        # Convert DataFrame to list of dictionaries for processing
                        tweets_to_process = df.to_dict('records')
                        total_tweets = len(tweets_to_process)
                        
                        status_text.text(f"Processing {total_tweets} tweets with {num_workers} workers...")
                        
                        start_time = time.time()
                        processed_tweets = []
                        completed = 0
                        
                        # Process in parallel
                        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
                            # Submit all tweets for processing
                            future_to_tweet = {executor.submit(process_tweet, tweet): tweet for tweet in tweets_to_process}
                            
                            # Process as they complete
                            for future in concurrent.futures.as_completed(future_to_tweet):
                                result = future.result()
                                processed_tweets.append(result)
                                
                                # Update progress
                                completed += 1
                                progress = completed / total_tweets
                                progress_bar.progress(progress)
                                elapsed = time.time() - start_time
                                tweets_per_second = completed / elapsed if elapsed > 0 else 0
                                remaining = (total_tweets - completed) / tweets_per_second if tweets_per_second > 0 else 0
                                
                                # Update status every few tweets to avoid UI slowdown
                                if completed % 5 == 0 or completed == total_tweets:
                                    status_text.text(
                                        f"Processed: {completed}/{total_tweets} tweets ({int(progress*100)}%) | "
                                        f"Speed: {tweets_per_second:.1f} tweets/sec | "
                                        f"Est. remaining: {int(remaining)} seconds"
                                    )
                        
                        # Create DataFrame from results
                        processed_df = pd.DataFrame(processed_tweets)
                        
                        # Store in session state
                        st.session_state.processed_data = processed_df
                        
                        # Clear progress indicators
                        progress_bar.empty()
                        
                        # Show success message
                        elapsed_time = time.time() - start_time
                        st.success(f"✅ Successfully processed {len(processed_df)} tweets in {elapsed_time:.1f} seconds!")
                        
                        # Show preview
                        st.write("### Processed Tweets Preview")
                        preview_columns = ['text', 'sentiment', 'time_horizon', 'trade_type']
                        available_cols = [col for col in preview_columns if col in processed_df.columns]
                        st.dataframe(processed_df[available_cols].head())
                        
                    except Exception as e:
                        st.error(f"Error in processing: {str(e)}")
                        st.error(traceback.format_exc())
        else:
            st.warning("⚠️ No tweets available. Please run tweet extraction first.")
    
    with upload_col:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<h3 style='color: #1DA1F2; margin-bottom: 15px;'>Upload to Database</h3>", unsafe_allow_html=True)
        
        # Check if we have processed data
        if st.session_state.processed_data is not None:
            st.markdown("""
            <div style="background-color: #E8F5E9; padding: 10px; border-radius: 5px; margin-bottom: 15px;">
                <p style="margin: 0; color: #2E7D32;"><strong>✅ Processed tweets available</strong></p>
                <p style="margin-top: 5px; color: #2E7D32;">Ready to filter and upload to database</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Upload button with improved styling and feedback
            if st.button("📤 Filter & Upload", use_container_width=True, type="primary"):
                try:
                    # Prepare tweets for filtering
                    with st.spinner("Preparing tweets for upload..."):
                        df = st.session_state.processed_data.copy()
                        
                        if len(df) == 0:
                            st.warning("No tweets available for processing.")
                            return
                        
                        # Create progress bar
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        status_text.text("Step 1/4: Preparing data...")
                        
                        # Add required columns if they don't exist
                        required_columns = {
                            'tweet_id': lambda x: float(x.name),
                            'author': lambda x: str(x.get('author', ''))[:100],
                            'text': lambda x: str(x.get('text', '')),
                            'created_at': lambda x: pd.Timestamp.now(),
                            'likes': lambda x: float(x.get('likes', 0)),
                            'retweets': lambda x: float(x.get('retweets', 0)),
                            'replies_count': lambda x: float(x.get('replies_count', 0)),
                            'views': lambda x: float(x.get('views', 0)),
                            'author_followers': lambda x: float(x.get('author_followers', 0)),
                            'author_following': lambda x: float(x.get('author_following', 0)),
                            'sentiment': lambda x: x.get('sentiment', 'bullish'),
                            'trade_type': lambda x: str(x.get('trade_type', 'unknown')),
                            'time_horizon': lambda x: str(x.get('time_horizon', 'unknown')),
                            'prediction_date': lambda x: pd.Timestamp.now(),
                            'tickers_mentioned': lambda x: ','.join([word.strip('$') for word in str(x.get('text', '')).split() if word.startswith('$')]),
                            'conversation_id': lambda x: float(x.name),
                            'tweet_type': 'parent',
                            'reply_to_tweet_id': lambda x: float(0),
                            'reply_to_user': lambda x: str(x.get('reply_to_user', ''))[:100],
                            'author_verified': False,
                            'author_blue_verified': False,
                            'created_date': lambda x: pd.Timestamp.now().date(),
                            'has_ticker': True,
                            'validated_ticker': lambda x: str(x.get('validated_ticker', ''))[:50]
                        }
                        
                        # Update progress
                        progress_bar.progress(25)
                        status_text.text("Step 2/4: Filtering actionable tweets...")
                        
                        # Add tweet_type column if it doesn't exist
                        if 'tweet_type' not in df.columns:
                            df['tweet_type'] = 'parent'  # Default all to parent tweets
                        
                        # Add tickers_mentioned if it doesn't exist
                        if 'tickers_mentioned' not in df.columns:
                            # Extract tickers from text
                            df['tickers_mentioned'] = df['text'].apply(
                                lambda x: ','.join([word.strip('$') for word in str(x).split() if word.startswith('$')])
                            )
                        
                        # Add sentiment if it doesn't exist
                        if 'sentiment' not in df.columns:
                            df['sentiment'] = 'unknown'  # Default sentiment
                        
                        # Now filter with safety checks
                        try:
                            # Filter tweets to only include actionable ones
                            actionable_tweets = df[
                                (df['tweet_type'] == 'parent') &
                                (df['tickers_mentioned'].notna()) & 
                                (df['tickers_mentioned'] != '') &
                                (df['sentiment'].isin(['bullish', 'bearish']))
                            ].copy()
                        except KeyError as e:
                            st.error(f"Missing column in data: {str(e)}")
                            st.info("Adding required columns and continuing...")
                            
                            # Add all required columns
                            for col in ['tweet_type', 'tickers_mentioned', 'sentiment']:
                                if col not in df.columns:
                                    if col == 'tweet_type':
                                        df[col] = 'parent'
                                    elif col == 'tickers_mentioned':
                                        df[col] = df['text'].apply(
                                            lambda x: ','.join([word.strip('$') for word in str(x).split() if word.startswith('$')])
                                        )
                                    elif col == 'sentiment':
                                        df[col] = 'unknown'
                            
                            # Try filtering again
                            actionable_tweets = df[
                                (df['tweet_type'] == 'parent') &
                                (df['tickers_mentioned'].notna()) & 
                                (df['tickers_mentioned'] != '') &
                                (df['sentiment'].isin(['bullish', 'bearish']))
                            ].copy()
                        
                        if len(actionable_tweets) == 0:
                            st.warning("No actionable tweets found after filtering.")
                            return
                        
                        # Add missing columns
                        actionable_tweets = actionable_tweets.assign(
                            prediction_correct=None,
                            start_price=None,
                            end_price=None,
                            start_date=None,
                            end_date=None,
                            company_names=None,
                            stocks=None,
                            price_change_pct=None,
                            actual_return=None,
                            prediction_score=None
                        )
                        
                        # Update progress
                        progress_bar.progress(50)
                        status_text.text("Step 3/4: Preparing data for upload...")
                        
                        # Display summary instead of full preview
                        st.info(f"Found {len(actionable_tweets)} actionable tweets to upload")
                        
                        # Show sentiment distribution
                        sentiment_counts = actionable_tweets['sentiment'].value_counts().to_dict()
                        sentiment_html = "<div style='margin: 10px 0;'><strong>Sentiment distribution:</strong><br>"
                        for sentiment, count in sentiment_counts.items():
                            color = "#4CAF50" if sentiment == "bullish" else "#F44336"
                            sentiment_html += f"<span style='color:{color};'>{sentiment}</span>: {count} tweets<br>"
                        sentiment_html += "</div>"
                        st.markdown(sentiment_html, unsafe_allow_html=True)
                        
                        # Update progress
                        progress_bar.progress(75)
                        status_text.text("Step 4/4: Uploading to database...")
                        
                        # Upload to database
                        try:
                            # Capture the output from upload_to_database for debugging
                            import io
                            import sys
                            original_stdout = sys.stdout
                            debug_output = io.StringIO()
                            sys.stdout = debug_output
                            
                            # Try the upload
                            success = upload_to_database(actionable_tweets)
                            
                            # Restore stdout
                            sys.stdout = original_stdout
                            
                            # Get the debug output
                            upload_logs = debug_output.getvalue()
                            
                            # Update progress
                            progress_bar.progress(100)
                            
                            # Check logs for success message even if function returns False
                            if success or "Database upload complete" in upload_logs:
                                # Show success message with animation
                                st.balloons()
                                st.success(f"✅ Successfully uploaded {len(actionable_tweets)} tweets to database!")
                                
                                # Show details in an expandable section
                                with st.expander("View Upload Details"):
                                    st.write(f"Tweet types: {actionable_tweets['tweet_type'].value_counts().to_dict()}")
                                    st.write(f"Time horizons: {actionable_tweets['time_horizon'].value_counts().to_dict()}")
                                    st.write(f"Trade types: {actionable_tweets['trade_type'].value_counts().to_dict()}")
                            else:
                                st.error("❌ Error during database upload. Check the logs for details.")
                                with st.expander("View Upload Logs"):
                                    st.code(upload_logs)
                        except Exception as e:
                            st.error(f"Error during upload: {str(e)}")
                            with st.expander("View Error Details"):
                                st.code(traceback.format_exc())
                        
                except Exception as e:
                    st.error(f"Error during upload: {str(e)}")
                    with st.expander("View Error Details"):
                        st.code(traceback.format_exc())
        else:
            st.warning("⚠️ No processed tweets available. Please run LLM processing first.")
    
    st.markdown("</div>", unsafe_allow_html=True)

# Function to run the prediction process
def run_prediction_process(df, output_dir='results'):
    """Run the full prediction process on the given dataframe"""
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Step 1: Filter actionable tweets
        print("Filtering actionable tweets...")
        filtered_data = filter_actionable_tweets(df)
        actionable_tweets = filtered_data['actionable_tweets']
        analysis_tweets = filtered_data['analysis_tweets']
        
        # Save filtered datasets
        actionable_tweets.to_csv(f'{output_dir}/actionable_tweets.csv', index=False)
        analysis_tweets.to_csv(f'{output_dir}/analysis_tweets.csv', index=False)
        
        # Step 2: Add market validation columns
        print("Adding market validation...")
        validated_tweets = add_market_validation_columns(actionable_tweets, all_tweets=df, output_dir=output_dir)
        validated_tweets.to_csv(f'{output_dir}/validated_tweets.csv', index=False)
        
        # Step 3: Analyze user accuracy
        print("Analyzing user accuracy...")
        user_accuracy = analyze_user_accuracy(validated_tweets)
        user_accuracy.to_csv(f'{output_dir}/user_accuracy.csv', index=False)
        
        # Step 4: Upload to database
        print("Uploading to database...")
        upload_result = upload_to_database(validated_tweets)
        
        return {
            'actionable_tweets': actionable_tweets,
            'validated_tweets': validated_tweets,
            'user_accuracy': user_accuracy,
            'upload_result': upload_result
        }
    
    except Exception as e:
        print(f"Error in prediction process: {e}")
        import traceback
        traceback.print_exc()
        return None

# Main app
def main():
    # DO NOT set page config here - it's already done at the top
    
    # Check password if required
    if st.session_state.get('password_required', True):
        if not check_password():
            return
    
    # Display title
    st.title("Twitter Trader Analysis Dashboard")
    
    # Add a sidebar with navigation and database tools
    with st.sidebar:
        st.title("Navigation")
        
        # Note about data
        st.info("Data is based on validated stock predictions from Twitter conversations.")
        
        # Add database management options for admins
        if st.session_state.get('password_correct', False):
            with st.expander("Database Management", expanded=False):
                cols = st.columns(2)
                
                with cols[0]:
                    if st.button("Reset Database"):
                        try:
                            import subprocess
                            result = subprocess.run(["python", "reset_database.py"], 
                                                  capture_output=True, text=True)
                            st.success("Database reset successful!")
                            st.code(result.stdout)
                            # Clear cache to reload data
                            st.cache_data.clear()
                        except Exception as e:
                            st.error(f"Error resetting database: {e}")
                
                with cols[1]:
                    if st.button("Reload Data"):
                        st.cache_data.clear()
                        st.success("Cache cleared! Data will reload on next access.")
        
        # Page selection
        page = st.radio("Select a page", [
            "Dashboard",
            "Trader Profile",
            "Raw Data",
            "Data Extraction"
        ])
        
        # Trader selection only for Trader Profile page
        selected_trader = None
        if page == "Trader Profile":
            # Try to load data first to get trader list
            try:
                df = load_data_from_db()
                if not df.empty:
                    traders = get_traders(df)
                    selected_trader = st.selectbox("Select Trader", traders)
            except Exception as e:
                st.error(f"Error loading traders: {e}")
    
    # Load data based on the selected page
    df = None
    if page in ["Dashboard", "Trader Profile", "Raw Data"]:
        df = load_data_from_db()
        
        # Add a data status indicator
        record_count = len(df) if df is not None else 0
        st.sidebar.info(f"Database contains {record_count:,} records")
    
    # Display the selected page
    if page == "Dashboard":
        create_overview_dashboard(df)
    elif page == "Trader Profile" and selected_trader:
        create_trader_profile(df, selected_trader)
    elif page == "Raw Data":
        create_raw_data_dashboard(df)
    elif page == "Data Extraction":
        create_data_extraction_dashboard()

# Run the app
if __name__ == "__main__":
    main()

def enrich_with_context(df):
    """
    Enriches tweets with context code decoding (e.g., "China" → $KTEC, $FXI for JediAnt)
    """
    if df is None or df.empty:
        return df
    
    # Create a copy to avoid modifying the original
    df = df.copy()
    
    # Define mappings for code words by user
    code_word_mappings = {
        'JediAnt': {
            'china': ['KTEC', 'FXI'],
            'China': ['KTEC', 'FXI'],
            # Add more code words as you discover them
        },
        # You can add more users with their specific code words
    }
    
    # Column to track added tickers
    if 'enriched_tickers' not in df.columns:
        df['enriched_tickers'] = ''
    
    # Process each row
    enriched_count = 0
    for idx, row in df.iterrows():
        author = row.get('author', '')
        text = row.get('text', '') if pd.notna(row.get('text', '')) else ''
        
        # Skip if no text or not an author we're tracking
        if not text or author not in code_word_mappings:
            continue
        
        # Get the code word mapping for this author
        mappings = code_word_mappings[author]
        
        # Check for each code word
        enriched_tickers = set()
        for code_word, tickers in mappings.items():
            if code_word in text:
                # Add these tickers to the enriched set
                enriched_tickers.update(tickers)
        
        # If we found code words, update the tickers_mentioned column
        if enriched_tickers:
            enriched_count += 1
            
            # Store the enriched tickers in a separate column
            df.at[idx, 'enriched_tickers'] = ','.join(enriched_tickers)
            
            # Get current tickers
            ticker_col = 'tickers_mentioned' if 'tickers_mentioned' in df.columns else 'tickers'
            current_tickers = set()
            
            if pd.notna(row.get(ticker_col, '')) and row.get(ticker_col, ''):
                current_tickers = set(row[ticker_col].split(','))
            
            # Combine with enriched tickers
            all_tickers = current_tickers.union(enriched_tickers)
            
            # Update the tickers column
            df.at[idx, ticker_col] = ','.join(all_tickers)
    
    return df

def analyze_tweets(df_user):
    """
    Analyze the tweets of a trader
    """
    if df_user.empty:
        return pd.DataFrame(), pd.DataFrame(), {}
    
    # Apply enrichment if JediAnt
    if 'author' in df_user.columns and df_user['author'].iloc[0] == 'JediAnt':
        df_user = enrich_with_context(df_user)
    
    # Get ticker column
    ticker_col = 'tickers_mentioned' if 'tickers_mentioned' in df_user.columns else 'tickers'
    
    # Only analyze tweets with tickers
    ticker_tweets = df_user[df_user[ticker_col].notna() & (df_user[ticker_col] != '')].copy()
    
    # Count mentions per ticker
    all_tickers = []
    
    for tickers in ticker_tweets[ticker_col]:
        if pd.notna(tickers) and tickers:
            all_tickers.extend(tickers.split(','))
    
    if not all_tickers:
        ticker_counts = pd.DataFrame(columns=['ticker', 'count'])
    else:
        ticker_counts = pd.DataFrame(all_tickers, columns=['ticker'])
        ticker_counts = ticker_counts['ticker'].value_counts().reset_index()
        ticker_counts.columns = ['ticker', 'count']
    
    # Calculate sentiment distribution by ticker
    sentiment_by_ticker = {}
    
    for ticker in ticker_counts['ticker']:
        # Find tweets mentioning this ticker
        ticker_mask = ticker_tweets[ticker_col].apply(lambda x: ticker in x.split(',') if pd.notna(x) else False)
        ticker_subset = ticker_tweets[ticker_mask]
        
        # Count sentiments for this ticker
        sentiment_counts = ticker_subset['sentiment'].value_counts()
        total = len(ticker_subset)
        
        sentiment_by_ticker[ticker] = {
            'bullish': sentiment_counts.get('bullish', 0) / total * 100 if total > 0 else 0,
            'bearish': sentiment_counts.get('bearish', 0) / total * 100 if total > 0 else 0,
            'neutral': sentiment_counts.get('neutral', 0) / total * 100 if total > 0 else 0,
            'total_mentions': total
        }
    
    # Check for contextual enrichment
    enriched_tickers = []
    if 'enriched_tickers' in df_user.columns:
        for tickers in df_user['enriched_tickers'].dropna():
            if tickers:
                enriched_tickers.extend(tickers.split(','))
        
        # Add note about contextually enriched tickers
        if enriched_tickers:
            st.info(f"Contextual mapping applied: found {len(enriched_tickers)} enriched ticker references")
            # Show the enriched tickers
            st.write("Contextually enriched tickers: " + ", ".join([f"${t}" for t in set(enriched_tickers)]))
    
    return ticker_counts, ticker_tweets, sentiment_by_ticker
