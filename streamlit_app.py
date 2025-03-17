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
warnings.filterwarnings('ignore')

# Set page configuration - this must be the first Streamlit command
st.set_page_config(
    page_title="Twitter Trader Analysis",
    page_icon="📊",
    layout="wide",  # Changed from "centered" to "wide" for dashboard views
    initial_sidebar_state="expanded"  # Changed to expanded for better dashboard navigation
)

# Function to check login credentials
def check_password():
    """Returns `True` if the user had the correct password."""

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if st.session_state["username"] == "TwitterProfiling" and st.session_state["password"] == "Twitter135":
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # Don't store password
            del st.session_state["username"]  # Don't store username
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        # For login page, override the layout to be centered
        st.markdown("""
        <style>
        /* Center the login form by limiting width */
        .main .block-container {
            max-width: 400px;
            margin-left: auto;
            margin-right: auto;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Custom CSS to create a clean login page
        st.markdown("""
        <style>
        /* Hide Streamlit elements */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        
        /* Make all containers match the background color */
        .stApp > div, 
        .stApp > header,
        .stApp [data-testid="stDecoration"], 
        .stApp [data-testid="stToolbar"],
        .stApp [data-testid="stAppViewBlockContainer"] > div:first-child,
        .stApp [data-testid="stAppViewBlockContainer"] > div,
        .stApp [data-testid="stVerticalBlock"] > div,
        .stApp [data-testid="stHorizontalBlock"] > div,
        .stApp [data-testid="element-container"] > div,
        .stApp [data-testid="stMarkdown"] > div {
            background-color: #f0f2f5 !important;
            border: none !important;
            box-shadow: none !important;
        }
        
        /* Page background */
        .stApp {
            background-color: #f0f2f5;
        }
        
        /* Remove padding */
        .main .block-container {
            padding-top: 0 !important;
            padding-bottom: 2rem;
            max-width: 400px;
            margin-top: 0 !important;
        }
        
        /* Login container */
        .login-container {
            background-color: white;
            border-radius: 12px;
            padding: 2rem;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
            text-align: center;
            margin-top: 2rem;
        }
        
        /* Logo */
        .logo-container {
            margin-bottom: 1.5rem;
            display: flex;
            justify-content: center;
            background-color: #f0f2f5; /* Match the background */
            border-radius: 50%;
            padding: 10px;
        }
        
        .logo {
            width: 60px;
            height: 60px;
            background-color: #1DA1F2;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        
        /* Title */
        .login-title {
            font-size: 24px;
            font-weight: 700;
            margin-bottom: 0.5rem;
            color: #14171A;
        }
        
        /* Subtitle */
        .login-subtitle {
            font-size: 14px;
            color: #657786;
            margin-bottom: 1.5rem;
        }
        
        /* Form fields */
        .form-label {
            font-size: 14px;
            font-weight: 600;
            color: #14171A;
            text-align: left;
            display: block;
            margin-bottom: 0.5rem;
        }
        
        /* Input styling */
        .stTextInput > div {
            margin-bottom: 1rem;
        }
        
        .stTextInput > div > div > input {
            border-radius: 8px;
            border: 1px solid #E1E8ED;
            padding: 0.75rem 1rem;
            font-size: 14px;
        }
        
        /* Button styling */
        .stButton > button {
            background-color: #1DA1F2;
            color: white;
            border: none;
            border-radius: 50px;
            padding: 0.75rem 1rem;
            font-size: 16px;
            font-weight: 600;
            width: 100%;
            transition: background-color 0.2s;
        }
        
        .stButton > button:hover {
            background-color: #1a91da;
        }
        
        /* Forgot password */
        .forgot-password {
            display: block;
            text-align: center;
            color: #1DA1F2;
            font-size: 14px;
            margin-top: 1rem;
            text-decoration: none;
        }
        
        /* Terms text */
        .terms-text {
            font-size: 12px;
            color: #657786;
            margin-top: 1.5rem;
            text-align: center;
        }
        
        /* Footer */
        .footer-text {
            font-size: 12px;
            color: #657786;
            margin-top: 2rem;
            text-align: center;
        }
        
        /* Links */
        a {
            color: #1DA1F2;
            text-decoration: none;
        }
        
        a:hover {
            text-decoration: underline;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Login container
        st.markdown('<div class="login-container">', unsafe_allow_html=True)
        
        # Logo
        st.markdown("""
        <div class="logo-container">
            <div class="logo">
                <svg width="30" height="24" viewBox="0 0 24 20" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <path d="M23.643 2.937C22.808 3.307 21.911 3.557 20.968 3.67C21.941 3.08 22.669 2.169 23.016 1.092C22.116 1.628 21.119 2.01 20.058 2.222C19.208 1.319 17.998 0.75 16.658 0.75C14.086 0.75 12 2.836 12 5.408C12 5.776 12.042 6.132 12.12 6.472C8.247 6.277 4.828 4.422 2.529 1.642C2.122 2.339 1.891 3.14 1.891 3.992C1.891 5.595 2.712 7.012 3.96 7.842C3.196 7.82 2.474 7.6 1.84 7.241C1.84 7.26 1.84 7.28 1.84 7.3C1.84 9.558 3.456 11.441 5.592 11.873C5.2 11.98 4.786 12.034 4.36 12.034C4.06 12.034 3.766 12.005 3.484 11.95C4.076 13.801 5.791 15.147 7.832 15.183C6.236 16.433 4.241 17.178 2.075 17.178C1.692 17.178 1.313 17.156 0.942 17.113C3.003 18.436 5.425 19.206 8.022 19.206C16.647 19.206 21.332 12.108 21.332 5.952C21.332 5.752 21.327 5.552 21.318 5.353C22.228 4.682 23.018 3.85 23.641 2.94L23.643 2.937Z" fill="white"/>
                </svg>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Title and subtitle
        st.markdown('<h1 class="login-title">Twitter Trader Analysis</h1>', unsafe_allow_html=True)
        st.markdown('<p class="login-subtitle">Enter your credentials to access the dashboard</p>', unsafe_allow_html=True)
        
        # Username field
        st.markdown('<label class="form-label">Username</label>', unsafe_allow_html=True)
        username = st.text_input("", key="username", placeholder="Enter username")
        
        # Password field
        st.markdown('<label class="form-label">Password</label>', unsafe_allow_html=True)
        password = st.text_input("", key="password", type="password", placeholder="Enter password")
        
        # Login button
        st.button("Sign In", on_click=password_entered)
        
        # Forgot password link
        st.markdown('<a href="#" class="forgot-password">Forgot password?</a>', unsafe_allow_html=True)
        
        # Terms text
        st.markdown("""
        <p class="terms-text">
            By signing in, you agree to our <a href="#">Terms of Service</a> and <a href="#">Privacy Policy</a>
        </p>
        """, unsafe_allow_html=True)
        
        # Footer
        st.markdown('<p class="footer-text">Twitter Trader Analysis v1.0.0 © 2023</p>', unsafe_allow_html=True)
        
        # Close container
        st.markdown('</div>', unsafe_allow_html=True)
        
        return False
    
    return st.session_state["password_correct"]

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.8rem;
        color: #1DA1F2;
        font-weight: 800;
        margin-bottom: 1.5rem;
        text-align: center;
        padding-bottom: 1rem;
        border-bottom: 2px solid #F5F8FA;
    }
    .sub-header {
        font-size: 1.8rem;
        color: #14171A;
        font-weight: 600;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        padding-left: 0.5rem;
        border-left: 4px solid #1DA1F2;
    }
    .card {
        border-radius: 15px;
        padding: 1.8rem;
        background-color: white;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.08);
        margin-bottom: 1.5rem;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    .card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 15px rgba(0, 0, 0, 0.1);
    }
    .metric-value {
        font-size: 2.5rem;
        font-weight: 800;
        color: #1DA1F2;
        text-align: center;
    }
    .metric-label {
        font-size: 1.1rem;
        color: #657786;
        text-align: center;
        margin-top: 0.5rem;
    }
    .positive {
        color: #17BF63;
    }
    .negative {
        color: #E0245E;
    }
    .neutral {
        color: #AAB8C2;
    }
    .highlight {
        background-color: #F5F8FA;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1DA1F2;
        margin-bottom: 1rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
        border-bottom: 2px solid #F5F8FA;
    }
    .stTabs [data-baseweb="tab"] {
        height: 4rem;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 4px 4px 0px 0px;
        gap: 1rem;
        padding-top: 0.5rem;
        padding-bottom: 0.5rem;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: #F5F8FA;
        border-bottom: 3px solid #1DA1F2;
    }
    .stDataFrame {
        border-radius: 10px;
        overflow: hidden;
    }
    .stDataFrame [data-testid="stTable"] {
        border-radius: 10px;
    }
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #F5F8FA;
    }
    .css-1v3fvcr {
        background-color: #FFFFFF;
    }
    /* Make the app background slightly off-white for better contrast */
    .main .block-container {
        background-color: #FAFAFA;
        padding: 2rem;
        border-radius: 15px;
    }
    /* Style the plotly charts */
    .js-plotly-plot {
        border-radius: 10px;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.05);
        background-color: white;
        padding: 1rem;
    }
    /* Style the selectbox */
    .stSelectbox > div > div > div {
        background-color: white;
        border-radius: 30px;
        padding: 0.2rem 1rem;
        border: 1px solid #AAB8C2;
    }
    /* Style the radio buttons */
    .stRadio > div {
        background-color: white;
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.05);
    }
    .metric-note {
        font-size: 0.8rem;
        color: #657786;
        text-align: center;
        margin-top: 0.2rem;
    }
</style>
""", unsafe_allow_html=True)

# Function to load data from database
@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_data_from_db():
    """
    Load data from the PostgreSQL database using direct psycopg2 connection
    similar to the approach in db_connection_test.py
    """
    try:
        # Database connection parameters
        host = "database-twitter.cdh6c6zr5mbr.us-east-1.rds.amazonaws.com"
        database = "postgres"
        user = "postgres"
        password = "DrorMai531"
        
        # Connect to the database
        conn = psycopg2.connect(
            host=host,
            database=database,
            user=user,
            password=password,
            connect_timeout=10
        )
        
        # Create a cursor
        cursor = conn.cursor()
        
        # Get column names from the database
        cursor.execute("""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name = 'tweets'
            ORDER BY ordinal_position;
        """)
        columns = [row[0] for row in cursor.fetchall()]
        
        # Get total count
        cursor.execute("SELECT COUNT(*) FROM tweets")
        total_count = cursor.fetchone()[0]
        print(f"Total tweets in database: {total_count}")
        
        # Fetch all data from the tweets table
        cursor.execute("""
            SELECT * FROM tweets
        """)
        
        # Fetch all rows
        rows = cursor.fetchall()
        
        # Create DataFrame
        df = pd.DataFrame(rows, columns=columns)
        
        # Close cursor and connection
        cursor.close()
        conn.close()
        
        print(f"Fetched {len(df)} rows from database")
        print(f"Processed dataframe has {len(df)} rows and columns: {df.columns.tolist()}")
        
        return df
        
    except Exception as e:
        st.error(f"Database connection error: {str(e)}")
        print(f"Error details: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()

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

# Function to filter data for a specific trader with better error handling
def filter_trader_data(df, trader_name, show_warnings=True):
    """
    Filter data for a specific trader.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The full dataset
    trader_name : str
        Name of the trader to filter for
    show_warnings : bool, default=True
        Whether to display warning messages for missing data
    
    Returns:
    --------
    df_user : pandas.DataFrame
        DataFrame with all tweets by this trader
    df_parent : pandas.DataFrame
        DataFrame with only parent tweets by this trader
    """
    df_user = df[df['author'] == trader_name].copy()
    
    if len(df_user) == 0:
        if show_warnings:
            st.error(f"No data found for trader: {trader_name}")
        return pd.DataFrame(), pd.DataFrame()
    
    conv_starter = df_user[df_user['tweet_type'] == 'parent']['conversation_id']
    
    if len(conv_starter) == 0:
        if show_warnings:
            st.warning(f"No parent tweets found for trader: {trader_name}")
        return df_user, pd.DataFrame()
    
    df_user = df_user.loc[df_user['conversation_id'].isin(conv_starter)]
    df_parent = df_user[df_user['tweet_type'] == 'parent']
    
    # Ensure prediction_correct is Boolean
    df_user['prediction_correct'] = (
        df_user['prediction_correct']
        .astype(str)
        .str.lower()
        .map({'true': True, 'false': False})
    )
    
    # Only calculate sentiment consistency if we have parent tweets
    if len(df_parent) > 0:
        # Calculate sentiment consistency per conversation
        parent_sentiment = df_user[df_user['tweet_type'] == 'parent'][['conversation_id', 'sentiment']].rename(
            columns={'sentiment': 'parent_sentiment'}
        )
        df_user = df_user.merge(parent_sentiment, on='conversation_id', how='left')
        df_user['consistent_sentiment'] = (df_user['sentiment'] == df_user['parent_sentiment']).astype(int)
        
        # Compute Weighted Profitability Score per conversation
        if 'prediction_score' in df_user.columns:
            prediction_score_sum = (
                df_user.groupby('conversation_id')['prediction_score']
                .sum()
                .reset_index(name='Weighted Profitability Score')
            )
            df_user = df_user.merge(prediction_score_sum, on='conversation_id', how='left')
    
    return df_user, df_parent

# Function to compute trader profile summary
def compute_profile_summary(df_user, df_parent):
    # Handle empty dataframes
    if df_user.empty or df_parent.empty:
        # Return a default profile summary with N/A values
        return {
            "Total Tweets": 0 if df_user.empty else len(df_user),
            "Total Conversations": 0 if df_user.empty else df_user['conversation_id'].nunique(),
            "Verified Account": "Unknown",
            "Followers": 0,
            "Following": 0,
            "Most Frequent Time Horizon": "N/A",
            "Most Frequent Trade Type": "N/A",
            "Most Frequent Sentiment": "N/A",
            "Most Mentioned Stock": "N/A",
            "Average Price Change (%)": 0,
            "Average Actual Return (%)": 0,
            "Avg Likes per Tweet": 0,
            "Avg Retweets per Tweet": 0,
            "Avg Replies per Tweet": 0,
            "Avg Views per Tweet": 0,
            "Prediction Accuracy (%)": "N/A",
            "Sentiment Consistency per Conversation (%)": 0,
            "Weighted Profitability Mean": 0,
        }
    
    # Calculate sentiment consistency if the column exists
    if 'consistent_sentiment' in df_user.columns:
        consistency_by_conv = df_user.groupby('conversation_id')['consistent_sentiment'].mean()
        consistency_mean = consistency_by_conv.mean() * 100
    else:
        consistency_mean = 0
    
    # Calculate weighted profitability if the column exists
    if 'Weighted Profitability Score' in df_user.columns:
        weighted_prof_mean = df_user.drop_duplicates(subset='conversation_id')[
            'Weighted Profitability Score'
        ].mean()
    else:
        weighted_prof_mean = 0
    
    profile_summary = {
        "Total Tweets": len(df_user),
        "Total Conversations": df_user['conversation_id'].nunique(),
        "Verified Account": (
            df_user['author_blue_verified'].mode()[0]
            if not df_user['author_blue_verified'].isna().all()
            else "Unknown"
        ),
        "Followers": df_user['author_followers'].max(),
        "Following": df_user['author_following'].max(),
        "Most Frequent Time Horizon": (
            df_parent['time_horizon'].mode()[0]
            if not df_parent['time_horizon'].isna().all() else "Unknown"
        ),
        "Most Frequent Trade Type": (
            df_parent['trade_type'].mode()[0]
            if not df_parent['trade_type'].isna().all() else "Unknown"
        ),
        "Most Frequent Sentiment": (
            df_parent['sentiment'].mode()[0]
            if not df_parent['sentiment'].isna().all() else "Unknown"
        ),
        "Most Mentioned Stock": (
            df_parent['validated_ticker'].mode()[0]
            if not df_parent['validated_ticker'].isna().all() else "Unknown"
        ),
        "Average Price Change (%)": df_parent['price_change_pct'].mean(),
        "Average Actual Return (%)": df_parent['actual_return'].mean(),
        "Avg Likes per Tweet": df_user['likes'].mean(),
        "Avg Retweets per Tweet": df_user['retweets'].mean(),
        "Avg Replies per Tweet": df_user['replies_count'].mean(),
        "Avg Views per Tweet": df_user['views'].mean(),
        "Prediction Accuracy (%)": (
            df_parent['prediction_correct'].mean() * 100
            if not df_parent['prediction_correct'].isna().all() else "N/A"
        ),
        "Sentiment Consistency per Conversation (%)": consistency_mean,
        "Weighted Profitability Mean": weighted_prof_mean,
    }
    return profile_summary

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
    st.markdown("<h1 class='main-header'>Twitter Trader Analysis Dashboard</h1>", unsafe_allow_html=True)
    
    # Get actionable tweets from traders with at least 3 parent tweets
    valid_traders = []
    invalid_traders = []
    
    # Find traders with at least 3 actionable parent tweets
    for trader in get_traders(df):
        # Count actionable parent tweets for this trader
        count = len(df[(df['author'] == trader) & 
                       (df['tweet_type'] == 'parent') & 
                       (df['sentiment'].isin(['bullish', 'bearish'])) &
                       (df['time_horizon'].notna() & (df['time_horizon'] != ''))])
        
        if count >= 3:
            valid_traders.append(trader)
        else:
            invalid_traders.append(trader)
    
    # Filter dataframe to only include valid traders
    filtered_df = df[df['author'].isin(valid_traders)]
    
    # Get parent tweets for statistics
    parent_tweets = filtered_df[filtered_df['tweet_type'] == 'parent']
    
    # Get metrics
    unique_traders = len(valid_traders)
    total_tweets = len(filtered_df)
    
    # Calculate accuracy and return only from parent tweets of valid traders 
    # with non-null prediction_correct values
    accuracy_df = parent_tweets[parent_tweets['prediction_correct'].notna()]
    overall_accuracy = accuracy_df['prediction_correct'].mean() * 100 if len(accuracy_df) > 0 else 0
    
    # Calculate average return from parent tweets
    avg_return = parent_tweets['actual_return'].mean() if 'actual_return' in parent_tweets.columns else 0
    
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value neutral">{unique_traders}</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Unique Traders</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value neutral">{total_tweets:,}</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Total Tweets</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        accuracy_class = "positive" if overall_accuracy > 50 else "negative"
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value {accuracy_class}">{overall_accuracy:.1f}%</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Overall Accuracy</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        return_class = "positive" if avg_return > 0 else "negative"
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value {return_class}">{avg_return:.2f}%</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Average Return (per prediction)</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-note">Based on actual price movement during prediction period</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Optional: Add an expandable section with filtered traders info instead of warnings
    with st.expander("Show traders with insufficient data"):
        st.info(f"{len(invalid_traders)} traders were filtered out due to having fewer than 3 actionable tweets.")
        if invalid_traders:
            st.write("Traders excluded from analysis:")
            chunks = [invalid_traders[i:i+5] for i in range(0, len(invalid_traders), 5)]
            for chunk in chunks:
                st.write(", ".join(chunk))
    
    # Analyze all traders (this already filters to traders with 3+ tweets)
    trader_metrics = analyze_all_traders(df)
    
    # Top traders section
    st.markdown('<div class="sub-header">Top Traders by Accuracy</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Top 10 traders by accuracy
        top_traders = trader_metrics.sort_values('accuracy', ascending=False).head(10)
        
        fig = px.bar(
            top_traders,
            x='trader',
            y='accuracy',
            color='avg_return',
            color_continuous_scale='RdYlGn',
            labels={'trader': 'Trader', 'accuracy': 'Prediction Accuracy (%)', 'avg_return': 'Avg Return (%)'},
            title='Top 10 Traders by Prediction Accuracy'
        )
        
        fig.update_layout(
            xaxis_title='Trader',
            yaxis_title='Accuracy (%)',
            coloraxis_colorbar_title='Avg Return (%)',
            height=500,
            hovermode='x unified'
        )
        
        # Add a horizontal line at 50% accuracy
        fig.add_shape(
            type='line',
            x0=-0.5,
            x1=9.5,
            y0=50,
            y1=50,
            line=dict(color='red', width=2, dash='dash')
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Top traders table
        st.markdown('<div class="highlight">', unsafe_allow_html=True)
        st.markdown("### Trader Leaderboard")
        st.markdown("Top traders ranked by prediction accuracy with minimum 3 predictions")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Format the table
        display_df = top_traders[['trader', 'accuracy', 'avg_return', 'total_tweets', 'followers']].copy()
        display_df.columns = ['Trader', 'Accuracy (%)', 'Avg Return (%)', 'Tweets', 'Followers']
        display_df['Accuracy (%)'] = display_df['Accuracy (%)'].round(1)
        display_df['Avg Return (%)'] = display_df['Avg Return (%)'].round(2)
        display_df['Followers'] = display_df['Followers'].apply(lambda x: f"{x:,}")
        
        st.dataframe(display_df, use_container_width=True, height=400)
    
    # Sentiment and stock analysis
    st.markdown('<div class="sub-header">Sentiment & Stock Analysis</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Sentiment distribution
        all_parent_tweets = filtered_df[filtered_df['tweet_type'] == 'parent']
        sentiment_counts = all_parent_tweets['sentiment'].value_counts()
        sentiment_pcts = sentiment_counts / sentiment_counts.sum() * 100

        # Create a color map that includes neutral
        sentiment_colors = {
            'bullish': '#17BF63',   # Green
            'bearish': '#E0245E',   # Red
            'neutral': '#AAB8C2',   # Grey
            'unknown': '#F5F8FA'    # Light grey
        }

        # Create the pie chart with all sentiment values
        fig = px.pie(
            values=sentiment_pcts,
            names=sentiment_pcts.index,
            title="Overall Sentiment Distribution",
            color=sentiment_pcts.index,
            color_discrete_map=sentiment_colors,
            hole=0.4
        )

        # Update layout
        fig.update_layout(
            legend_title="Sentiment",
            margin=dict(t=30, b=0, l=0, r=0),
            height=400,
            font=dict(size=14)
        )

        # Add percentage labels
        fig.update_traces(
            textposition='inside',
            textinfo='percent+label',
            insidetextfont=dict(color='white')
        )

        # Display the chart
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Top stocks by mention
        stock_counts = df['validated_ticker'].value_counts().nlargest(10).reset_index()
        stock_counts.columns = ['Stock', 'Count']
        
        fig = px.bar(
            stock_counts,
            x='Count',
            y='Stock',
            orientation='h',
            color='Count',
            color_continuous_scale='Viridis',
            title='Top 10 Most Mentioned Stocks'
        )
        
        fig.update_layout(height=400, yaxis={'categoryorder': 'total ascending'})
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Time series analysis
    st.markdown('<div class="sub-header">Time Series Analysis</div>', unsafe_allow_html=True)
    
    # Prepare time series data
    df['month'] = df['created_date'].dt.to_period('M').astype(str)
    monthly_data = df.groupby('month').agg(
        tweet_count=('conversation_id', 'count'),
        avg_accuracy=('prediction_correct', lambda x: x.mean() * 100 if x.notna().any() else None),
        avg_return=('actual_return', 'mean')
    ).reset_index()
    
    # Plot time series
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add tweet count bars
    fig.add_trace(
        go.Bar(
            x=monthly_data['month'],
            y=monthly_data['tweet_count'],
            name='Tweet Count',
            marker_color='lightblue'
        ),
        secondary_y=False
    )
    
    # Add accuracy line
    fig.add_trace(
        go.Scatter(
            x=monthly_data['month'],
            y=monthly_data['avg_accuracy'],
            name='Avg Accuracy (%)',
            line=dict(color='green', width=3)
        ),
        secondary_y=True
    )
    
    # Add return line
    fig.add_trace(
        go.Scatter(
            x=monthly_data['month'],
            y=monthly_data['avg_return'],
            name='Avg Return (%)',
            line=dict(color='red', width=3, dash='dot')
        ),
        secondary_y=True
    )
    
    # Add a horizontal line at 50% accuracy
    fig.add_shape(
        type='line',
        x0=monthly_data['month'].iloc[0],
        x1=monthly_data['month'].iloc[-1],
        y0=50,
        y1=50,
        line=dict(color='green', width=2, dash='dash'),
        yref='y2'
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
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Trader comparison table
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
    display_df['Followers'] = display_df['Followers'].apply(lambda x: f"{x:,}")
    
    st.dataframe(display_df, use_container_width=True, height=400)

# Function to create trader profile dashboard
def create_trader_profile(df, trader_name):
    df_user, df_parent = filter_trader_data(df, trader_name)
    
    if len(df_user) == 0:
        st.error(f"No data found for trader: {trader_name}")
        return
    
    # Calculate profile summary
    profile_summary = compute_profile_summary(df_user, df_parent)
    
    # Header with trader info
    st.markdown(f'<div class="main-header">Trader Profile: {trader_name}</div>', unsafe_allow_html=True)
    
    # Check if we have any parent tweets after filtering
    if len(df_parent) == 0:
        st.warning(f"No actionable parent tweets found for {trader_name}. Unable to generate detailed profile.")
        
        # Show basic stats if available
        if len(df_user) > 0:
            st.markdown("### Basic Information")
            st.markdown(f"**Total Tweets:** {profile_summary['Total Tweets']}")
            
            if profile_summary['Followers'] > 0:
                st.markdown(f"**Followers:** {profile_summary['Followers']:,}")
            
            if profile_summary['Following'] > 0:
                st.markdown(f"**Following:** {profile_summary['Following']:,}")
            
            st.markdown("### Note")
            st.markdown(
                "This trader does not have any parent tweets that meet the actionable criteria. "
                "Actionable tweets must be parent tweets with bullish or bearish sentiment and a defined time horizon."
            )
        return
    
    # Check if we have enough parent tweets for reliable analysis
    if len(df_parent) < 3:
        st.warning(f"Insufficient data for reliable analysis. {trader_name} has only {len(df_parent)} actionable parent tweets.")
        
        # Show basic stats
        st.markdown("### Basic Information")
        st.markdown(f"**Total Tweets:** {profile_summary['Total Tweets']}")
        st.markdown(f"**Actionable Parent Tweets:** {len(df_parent)}")
        st.markdown(f"**Conversations:** {profile_summary['Total Conversations']}")
        
        if profile_summary['Followers'] > 0:
            st.markdown(f"**Followers:** {profile_summary['Followers']:,}")
        
        if profile_summary['Following'] > 0:
            st.markdown(f"**Following:** {profile_summary['Following']:,}")
        
        if profile_summary['Most Mentioned Stock'] != "N/A":
            st.markdown(f"**Most Mentioned Stock:** {profile_summary['Most Mentioned Stock']}")
        
        st.markdown("### Note")
        st.markdown(
            "This trader does not have enough actionable parent tweets for reliable analysis. "
            "We recommend at least 3 actionable tweets to generate meaningful metrics."
        )
        return
    
    # Continue with the normal profile display for traders with sufficient actionable tweets
    # Profile summary cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        accuracy = profile_summary["Prediction Accuracy (%)"]
        if isinstance(accuracy, str):
            accuracy_display = accuracy
            color_class = "neutral"
        else:
            accuracy_display = f"{accuracy:.1f}%"
            color_class = "positive" if accuracy > 50 else "negative"
        st.markdown(f'<div class="metric-value {color_class}">{accuracy_display}</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Prediction Accuracy</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        avg_return = profile_summary["Average Actual Return (%)"]
        color_class = "positive" if avg_return > 0 else "negative"
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value {color_class}">{avg_return:.2f}%</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Average Return</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{profile_summary["Total Conversations"]}</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Total Conversations</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        followers = profile_summary["Followers"]
        followers_display = f"{followers:,}" if not pd.isna(followers) else "Unknown"
        st.markdown(f'<div class="metric-value">{followers_display}</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Followers</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Create tabs for different analyses
    tab1, tab2, tab3, tab4 = st.tabs(["Profile Summary", "Prediction Analysis", "Sentiment Analysis", "Stock Performance"])
    
    with tab1:
        # Profile summary
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown('<div class="sub-header">Trader Information</div>', unsafe_allow_html=True)
            
            # Format profile info
            profile_info = {
                "Total Tweets": profile_summary["Total Tweets"],
                "Total Conversations": profile_summary["Total Conversations"],
                "Verified Account": profile_summary["Verified Account"],
                "Followers": f"{profile_summary['Followers']:,}" if not pd.isna(profile_summary['Followers']) else "Unknown",
                "Following": f"{profile_summary['Following']:,}" if not pd.isna(profile_summary['Following']) else "Unknown",
                "Most Frequent Time Horizon": profile_summary["Most Frequent Time Horizon"],
                "Most Frequent Trade Type": profile_summary["Most Frequent Trade Type"],
                "Most Frequent Sentiment": profile_summary["Most Frequent Sentiment"],
                "Most Mentioned Stock": profile_summary["Most Mentioned Stock"]
            }
            
            # Display as a styled table
            for key, value in profile_info.items():
                st.markdown(f"**{key}:** {value}")
        
        with col2:
            st.markdown('<div class="sub-header">Performance Metrics</div>', unsafe_allow_html=True)
            
            # Format performance metrics
            performance_metrics = {
                "Prediction Accuracy": f"{profile_summary['Prediction Accuracy (%)']:.1f}%" if isinstance(profile_summary['Prediction Accuracy (%)'], float) else profile_summary['Prediction Accuracy (%)'],
                "Average Price Change": f"{profile_summary['Average Price Change (%)']:.2f}%",
                "Average Actual Return": f"{profile_summary['Average Actual Return (%)']:.2f}%",
                "Sentiment Consistency": f"{profile_summary['Sentiment Consistency per Conversation (%)']:.1f}%",
                "Weighted Profitability": f"{profile_summary['Weighted Profitability Mean']:.2f}",
                "Avg Likes per Tweet": f"{profile_summary['Avg Likes per Tweet']:.1f}",
                "Avg Retweets per Tweet": f"{profile_summary['Avg Retweets per Tweet']:.1f}",
                "Avg Replies per Tweet": f"{profile_summary['Avg Replies per Tweet']:.1f}",
                "Avg Views per Tweet": f"{profile_summary['Avg Views per Tweet']:.1f}" if not pd.isna(profile_summary['Avg Views per Tweet']) else "Unknown"
            }
            
            # Display as a styled table
            for key, value in performance_metrics.items():
                st.markdown(f"**{key}:** {value}")
        
        # Categorical features distribution
        st.markdown('<div class="sub-header">Feature Distributions</div>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Time Horizon distribution
            time_horizon_counts = df_parent['time_horizon'].value_counts()
            time_horizon_pcts = time_horizon_counts / time_horizon_counts.sum() * 100

            # Remove 'unknown' or empty values for visualization if they somehow exist
            if '' in time_horizon_pcts.index:
                time_horizon_pcts = time_horizon_pcts.drop('')
            if 'unknown' in time_horizon_pcts.index:
                time_horizon_pcts = time_horizon_pcts.drop('unknown')

            # Create the pie chart with valid time horizons only
            fig1 = px.pie(
                values=time_horizon_pcts,
                names=time_horizon_pcts.index,
                title="Time Horizon Distribution",
                color=time_horizon_pcts.index,
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            
            fig1.update_traces(textposition='inside', textinfo='percent+label')
            fig1.update_layout(height=350)
            
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            # Trade Type distribution
            trade_type_counts = df_parent['trade_type'].value_counts(normalize=True) * 100
            
            fig = px.pie(
                values=trade_type_counts.values,
                names=trade_type_counts.index,
                title='Trade Type Distribution',
                color_discrete_sequence=px.colors.qualitative.Pastel2
            )
            
            fig.update_traces(textposition='inside', textinfo='percent+label')
            fig.update_layout(height=350)
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col3:
            # Sentiment distribution
            sentiment_counts = df_parent['sentiment'].value_counts(normalize=True) * 100
            
            colors = {'bullish': '#17BF63', 'bearish': '#E0245E', 'neutral': '#AAB8C2'}
            
            fig = px.pie(
                values=sentiment_counts.values,
                names=sentiment_counts.index,
                title='Sentiment Distribution',
                color=sentiment_counts.index,
                color_discrete_map=colors
            )
            
            fig.update_traces(textposition='inside', textinfo='percent+label')
            fig.update_layout(height=350)
            
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        # Prediction Analysis
        st.markdown('<div class="sub-header">Prediction Performance</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Prediction accuracy by time horizon
            accuracy_by_horizon = df_parent.groupby('time_horizon')['prediction_correct'].mean() * 100
            
            fig = px.bar(
                x=accuracy_by_horizon.index,
                y=accuracy_by_horizon.values,
                labels={'x': 'Time Horizon', 'y': 'Accuracy (%)'},
                title='Prediction Accuracy by Time Horizon',
                color_discrete_sequence=px.colors.diverging.RdYlGn,
            )
            
            # Color the bars based on whether they're above 50%
            colors = ['#E0245E' if val < 50 else '#17BF63' for val in accuracy_by_horizon.values]
            fig.data[0].marker.color = colors
            
            fig.update_layout(height=400)
            
            # Add a horizontal line at 50% accuracy
            fig.add_shape(
                type='line',
                x0=-0.5,
                x1=len(accuracy_by_horizon) - 0.5,
                y0=50,
                y1=50,
                line=dict(color='red', width=2, dash='dash')
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Prediction accuracy by sentiment
            accuracy_by_sentiment = df_parent.groupby('sentiment')['prediction_correct'].mean() * 100
            
            colors = {'bullish': '#17BF63', 'bearish': '#E0245E', 'neutral': '#AAB8C2'}
            
            fig = px.bar(
                x=accuracy_by_sentiment.index,
                y=accuracy_by_sentiment.values,
                labels={'x': 'Sentiment', 'y': 'Accuracy (%)'},
                title='Prediction Accuracy by Sentiment',
                color=accuracy_by_sentiment.index,
                color_discrete_map=colors
            )
            
            fig.update_layout(height=400)
            
            # Add a horizontal line at 50% accuracy
            fig.add_shape(
                type='line',
                x0=-0.5,
                x1=len(accuracy_by_sentiment) - 0.5,
                y0=50,
                y1=50,
                line=dict(color='red', width=2, dash='dash')
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Rolling accuracy over time
        st.markdown('<div class="sub-header">Prediction Accuracy Over Time</div>', unsafe_allow_html=True)
        
        # Sort by date and calculate rolling accuracy
        df_parent_sorted = df_parent.sort_values('created_date')
        df_parent_sorted['rolling_accuracy'] = df_parent_sorted['prediction_correct'].rolling(window=10, min_periods=3).mean() * 100
        
        fig = px.line(
            df_parent_sorted,
            x='created_date',
            y='rolling_accuracy',
            labels={'created_date': 'Date', 'rolling_accuracy': 'Rolling Accuracy (%)'},
            title=f'Rolling Prediction Accuracy (10-tweet window) for {trader_name}',
            color_discrete_sequence=['#1DA1F2']
        )
        
        # Add overall accuracy line
        overall_accuracy = df_parent['prediction_correct'].mean() * 100
        fig.add_hline(y=overall_accuracy, line_dash="dash", line_color="green", 
                      annotation_text=f"Overall: {overall_accuracy:.1f}%")
        
        # Add 50% reference line
        fig.add_hline(y=50, line_dash="dot", line_color="red")
        
        fig.update_layout(height=400)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Price change vs prediction correctness
        st.markdown('<div class="sub-header">Price Change vs Prediction Correctness</div>', unsafe_allow_html=True)
        
        fig = px.scatter(
            df_parent,
            x='price_change_pct',
            y='actual_return',
            color='prediction_correct',
            color_discrete_map={True: '#17BF63', False: '#E0245E'},
            labels={
                'price_change_pct': 'Price Change (%)',
                'actual_return': 'Actual Return (%)',
                'prediction_correct': 'Prediction Correct'
            },
            title='Price Change vs Actual Return by Prediction Correctness',
            hover_data=['validated_ticker', 'sentiment', 'created_date']
        )
        
        fig.update_layout(height=500)
        
        # Add reference lines
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        fig.add_vline(x=0, line_dash="dash", line_color="gray")
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        # Sentiment Analysis
        st.markdown('<div class="sub-header">Sentiment Analysis</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Sentiment over time
            df_parent['month'] = df_parent['created_date'].dt.to_period('M').astype(str)
            sentiment_over_time = df_parent.groupby(['month', 'sentiment']).size().unstack(fill_value=0)
            
            # Convert to percentage
            sentiment_pct = sentiment_over_time.div(sentiment_over_time.sum(axis=1), axis=0) * 100
            
            # Create a stacked area chart
            fig = go.Figure()
            
            colors = {'bullish': '#17BF63', 'bearish': '#E0245E', 'neutral': '#AAB8C2'}
            
            for sentiment in sentiment_pct.columns:
                if sentiment in colors:
                    fig.add_trace(go.Scatter(
                        x=sentiment_pct.index,
                        y=sentiment_pct[sentiment],
                        mode='lines',
                        stackgroup='one',
                        name=sentiment,
                        line=dict(width=0.5),
                        fillcolor=colors[sentiment]
                    ))
            
            fig.update_layout(
                title=f'Sentiment Distribution Over Time for {trader_name}',
                xaxis_title='Month',
                yaxis_title='Percentage (%)',
                hovermode='x unified',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Sentiment vs engagement
            engagement_by_sentiment = df_parent.groupby('sentiment').agg({
                'likes': 'mean',
                'retweets': 'mean',
                'replies_count': 'mean'
            }).reset_index()
            
            # Melt the dataframe for easier plotting
            engagement_melted = pd.melt(
                engagement_by_sentiment,
                id_vars='sentiment',
                value_vars=['likes', 'retweets', 'replies_count'],
                var_name='Engagement Type',
                value_name='Average Count'
            )
            
            fig = px.bar(
                engagement_melted,
                x='sentiment',
                y='Average Count',
                color='Engagement Type',
                barmode='group',
                title=f'Average Engagement by Sentiment for {trader_name}',
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            
            fig.update_layout(height=400)
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Sentiment consistency analysis
        st.markdown('<div class="sub-header">Sentiment Consistency Analysis</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Sentiment consistency per conversation
            consistency_by_conv = df_user.groupby('conversation_id')['consistent_sentiment'].mean() * 100
            
            fig = px.histogram(
                consistency_by_conv,
                nbins=20,
                labels={'value': 'Sentiment Consistency (%)'},
                title=f'Distribution of Sentiment Consistency per Conversation for {trader_name}',
                color_discrete_sequence=['#1DA1F2']
            )
            
            fig.update_layout(height=400)
            
            # Add a vertical line at the mean
            mean_consistency = consistency_by_conv.mean()
            fig.add_vline(x=mean_consistency, line_dash="dash", line_color="red",
                         annotation_text=f"Mean: {mean_consistency:.1f}%")
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Sentiment vs return
            sentiment_return = df_parent.groupby('sentiment').agg({
                'actual_return': ['mean', 'std', 'count']
            })
            sentiment_return.columns = ['mean_return', 'std_return', 'count']
            sentiment_return = sentiment_return.reset_index()
            
            # Create error bars
            fig = go.Figure()
            
            colors = {'bullish': '#17BF63', 'bearish': '#E0245E', 'neutral': '#AAB8C2'}
            
            for sentiment in sentiment_return['sentiment']:
                row = sentiment_return[sentiment_return['sentiment'] == sentiment].iloc[0]
                color = colors.get(sentiment, '#1DA1F2')
                
                fig.add_trace(go.Bar(
                    x=[sentiment],
                    y=[row['mean_return']],
                    error_y=dict(type='data', array=[row['std_return']]),
                    name=f"{sentiment} (n={row['count']})",
                    marker_color=color
                ))
            
            fig.update_layout(
                title=f'Average Return by Sentiment for {trader_name}',
                xaxis_title='Sentiment',
                yaxis_title='Average Return (%)',
                height=400
            )
            
            # Add a horizontal line at 0%
            fig.add_hline(y=0, line_dash="dash", line_color="gray")
            
            st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        # Stock Performance
        st.markdown('<div class="sub-header">Stock Performance Analysis</div>', unsafe_allow_html=True)
        
        # Top stocks by mention
        top_stocks = df_parent['validated_ticker'].value_counts().nlargest(10)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Stock mention frequency
            fig = px.bar(
                x=top_stocks.index,
                y=top_stocks.values,
                labels={'x': 'Stock', 'y': 'Mention Count'},
                title=f'Top 10 Most Mentioned Stocks by {trader_name}',
                color=top_stocks.values,
                color_continuous_scale='Viridis'
            )
            
            fig.update_layout(height=400)
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Stock prediction accuracy
            stock_accuracy = df_parent.groupby('validated_ticker')['prediction_correct'].agg(['mean', 'count'])
            stock_accuracy = stock_accuracy[stock_accuracy['count'] >= 3]  # At least 3 predictions
            stock_accuracy['mean'] = stock_accuracy['mean'] * 100
            stock_accuracy = stock_accuracy.sort_values('mean', ascending=False)
            
            # Create a DataFrame for plotting instead of passing x and y directly
            plot_df = stock_accuracy.reset_index()
            plot_df.columns = ['Stock', 'Accuracy', 'Count']
            # Round accuracy to 2 decimal places
            plot_df['Accuracy'] = plot_df['Accuracy'].round(2)
            
            fig = px.bar(
                plot_df,
                x='Stock',
                y='Accuracy',
                labels={'Stock': 'Stock', 'Accuracy': 'Accuracy (%)'},
                title=f'Prediction Accuracy by Stock for {trader_name} (min. 3 predictions)',
                color='Accuracy',
                color_continuous_scale='RdYlGn',
                text=plot_df['Count'].apply(lambda x: f"n={x}")
            )
            
            # Format the color bar ticks to show only 2 decimal places
            fig.update_layout(
                height=400,
                coloraxis_colorbar=dict(
                    tickformat='.2f'
                )
            )
            
            # Add a horizontal line at 50% accuracy
            fig.add_hline(y=50, line_dash="dash", line_color="red")
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Stock return analysis
        st.markdown('<div class="sub-header">Stock Return Analysis</div>', unsafe_allow_html=True)
        
        # Calculate average return by stock
        stock_return = df_parent.groupby('validated_ticker').agg({
            'actual_return': ['mean', 'std', 'count'],
            'price_change_pct': 'mean'
        })
        stock_return.columns = ['mean_return', 'std_return', 'return_count', 'price_change']
        stock_return = stock_return[stock_return['return_count'] >= 3]  # At least 3 predictions
        stock_return = stock_return.sort_values('mean_return', ascending=False)
        
        # Create a scatter plot of return vs accuracy
        stock_combined = stock_accuracy.join(stock_return, how='inner', lsuffix='_accuracy', rsuffix='_return')
        
        # Create a DataFrame for plotting
        scatter_df = pd.DataFrame({
            'accuracy': stock_combined['mean'],
            'return': stock_combined['mean_return'],
            'count': stock_combined['count'],
            'price_change': stock_combined['price_change'],
            'stock': stock_combined.index
        })

        fig = px.scatter(
            scatter_df,
            x='accuracy',
            y='return',
            size='count',
            color='price_change',
            color_continuous_scale='RdYlGn',
            labels={
                'accuracy': 'Prediction Accuracy (%)',
                'return': 'Average Return (%)',
                'count': 'Number of Predictions',
                'price_change': 'Avg Price Change (%)'
            },
            title=f'Stock Performance: Accuracy vs Return for {trader_name}',
            hover_name='stock'
        )
        
        # Add quadrant lines
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        fig.add_vline(x=50, line_dash="dash", line_color="gray")
        
        # Add annotations for quadrants
        fig.add_annotation(x=25, y=stock_combined['mean_return'].max() * 0.75,
                          text="Low Accuracy<br>High Return", showarrow=False)
        fig.add_annotation(x=75, y=stock_combined['mean_return'].max() * 0.75,
                          text="High Accuracy<br>High Return", showarrow=False)
        fig.add_annotation(x=25, y=stock_combined['mean_return'].min() * 0.75,
                          text="Low Accuracy<br>Low Return", showarrow=False)
        fig.add_annotation(x=75, y=stock_combined['mean_return'].min() * 0.75,
                          text="High Accuracy<br>Low Return", showarrow=False)
        
        fig.update_layout(height=600)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display top and bottom performing stocks
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Top 5 Performing Stocks")
            top_performing = stock_return.head(5).reset_index()
            top_performing.columns = ['Stock', 'Avg Return (%)', 'Std Dev (%)', 'Count', 'Price Change (%)']
            top_performing['Avg Return (%)'] = top_performing['Avg Return (%)'].round(2)
            top_performing['Price Change (%)'] = top_performing['Price Change (%)'].round(2)
            st.dataframe(top_performing, use_container_width=True)
        
        with col2:
            st.markdown("### Bottom 5 Performing Stocks")
            bottom_performing = stock_return.tail(5).reset_index()
            bottom_performing.columns = ['Stock', 'Avg Return (%)', 'Std Dev (%)', 'Count', 'Price Change (%)']
            bottom_performing['Avg Return (%)'] = bottom_performing['Avg Return (%)'].round(2)
            bottom_performing['Price Change (%)'] = bottom_performing['Price Change (%)'].round(2)
            st.dataframe(bottom_performing, use_container_width=True)

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

# Main app
def main():
    # Load data
    df = load_data()
    
    if df is None or df.empty:
        st.error("Failed to load data. Please check the database connection and table structure.")
        return
    
    # Sidebar
    st.sidebar.title("Navigation")
    
    # Add logo or image
    st.sidebar.image("https://www.iconpacks.net/icons/2/free-twitter-logo-icon-2429-thumb.png", width=100)
    
    # Dashboard selection
    dashboard = st.sidebar.radio("Select Dashboard", ["Overview", "Trader Profile", "Raw Data Explorer"])
    
    if dashboard == "Overview":
        create_overview_dashboard(df)
    elif dashboard == "Trader Profile":
        # Trader selection
        traders = get_traders(df)
        selected_trader = st.sidebar.selectbox("Select Trader", traders)
        
        # Display trader profile
        create_trader_profile(df, selected_trader)
    else:  # Raw Data Explorer
        create_raw_data_dashboard(df)
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("### About")
    st.sidebar.info(
        "This dashboard analyzes Twitter traders' prediction accuracy "
        "and performance metrics. Data is based on validated stock predictions "
        "from Twitter conversations."
    )
    st.sidebar.markdown("2025 Twitter Trader Analysis")

# Only show the app if the user has entered the correct password
if check_password():
    if __name__ == "__main__":
        main()
