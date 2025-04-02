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
            if row and len(row) > 0 and row[0]:
                list_id = extract_username(row[0])
                if list_id:
                    twitter_lists.append(list_id)
                    print(f"Added list: {list_id}")
        
        # Process Individual Accounts (Column B)
        for row in data[1:]:  # Skip header row
            if row and len(row) > 1 and row[1]:
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
        return [], []

# Set environment variables for the prediction process
os.environ['FILTER_BY_TIME_HORIZON'] = 'True'
os.environ['FILTER_BY_TICKER_COUNT'] = 'True'
os.environ['MAX_TICKERS'] = '5'  # Default value, can be overridden by UI

# Import twitter-predictions.py using importlib
try:
    spec = importlib.util.spec_from_file_location("twitter_predictions", "twitter-predictions.py")
    twitter_predictions = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(twitter_predictions)
    
    filter_actionable_tweets = twitter_predictions.filter_actionable_tweets
    add_market_validation_columns = twitter_predictions.add_market_validation_columns
    analyze_user_accuracy = twitter_predictions.analyze_user_accuracy
    upload_to_database = twitter_predictions.upload_to_database
    
    print("Successfully imported twitter-predictions.py module")
except Exception as e:
    print(f"Error importing twitter predictions module: {e}")
    
    def filter_actionable_tweets(df):
        st.error(f"Failed to import twitter-predictions.py: {e}")
        return {"actionable_tweets": pd.DataFrame(), "analysis_tweets": pd.DataFrame()}
    
    def add_market_validation_columns(df, all_tweets=None, output_dir=None):
        return pd.DataFrame()
    
    def analyze_user_accuracy(df, min_tweets=5):
        return pd.DataFrame()
    
    def upload_to_database(df):
        return False

def check_password():
    """Returns `True` if the user had the correct password."""
    def password_entered():
        if st.session_state["username"] == "TwitterProfiling" and st.session_state["password"] == "Twitter135":
            st.session_state["password_correct"] = True
            del st.session_state["password"]
            del st.session_state["username"]
        else:
            st.session_state["password_correct"] = False
    
    if "password_correct" not in st.session_state:
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
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown('<div class="login-container">', unsafe_allow_html=True)
            st.markdown(f'<img src="https://www.iconpacks.net/icons/2/free-twitter-logo-icon-2429-thumb.png" class="login-logo">', unsafe_allow_html=True)
            st.markdown('<h1 class="login-title">Twitter Trader Analysis</h1>', unsafe_allow_html=True)
            st.markdown('<p class="login-subtitle">Enter your credentials to access the dashboard</p>', unsafe_allow_html=True)
            st.markdown('<label class="form-label">Username</label>', unsafe_allow_html=True)
            username = st.text_input("Username", key="username", placeholder="Enter username", label_visibility="collapsed")
            st.markdown('<label class="form-label">Password</label>', unsafe_allow_html=True)
            password = st.text_input("Password", key="password", type="password", placeholder="Enter password", label_visibility="collapsed")
            st.button("Sign In", on_click=password_entered)
            st.markdown('</div>', unsafe_allow_html=True)
        return False
    return st.session_state["password_correct"]

@st.cache_data(ttl=3600)
def load_data_from_db():
    try:
        host = "database-twitter.cdh6c6zr5mbr.us-east-1.rds.amazonaws.com"
        database = "postgres"
        user = "postgres"
        password = "DrorMai531"
        
        conn = psycopg2.connect(
            host=host,
            database=database,
            user=user,
            password=password,
            connect_timeout=10
        )
        
        cursor = conn.cursor()
        cursor.execute("""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name = 'tweets'
            ORDER BY ordinal_position;
        """)
        columns = [row[0] for row in cursor.fetchall()]
        print(f"Columns: {columns}")
        
        cursor.execute("SELECT COUNT(*) FROM tweets")
        total_count = cursor.fetchone()[0]
        print(f"Total tweets in database: {total_count}")
        
        cursor.execute("SELECT * FROM tweets")
        rows = cursor.fetchall()
        df = pd.DataFrame(rows, columns=columns)
        
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

@st.cache_data
def load_data(filepath=None):
    df = load_data_from_db()
    
    if df.empty:
        st.error("Failed to load data from database. Please check the database connection and table structure.")
        return None
    
    try:
        date_columns = ['created_at', 'created_date', 'prediction_date', 'start_date', 'end_date']
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        
        numeric_columns = [
            'likes', 'retweets', 'replies_count', 'views', 
            'author_followers', 'author_following',
            'price_change_pct', 'actual_return', 'start_price', 'end_price'
        ]
        
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        if 'prediction_correct' in df.columns:
            df['prediction_correct'] = df['prediction_correct'].map(
                {True: True, 'True': True, 'true': True, 
                 False: False, 'False': False, 'false': False}
            )
        
        if 'prediction_date' in df.columns and 'prediction_correct' in df.columns:
            df.loc[df['prediction_date'] > pd.Timestamp.today(), 'prediction_correct'] = None
        
        if 'tweet_type' not in df.columns:
            df['tweet_type'] = 'parent'
        
        if 'trader' not in df.columns:
            df['trader'] = df['author']
        
        if 'prediction_score' not in df.columns and 'prediction_correct' in df.columns and 'price_change_pct' in df.columns:
            def calculate_prediction_score(row):
                if pd.isna(row['prediction_correct']) or pd.isna(row['price_change_pct']):
                    return None
                multiplier = 1 if row['prediction_correct'] else -1
                return abs(float(row['price_change_pct'])) * multiplier
            df['prediction_score'] = df.apply(calculate_prediction_score, axis=1)
        
        full_df = df.copy()
        actionable_df = df[
            (df['tweet_type'] == 'parent') &
            (df['sentiment'].isin(['bullish', 'bearish'])) &
            (df['time_horizon'].notna() & (df['time_horizon'] != ''))
        ]
        actionable_conv_ids = actionable_df['conversation_id'].unique()
        df = full_df[full_df['conversation_id'].isin(actionable_conv_ids)]
        
        print(f"Full dataset: {len(full_df)} tweets")
        print(f"Actionable parent tweets: {len(actionable_df)} tweets")
        print(f"Filtered dataset (actionable conversations): {len(df)} tweets")
        
        return df
    
    except Exception as e:
        st.error(f"Error processing data: {e}")
        import traceback
        traceback.print_exc()
        return None

def get_traders(df):
    if 'author' not in df.columns:
        st.error("Author column missing from data")
        return []
    
    all_authors = df['author'].dropna().unique()
    valid_traders = []
    for author in all_authors:
        if not author or not isinstance(author, str):
            continue
        if author.startswith('$'):
            continue
        if len(author) < 2:
            continue
        if author.isdigit():
            continue
        valid_traders.append(author)
    
    print(f"Found {len(valid_traders)} valid traders out of {len(all_authors)} unique authors")
    
    return sorted(valid_traders)

def filter_trader_data(df, trader_name, show_warnings=True):
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
    
    df_user['prediction_correct'] = (
        df_user['prediction_correct']
        .astype(str)
        .str.lower()
        .map({'true': True, 'false': False})
    )
    
    if len(df_parent) > 0:
        parent_sentiment = df_user[df_user['tweet_type'] == 'parent'][['conversation_id', 'sentiment']].rename(
            columns={'sentiment': 'parent_sentiment'}
        )
        df_user = df_user.merge(parent_sentiment, on='conversation_id', how='left')
        df_user['consistent_sentiment'] = (df_user['sentiment'] == df_user['parent_sentiment']).astype(int)
        
        if 'prediction_score' in df_user.columns:
            prediction_score_sum = (
                df_user.groupby('conversation_id')['prediction_score']
                .sum()
                .reset_index(name='Weighted Profitability Score')
            )
            df_user = df_user.merge(prediction_score_sum, on='conversation_id', how='left')
    
    return df_user, df_parent

def compute_profile_summary(df_user, df_parent):
    if df_user.empty or df_parent.empty:
        return {
            "Total Tweets": 0 if df_user.empty else len(df_user),
            "Total Conversations": 0 if df_user.empty else df_user['conversation_id'].nunique(),
            "Verified Account": "Unknown",
            "Author Blue Verified": "Unknown",
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
            "Sentiment Consistency (%)": 0,
            "Weighted Profitability Mean": 0,
            "Total Predictions": 0,
            "Successful Predictions": 0,
            "Failed Predictions": 0,
            "Pending Predictions": 0,
            "Average Prediction Score": 0,
            "Best Performing Stock": "N/A",
            "Worst Performing Stock": "N/A",
            "Bullish Accuracy (%)": 0,
            "Bearish Accuracy (%)": 0,
            "Average Hold Duration (days)": 0
        }
    
    profile_summary = {
        "Total Tweets": len(df_user),
        "Total Conversations": df_user['conversation_id'].nunique(),
        "Verified Account": df_user['author_verified'].iloc[0] if 'author_verified' in df_user.columns else "Unknown",
        "Author Blue Verified": df_user['author_blue_verified'].iloc[0] if 'author_blue_verified' in df_user.columns else "Unknown",
        "Followers": df_user['author_followers'].max(),
        "Following": df_user['author_following'].max(),
        "Most Frequent Time Horizon": df_parent['time_horizon'].mode().iloc[0] if not df_parent['time_horizon'].isna().all() else "Unknown",
        "Most Frequent Trade Type": df_parent['trade_type'].mode().iloc[0] if not df_parent['trade_type'].isna().all() else "Unknown",
        "Most Frequent Sentiment": df_parent['sentiment'].mode().iloc[0] if not df_parent['sentiment'].isna().all() else "Unknown",
        "Most Mentioned Stock": df_parent['validated_ticker'].mode().iloc[0] if not df_parent['validated_ticker'].isna().all() else "Unknown",
    }
    
    profile_summary.update({
        "Avg Likes per Tweet": df_user['likes'].mean(),
        "Avg Retweets per Tweet": df_user['retweets'].mean(),
        "Avg Replies per Tweet": df_user['replies_count'].mean(),
        "Avg Views per Tweet": df_user['views'].mean() if 'views' in df_user.columns else 0
    })
    
    prediction_stats = df_parent['prediction_correct'].value_counts()
    total_validated = prediction_stats.sum()
    
    profile_summary.update({
        "Total Predictions": len(df_parent),
        "Successful Predictions": prediction_stats.get(True, 0),
        "Failed Predictions": prediction_stats.get(False, 0),
        "Pending Predictions": len(df_parent) - total_validated,
        "Prediction Accuracy (%)": (prediction_stats.get(True, 0) / total_validated * 100) if total_validated > 0 else 0,
        "Validated Predictions": f"{total_validated}/{len(df_parent)} ({(total_validated/len(df_parent)*100):.1f}%)"
    })
    
    profile_summary.update({
        "Average Price Change (%)": df_parent['price_change_pct'].mean(),
        "Average Actual Return (%)": df_parent['actual_return'].mean(),
        "Average Prediction Score": df_parent['prediction_score'].mean() if 'prediction_score' in df_parent.columns else 0
    })
    
    if 'validated_ticker' in df_parent.columns and 'actual_return' in df_parent.columns:
        stock_performance = df_parent.groupby('validated_ticker')['actual_return'].agg(['mean', 'count'])
        stock_performance = stock_performance[stock_performance['count'] >= 3]
        
        if not stock_performance.empty:
            best_stock = stock_performance.nlargest(1, 'mean').index[0]
            worst_stock = stock_performance.nsmallest(1, 'mean').index[0]
            profile_summary.update({
                "Best Performing Stock": f"{best_stock} ({stock_performance.loc[best_stock, 'mean']:.1f}%)",
                "Worst Performing Stock": f"{worst_stock} ({stock_performance.loc[worst_stock, 'mean']:.1f}%)"
            })
    
    bullish_tweets = df_parent[df_parent['sentiment'] == 'bullish']
    bearish_tweets = df_parent[df_parent['sentiment'] == 'bearish']
    
    # Only consider validated predictions (where prediction_correct is not NaN)
    validated_bullish = bullish_tweets[bullish_tweets['prediction_correct'].notna()]
    validated_bearish = bearish_tweets[bearish_tweets['prediction_correct'].notna()]
    
    # Calculate accuracy only if we have enough validated predictions
    bullish_accuracy = (validated_bullish['prediction_correct'].mean() * 100) if len(validated_bullish) >= 3 else 0
    bearish_accuracy = (validated_bearish['prediction_correct'].mean() * 100) if len(validated_bearish) >= 3 else 0
    
    # Add counts to make interpretation easier
    bullish_info = f"{bullish_accuracy:.1f}% ({len(validated_bullish)}/{len(bullish_tweets)} validated)"
    bearish_info = f"{bearish_accuracy:.1f}% ({len(validated_bearish)}/{len(bearish_tweets)} validated)"
    
    profile_summary.update({
        "Bullish Accuracy (%)": bullish_accuracy,
        "Bearish Accuracy (%)": bearish_accuracy,
        "Bullish Accuracy Info": bullish_info,
        "Bearish Accuracy Info": bearish_info
    })
    
    if 'start_date' in df_parent.columns and 'end_date' in df_parent.columns:
        df_parent['hold_duration'] = (pd.to_datetime(df_parent['end_date']) - pd.to_datetime(df_parent['start_date'])).dt.days
        profile_summary["Average Hold Duration (days)"] = df_parent['hold_duration'].mean()
    
    if 'consistent_sentiment' in df_user.columns:
        consistency_by_conv = df_user.groupby('conversation_id')['consistent_sentiment'].mean() * 100
        profile_summary["Sentiment Consistency (%)"] = consistency_by_conv.mean() * 100
    
    return profile_summary

def format_and_display_data(df, title="Show Raw Data", relevant_columns=None):
    """Helper function to format and display a dataframe in an expander."""
    if df is None or df.empty:
        st.warning("No data available to display.")
        return

    display_df = df.copy()

    # Define default core columns if specific ones aren't requested
    default_core_columns = [
        'tweet_id', 'tweet_link', 'created_date', 'author', 'text', 'Result Flag'
    ]
    
    target_columns = relevant_columns if relevant_columns else default_core_columns
    target_columns = target_columns[:] # Create a copy to modify

    # --- Ensure necessary columns exist for link and flag generation ---
    can_make_link = 'tweet_id' in display_df.columns and 'author' in display_df.columns
    can_make_flag = 'prediction_correct' in display_df.columns

    # Generate tweet_link if possible and requested/default
    if 'tweet_link' in target_columns and can_make_link and 'tweet_link' not in display_df.columns:
        display_df['tweet_link'] = display_df.apply(
            lambda row: f"https://twitter.com/{row['author']}/status/{row['tweet_id']}" if pd.notna(row['tweet_id']) and pd.notna(row['author']) else None, 
            axis=1
        )
    elif 'tweet_link' in target_columns and not can_make_link:
         target_columns.remove('tweet_link') # Remove if base columns missing

    # Generate Result Flag if possible and requested/default
    if 'Result Flag' in target_columns and can_make_flag and 'Result Flag' not in display_df.columns:
        def correctness_flag(val):
            if pd.isna(val):
                return "⏳ Pending"
            elif val == True:
                return "✅ Correct"
            elif val == False:
                return "❌ Incorrect"
            else:
                return "❓ Unknown"
        display_df['Result Flag'] = display_df['prediction_correct'].apply(correctness_flag)
    elif 'Result Flag' in target_columns and not can_make_flag:
        target_columns.remove('Result Flag') # Remove if base columns missing

    # --- Filter to available and requested display columns ---
    available_display_columns = [col for col in target_columns if col in display_df.columns]
    
    # Ensure tweet_id is string if present
    if 'tweet_id' in available_display_columns:
        display_df['tweet_id'] = display_df['tweet_id'].apply(lambda x: str(int(x)) if pd.notna(x) and isinstance(x, (int, float)) else str(x))

    # Select only the final available columns for display
    final_df_to_display = display_df[available_display_columns]

    # --- Display in Expander ---
    with st.expander(title):
        if final_df_to_display.empty:
             st.warning("No data available with the selected columns.")
        else:
            st.dataframe(final_df_to_display, use_container_width=True, 
                           column_config={
                               "tweet_link": st.column_config.LinkColumn("Tweet Link", display_text="🔗")
                           })
            
            # Add a note linking to the main Raw Data page
            st.caption("For more filtering options and all columns, use the main 'Raw Data' page.")

def analyze_all_traders(df):
    all_traders = get_traders(df)
    trader_metrics = []
    
    for trader in all_traders:
        df_user, df_parent = filter_trader_data(df, trader, show_warnings=False)
        
        if len(df_parent) < 3:
            continue
            
        accuracy = df_parent['prediction_correct'].mean() * 100 if not df_parent['prediction_correct'].isna().all() else 0
        avg_return = df_parent['actual_return'].mean()
        total_tweets = len(df_user)
        total_convs = df_user['conversation_id'].nunique()
        followers = df_user['author_followers'].max()
        
        sentiment_counts = df_parent['sentiment'].value_counts(normalize=True) * 100
        bullish_pct = sentiment_counts.get('bullish', 0)
        bearish_pct = sentiment_counts.get('bearish', 0)
        
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

def create_overview_dashboard(df):
    st.markdown("<h1 class='main-header'>Twitter Trader Analysis Dashboard</h1>", unsafe_allow_html=True)
    
    st.markdown("<h2>Key Metrics</h2>", unsafe_allow_html=True)
    
    metric_cols = st.columns(4)
    
    with metric_cols[0]:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">{}</div>
            <div class="metric-label">Total Traders</div>
        </div>
        """.format(df['author'].nunique()), unsafe_allow_html=True)
    
    with metric_cols[1]:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">{}</div>
            <div class="metric-label">Total Predictions</div>
        </div>
        """.format(len(df)), unsafe_allow_html=True)
    
    with metric_cols[2]:
        correct_preds = df[df['prediction_correct'] == True].shape[0]
        accuracy = correct_preds / len(df) * 100 if len(df) > 0 else 0
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">{:.1f}%</div>
            <div class="metric-label">Overall Accuracy</div>
        </div>
        """.format(accuracy), unsafe_allow_html=True)
    
    with metric_cols[3]:
        avg_return = df['actual_return'].mean() if 'actual_return' in df.columns else 0
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">{:.2f}%</div>
            <div class="metric-label">Avg. Return</div>
        </div>
        """.format(avg_return), unsafe_allow_html=True)
    
    valid_traders = []
    invalid_traders = []
    
    for trader in get_traders(df):
        count = len(df[(df['author'] == trader) & 
                       (df['tweet_type'] == 'parent') & 
                       (df['sentiment'].isin(['bullish', 'bearish'])) &
                       (df['time_horizon'].notna() & (df['time_horizon'] != ''))])
        
        if count >= 3:
            valid_traders.append(trader)
        else:
            invalid_traders.append(trader)
    
    filtered_df = df[df['author'].isin(valid_traders)]
    parent_tweets = filtered_df[filtered_df['tweet_type'] == 'parent']
    
    unique_traders = len(valid_traders)
    total_tweets = len(filtered_df)
    
    accuracy_df = parent_tweets[parent_tweets['prediction_correct'].notna()]
    overall_accuracy = accuracy_df['prediction_correct'].mean() * 100 if len(accuracy_df) > 0 else 0
    avg_return = parent_tweets['actual_return'].mean() if 'actual_return' in parent_tweets.columns else 0
    
    validation_rate = len(accuracy_df) / len(parent_tweets) * 100 if len(parent_tweets) > 0 else 0
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
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
        st.markdown('<div class="metric-label">Average Return</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col5:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value neutral">{validation_rate:.1f}%</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Validation Rate</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-note">{len(accuracy_df)}/{len(parent_tweets)} predictions</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with st.expander("Show traders with insufficient data"):
        st.info(f"{len(invalid_traders)} traders were filtered out due to having fewer than 3 actionable tweets.")
        if invalid_traders:
            st.write("Traders excluded from analysis:")
            chunks = [invalid_traders[i:i+5] for i in range(0, len(invalid_traders), 5)]
            for chunk in chunks:
                st.write(", ".join(chunk))
    
    trader_metrics = analyze_all_traders(df)
    
    st.markdown('<div class="sub-header">Top Traders by Accuracy</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
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
        
        fig.add_shape(
            type='line',
            x0=-0.5,
            x1=9.5,
            y0=50,
            y1=50,
            line=dict(color='red', width=2, dash='dash')
        )
        
        st.plotly_chart(fig, use_container_width=True)

        # Show the aggregated top_traders data used in the plot
        format_and_display_data(top_traders, 
                                title="Show Aggregated Data for Top Traders Plot",
                                relevant_columns=['trader', 'accuracy', 'avg_return', 'total_tweets', 'followers'])
    
    with col2:
        st.markdown('<div class="highlight">', unsafe_allow_html=True)
        st.markdown("### Trader Leaderboard")
        st.markdown("Top traders ranked by prediction accuracy with minimum 3 predictions")
        st.markdown('</div>', unsafe_allow_html=True)
        
        display_df = top_traders[['trader', 'accuracy', 'avg_return', 'total_tweets', 'followers']].copy()
        display_df.columns = ['Trader', 'Accuracy (%)', 'Avg Return (%)', 'Tweets', 'Followers']
        display_df['Accuracy (%)'] = display_df['Accuracy (%)'].round(1)
        display_df['Avg Return (%)'] = display_df['Avg Return (%)'].round(2)
        display_df['Followers'] = display_df['Followers'].apply(lambda x: f"{x:,}")
        
        st.dataframe(display_df, use_container_width=True, height=400)
    
    st.markdown('<div class="sub-header">Sentiment & Stock Analysis</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        all_parent_tweets = filtered_df[filtered_df['tweet_type'] == 'parent']
        sentiment_counts = all_parent_tweets['sentiment'].value_counts()
        sentiment_pcts = sentiment_counts / sentiment_counts.sum() * 100

        fig = px.pie(
            values=sentiment_pcts,
            names=sentiment_pcts.index,
            title="Overall Sentiment Distribution",
            color=sentiment_pcts.index,
            color_discrete_map={'bullish': '#17BF63', 'bearish': '#E0245E', 'neutral': '#AAB8C2'},
            hole=0.4
        )

        fig.update_layout(
            legend_title="Sentiment",
            margin=dict(t=30, b=0, l=0, r=0),
            height=400,
            font=dict(size=14)
        )

        fig.update_traces(
            textposition='inside',
            textinfo='percent+label',
            insidetextfont=dict(color='white')
        )

        st.plotly_chart(fig, use_container_width=True)

        # Show raw data with relevant columns
        format_and_display_data(all_parent_tweets, 
                                title="Show Raw Data for Sentiment Distribution",
                                relevant_columns=['tweet_id', 'author', 'created_date', 'sentiment'])
    
    with col2:
        stock_counts = df['validated_ticker'].value_counts().nlargest(10).reset_index()
        # Rename columns to match plot expectations
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

        # Show filtered raw data with relevant columns
        top_stocks_list = stock_counts['Stock'].tolist()
        filtered_raw_stock_data = parent_tweets[parent_tweets['validated_ticker'].isin(top_stocks_list)]
        format_and_display_data(filtered_raw_stock_data, 
                                title="Show Raw Data for Top Mentioned Stocks",
                                relevant_columns=['tweet_id', 'author', 'created_date', 'validated_ticker'])
    
    st.markdown('<div class="sub-header">Time Series Analysis</div>', unsafe_allow_html=True)
    
    df['month'] = df['created_date'].dt.to_period('M').astype(str)
    monthly_data = df.groupby('month').agg(
        tweet_count=('conversation_id', 'count'),
        avg_accuracy=('prediction_correct', lambda x: x.mean() * 100 if x.notna().any() else None),
        avg_return=('actual_return', 'mean')
    ).reset_index()
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig.add_trace(
        go.Bar(
            x=monthly_data['month'],
            y=monthly_data['tweet_count'],
            name='Tweet Count',
            marker_color='lightblue'
        ),
        secondary_y=False
    )
    
    fig.add_trace(
        go.Scatter(
            x=monthly_data['month'],
            y=monthly_data['avg_accuracy'],
            name='Avg Accuracy (%)',
            line=dict(color='green', width=3)
        ),
        secondary_y=True
    )
    
    fig.add_trace(
        go.Scatter(
            x=monthly_data['month'],
            y=monthly_data['avg_return'],
            name='Avg Return (%)',
            line=dict(color='red', width=3, dash='dot')
        ),
        secondary_y=True
    )
    
    fig.add_shape(
        type='line',
        x0=monthly_data['month'].iloc[0],
        x1=monthly_data['month'].iloc[-1],
        y0=50,
        y1=50,
        line=dict(color='green', width=2, dash='dash'),
        yref='y2'
    )
    
    fig.add_shape(
        type='line',
        x0=monthly_data['month'].iloc[0],
        x1=monthly_data['month'].iloc[-1],
        y0=0,
        y1=0,
        line=dict(color='red', width=2, dash='dash'),
        yref='y2'
    )
    
    fig.update_layout(
        title='Monthly Tweet Count, Accuracy, and Return',
        xaxis_title='Month',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        height=500
    )
    
    fig.update_yaxes(title_text='Tweet Count', secondary_y=False)
    fig.update_yaxes(title_text='Percentage (%)', secondary_y=True)
    
    st.plotly_chart(fig, use_container_width=True)

    # Show raw data with relevant columns used in aggregation
    format_and_display_data(df, 
                                title="Show Raw Data for Monthly Analysis", 
                                relevant_columns=['tweet_id', 'author', 'created_date', 'prediction_correct', 'actual_return', 'conversation_id'])
    
    st.markdown('<div class="sub-header">Trader Comparison</div>', unsafe_allow_html=True)
    
    
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

def create_trader_profile(df, trader_name):
    df_user, df_parent = filter_trader_data(df, trader_name)
    
    if len(df_user) == 0:
        st.error(f"No data found for trader: {trader_name}")
        return
    
    profile_summary = compute_profile_summary(df_user, df_parent)
    
    st.markdown(f'<div class="main-header">Trader Profile: {trader_name}</div>', unsafe_allow_html=True)
    
    if len(df_parent) == 0:
        st.warning(f"No actionable parent tweets found for {trader_name}. Unable to generate detailed profile.")
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
    
    if len(df_parent) < 3:
        st.warning(f"Insufficient data for reliable analysis. {trader_name} has only {len(df_parent)} actionable parent tweets.")
        
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
    col1, col2, col3 = st.columns(3)
    
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
        st.markdown(f'<div class="metric-value">{profile_summary["Total Predictions"]}</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Total Conversations</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    tab1, tab2, tab3, tab4 = st.tabs(["Profile Summary", "Prediction Analysis", "Sentiment Analysis", "Stock Performance"])
    
    with tab1:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown('<div class="sub-header">Trader Information</div>', unsafe_allow_html=True)
            
            profile_info = {
                "Total Conversations": profile_summary["Total Predictions"],  # Changed to use Total Predictions
                "Total Predictions": profile_summary["Total Predictions"],
                "Verified Account": "✅" if profile_summary["Verified Account"] else "❌",
                "Twitter Blue": "✅" if profile_summary["Author Blue Verified"] else "❌",
                "Following": f"{profile_summary['Following']:,}" if not pd.isna(profile_summary['Following']) else "Unknown",
                "Most Frequent Time Horizon": profile_summary["Most Frequent Time Horizon"],
                "Most Frequent Trade Type": profile_summary["Most Frequent Trade Type"],
                "Most Frequent Sentiment": profile_summary["Most Frequent Sentiment"],
                "Most Mentioned Stock": profile_summary["Most Mentioned Stock"]
            }
            
            for key, value in profile_info.items():
                st.markdown(f"**{key}:** {value}")
        
        with col2:
            st.markdown('<div class="sub-header">Performance Metrics</div>', unsafe_allow_html=True)
            
            performance_metrics = {
                "Prediction Accuracy": f"{profile_summary['Prediction Accuracy (%)']:.1f}%" if isinstance(profile_summary['Prediction Accuracy (%)'], float) else profile_summary['Prediction Accuracy (%)'],
                "Validated Predictions": profile_summary["Validated Predictions"],
                "Bullish Accuracy": profile_summary["Bullish Accuracy Info"],
                "Bearish Accuracy": profile_summary["Bearish Accuracy Info"],
                "Successful Predictions": profile_summary["Successful Predictions"],
                "Failed Predictions": profile_summary["Failed Predictions"],
                "Pending Predictions": profile_summary["Pending Predictions"],
                "Average Prediction Score": f"{profile_summary['Average Prediction Score']:.2f}",
                "Average Price Change": f"{profile_summary['Average Price Change (%)']:.2f}%",
                "Average Actual Return": f"{profile_summary['Average Actual Return (%)']:.2f}%",
                "Average Hold Duration": f"{profile_summary['Average Hold Duration (days)']:.1f} days",
                "Best Performing Stock": profile_summary["Best Performing Stock"],
                "Worst Performing Stock": profile_summary["Worst Performing Stock"]
            }
            
            for key, value in performance_metrics.items():
                color = ""
                if "Accuracy" in key or "Return" in key or "Change" in key:
                    try:
                        num_value = float(str(value).rstrip('%'))
                        if num_value > 50:
                            color = "color: green"
                        elif num_value < 50:
                            color = "color: red"
                    except:
                        pass
                st.markdown(f"**{key}:** <span style='{color}'>{value}</span>", unsafe_allow_html=True)
        
        # Tweet Distribution Analysis with two pie charts
        st.markdown('<div class="sub-header">Tweet Distribution Analysis</div>', unsafe_allow_html=True)
        dist_col1, dist_col2 = st.columns(2)
        
        with dist_col1:
            sentiment_counts = df_parent['sentiment'].value_counts()
            fig_sentiment = px.pie(
                values=sentiment_counts.values,
                names=sentiment_counts.index,
                title="Sentiment Distribution",
                hole=0.4,
                color=sentiment_counts.index,
                color_discrete_map={'bullish': '#17BF63', 'bearish': '#E0245E', 'neutral': '#AAB8C2'}
            )
            fig_sentiment.update_traces(textposition='inside', textinfo='percent+label')
            fig_sentiment.update_layout(height=400)
            st.plotly_chart(fig_sentiment, use_container_width=True)

            # Show raw data with relevant columns
            format_and_display_data(df_parent, 
                                    title="Show Raw Data for Sentiment Distribution",
                                    relevant_columns=['tweet_id', 'author', 'created_date', 'sentiment'])
        
        with dist_col2:
            time_horizon_counts = df_parent['time_horizon'].value_counts()
            fig_time = px.pie(
                values=time_horizon_counts.values,
                names=time_horizon_counts.index,
                title="Time Horizon Distribution",
                hole=0.4,
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            fig_time.update_traces(textposition='inside', textinfo='percent+label')
            fig_time.update_layout(height=400)
            st.plotly_chart(fig_time, use_container_width=True)

            # Show raw data with relevant columns
            format_and_display_data(df_parent, 
                                    title="Show Raw Data for Time Horizon Distribution",
                                    relevant_columns=['tweet_id', 'author', 'created_date', 'time_horizon'])
        
        # Add the Raw Data table
        st.markdown('<div class="sub-header">Raw Data Used for Analysis</div>', unsafe_allow_html=True)
        st.dataframe(df_parent, use_container_width=True)
    
    with tab2:
        st.markdown('<div class="sub-header">Prediction Performance</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            accuracy_by_horizon = df_parent.groupby('time_horizon')['prediction_correct'].mean() * 100
            
            fig = px.bar(
                x=accuracy_by_horizon.index,
                y=accuracy_by_horizon.values,
                labels={'x': 'Time Horizon', 'y': 'Accuracy (%)'},
                title='Prediction Accuracy by Time Horizon',
                color_discrete_sequence=px.colors.diverging.RdYlGn,
            )
            
            colors = ['#E0245E' if val < 50 else '#17BF63' for val in accuracy_by_horizon.values]
            fig.data[0].marker.color = colors
            
            fig.update_layout(height=400)
            
            fig.add_shape(
                type='line',
                x0=-0.5,
                x1=len(accuracy_by_horizon) - 0.5,
                y0=50,
                y1=50,
                line=dict(color='red', width=2, dash='dash')
            )
            
            st.plotly_chart(fig, use_container_width=True)

            # Show raw data with relevant columns
            format_and_display_data(df_parent, 
                                    title="Show Raw Data for Accuracy by Time Horizon",
                                    relevant_columns=['tweet_id', 'author', 'created_date', 'time_horizon', 'prediction_correct', 'Result Flag'])
        
        with col2:
            # Filter for validated predictions only
            validated_df = df_parent[df_parent['prediction_correct'].notna()]
            
            if len(validated_df) > 0:
                # Prediction accuracy by sentiment with counts
                accuracy_by_sentiment = validated_df.groupby('sentiment')['prediction_correct'].mean() * 100
                
                # Count number of predictions for each sentiment
                sentiment_counts = validated_df.groupby('sentiment').size()
                total_counts = df_parent.groupby('sentiment').size()
                
                # Add counts to hover text
                hover_data = pd.DataFrame({
                    'sentiment': accuracy_by_sentiment.index,
                    'count': [f"{sentiment_counts.get(sent, 0)}/{total_counts.get(sent, 0)} validated" for sent in accuracy_by_sentiment.index]
                })
                
                colors = {'bullish': '#17BF63', 'bearish': '#E0245E', 'neutral': '#AAB8C2'}
                
                fig = px.bar(
                    x=accuracy_by_sentiment.index,
                    y=accuracy_by_sentiment.values,
                    labels={'x': 'Sentiment', 'y': 'Accuracy (%)'},
                    title='Prediction Accuracy by Sentiment (Validated Predictions Only)',
                    color=accuracy_by_sentiment.index,
                    color_discrete_map=colors,
                    hover_data=[hover_data.set_index('sentiment')['count']]
                )
                
                # Add annotations with counts
                for i, sentiment in enumerate(accuracy_by_sentiment.index):
                    fig.add_annotation(
                        x=sentiment,
                        y=accuracy_by_sentiment[sentiment] + 5,  # Position above the bar
                        text=f"{sentiment_counts.get(sentiment, 0)}/{total_counts.get(sentiment, 0)}",
                        showarrow=False
                    )
                
                fig.update_layout(height=400)
                
                fig.add_shape(
                    type='line',
                    x0=-0.5,
                    x1=len(accuracy_by_sentiment) - 0.5,
                    y0=50,
                    y1=50,
                    line=dict(color='red', width=2, dash='dash')
                )
                
                st.plotly_chart(fig, use_container_width=True)

                # Show raw data with relevant columns
                format_and_display_data(df_parent, 
                                        title="Show Raw Data for Accuracy by Sentiment",
                                        relevant_columns=['tweet_id', 'author', 'created_date', 'sentiment', 'prediction_correct', 'Result Flag'])
            else:
                st.warning("No validated predictions available to calculate accuracy by sentiment.")
        
        st.markdown('<div class="sub-header">Prediction Accuracy Over Time</div>', unsafe_allow_html=True)
        
        # Filter to only include validated predictions for the chart
        validated_parent = df_parent[df_parent['prediction_correct'].notna()].copy()
        
        if len(validated_parent) > 0:
            # Sort by date and calculate rolling accuracy (increase window for more smoothing)
            df_parent_sorted = validated_parent.sort_values('created_date')
            
            # Adjust window size based on amount of data
            window_size = min(30, max(10, len(df_parent_sorted) // 5))
            min_periods = min(5, max(3, window_size // 5))
            
            df_parent_sorted['rolling_accuracy'] = df_parent_sorted['prediction_correct'].rolling(
                window=window_size, 
                min_periods=min_periods
            ).mean() * 100
            
            # Get date range for title
            date_range = f"{df_parent_sorted['created_date'].min().strftime('%b %Y')} - {df_parent_sorted['created_date'].max().strftime('%b %Y')}"
            
            fig = px.line(
                df_parent_sorted,
                x='created_date',
                y='rolling_accuracy',
                labels={'created_date': 'Date', 'rolling_accuracy': 'Rolling Accuracy (%)'},
                title=f'Rolling Prediction Accuracy ({window_size}-tweet window) for {trader_name} ({date_range})',
                color_discrete_sequence=['#1DA1F2']
            )
            
            # Add data count annotation
            fig.add_annotation(
                xref="paper", yref="paper",
                x=0.01, y=0.99,
                text=f"{len(validated_parent)}/{len(df_parent)} tweets validated",
                showarrow=False,
                font=dict(size=12),
                align="left"
            )
            
            # Add overall accuracy line
            overall_accuracy = validated_parent['prediction_correct'].mean() * 100
            fig.add_hline(y=overall_accuracy, line_dash="dash", line_color="green", 
                        annotation_text=f"Overall: {overall_accuracy:.1f}%")
            fig.add_hline(y=50, line_dash="dot", line_color="red")
            
            # Set date range to focus on the data we have
            fig.update_xaxes(range=[
                df_parent_sorted['created_date'].min() - pd.Timedelta(days=1),
                df_parent_sorted['created_date'].max() + pd.Timedelta(days=1)
            ])
            
            fig.update_layout(height=400)
            
            st.plotly_chart(fig, use_container_width=True)

            # Show raw data with relevant columns
            # Pass df_parent_sorted as it's the relevant subset for rolling calc
            format_and_display_data(df_parent_sorted, 
                                    title="Show Raw Data for Rolling Accuracy", 
                                    relevant_columns=['tweet_id', 'author', 'created_date', 'prediction_correct', 'Result Flag'])
        else:
            st.warning("No validated predictions available to calculate accuracy over time.")
        
        st.markdown('<div class="sub-header">Price Change vs Prediction Correctness</div>', unsafe_allow_html=True)
        
        # Filter to only include validated predictions
        validated_df = df_parent[df_parent['prediction_correct'].notna()]
        
        if len(validated_df) > 0:
            fig = px.scatter(
                validated_df,
                x='price_change_pct',
                y='actual_return',
                color='prediction_correct',
                color_discrete_map={True: '#17BF63', False: '#E0245E'},
                labels={
                    'price_change_pct': 'Price Change (%)',
                    'actual_return': 'Actual Return (%)',
                    'prediction_correct': 'Prediction Correct'
                },
                title=f'Price Change vs Actual Return ({len(validated_df)} validated predictions)',
                hover_data=['validated_ticker', 'sentiment', 'created_date']
            )
            
            fig.update_layout(height=500)
            
            fig.add_hline(y=0, line_dash="dash", line_color="gray")
            fig.add_vline(x=0, line_dash="dash", line_color="gray")
            
            st.plotly_chart(fig, use_container_width=True)

            # Show raw data with relevant columns
            format_and_display_data(validated_df, # Use validated subset shown in plot
                                    title="Show Raw Data for Price Change vs Return",
                                    relevant_columns=['tweet_id', 'author', 'created_date', 'validated_ticker', 'sentiment', 'prediction_correct', 'Result Flag', 'price_change_pct', 'actual_return'])
        else:
            st.warning("No validated predictions available for price change analysis.")
    
    with tab3:
        st.markdown('<div class="sub-header">Sentiment Analysis</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            df_parent['month'] = df_parent['created_date'].dt.to_period('M').astype(str)
            sentiment_over_time = df_parent.groupby(['month', 'sentiment']).size().unstack(fill_value=0)
            
            sentiment_pct = sentiment_over_time.div(sentiment_over_time.sum(axis=1), axis=0) * 100
            
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

            # Show raw data with relevant columns
            format_and_display_data(df_parent, 
                                    title="Show Raw Data for Sentiment Over Time",
                                    relevant_columns=['tweet_id', 'author', 'created_date', 'sentiment'])
        
        with col2:
            engagement_by_sentiment = df_parent.groupby('sentiment').agg({
                'likes': 'mean',
                'retweets': 'mean',
                'replies_count': 'mean'
            }).reset_index()
            
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

            # Show raw data with relevant columns
            format_and_display_data(df_parent, 
                                    title="Show Raw Data for Average Engagement",
                                    relevant_columns=['tweet_id', 'author', 'created_date', 'sentiment', 'likes', 'retweets', 'replies_count'])
        
        st.markdown('<div class="sub-header">Sentiment Consistency Analysis</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            consistency_by_conv = df_user.groupby('conversation_id')['consistent_sentiment'].mean() * 100
            
            fig = px.histogram(
                consistency_by_conv,
                nbins=20,
                labels={'value': 'Sentiment Consistency (%)'},
                title=f'Distribution of Sentiment Consistency per Conversation for {trader_name}',
                color_discrete_sequence=['#1DA1F2']
            )
            
            fig.update_layout(height=400)
            
            mean_consistency = consistency_by_conv.mean()
            fig.add_vline(x=mean_consistency, line_dash="dash", line_color="red",
                         annotation_text=f"Mean: {mean_consistency:.1f}%")
            
            st.plotly_chart(fig, use_container_width=True)

            # Show raw data (df_user includes replies) with relevant columns
            format_and_display_data(df_user, 
                                    title="Show Raw Data for Sentiment Consistency",
                                    relevant_columns=['tweet_id', 'author', 'created_date', 'conversation_id', 'sentiment', 'parent_sentiment', 'consistent_sentiment'])
        
        with col2:
            sentiment_return = df_parent.groupby('sentiment').agg({
                'actual_return': ['mean', 'std', 'count']
            })
            sentiment_return.columns = ['mean_return', 'std_return', 'count']
            sentiment_return = sentiment_return.reset_index()
            
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
            
            fig.add_hline(y=0, line_dash="dash", line_color="gray")
            
            st.plotly_chart(fig, use_container_width=True)

            # Show raw data with relevant columns
            format_and_display_data(df_parent, 
                                    title="Show Raw Data for Average Return by Sentiment",
                                    relevant_columns=['tweet_id', 'author', 'created_date', 'sentiment', 'actual_return', 'prediction_correct', 'Result Flag'])
    
    with tab4:
        st.markdown('<div class="sub-header">Stock Performance Analysis</div>', unsafe_allow_html=True)
        
        top_stocks = df_parent['validated_ticker'].value_counts().nlargest(10)
        
        col1, col2 = st.columns(2)
        
        with col1:
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

            # Show raw data filtered to top stocks
            top_stocks_list = top_stocks.index.tolist() # Get tickers from top_stocks series
            filtered_data = df_parent[df_parent['validated_ticker'].isin(top_stocks_list)]
            format_and_display_data(filtered_data, 
                                    title="Show Raw Data for Top Mentioned Stocks",
                                    relevant_columns=['tweet_id', 'author', 'created_date', 'validated_ticker'])
        
        with col2:
            stock_accuracy = df_parent.groupby('validated_ticker')['prediction_correct'].agg(['mean', 'count'])
            stock_accuracy = stock_accuracy[stock_accuracy['count'] >= 3]
            stock_accuracy['mean'] = stock_accuracy['mean'] * 100
            stock_accuracy = stock_accuracy.sort_values('mean', ascending=False)
            
            plot_df = stock_accuracy.reset_index()
            plot_df.columns = ['Stock', 'Accuracy', 'Count']
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
            
            fig.update_layout(
                height=400,
                coloraxis_colorbar=dict(
                    tickformat='.2f'
                )
            )
            
            fig.add_hline(y=50, line_dash="dash", line_color="red")
            
            st.plotly_chart(fig, use_container_width=True)

            # Show raw data filtered to stocks with >= 3 preds
            stocks_in_plot = plot_df['Stock'].tolist() # Get tickers from plot_df
            filtered_data = df_parent[df_parent['validated_ticker'].isin(stocks_in_plot)]
            format_and_display_data(filtered_data, 
                                    title="Show Raw Data for Accuracy by Stock",
                                    relevant_columns=['tweet_id', 'author', 'created_date', 'validated_ticker', 'prediction_correct', 'Result Flag'])
        
        st.markdown('<div class="sub-header">Stock Return Analysis</div>', unsafe_allow_html=True)
        
        stock_return = df_parent.groupby('validated_ticker').agg({
            'actual_return': ['mean', 'std', 'count'],
            'price_change_pct': 'mean'
        })
        stock_return.columns = ['mean_return', 'std_return', 'return_count', 'price_change']
        stock_return = stock_return[stock_return['return_count'] >= 3]
        stock_return = stock_return.sort_values('mean_return', ascending=False)
        
        stock_combined = stock_accuracy.join(stock_return, how='inner', lsuffix='_accuracy', rsuffix='_return')
        
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
        
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        fig.add_vline(x=50, line_dash="dash", line_color="gray")
        
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

        # Show raw data filtered to stocks with >= 3 preds
        stocks_in_plot = scatter_df['stock'].tolist() # Get tickers from scatter_df
        filtered_data = df_parent[df_parent['validated_ticker'].isin(stocks_in_plot)]
        format_and_display_data(filtered_data, 
                                title="Show Raw Data for Accuracy vs Return", 
                                relevant_columns=['tweet_id', 'author', 'created_date', 'validated_ticker', 'prediction_correct', 'Result Flag', 'actual_return', 'price_change_pct'])
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Top 5 Performing Stocks")
            top_performing = stock_return.head(5).reset_index()
            top_performing.columns = ['Stock', 'Avg Return (%)', 'Std Dev (%)', 'Count', 'Price Change (%)']
            top_performing['Avg Return (%)'] = top_performing['Avg Return (%)'].round(2)
            top_performing['Price Change (%)'] = top_performing['Price Change (%)'].round(2)
            st.dataframe(top_performing, use_container_width=True)

            # Show raw data filtered to top 5 stocks
            top_5_stocks = top_performing['Stock'].tolist()
            filtered_data = df_parent[df_parent['validated_ticker'].isin(top_5_stocks)]
            format_and_display_data(filtered_data, 
                                    title="Show Raw Data for Top Performing Stocks",
                                    relevant_columns=['tweet_id', 'author', 'created_date', 'validated_ticker', 'actual_return', 'price_change_pct', 'prediction_correct', 'Result Flag'])
        
        with col2:
            st.markdown("### Bottom 5 Performing Stocks")
            bottom_performing = stock_return.tail(5).reset_index()
            bottom_performing.columns = ['Stock', 'Avg Return (%)', 'Std Dev (%)', 'Count', 'Price Change (%)']
            bottom_performing['Avg Return (%)'] = bottom_performing['Avg Return (%)'].round(2)
            bottom_performing['Price Change (%)'] = bottom_performing['Price Change (%)'].round(2)
            st.dataframe(bottom_performing, use_container_width=True)

            # Show raw data filtered to bottom 5 stocks
            bottom_5_stocks = bottom_performing['Stock'].tolist()
            filtered_data = df_parent[df_parent['validated_ticker'].isin(bottom_5_stocks)]
            format_and_display_data(filtered_data, 
                                    title="Show Raw Data for Bottom Performing Stocks",
                                    relevant_columns=['tweet_id', 'author', 'created_date', 'validated_ticker', 'actual_return', 'price_change_pct', 'prediction_correct', 'Result Flag'])
        
        # ... (Remove the Feature Distributions section at the end as it duplicates tab1 plots) ...
        # st.markdown('<div class="sub-header">Feature Distributions</div>', unsafe_allow_html=True)
        # format_and_display_data(df_parent, "Show Raw Data for Feature Distributions")

def create_data_extraction_dashboard():
    if 'twitter_data' not in st.session_state:
        st.session_state.twitter_data = None
    if 'processed_data' not in st.session_state:
        st.session_state.processed_data = None
    
    st.markdown("<h1 class='main-header'>Twitter Data Extraction Dashboard</h1>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<h3 style='color: #1DA1F2; margin-bottom: 15px;'>Run Tweet Extraction</h3>", unsafe_allow_html=True)
        
        extraction_options_col1, extraction_options_col2 = st.columns(2)
        
        with extraction_options_col1:
            sample_mode = st.checkbox("Sample Mode (100 tweets)", value=True, 
                                     help="Enable to fetch only 100 tweets for testing")
        
        with extraction_options_col2:
            extraction_type = st.radio(
                "Extraction Type",
                [
                    "Daily Only", 
                    "Weekly (Last 7 Days)",
                    "New Handles Only", 
                    "Complete Extraction"
                ],
                index=0
            )
        
        button_col1, button_col2, button_col3 = st.columns([1, 2, 1])
        
        with button_col2:
            if st.button("🚀 Start Tweet Extraction", type="primary"):
                log_output = st.empty()
                
                st.markdown("""
                <div style="padding: 10px; background-color: #f8f9fa; border-radius: 5px; margin-top: 10px;">
                    <p style="margin: 0; color: #1DA1F2;"><strong>⏳ Initializing extraction process...</strong></p>
                </div>
                """, unsafe_allow_html=True)
                
                def run_extraction_with_logs():
                    import sys
                    from io import StringIO
                    
                    old_stdout = sys.stdout
                    mystdout = StringIO()
                    sys.stdout = mystdout
                    
                    try:
                        os.environ["SAMPLE_MODE"] = "True" if sample_mode else "False"
                        print(f"🔧 Sample mode: {'Enabled (100 tweets)' if sample_mode else 'Disabled (100,000 tweets)'}")
                        os.environ["EXTRACTION_TYPE"] = extraction_type
                        
                        twitter_handles, twitter_lists = extract_handles_directly()
                        
                        if not twitter_handles:
                            print("❌ No Twitter handles found. Please check the Google Sheet connection.")
                            return mystdout.getvalue(), 0
                        
                        print(f"✅ Using {len(twitter_handles)} Twitter handles for processing")
                        
                        from APIFY_tweet_extraction import TwitterScraper
                        
                        api_token = "apify_api_kdevcdwOVQ5K4HugmeLYlaNgaxeOGG3dkcwc"
                        actor_id = "nfp1fpt5gUlBwPcor"
                        scraper = TwitterScraper(api_token, actor_id=actor_id)
                        
                        max_items = 100 if sample_mode else 100000
                        
                        today = datetime.now().strftime("%Y-%m-%d")
                        start_date = "2023-01-01"
                        
                        tweets = scraper.extract_tweets_with_retry(
                            twitter_handles=twitter_handles,
                            start_date=start_date,
                            end_date=today,
                            max_items=max_items,
                            retries=3
                        )
                        
                        if not tweets:
                            print("❌ No tweets were extracted. Please check the API connection and handles.")
                            return mystdout.getvalue(), 0
                        
                        print(f"✅ Successfully extracted {len(tweets)} tweets")
                        
                    except Exception as e:
                        print(f"Error during extraction: {str(e)}")
                        import traceback
                        traceback.print_exc(file=mystdout)
                        return mystdout.getvalue(), 0
                    finally:
                        sys.stdout = old_stdout
                    
                    return mystdout.getvalue(), 0
                
                with st.spinner("Running tweet extraction... This may take a few minutes."):
                    logs, tweet_count = run_extraction_with_logs()
                    
                    formatted_logs = logs.replace("\n", "<br>")
                    formatted_logs = formatted_logs.replace("✅", "<span style='color: #17BF63'>✅</span>")
                    formatted_logs = formatted_logs.replace("❌", "<span style='color: #E0245E'>❌</span>")
                    formatted_logs = formatted_logs.replace("⚠️", "<span style='color: #FFAD1F'>⚠️</span>")
                    formatted_logs = formatted_logs.replace("🔄", "<span style='color: #1DA1F2'>🔄</span>")
                    formatted_logs = formatted_logs.replace("🆕", "<span style='color: #17BF63'>🆕</span>")
                    formatted_logs = formatted_logs.replace("📊", "<span style='color: #794BC4'>📊</span>")
                    formatted_logs = formatted_logs.replace("📝", "<span style='color: #1DA1F2'>📝</span>")
                    
                    log_output.markdown(f"""
                    <div class='log-container' style='background-color: #1e1e1e; color: #d4d4d4; padding: 15px; border-radius: 8px; font-family: "Consolas", monospace; line-height: 1.5; max-height: 500px; overflow-y: auto;'>
                        {formatted_logs}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if tweet_count > 0:
                        st.success(f"Tweet extraction completed successfully! Extracted {tweet_count} tweets.")
                    else:
                        st.warning("Tweet extraction completed but no tweets were found.")
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<h3 style='color: #1DA1F2; margin-bottom: 15px;'>Twitter Handles</h3>", unsafe_allow_html=True)
        
        try:
            twitter_handles, twitter_lists = extract_handles_directly()
            
            st.markdown(f"""
            <div style="background-color: #E8F5E9; padding: 10px; border-radius: 5px; margin-bottom: 15px;">
                <p style="margin: 0; color: #2E7D32;"><strong>✅ Found {len(twitter_handles)} Twitter handles in Google Sheet</strong></p>
            </div>
            """, unsafe_allow_html=True)
            
            
        except Exception as e:
            st.error(f"Error fetching Twitter handles: {str(e)}")
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("<h2 class='sub-header'>Data Processing & Database Upload</h2>", unsafe_allow_html=True)
    
    process_col, upload_col = st.columns(2)
    
    with process_col:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<h3 style='color: #1DA1F2; margin-bottom: 15px;'>Process Tweets</h3>", unsafe_allow_html=True)
        
        if st.session_state.twitter_data is not None:
            st.markdown(f"""
            <div style="background-color: #E8F5E9; padding: 10px; border-radius: 5px; margin-bottom: 15px;">
                <p style="margin: 0; color: #2E7D32;"><strong>✅ Tweets available for processing</strong></p>
                <p style="margin-top: 5px; color: #2E7D32;">Ready to process {len(st.session_state.twitter_data)} tweets with LLM analysis</p>
            </div>
            """, unsafe_allow_html=True)
            
            openai_api_key = st.text_input(
                "OpenAI API Key", 
                type="password",
                help="Enter your OpenAI API key to use for LLM processing",
                placeholder="sk-..."
            )
            
            num_workers = st.slider(
                "Number of parallel workers", 
                min_value=1, 
                max_value=16, 
                value=8,
                help="Higher values may process faster but use more resources"
            )
            
            st.info(f"""
            💡 **Worker Configuration:**
            - Using {num_workers} parallel workers for processing
            - More workers = faster processing but higher resource usage
            - Recommended: 4-8 workers for most systems
            """)
            
            if st.button("🧠 Process with LLM", use_container_width=True, disabled=not openai_api_key):
                if not openai_api_key:
                    st.error("Please enter your OpenAI API key to continue.")
                else:
                    try:
                        if 'twitter_data' not in st.session_state or st.session_state.twitter_data is None:
                            st.error("❌ No data in session state! Please run extraction first.")
                            return
                        
                        df = st.session_state.twitter_data
                        
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        import openai
                        client = openai.OpenAI(api_key=openai_api_key)
                        
                        num_workers = st.session_state.get("num_workers", 8)
                        
                        def process_tweet(tweet_data):
                            try:
                                text = tweet_data.get('text', '')
                                if not text:
                                    text = tweet_data.get('fullText', '')
                                
                                response = client.chat.completions.create(
                                    model="gpt-4o-mini",
                                    messages=[{"role": "system", "content": """
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
                                    response_format={"type": "json_object"}
                                )
                                
                                llm_response = response.choices[0].message.content
                                author_info = tweet_data.get('author', {})
                                author_name = author_info.get('userName', '') if isinstance(author_info, dict) else ''
                                
                                result = {
                                    'id': tweet_data.get('id', ''),
                                    'text': text,
                                    'author': author_name,
                                    'created_at': tweet_data.get('createdAt', ''),
                                    'llm_response': llm_response,
                                }
                                
                                try:
                                    import json
                                    data = json.loads(llm_response)
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
                        
                        import concurrent.futures
                        import time
                        
                        tweets_to_process = df.to_dict('records')
                        total_tweets = len(tweets_to_process)
                        
                        status_text.text(f"Processing {total_tweets} tweets with {num_workers} workers...")
                        
                        start_time = time.time()
                        processed_tweets = []
                        completed = 0
                        
                        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
                            future_to_tweet = {executor.submit(process_tweet, tweet): tweet for tweet in tweets_to_process}
                            
                            for future in concurrent.futures.as_completed(future_to_tweet):
                                result = future.result()
                                processed_tweets.append(result)
                                
                                completed += 1
                                progress = completed / total_tweets
                                progress_bar.progress(progress)
                                elapsed = time.time() - start_time
                                tweets_per_second = completed / elapsed if elapsed > 0 else 0
                                remaining = (total_tweets - completed) / tweets_per_second if tweets_per_second > 0 else 0
                                
                                if completed % 5 == 0 or completed == total_tweets:
                                    status_text.text(
                                        f"Processed: {completed}/{total_tweets} tweets ({int(progress*100)}%) | "
                                        f"Speed: {tweets_per_second:.1f} tweets/sec | "
                                        f"Est. remaining: {int(remaining)} seconds"
                                    )
                        
                        processed_df = pd.DataFrame(processed_tweets)
                        
                        st.session_state.processed_data = processed_df
                        
                        progress_bar.empty()
                        
                        elapsed_time = time.time() - start_time
                        st.success(f"✅ Successfully processed {len(processed_df)} tweets in {elapsed_time:.1f} seconds!")
                        
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
        
        if st.session_state.processed_data is not None:
            st.markdown("""
            <div style="background-color: #E8F5E9; padding: 10px; border-radius: 5px; margin-bottom: 15px;">
                <p style="margin: 0; color: #2E7D32;"><strong>✅ Processed tweets available</strong></p>
                <p style="margin-top: 5px; color: #2E7D32;">Ready to filter and upload to database</p>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("📤 Filter & Upload", use_container_width=True, type="primary"):
                try:
                    with st.spinner("Preparing tweets for upload..."):
                        df = st.session_state.processed_data.copy()
                        
                        if len(df) == 0:
                            st.warning("No tweets available for processing.")
                            return
                        
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        status_text.text("Step 1/4: Preparing data...")
                        
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
                        
                        progress_bar.progress(25)
                        status_text.text("Step 2/4: Filtering actionable tweets...")
                        
                        if 'tweet_type' not in df.columns:
                            df['tweet_type'] = 'parent'
                        
                        if 'tickers_mentioned' not in df.columns:
                            df['tickers_mentioned'] = df['text'].apply(
                                lambda x: ','.join([word.strip('$') for word in str(x).split() if word.startswith('$')])
                            )
                        
                        if 'sentiment' not in df.columns:
                            df['sentiment'] = 'unknown'
                        
                        try:
                            actionable_tweets = df[
                                (df['tweet_type'] == 'parent') &
                                (df['tickers_mentioned'].notna()) & 
                                (df['tickers_mentioned'] != '') &
                                (df['sentiment'].isin(['bullish', 'bearish']))
                            ].copy()
                        except KeyError as e:
                            st.error(f"Missing column in data: {str(e)}")
                            st.info("Adding required columns and continuing...")
                            
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
                            
                            actionable_tweets = df[
                                (df['tweet_type'] == 'parent') &
                                (df['tickers_mentioned'].notna()) & 
                                (df['tickers_mentioned'] != '') &
                                (df['sentiment'].isin(['bullish', 'bearish']))
                            ].copy()
                        
                        if len(actionable_tweets) == 0:
                            st.warning("No actionable tweets found after filtering.")
                            return
                        
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
                        
                        progress_bar.progress(50)
                        status_text.text("Step 3/4: Preparing data for upload...")
                        
                        st.info(f"Found {len(actionable_tweets)} actionable tweets to upload")
                        
                        sentiment_counts = actionable_tweets['sentiment'].value_counts().to_dict()
                        sentiment_html = "<div style='margin: 10px 0;'><strong>Sentiment distribution:</strong><br>"
                        for sentiment, count in sentiment_counts.items():
                            color = "#4CAF50" if sentiment == "bullish" else "#F44336"
                            sentiment_html += f"<span style='color:{color};'>{sentiment}</span>: {count} tweets<br>"
                        sentiment_html += "</div>"
                        st.markdown(sentiment_html, unsafe_allow_html=True)
                        
                        progress_bar.progress(75)
                        status_text.text("Step 4/4: Uploading to database...")
                        
                        try:
                            import io
                            import sys
                            original_stdout = sys.stdout
                            debug_output = io.StringIO()
                            sys.stdout = debug_output
                            
                            success = upload_to_database(actionable_tweets)
                            
                            sys.stdout = original_stdout
                            
                            upload_logs = debug_output.getvalue()
                            
                            progress_bar.progress(100)
                            
                            if success or "Database upload complete" in upload_logs:
                                st.balloons()
                                st.success(f"✅ Successfully uploaded {len(actionable_tweets)} tweets to database!")
                                
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

def run_prediction_process(df, output_dir='results'):
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        print("Filtering actionable tweets...")
        filtered_data = filter_actionable_tweets(df)
        actionable_tweets = filtered_data['actionable_tweets']
        analysis_tweets = filtered_data['analysis_tweets']
        
        actionable_tweets.to_csv(f'{output_dir}/actionable_tweets.csv', index=False)
        analysis_tweets.to_csv(f'{output_dir}/analysis_tweets.csv', index=False)
        
        print("Adding market validation...")
        validated_tweets = add_market_validation_columns(actionable_tweets, all_tweets=df, output_dir=output_dir)
        validated_tweets.to_csv(f'{output_dir}/validated_tweets.csv', index=False)
        
        print("Analyzing user accuracy...")
        user_accuracy = analyze_user_accuracy(validated_tweets)
        user_accuracy.to_csv(f'{output_dir}/user_accuracy.csv', index=False)
        
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

def main():
    if 'df' not in st.session_state:
        st.session_state.df = None
    
    st.sidebar.title("Navigation")
    st.sidebar.image("https://www.iconpacks.net/icons/2/free-twitter-logo-icon-2429-thumb.png", width=100)
    
    page = st.sidebar.radio(
        "Select a page",
        ["Dashboard", "Trader Profile", "Raw Data", "Data Extraction"]
    )
    
    if page != "Data Extraction":
        try:
            df = load_data()
            st.session_state.df = df
            
            if df is None or df.empty:
                st.warning("No data available in database. Please extract data first.")
                page = "Data Extraction"
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            page = "Data Extraction"
    
    if page == "Dashboard":
        if st.session_state.df is not None:
            create_overview_dashboard(st.session_state.df)
        else:
            st.warning("Please extract data first using the Data Extraction page.")
    elif page == "Trader Profile":
        if st.session_state.df is not None:
            traders = get_traders(st.session_state.df)
            selected_trader = st.sidebar.selectbox("Select Trader", traders)
            create_trader_profile(st.session_state.df, selected_trader)
        else:
            st.warning("Please extract data first using the Data Extraction page.")
    elif page == "Raw Data":
        if st.session_state.df is not None:
            create_raw_data_dashboard(st.session_state.df)
        else:
            st.warning("Please extract data first using the Data Extraction page.")
    elif page == "Data Extraction":
        create_data_extraction_dashboard()
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### About")
    st.sidebar.info(
        "This dashboard analyzes Twitter traders' prediction accuracy "
        "and performance metrics. Data is based on validated stock predictions "
        "from Twitter conversations."
    )
    st.sidebar.markdown("2025 Twitter Trader Analysis")

# -------- INSERT create_raw_data_dashboard HERE --------
def create_raw_data_dashboard(df):
    st.markdown("<h1 class='main-header'>Raw Data Explorer</h1>", unsafe_allow_html=True)
    
    st.markdown("<h2 class='sub-header'>Filters</h2>", unsafe_allow_html=True)

    # --- Filter Row 1 ---
    filter_cols_1 = st.columns(4)
    with filter_cols_1[0]:
        if 'author' in df.columns:
            authors = ["All"] + sorted(df["author"].dropna().unique().tolist())
            selected_author = st.selectbox("Filter by Trader", authors)
        else:
            selected_author = "All"
            st.info("No author column found")

    with filter_cols_1[1]:
        if 'validated_ticker' in df.columns:
            tickers = ["All"] + sorted(df["validated_ticker"].dropna().unique().tolist())
            selected_ticker = st.selectbox("Filter by Ticker", tickers)
        elif 'tickers_mentioned' in df.columns:
             # Handle comma-separated tickers if necessary
            all_tickers = set()
            df['tickers_mentioned'].dropna().apply(lambda x: all_tickers.update(t.strip() for t in str(x).split(',') if t.strip()))
            tickers = ["All"] + sorted(list(all_tickers))
            selected_ticker = st.selectbox("Filter by Ticker", tickers)
        else:
            selected_ticker = "All"
            st.info("No ticker column found")

    with filter_cols_1[2]:
        sentiments = ["All"] + df['sentiment'].dropna().unique().tolist()
        selected_sentiment = st.selectbox("Filter by Sentiment", sentiments)

    with filter_cols_1[3]:
        correctness_options = {
            "All": None, 
            "Correct": True, 
            "Incorrect": False, 
            "Pending": pd.NA
        }
        selected_correctness_label = st.selectbox(
            "Filter by Prediction Correctness", 
            options=list(correctness_options.keys()),
            index=0
        )
        selected_correctness = correctness_options[selected_correctness_label]

    # --- Filter Row 2 ---
    filter_cols_2 = st.columns(4)
    with filter_cols_2[0]:
        if 'created_date' in df.columns:
            min_date = df['created_date'].min().date()
            max_date = df['created_date'].max().date()
            
            # Check if min_date and max_date are the same
            if min_date == max_date:
                st.info(f"Data available only for {min_date}")
                selected_date_range = (min_date, max_date)
            elif min_date > max_date: # Handle potential data issue
                 st.warning("Min date is after max date. Check data.")
                 selected_date_range = (min_date, min_date) # Default to min_date
            else:
                selected_date_range = st.date_input(
                    "Filter by Date Range",
                    value=(min_date, max_date),
                    min_value=min_date,
                    max_value=max_date,
                )
        else:
            selected_date_range = None
            st.info("No date column for filtering")

    # --- Apply Filters ---
    filtered_df = df.copy()

    if selected_author != "All":
        if 'author' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df["author"] == selected_author]

    if selected_ticker != "All":
        if 'validated_ticker' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df["validated_ticker"] == selected_ticker]
        elif 'tickers_mentioned' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['tickers_mentioned'].apply(
                lambda x: selected_ticker in [t.strip() for t in str(x).split(',')] if pd.notna(x) else False
            )]

    if selected_sentiment != "All":
        filtered_df = filtered_df[filtered_df["sentiment"] == selected_sentiment]

    # Handle the prediction correctness filter, including NA for Pending
    if selected_correctness_label != "All":
        if pd.isna(selected_correctness):
            filtered_df = filtered_df[filtered_df['prediction_correct'].isna()]
        else:
            filtered_df = filtered_df[filtered_df['prediction_correct'] == selected_correctness]

    if selected_date_range and 'created_date' in filtered_df.columns and len(selected_date_range) == 2:
        start_date, end_date = selected_date_range
        # Convert to datetime if they are date objects
        start_datetime = pd.to_datetime(start_date)
        end_datetime = pd.to_datetime(end_date) + pd.Timedelta(days=1) # Make end date inclusive
        filtered_df = filtered_df[(filtered_df['created_date'] >= start_datetime) & (filtered_df['created_date'] < end_datetime)]

    st.markdown("<h2 class='sub-header'>Filtered Data</h2>", unsafe_allow_html=True)

    # Use helper function to display the filtered data
    # Note: Pass the filtered_df to the helper
    format_and_display_data(filtered_df, "Filtered Raw Data Details") 
    
    # --- Recalculate display columns for download ---
    display_df_for_download = filtered_df.copy()
    display_columns = [
        'tweet_id', 'tweet_link', 'created_date', 'author', 'text', 
        'sentiment', 'time_horizon', 'trade_type', 'validated_ticker',
        'prediction_correct', 'Result Flag', 'actual_return', 'price_change_pct',
        'start_date', 'end_date', 'start_price', 'end_price'
    ]
    if 'tweet_id' in display_df_for_download.columns and 'author' in display_df_for_download.columns and 'tweet_link' not in display_df_for_download.columns:
        display_df_for_download['tweet_link'] = display_df_for_download.apply(lambda row: f"https://twitter.com/{row['author']}/status/{row['tweet_id']}" if pd.notna(row['tweet_id']) and pd.notna(row['author']) else None, axis=1)
    if 'prediction_correct' in display_df_for_download.columns and 'Result Flag' not in display_df_for_download.columns:
        def correctness_flag(val): # Redefine locally for robustness
            if pd.isna(val): return "⏳ Pending"
            elif val == True: return "✅ Correct"
            elif val == False: return "❌ Incorrect"
            else: return "❓ Unknown"
        display_df_for_download['Result Flag'] = display_df_for_download['prediction_correct'].apply(correctness_flag)
    
    # Filter to only available columns for download
    available_columns_for_download = [col for col in display_columns if col in display_df_for_download.columns]
    
    if 'tweet_id' in display_df_for_download.columns:
         display_df_for_download['tweet_id'] = display_df_for_download['tweet_id'].apply(lambda x: str(int(x)) if pd.notna(x) and isinstance(x, (int, float)) else str(x))
    
    display_df_for_download = display_df_for_download[available_columns_for_download]
    # --- End Recalculation ---

    csv = display_df_for_download.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Filtered Data as CSV",
        data=csv,
        file_name=f"twitter_filtered_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
    )
# -------- END INSERT --------

def load_twitter_llm_module():
    try:
        spec = importlib.util.spec_from_file_location("twitter_llm", "twitter-llms-optimized.py")
        twitter_llm = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(twitter_llm)
        return twitter_llm
    except Exception as e:
        st.error(f"Error loading Twitter LLM module: {str(e)}")
        return None

if __name__ == "__main__":
    if check_password():
        main()
