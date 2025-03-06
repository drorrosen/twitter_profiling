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
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Twitter Trader Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
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
        # First run, show inputs for username + password.
        st.markdown("""
        <style>
        /* Remove the white container */
        div.css-1r6slb0.e1tzin5v2 {
            background-color: transparent;
            border: none;
            box-shadow: none;
        }
        
        /* Set page background */
        .stApp {
            background-color: #f5f8fa;
        }
        
        .login-container {
            max-width: 450px;
            margin: 0 auto;
            padding: 2.5rem;
            border-radius: 15px;
            box-shadow: 0 8px 20px rgba(0, 0, 0, 0.15);
            background-color: white;
            margin-top: 3rem;
        }
        .login-header {
            text-align: center;
            margin-bottom: 2rem;
            color: #1DA1F2;
            font-size: 2.2rem;
            font-weight: 700;
        }
        .login-subheader {
            text-align: center;
            margin-bottom: 2rem;
            color: #657786;
            font-size: 1.1rem;
        }
        .login-logo {
            display: block;
            margin: 0 auto 1.5rem auto;
            width: 80px;
            height: 80px;
        }
        .stButton > button {
            background-color: #1DA1F2;
            color: white;
            font-weight: bold;
            border: none;
            border-radius: 30px;
            padding: 0.5rem 1rem;
            width: 100%;
            margin-top: 1rem;
        }
        .stButton > button:hover {
            background-color: #0c85d0;
        }
        .stTextInput > div > div > input {
            border-radius: 30px;
            padding: 1rem;
            border: 1px solid #AAB8C2;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Use empty to create space at the top
        st.empty()
        
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            st.markdown('<div class="login-container">', unsafe_allow_html=True)
            st.markdown('<img src="https://www.iconpacks.net/icons/2/free-twitter-logo-icon-2429-thumb.png" class="login-logo">', unsafe_allow_html=True)
            st.markdown('<h1 class="login-header">Twitter Trader Analysis</h1>', unsafe_allow_html=True)
            st.markdown('<p class="login-subheader">Enter your credentials to access the dashboard</p>', unsafe_allow_html=True)
            
            st.text_input("Username", key="username", placeholder="Enter username")
            st.text_input("Password", type="password", key="password", placeholder="Enter password")
            st.button("Login", on_click=password_entered)
            
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
</style>
""", unsafe_allow_html=True)

# Function to load data
@st.cache_data
def load_data(filepath='results/validated_predictions.csv'):
    try:
        df = pd.read_csv(filepath, low_memory=False)
        # Convert date columns
        df['created_date'] = pd.to_datetime(df['created_date'], errors='coerce')
        df['prediction_date'] = pd.to_datetime(df['prediction_date'], errors='coerce')
        # Convert selected columns to numeric
        numeric_columns = [
            'confidence', 'price_change_pct', 'actual_return',
            'likes', 'retweets', 'replies_count',
            'views', 'author_followers', 'author_following'
        ]
        df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')
        
        # Convert prediction_correct to boolean
        df['prediction_correct'] = df['prediction_correct'].map(
            {True: True, 'True': True, 'true': True, 
             False: False, 'False': False, 'false': False}
        )
        
        # Set prediction_correct to None for future predictions
        df.loc[df['prediction_date'] > pd.Timestamp.today(), 'prediction_correct'] = None
        
        # Remove conversation IDs where parent tweet has prediction_correct as None
        conv_to_remove = df[
            (df['prediction_correct'].isna()) & (df['tweet_type'] == 'parent')
        ]['conversation_id']
        df = df[~df['conversation_id'].isin(conv_to_remove)]
        
        # Remove parent tweets with missing validated ticker
        df = df[~((df['validated_ticker'].isna()) & (df['tweet_type'] == 'parent'))]
        
        # Create a new feature for prediction score - safely handling NaN values
        def calculate_prediction_score(row):
            if pd.isna(row['prediction_correct']) or pd.isna(row['price_change_pct']):
                return None
            
            multiplier = 1 if row['prediction_correct'] else -1
            return abs(float(row['price_change_pct'])) * multiplier
        
        df['prediction_score'] = df.apply(calculate_prediction_score, axis=1)
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# Function to get unique traders
def get_traders(df):
    return sorted(df['author'].unique())

# Function to filter data for a specific trader
def filter_trader_data(df, trader_name):
    df_user = df[df['author'] == trader_name].copy()
    conv_starter = df_user[df_user['tweet_type'] == 'parent']['conversation_id']
    df_user = df_user.loc[df_user['conversation_id'].isin(conv_starter)]
    df_parent = df_user[df_user['tweet_type'] == 'parent']
    
    # Ensure prediction_correct is Boolean
    df_user['prediction_correct'] = (
        df_user['prediction_correct']
        .astype(str)
        .str.lower()
        .map({'true': True, 'false': False})
    )
    
    # Calculate sentiment consistency per conversation
    parent_sentiment = df_user[df_user['tweet_type'] == 'parent'][['conversation_id', 'sentiment']].rename(
        columns={'sentiment': 'parent_sentiment'}
    )
    df_user = df_user.merge(parent_sentiment, on='conversation_id', how='left')
    df_user['consistent_sentiment'] = (df_user['sentiment'] == df_user['parent_sentiment']).astype(int)
    
    # Compute Weighted Profitability Score per conversation
    prediction_score_sum = (
        df_user.groupby('conversation_id')['prediction_score']
        .sum()
        .reset_index(name='Weighted Profitability Score')
    )
    df_user = df_user.merge(prediction_score_sum, on='conversation_id', how='left')
    
    return df_user, df_parent

# Function to compute trader profile summary
def compute_profile_summary(df_user, df_parent):
    consistency_by_conv = df_user.groupby('conversation_id')['consistent_sentiment'].mean()
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
        "Sentiment Consistency per Conversation (%)": consistency_by_conv.mean() * 100,
        "Weighted Profitability Mean": df_user.drop_duplicates(subset='conversation_id')[
            'Weighted Profitability Score'
        ].mean(),
    }
    return profile_summary

# Function to analyze all traders
def analyze_all_traders(df):
    all_traders = get_traders(df)
    trader_metrics = []
    
    for trader in all_traders:
        df_user, df_parent = filter_trader_data(df, trader)
        
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
    st.markdown('<div class="main-header">Twitter Trader Analysis Dashboard</div>', unsafe_allow_html=True)
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{len(df["author"].unique())}</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Unique Traders</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{len(df)}</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Total Tweets</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        accuracy = df[df['prediction_correct'].notna()]['prediction_correct'].mean() * 100
        color_class = "positive" if accuracy > 50 else "negative"
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value {color_class}">{accuracy:.1f}%</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Overall Accuracy</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        avg_return = df[df['actual_return'].notna()]['actual_return'].mean()
        color_class = "positive" if avg_return > 0 else "negative"
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value {color_class}">{avg_return:.2f}%</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Average Return</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Analyze all traders
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
        sentiment_counts = df['sentiment'].value_counts().reset_index()
        sentiment_counts.columns = ['Sentiment', 'Count']
        
        colors = {'bullish': '#17BF63', 'bearish': '#E0245E', 'neutral': '#AAB8C2'}
        
        fig = px.pie(
            sentiment_counts,
            values='Count',
            names='Sentiment',
            color='Sentiment',
            color_discrete_map=colors,
            title='Overall Sentiment Distribution'
        )
        
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(height=400)
        
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
            time_horizon_counts = df_parent['time_horizon'].value_counts(normalize=True) * 100
            
            fig = px.pie(
                values=time_horizon_counts.values,
                names=time_horizon_counts.index,
                title='Time Horizon Distribution',
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            
            fig.update_traces(textposition='inside', textinfo='percent+label')
            fig.update_layout(height=350)
            
            st.plotly_chart(fig, use_container_width=True)
        
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

# Main app
def main():
    # Load data
    df = load_data()
    
    if df is None:
        st.error("Failed to load data. Please check the data file path.")
        return
    
    # Sidebar
    st.sidebar.title("Navigation")
    
    # Add logo or image
    st.sidebar.image("https://www.iconpacks.net/icons/2/free-twitter-logo-icon-2429-thumb.png", width=100)
    
    # Dashboard selection
    dashboard = st.sidebar.radio("Select Dashboard", ["Overview", "Trader Profile"])
    
    if dashboard == "Overview":
        create_overview_dashboard(df)
    else:
        # Trader selection
        traders = get_traders(df)
        selected_trader = st.sidebar.selectbox("Select Trader", traders)
        
        # Display trader profile
        create_trader_profile(df, selected_trader)
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("### About")
    st.sidebar.info(
        "This dashboard analyzes Twitter traders' prediction accuracy "
        "and performance metrics. Data is based on validated stock predictions "
        "from Twitter conversations."
    )
    st.sidebar.markdown("© 2023 Twitter Trader Analysis")

# Only show the app if the user has entered the correct password
if check_password():
    if __name__ == "__main__":
        main()