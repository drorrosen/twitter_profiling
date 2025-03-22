import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import os
import re
import yfinance as yf
import time
import random
import concurrent.futures
from functools import partial
import psycopg2
from sqlalchemy import create_engine, text, Table, MetaData, select, insert
from psycopg2.extras import execute_values

def load_processed_data(file_path='all_tweets_analysis.csv'):
    """
    Load the processed tweet data from the CSV file.
    """
    print(f"Loading data from {file_path}...")
    
    try:
        df = pd.read_csv(file_path)
        print(f"Loaded {len(df)} tweets.")
        return df
    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
        return None
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def filter_actionable_tweets(df):
    """
    Filter tweets to find those with actionable trading signals.
    Modified to retain more records while maintaining quality.
    
    Parent tweets must have:
    1. Tickers mentioned
    2. Sentiment (now includes neutral along with bullish/bearish)
    
    Time horizon and ticker count filtering are now optional and configurable.
    """
    print("\nFiltering tweets for actionable trading signals...")
    
    # Configuration parameters - adjust these to control filtering strictness
    FILTER_BY_TIME_HORIZON = False  # Set to True to enforce time horizon filter
    FILTER_BY_TICKER_COUNT = False  # Set to True to enforce maximum ticker count
    INCLUDE_NEUTRAL_SENTIMENT = True  # Set to True to include neutral sentiment
    MAX_TICKERS = 10  # Maximum number of tickers in a tweet (if filtering by ticker count)
    
    # Count original parent tweets
    original_parents = df[df['tweet_type'] == 'parent']
    print(f"Original parent tweets: {len(original_parents)} ({len(original_parents)/len(df)*100:.1f}%)")
    
    # 1. Filter for parent tweets that mention stock tickers (required)
    has_tickers = df['tickers_mentioned'].notna() & (df['tickers_mentioned'] != '')
    parent_ticker_tweets = original_parents[has_tickers]
    
    print(f"Parent tweets with tickers: {len(parent_ticker_tweets)} ({len(parent_ticker_tweets)/len(original_parents)*100:.1f}% of original parents)")
    print(f"Parents lost in ticker filtering: {len(original_parents) - len(parent_ticker_tweets)}")
    
    # 2. Filter for actionable sentiment (can include neutral now)
    if INCLUDE_NEUTRAL_SENTIMENT:
        actionable_sentiment = ['bullish', 'bearish', 'neutral']
        print("Including neutral sentiment in actionable tweets")
    else:
        actionable_sentiment = ['bullish', 'bearish']
    
    parent_sentiment_tweets = parent_ticker_tweets[parent_ticker_tweets['sentiment'].isin(actionable_sentiment)]
    
    print(f"Parent tweets with actionable sentiment: {len(parent_sentiment_tweets)} ({len(parent_sentiment_tweets)/len(parent_ticker_tweets)*100:.1f}% of parents with tickers)")
    print(f"Parents lost in sentiment filtering: {len(parent_ticker_tweets) - len(parent_sentiment_tweets)}")
    
    # Save parent tweets after essential filtering (tickers + sentiment)
    filtered_parents = parent_sentiment_tweets
    
    # 3. Optional: Filter for tweets with valid time horizons
    if FILTER_BY_TIME_HORIZON:
        # Include all valid time horizons from the LLM classification
        valid_time_horizons = [
            'intraday', 'daily', 'weekly', 
            'short_term', 'medium_term', 'long_term'
        ]
        parent_horizon_tweets = filtered_parents[
            filtered_parents['time_horizon'].notna() & 
            filtered_parents['time_horizon'].isin(valid_time_horizons)
        ]
        
        print(f"Parent tweets with valid time horizons: {len(parent_horizon_tweets)} ({len(parent_horizon_tweets)/len(filtered_parents)*100:.1f}% of filtered parents)")
        print(f"Parents lost in time horizon filtering: {len(filtered_parents) - len(parent_horizon_tweets)}")
        
        filtered_parents = parent_horizon_tweets
    
    # 4. Optional: Filter out tweets with too many tickers (likely spam)
    if FILTER_BY_TICKER_COUNT:
        # Count the number of tickers in each tweet
        def count_tickers(ticker_str):
            if not isinstance(ticker_str, str) or not ticker_str:
                return 0
            return len(ticker_str.split(','))
        
        filtered_parents['ticker_count'] = filtered_parents['tickers_mentioned'].apply(count_tickers)
        parent_focused_tweets = filtered_parents[filtered_parents['ticker_count'] <= MAX_TICKERS]
        
        print(f"Parent tweets with focused ticker count (≤{MAX_TICKERS}): {len(parent_focused_tweets)} ({len(parent_focused_tweets)/len(filtered_parents)*100:.1f}% of filtered parents)")
        print(f"Parents filtered out as potential spam: {len(filtered_parents) - len(parent_focused_tweets)}")
        
        filtered_parents = parent_focused_tweets
    
    # Count distribution of sentiment among filtered parent tweets
    parent_sentiment_counts = filtered_parents['sentiment'].value_counts()
    print("\nSentiment distribution among filtered parent tweets:")
    for sentiment, count in parent_sentiment_counts.items():
        print(f"  - {sentiment}: {count} ({count/len(filtered_parents)*100:.1f}%)")
    
    # Final set of actionable parent tweets
    actionable_parents = filtered_parents
    
    # Find ALL replies to actionable conversations, regardless of content
    actionable_convs = set(actionable_parents['conversation_id'].unique())
    all_replies = df[
        (df['tweet_type'] == 'reply') & 
        (df['conversation_id'].isin(actionable_convs))
    ]
    print(f"All reply tweets in actionable conversations: {len(all_replies)} ({len(all_replies)/len(df)*100:.1f}%)")
    
    # Include all replies with tickers mentioned for the database set
    ticker_replies = all_replies[all_replies['tickers_mentioned'].notna() & (all_replies['tickers_mentioned'] != '')]
    print(f"Reply tweets with tickers: {len(ticker_replies)} ({len(ticker_replies)/len(all_replies)*100:.1f}% of replies)")
    
    # Create two sets:
    # 1. Database set: Parents + replies with tickers (meets database constraints)
    database_tweets = pd.concat([actionable_parents, ticker_replies]).drop_duplicates()
    
    # For analysis, include ALL replies regardless of whether they have tickers
    analysis_tweets = pd.concat([actionable_parents, all_replies]).drop_duplicates()
    
    print(f"\nTotal tweets for database upload: {len(database_tweets)}")
    print(f"Total tweets for analysis: {len(analysis_tweets)}")
    
    # Final summary
    print("\n=== SUMMARY OF TWEET FILTERING ===")
    print(f"Original parent tweets: {len(original_parents)}")
    print(f"Parents with tickers: {len(parent_ticker_tweets)}")
    print(f"Parents with actionable sentiment: {len(parent_sentiment_tweets)}")
    print(f"Final actionable parents: {len(actionable_parents)}")
    print(f"Percentage retained: {len(actionable_parents)/len(original_parents)*100:.1f}%")
    print(f"Total database tweets: {len(database_tweets)}")
    
    # Count tweet types in database set
    database_tweet_types = database_tweets['tweet_type'].value_counts()
    print("\nTweet types in database set:")
    for tweet_type, count in database_tweet_types.items():
        print(f"  - {tweet_type}: {count} ({count/len(database_tweets)*100:.1f}%)")
    
    # Return both sets
    return {
        'actionable_tweets': database_tweets,  # For database upload (has tickers)
        'analysis_tweets': analysis_tweets     # For analysis (includes all replies)
    }

def analyze_user_accuracy(df, min_tweets=5):
    """
    Analyze users based on their tweet frequency, sentiment consistency, and prediction accuracy.
    """
    if df is None or df.empty:
        print("No data to analyze.")
        return None
    
    print("\nAnalyzing user accuracy and consistency...")
    
    # Group by user and count their tweets
    user_counts = df['author'].value_counts()
    active_users = user_counts[user_counts >= min_tweets].index.tolist()
    
    print(f"Found {len(active_users)} users with at least {min_tweets} actionable tweets.")
    
    # Create a dataframe to store user metrics
    user_metrics = []
    
    for user in active_users:
        user_tweets = df[df['author'] == user]
        
        # Calculate sentiment distribution
        sentiment_counts = user_tweets['sentiment'].value_counts(normalize=True)
        bullish_pct = sentiment_counts.get('bullish', 0) * 100
        bearish_pct = sentiment_counts.get('bearish', 0) * 100
        
        # Calculate consistency (how often they stick with one sentiment)
        consistency = max(bullish_pct, bearish_pct)
        
        # Calculate prediction accuracy if available
        prediction_accuracy = None
        if 'prediction_correct' in user_tweets.columns:
            validated_tweets = user_tweets[user_tweets['prediction_correct'].notna()]
            if len(validated_tweets) >= 3:  # Require at least 3 validated predictions
                prediction_accuracy = validated_tweets['prediction_correct'].mean() * 100
        
        # Store metrics
        metrics = {
            'user': user,
            'tweet_count': len(user_tweets),
            'bullish_pct': bullish_pct,
            'bearish_pct': bearish_pct,
            'consistency': consistency,
            'most_mentioned_tickers': get_top_tickers(user_tweets, 3)
        }
        
        # Add prediction accuracy if available
        if prediction_accuracy is not None:
            metrics['prediction_accuracy'] = prediction_accuracy
            metrics['validated_count'] = len(validated_tweets)
        
        user_metrics.append(metrics)
    
    # Convert to dataframe and sort by prediction accuracy if available, otherwise by consistency
    user_df = pd.DataFrame(user_metrics)
    if 'prediction_accuracy' in user_df.columns and user_df['prediction_accuracy'].notna().any():
        user_df = user_df.sort_values('prediction_accuracy', ascending=False)
    else:
        user_df = user_df.sort_values('consistency', ascending=False)
    
    return user_df

def get_top_tickers(df, n=5):
    """
    Get the top N most mentioned tickers in a dataframe.
    """
    all_tickers = []
    
    for tickers_str in df['tickers_mentioned']:
        if isinstance(tickers_str, str) and tickers_str:
            all_tickers.extend(tickers_str.split(','))
    
    if not all_tickers:
        return ""
    
    ticker_counts = pd.Series(all_tickers).value_counts()
    top_tickers = ticker_counts.head(n).index.tolist()
    
    return ','.join(top_tickers)

def analyze_ticker_sentiment(df):
    """
    Analyze sentiment for each ticker mentioned in tweets.
    """
    if df is None or df.empty:
        print("No data to analyze ticker sentiment.")
        return None
    
    print("\nAnalyzing ticker sentiment...")
    
    # Explode the tickers column to get one row per ticker
    ticker_df = df.copy()
    ticker_df['ticker'] = ticker_df['tickers_mentioned'].str.split(',')
    ticker_df = ticker_df.explode('ticker')
    
    # Remove empty tickers
    ticker_df = ticker_df[ticker_df['ticker'].notna() & (ticker_df['ticker'] != '')]
    
    # Get the most mentioned tickers
    ticker_counts = ticker_df['ticker'].value_counts()
    popular_tickers = ticker_counts[ticker_counts >= 5].index.tolist()
    
    print(f"Found {len(popular_tickers)} tickers with 5+ mentions")
    
    # Analyze sentiment for each popular ticker
    ticker_sentiment = []
    
    for ticker in popular_tickers:
        ticker_tweets = ticker_df[ticker_df['ticker'] == ticker]
        
        # Calculate sentiment distribution
        sentiment_counts = ticker_tweets['sentiment'].value_counts(normalize=True)
        bullish_pct = sentiment_counts.get('bullish', 0) * 100
        bearish_pct = sentiment_counts.get('bearish', 0) * 100
        
        # Calculate sentiment ratio (bullish to bearish)
        if bearish_pct > 0:
            sentiment_ratio = bullish_pct / bearish_pct
        else:
            sentiment_ratio = float('inf') if bullish_pct > 0 else 0
        
        # Calculate weighted sentiment
        weighted_sentiment = 0
        for _, tweet in ticker_tweets.iterrows():
            if tweet['sentiment'] == 'bullish':
                weighted_sentiment += 1
            elif tweet['sentiment'] == 'bearish':
                weighted_sentiment -= 1
        
        weighted_sentiment = weighted_sentiment / len(ticker_tweets) * 100
        
        ticker_sentiment.append({
            'ticker': ticker,
            'mentions': len(ticker_tweets),
            'bullish_pct': bullish_pct,
            'bearish_pct': bearish_pct,
            'neutral_pct': 100 - bullish_pct - bearish_pct,
            'sentiment_ratio': sentiment_ratio,
            'weighted_sentiment': weighted_sentiment
        })
    
    # Convert to dataframe and sort by weighted sentiment
    ticker_df = pd.DataFrame(ticker_sentiment)
    ticker_df = ticker_df.sort_values('weighted_sentiment', ascending=False)
    
    return ticker_df

def generate_trading_signals(ticker_df, user_df):
    """
    Generate trading signals based on ticker sentiment and user accuracy.
    """
    if ticker_df is None or ticker_df.empty:
        print("No ticker data to generate signals.")
        return None
    
    print("\nGenerating trading signals...")
    
    # Filter for tickers with strong sentiment
    strong_bullish = ticker_df[ticker_df['weighted_sentiment'] > 50]
    strong_bearish = ticker_df[ticker_df['weighted_sentiment'] < -50]
    
    # Generate buy signals
    buy_signals = strong_bullish.copy()
    buy_signals['signal'] = 'BUY'
    buy_signals['strength'] = buy_signals['weighted_sentiment'].apply(
        lambda x: 'Strong' if x > 75 else 'Moderate')
    
    # Generate sell signals
    sell_signals = strong_bearish.copy()
    sell_signals['signal'] = 'SELL'
    sell_signals['strength'] = sell_signals['weighted_sentiment'].apply(
        lambda x: 'Strong' if x < -75 else 'Moderate')
    
    # Combine signals
    signals = pd.concat([buy_signals, sell_signals])
    
    # Sort by absolute weighted sentiment
    signals['abs_sentiment'] = signals['weighted_sentiment'].abs()
    signals = signals.sort_values('abs_sentiment', ascending=False)
    
    # Add user consensus information if available
    if user_df is not None and not user_df.empty:
        top_users = user_df[user_df['consistency'] > 70]['user'].tolist()
        signals['top_user_consensus'] = signals['ticker'].apply(
            lambda ticker: check_user_consensus(ticker, top_users, ticker_df))
    
    return signals[['ticker', 'signal', 'strength', 'weighted_sentiment', 
                   'bullish_pct', 'bearish_pct', 'mentions']]

def check_user_consensus(ticker, top_users, ticker_df):
    """
    Check if top users agree on the sentiment for a ticker.
    """
    # This is a placeholder - in a real implementation, you would check
    # if the top users have tweeted about this ticker and if they agree
    return "Unknown"  # Would return "Agree", "Disagree", or "Mixed"

def visualize_results(ticker_sentiment, user_metrics, df=None, output_dir='results'):
    """
    Create visualizations of the analysis results.
    """
    if ticker_sentiment is None or user_metrics is None:
        print("No data to visualize.")
        return
    
    print("\nCreating visualizations...")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Top tickers by sentiment
    plt.figure(figsize=(12, 8))
    top_positive = ticker_sentiment.nlargest(10, 'weighted_sentiment')
    top_negative = ticker_sentiment.nsmallest(10, 'weighted_sentiment')
    
    combined = pd.concat([top_positive, top_negative])
    sns.barplot(x='ticker', y='weighted_sentiment', data=combined)
    plt.title('Top Bullish and Bearish Tickers')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/top_tickers_sentiment.png')
    
    # 2. User consistency chart
    plt.figure(figsize=(12, 8))
    top_users = user_metrics.nlargest(15, 'consistency')
    sns.barplot(x='user', y='consistency', data=top_users)
    plt.title('Most Consistent Users')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/user_consistency.png')
    
    # 3. Sentiment distribution for top tickers
    plt.figure(figsize=(14, 10))
    top_mentioned = ticker_sentiment.nlargest(15, 'mentions')
    
    # Create a stacked bar chart
    top_mentioned_sorted = top_mentioned.sort_values('bullish_pct', ascending=False)
    
    # Create the stacked bars
    bar_width = 0.8
    indices = np.arange(len(top_mentioned_sorted))
    
    plt.bar(indices, top_mentioned_sorted['bullish_pct'], bar_width, 
            label='Bullish', color='green', alpha=0.7)
    plt.bar(indices, top_mentioned_sorted['bearish_pct'], bar_width,
            bottom=top_mentioned_sorted['bullish_pct'], 
            label='Bearish', color='red', alpha=0.7)
    plt.bar(indices, top_mentioned_sorted['neutral_pct'], bar_width,
            bottom=top_mentioned_sorted['bullish_pct'] + top_mentioned_sorted['bearish_pct'],
            label='Neutral', color='gray', alpha=0.7)
    
    plt.xlabel('Ticker')
    plt.ylabel('Percentage')
    plt.title('Sentiment Distribution for Most Mentioned Tickers')
    plt.xticks(indices, top_mentioned_sorted['ticker'], rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{output_dir}/ticker_sentiment_distribution.png')
    
    # 4. Prediction timeline visualizations (if data available)
    if df is not None and 'prediction_date' in df.columns:
        visualize_prediction_timeline(df, output_dir)
    
    print(f"Visualizations saved to {output_dir} directory.")

def save_results(filtered_data, user_metrics, ticker_sentiment, trading_signals, conversation_analysis=None, output_dir='results'):
    """
    Save analysis results to CSV files.
    """
    if filtered_data is None:
        print("No filtered data to save.")
        return
    
    print("\nSaving results to CSV files...")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save filtered tweets
    filtered_data['actionable_tweets'].to_csv(f'{output_dir}/actionable_tweets.csv', index=False)
    
    # Save user metrics
    if user_metrics is not None and not user_metrics.empty:
        user_metrics.to_csv(f'{output_dir}/user_metrics.csv', index=False)
    
    # Save ticker sentiment
    if ticker_sentiment is not None and not ticker_sentiment.empty:
        ticker_sentiment.to_csv(f'{output_dir}/ticker_sentiment.csv', index=False)
    
    # Save trading signals
    if trading_signals is not None and not trading_signals.empty:
        trading_signals.to_csv(f'{output_dir}/trading_signals.csv', index=False)
    
    # Save conversation analysis
    if conversation_analysis is not None and not conversation_analysis.empty:
        conversation_analysis.to_csv(f'{output_dir}/conversation_analysis.csv', index=False)
    
    print(f"Results saved to {output_dir} directory.")

def explore_data_columns(df):
    """
    Explore the unique values in each column of the dataframe.
    """
    print("\n=== Column Exploration ===")
    
    for column in df.columns:
        try:
            # For columns that might contain lists or complex data
            if column in ['tickers_mentioned', 'company_names', 'stocks']:
                print(f"\nColumn: {column}")
                # Count non-empty values
                non_empty = df[column].apply(lambda x: len(str(x)) > 0 and str(x) != 'nan').sum()
                print(f"Non-empty values: {non_empty} ({non_empty/len(df)*100:.1f}%)")
                
                # Get sample of unique values
                unique_samples = []
                for val in df[column].dropna().sample(min(10, df[column].count())):
                    if val and str(val) != 'nan':
                        unique_samples.append(val)
                
                print(f"Sample values: {unique_samples}")
            else:
                # For regular columns
                unique_vals = df[column].nunique()
                print(f"\nColumn: {column}")
                print(f"Unique values: {unique_vals}")
                
                # For columns with few unique values, show them all
                if unique_vals <= 20:
                    value_counts = df[column].value_counts(dropna=False)
                    for val, count in value_counts.items():
                        print(f"  {val}: {count} ({count/len(df)*100:.1f}%)")
                # For columns with many unique values, show top 10
                else:
                    print("Top 10 most common values:")
                    value_counts = df[column].value_counts(dropna=False).head(10)
                    for val, count in value_counts.items():
                        print(f"  {val}: {count} ({count/len(df)*100:.1f}%)")
        except Exception as e:
            print(f"\nColumn: {column}")
            print(f"Error exploring column: {e}")
    
    print("\n=== End of Column Exploration ===")

def analyze_conversation_threads(df):
    """
    Analyze conversation threads to identify influential parent tweets and their replies.
    """
    if df is None or df.empty:
        print("No data to analyze.")
        return None
    
    print("\nAnalyzing conversation threads...")
    
    # Group by conversation_id
    conversation_groups = df.groupby('conversation_id')
    
    # Collect conversation metrics
    conversation_metrics = []
    
    for conv_id, conv_tweets in conversation_groups:
        # Find the parent tweet
        parent_tweets = conv_tweets[conv_tweets['tweet_type'] == 'parent']
        
        if len(parent_tweets) == 0:
            continue  # Skip if no parent tweet found
        
        # Use the first parent tweet if multiple exist
        parent_tweet = parent_tweets.iloc[0]
        
        # Count replies
        replies = conv_tweets[conv_tweets['tweet_type'] == 'reply']
        reply_count = len(replies)
        
        # Calculate engagement metrics
        total_likes = conv_tweets['likes'].sum()
        total_retweets = conv_tweets['retweets'].sum()
        
        # Calculate sentiment distribution in replies
        if reply_count > 0:
            bullish_replies = len(replies[replies['sentiment'] == 'bullish'])
            bearish_replies = len(replies[replies['sentiment'] == 'bearish'])
            neutral_replies = len(replies[replies['sentiment'] == 'neutral'])
            
            bullish_pct = bullish_replies / reply_count * 100 if reply_count > 0 else 0
            bearish_pct = bearish_replies / reply_count * 100 if reply_count > 0 else 0
            neutral_pct = neutral_replies / reply_count * 100 if reply_count > 0 else 0
        else:
            bullish_pct = bearish_pct = neutral_pct = 0
        
        # Check if parent tweet has actionable content
        has_ticker = pd.notna(parent_tweet['tickers_mentioned']) and isinstance(parent_tweet['tickers_mentioned'], str) and parent_tweet['tickers_mentioned'].strip() != ''
        
        actionable_trade_types = ['trade_suggestion', 'portfolio_update', 'analysis']
        is_actionable = parent_tweet['trade_type'] in actionable_trade_types
        
        # Store metrics
        conversation_metrics.append({
            'conversation_id': conv_id,
            'parent_author': parent_tweet['author'],
            'parent_text': parent_tweet['text'][:100] + '...' if len(parent_tweet['text']) > 100 else parent_tweet['text'],
            'parent_sentiment': parent_tweet['sentiment'],
            'parent_trade_type': parent_tweet['trade_type'],
            'parent_tickers': parent_tweet['tickers_mentioned'],
            'reply_count': reply_count,
            'total_likes': total_likes,
            'total_retweets': total_retweets,
            'bullish_reply_pct': bullish_pct,
            'bearish_reply_pct': bearish_pct,
            'neutral_reply_pct': neutral_pct,
            'has_ticker': has_ticker,
            'is_actionable': is_actionable
        })
    
    # Convert to dataframe
    conv_df = pd.DataFrame(conversation_metrics)
    
    if conv_df.empty:
        print("No conversation threads found.")
        return conv_df
    
    # Filter for conversations with actionable parent tweets that mention tickers
    actionable_convs = conv_df[(conv_df['has_ticker']) & (conv_df['is_actionable'])]
    
    print(f"Found {len(conv_df)} total conversation threads")
    print(f"Found {len(actionable_convs)} actionable conversation threads with ticker mentions")
    
    # Sort by engagement (likes + retweets)
    actionable_convs['total_engagement'] = actionable_convs['total_likes'] + actionable_convs['total_retweets']
    actionable_convs = actionable_convs.sort_values('total_engagement', ascending=False)
    
    return actionable_convs

def standardize_data(df):
    """
    Standardize data values for consistency before analysis.
    """
    if df is None or df.empty:
        print("No data to standardize.")
        return df
    
    print("\nStandardizing data values...")
    
    # Standardize trade_type values
    trade_type_mapping = {
        'analysi': 'analysis',
        'portfolio_updates': 'portfolio_update',
        'trading_suggestion': 'trade_suggestion',
        'generaldiscussion': 'general_discussion'
    }
    
    df['trade_type'] = df['trade_type'].replace(trade_type_mapping)
    
    # Convert created_at to datetime for time-based analysis
    try:
        df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')
        print(f"Converted created_at to datetime. Date range: {df['created_at'].min()} to {df['created_at'].max()}")
        
        # Extract just the date part (no time) for simplicity
        df['created_date'] = df['created_at'].dt.date
        print(f"Created date-only column for easier analysis")
    except Exception as e:
        print(f"Warning: Could not convert created_at to datetime: {e}")
    
    # Ensure numeric columns are properly typed
    numeric_cols = ['likes', 'retweets', 'replies_count', 'views', 'author_followers', 'author_following', 'confidence']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Calculate prediction dates based on time horizons
    df = calculate_prediction_dates(df)
    
    print("Data standardization complete.")
    return df

def calculate_prediction_dates(df):
    """
    Calculate prediction dates based on time horizon and created_at date.
    
    Short-term: +1 month
    Medium-term: +3 months
    Long-term: +6 months
    """
    if df is None or df.empty:
        print("No data to calculate prediction dates.")
        return df
    
    print("\nCalculating prediction dates based on time horizons...")
    
    # Make sure created_at is in datetime format
    if not pd.api.types.is_datetime64_any_dtype(df['created_at']):
        try:
            df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')
        except Exception as e:
            print(f"Warning: Could not convert created_at to datetime: {e}")
            return df
    
    # Create a new column for prediction date (date only, no time)
    df['prediction_date'] = df['created_at'].dt.date
    
    # Add time based on horizon
    # Short-term: +1 month
    short_term_mask = df['time_horizon'] == 'short_term'
    df.loc[short_term_mask, 'prediction_date'] = pd.to_datetime(df.loc[short_term_mask, 'created_at']) + pd.DateOffset(months=1)
    
    # Medium-term: +3 months
    medium_term_mask = df['time_horizon'] == 'medium_term'
    df.loc[medium_term_mask, 'prediction_date'] = pd.to_datetime(df.loc[medium_term_mask, 'created_at']) + pd.DateOffset(months=3)
    
    # Long-term: +6 months
    long_term_mask = df['time_horizon'] == 'long_term'
    df.loc[long_term_mask, 'prediction_date'] = pd.to_datetime(df.loc[long_term_mask, 'created_at']) + pd.DateOffset(months=6)
    
    # For unknown time horizons, default to +1 month
    unknown_mask = ~(short_term_mask | medium_term_mask | long_term_mask)
    df.loc[unknown_mask, 'prediction_date'] = pd.to_datetime(df.loc[unknown_mask, 'created_at']) + pd.DateOffset(months=1)
    
    # Convert prediction_date to date only (no time)
    df['prediction_date'] = pd.to_datetime(df['prediction_date']).dt.date
    
    # Count predictions by time horizon
    horizon_counts = df['time_horizon'].value_counts()
    print("Prediction counts by time horizon:")
    for horizon, count in horizon_counts.items():
        print(f"  {horizon}: {count} ({count/len(df)*100:.1f}%)")
    
    # Calculate how many predictions are in the future vs. past
    try:
        today = datetime.now().date()
        future_predictions = (df['prediction_date'] > today).sum()
        past_predictions = (df['prediction_date'] <= today).sum()
        
        print(f"Future predictions: {future_predictions} ({future_predictions/len(df)*100:.1f}%)")
        print(f"Past predictions: {past_predictions} ({past_predictions/len(df)*100:.1f}%)")
    except Exception as e:
        print(f"Warning: Could not compare prediction dates with current time: {e}")
        print("Skipping future/past prediction count.")
    
    return df

def visualize_prediction_timeline(df, output_dir='results'):
    """
    Create visualizations of prediction timelines.
    """
    if df is None or df.empty or 'prediction_date' not in df.columns:
        print("No prediction date data to visualize.")
        return
    
    print("\nCreating prediction timeline visualizations...")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Filter for actionable tweets with tickers
    # Create a has_ticker flag based on tickers_mentioned column
    df['has_ticker'] = df['tickers_mentioned'].notna() & (df['tickers_mentioned'] != '')
    
    # Now filter for actionable tweets
    actionable_df = df[df['has_ticker'] & df['trade_type'].isin(['trade_suggestion', 'portfolio_update', 'analysis'])]
    
    if actionable_df.empty:
        print("No actionable tweets with prediction dates to visualize.")
        return
    
    # Convert prediction_date to datetime if it's not already
    try:
        # Check if prediction_date is already datetime
        if not pd.api.types.is_datetime64_any_dtype(actionable_df['prediction_date']):
            print("Converting prediction_date to datetime for visualization...")
            actionable_df['prediction_date'] = pd.to_datetime(actionable_df['prediction_date'])
        
        # 1. Prediction timeline by time horizon
        plt.figure(figsize=(14, 8))
        
        # Group by month and time horizon
        actionable_df['month'] = actionable_df['prediction_date'].dt.to_period('M')
        timeline_data = actionable_df.groupby(['month', 'time_horizon']).size().unstack(fill_value=0)
        
        # Plot
        timeline_data.plot(kind='bar', stacked=True, ax=plt.gca())
        plt.title('Prediction Timeline by Time Horizon')
        plt.xlabel('Month')
        plt.ylabel('Number of Predictions')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/prediction_timeline.png')
        
        # 2. Sentiment distribution over time
        plt.figure(figsize=(14, 8))
        
        # Group by month and sentiment
        sentiment_timeline = actionable_df.groupby(['month', 'sentiment']).size().unstack(fill_value=0)
        
        # Calculate percentages
        sentiment_pct = sentiment_timeline.div(sentiment_timeline.sum(axis=1), axis=0) * 100
        
        # Plot
        sentiment_pct.plot(kind='line', marker='o', ax=plt.gca())
        plt.title('Sentiment Distribution Over Time')
        plt.xlabel('Month')
        plt.ylabel('Percentage')
        plt.ylim(0, 100)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/sentiment_timeline.png')
        
        print(f"Prediction timeline visualizations saved to {output_dir} directory.")
    except Exception as e:
        print(f"Error creating timeline visualizations: {e}")
        print("Skipping timeline visualizations.")

def clear_cache(cache_dir):
    """Clear the stock data cache directory"""
    if os.path.exists(cache_dir):
        print(f"Clearing cache directory: {cache_dir}")
        for file in os.listdir(cache_dir):
            file_path = os.path.join(cache_dir, file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(f"Error deleting {file_path}: {e}")
        print("Cache cleared successfully")

def add_market_validation_columns(df, all_tweets=None, output_dir='results', max_workers=4):
    """
    Add market validation columns to actionable tweets.
    """
    if df is None or df.empty:
        print("No data to validate.")
        return df
    
    print("\nAdding market validation columns...")
    print(f"Processing {len(df)} actionable tweets")
    
    # We'll only upload actionable tweets to the database
    print(f"Will upload {len(df)} actionable tweets to database")
    upload_df = df
    
    print(f"Tweet types in upload data: {upload_df['tweet_type'].value_counts().to_dict()}")
    
    # Create cache directory
    cache_dir = f"{output_dir}/stock_data_cache"
    os.makedirs(cache_dir, exist_ok=True)
    
    # Clear existing cache
    clear_cache(cache_dir)
    
    # Create a dictionary to store ticker data
    ticker_data = {}
    
    # Extract unique tickers and their required date ranges
    tickers = set()
    date_ranges = {}
    
    for _, row in df.iterrows():
        if pd.notna(row['tickers_mentioned']):
            tickers_mentioned = row['tickers_mentioned'].split(',')
            for ticker in tickers_mentioned:
                tickers.add(ticker)
                if ticker not in date_ranges:
                    date_ranges[ticker] = {'start': row['created_date'], 'end': row['prediction_date']}
                else:
                    date_ranges[ticker]['start'] = min(date_ranges[ticker]['start'], row['created_date'])
                    date_ranges[ticker]['end'] = max(date_ranges[ticker]['end'], row['prediction_date'])
    
    # Download data for each unique ticker
    for ticker in tickers:
        start_date = date_ranges[ticker]['start']
        end_date = date_ranges[ticker]['end']
        
        # Ensure end_date is not in the future
        end_date = min(end_date, datetime.now().date())
        
        # Download data for the required date range
        data = download_stock_data(ticker, start_date, end_date)
        
        # When saving to cache, ensure the index format is consistent
        if not data.empty and 'Close' in data.columns:
            # Convert index to string with consistent format before saving
            data.index = data.index.strftime('%Y-%m-%d')
            cache_file = os.path.join(cache_dir, f"{ticker}.csv")
            data.to_csv(cache_file)
            
            # Convert back to datetime for in-memory use
            data.index = pd.to_datetime(data.index)
            ticker_data[ticker] = data
    
    # Add market validation columns to the actionable tweets
    df['start_price'] = None
    df['end_price'] = None
    df['start_date'] = None
    df['end_date'] = None
    df['prediction_correct'] = None
    df['price_change_pct'] = None
    df['actual_return'] = None
    df['validated_ticker'] = None
    
    # Process all actionable tweets
    for idx, row in df.iterrows():
        if pd.notna(row['tickers_mentioned']):
            tickers_mentioned = row['tickers_mentioned'].split(',')
            for ticker in tickers_mentioned:
                if ticker in ticker_data:
                    data = ticker_data[ticker]
                    start_date = row['created_date']
                    end_date = row['prediction_date']
                    
                    # Find the closest trading days
                    start_date = find_closest_trading_day(data, start_date)
                    end_date = find_closest_trading_day(data, end_date)
                    
                    if start_date and end_date:
                        # Get the prices as scalar values, not Series - FIX FOR FUTUREWARNING
                        start_price = float(data.loc[start_date, 'Close'].iloc[0] if isinstance(data.loc[start_date, 'Close'], pd.Series) else data.loc[start_date, 'Close'])
                        end_price = float(data.loc[end_date, 'Close'].iloc[0] if isinstance(data.loc[end_date, 'Close'], pd.Series) else data.loc[end_date, 'Close'])
                        
                        # Calculate price change percentage
                        price_change_pct = ((end_price - start_price) / start_price) * 100
                        
                        # Calculate actual return based on sentiment
                        if row['sentiment'] == 'bullish':
                            actual_return = price_change_pct
                        elif row['sentiment'] == 'bearish':
                            actual_return = -price_change_pct
                        else:
                            actual_return = 0
                        
                        # Update the row with the closest trading dates and prices
                        df.at[idx, 'start_price'] = start_price
                        df.at[idx, 'end_price'] = end_price
                        df.at[idx, 'start_date'] = start_date
                        df.at[idx, 'end_date'] = end_date
                        df.at[idx, 'price_change_pct'] = price_change_pct
                        df.at[idx, 'actual_return'] = actual_return
                        df.at[idx, 'validated_ticker'] = ticker
                        
                        # Determine if the prediction was correct
                        if end_price > start_price:
                            df.at[idx, 'prediction_correct'] = row['sentiment'] == 'bullish'
                        elif end_price < start_price:
                            df.at[idx, 'prediction_correct'] = row['sentiment'] == 'bearish'
                        else:
                            df.at[idx, 'prediction_correct'] = None
                        
                        # Break the loop since we've found a valid ticker
                        break
    
    # Convert prediction_correct to boolean where possible
    df['prediction_correct'] = df['prediction_correct'].astype('object')
    
    # Count validated predictions
    validated_predictions = df[df['prediction_correct'].notna()]
    validated_count = len(validated_predictions)
    validated_pct = (validated_count / len(df)) * 100
    
    # Count correct predictions
    correct_predictions = validated_predictions[validated_predictions['prediction_correct'] == True]
    correct_count = len(correct_predictions)
    correct_pct = (correct_count / validated_count) * 100
    
    # Count incorrect predictions
    incorrect_predictions = validated_predictions[validated_predictions['prediction_correct'] == False]
    incorrect_count = len(incorrect_predictions)
    incorrect_pct = (incorrect_count / validated_count) * 100
    
    # Count unknown predictions
    unknown_predictions = df[df['prediction_correct'].isna()]
    unknown_count = len(unknown_predictions)
    unknown_pct = (unknown_count / len(df)) * 100
    
    # Save summary statistics
    summary_df = pd.DataFrame({
        'Metric': ['Total Predictions', 'Validated', 'Correct', 'Incorrect', 'Unknown'],
        'Count': [len(df), validated_count, correct_count, incorrect_count, unknown_count],
        'Percentage': [100.0, validated_pct, correct_pct, incorrect_pct, unknown_pct]
    })
    summary_df.to_csv(f'{output_dir}/prediction_summary.csv', index=False)
    
    # Save unknown predictions for further analysis
    unknown_df = df[df['prediction_correct'].isna()]
    unknown_df.to_csv(f'{output_dir}/unknown_predictions.csv', index=False)
    
    # Save summary of unknown predictions by ticker
    unknown_summary = unknown_df.groupby('tickers_mentioned').size().reset_index(name='count')
    unknown_summary = unknown_summary.sort_values('count', ascending=False)
    unknown_summary.to_csv(f'{output_dir}/unknown_predictions_summary.csv', index=False)
    
    return df

def download_stock_data(ticker, start_date, end_date):
    """
    Download stock data for a given ticker and date range.
    """
    print(f"Downloading data for {ticker} from {start_date} to {end_date}...")
    
    # Use yfinance to download data
    data = yf.download(ticker, start=start_date, end=end_date)
    
    if data.empty:
        print(f"{ticker}: No data found for {start_date} to {end_date}")
    else:
        print(f"Downloaded {len(data)} rows of data for {ticker}")
    
    return data

def find_closest_trading_day(data, target_date):
    """
    Find the closest trading day to a given target date.
    """
    # Convert target_date to datetime if it's not already
    if not isinstance(target_date, datetime):
        target_date = pd.to_datetime(target_date)
    
    # Find the closest trading day
    try:
        # Try the direct approach first
        if target_date in data.index:
            return target_date
        
        # If target date is not in index, find the closest date
        # Convert to numpy array for easier manipulation
        dates = data.index.to_numpy()
        
        # Calculate absolute differences
        differences = np.abs(dates - np.datetime64(target_date))
        
        # Find the index of the minimum difference
        closest_idx = np.argmin(differences)
        
        # Return the closest date
        return data.index[closest_idx]
    except Exception as e:
        print(f"Error finding closest trading day: {e}")
        # If all else fails, return the first date in the index if available
        if not data.empty:
            return data.index[0]
        return None

def upload_to_database(df):
    """
    Upload the actionable tweets to the database.
    Ensures the database schema accepts all sentiment types first.
    """
    if df is None or df.empty:
        print("No data to upload.")
        return
    
    print(f"\nUploading {len(df)} actionable tweets to database")
    
    # First, ensure the database schema accepts all sentiment types
    if not update_database_schema():
        print("WARNING: Failed to update database schema. Non-standard sentiment tweets may not be uploaded.")
    
    # Prepare data for upload
    print(f"\nUploading {len(df)} tweets to database...")
    
    # Print tweet types distribution
    tweet_types = df['tweet_type'].value_counts().to_dict()
    print(f"Tweet types: {tweet_types}")
    
    # Print sentiment distribution before filtering
    sentiment_counts = df['sentiment'].value_counts()
    print(f"Sentiment distribution before filtering: {sentiment_counts.to_dict()}")
    
    # Create a copy of the dataframe to avoid SettingWithCopyWarning
    upload_df = df.copy()
    
    # Map non-standard sentiment values to standard ones
    sentiment_mapping = {
        'bulllish': 'bullish',
        'bulish': 'bullish',
        'mixed': 'neutral',
        # Keep 'unknown' as is since we're updating the schema to accept it
    }
    
    # Apply the mapping
    upload_df['sentiment'] = upload_df['sentiment'].replace(sentiment_mapping)
    
    # Print sentiment distribution after mapping
    sentiment_counts_after = upload_df['sentiment'].value_counts()
    print(f"Sentiment distribution after mapping: {sentiment_counts_after.to_dict()}")
    
    # Get all columns that exist in both the dataframe and the database table
    valid_columns = [
        'tweet_id', 'conversation_id', 'tweet_type', 'author', 'text', 
        'created_at', 'tickers_mentioned', 'company_names', 'stocks',
        'time_horizon', 'trade_type', 'sentiment', 'reply_to_tweet_id', 
        'reply_to_user', 'likes', 'retweets', 'replies_count', 'views',
        'author_followers', 'author_following', 'author_verified', 
        'author_blue_verified', 'created_date', 'prediction_date',
        'start_price', 'end_price', 'start_date', 'end_date',
        'prediction_correct', 'price_change_pct', 'actual_return',
        'validated_ticker'
    ]
    
    valid_columns = [col for col in valid_columns if col in upload_df.columns]
    print(f"Valid columns for upload: {valid_columns}")
    
    # Filter to only include valid columns
    upload_df = upload_df[valid_columns]
    
    # Convert prediction_score to numeric if it exists
    if 'prediction_score' in upload_df.columns:
        upload_df['prediction_score'] = upload_df['actual_return'].fillna(0)
    
    try:
        # Connect to the database
        conn = psycopg2.connect(
            host="database-twitter.cdh6c6zr5mbr.us-east-1.rds.amazonaws.com",
            database="postgres",
            user="postgres",
            password="DrorMai531"
        )
        
        # Create a cursor
        cursor = conn.cursor()
        
        # Check if the schema update was successful by querying the constraint
        try:
            cursor.execute("""
            SELECT pg_get_constraintdef(oid) 
            FROM pg_constraint 
            WHERE conname = 'check_has_sentiment'
            """)
            constraint_def = cursor.fetchone()[0]
            print(f"Current sentiment constraint: {constraint_def}")
            
            # Get the allowed sentiment values from the constraint
            allowed_sentiments = re.findall(r"'([^']*)'", constraint_def)
            print(f"Allowed sentiment values: {allowed_sentiments}")
            
            # Filter out tweets with sentiment values not in the constraint
            if allowed_sentiments:
                original_count = len(upload_df)
                upload_df = upload_df[upload_df['sentiment'].isin(allowed_sentiments)]
                filtered_count = len(upload_df)
                if filtered_count < original_count:
                    print(f"Filtered out {original_count - filtered_count} tweets with unsupported sentiment values")
                    print(f"Remaining tweets for upload: {filtered_count}")
        except Exception as e:
            print(f"Could not check sentiment constraint: {e}")
            # To be safe, filter to standard sentiment values only
            upload_df = upload_df[upload_df['sentiment'].isin(['bullish', 'bearish', 'neutral'])]
            print(f"Filtered to {len(upload_df)} tweets with standard sentiment values.")
        
        # Check for the ticker constraint
        try:
            cursor.execute("""
            SELECT pg_get_constraintdef(oid) 
            FROM pg_constraint 
            WHERE conname = 'check_has_ticker'
            """)
            ticker_constraint = cursor.fetchone()
            if ticker_constraint:
                print(f"Ticker constraint exists: {ticker_constraint[0]}")
                # Make sure all tweets have tickers
                original_count = len(upload_df)
                upload_df = upload_df[upload_df['tickers_mentioned'].notna() & (upload_df['tickers_mentioned'] != '')]
                filtered_count = len(upload_df)
                if filtered_count < original_count:
                    print(f"Filtered out {original_count - filtered_count} tweets without tickers")
                    print(f"Remaining tweets for upload: {filtered_count}")
        except Exception as e:
            print(f"Could not check ticker constraint: {e}")
        
        # Convert NumPy types to Python native types
        for col in upload_df.columns:
            if upload_df[col].dtype.name.startswith('int') or upload_df[col].dtype.name.startswith('float'):
                upload_df[col] = upload_df[col].astype(float)
                
        # Convert DataFrame to list of tuples (using values.tolist() to avoid NumPy types)
        data = [tuple(x) for x in upload_df[valid_columns].values.tolist()]
        
        # Get column names and placeholders for the SQL query
        columns = ', '.join(valid_columns)
        placeholders = ', '.join(['%s'] * len(valid_columns))
        
        # Check for the unique constraint
        cursor.execute("""
        SELECT conname, pg_get_constraintdef(oid) 
        FROM pg_constraint 
        WHERE conrelid = 'tweets'::regclass
        AND contype = 'u'
        """)
        unique_constraints = cursor.fetchall()
        
        print("Unique constraints found:")
        for name, definition in unique_constraints:
            print(f"  - {name}: {definition}")
        
        # Determine if we need to use ON CONFLICT
        use_on_conflict = False
        conflict_columns = []
        
        for name, definition in unique_constraints:
            if 'tweet_id' in definition:
                use_on_conflict = True
                # Extract column names from the constraint definition
                match = re.search(r'UNIQUE\s*\(([^)]+)\)', definition)
                if match:
                    conflict_columns = [col.strip() for col in match.group(1).split(',')]
                    print(f"Found conflict columns: {conflict_columns}")
                    break
        
        # Prepare the SQL query - with or without ON CONFLICT
        if use_on_conflict and conflict_columns:
            conflict_clause = ', '.join(conflict_columns)
            sql = f"""
            INSERT INTO tweets ({columns})
            VALUES ({placeholders})
            ON CONFLICT ({conflict_clause}) DO NOTHING
            """
            print(f"Using ON CONFLICT ({conflict_clause}) DO NOTHING")
        else:
            # Simple insert without conflict handling
            sql = f"""
            INSERT INTO tweets ({columns})
            VALUES ({placeholders})
            """
            print("Using simple INSERT without conflict handling")
        
        # Upload in batches to avoid memory issues
        batch_size = 500  # Smaller batch size to reduce errors
        total_records = len(data)
        successful_inserts = 0
        
        print(f"Uploading {total_records} records in batches of {batch_size}...")
        
        # Import execute_batch for batch inserts
        from psycopg2.extras import execute_batch
        
        for i in range(0, total_records, batch_size):
            batch = data[i:i+batch_size]
            batch_num = i // batch_size + 1
            
            try:
                # Try to insert the batch using execute_batch
                execute_batch(cursor, sql, batch, page_size=10)  # Smaller page size
                conn.commit()
                successful_inserts += len(batch)
                print(f"Batch {batch_num}: Inserted {len(batch)} records")
            except Exception as e:
                conn.rollback()
                print(f"Error in batch {batch_num}: {e}")
                
                # Try inserting one by one to identify problematic records
                print("Attempting to insert records one by one...")
                for j, record in enumerate(batch):
                    try:
                        # Create a new connection for each record to avoid connection issues
                        new_conn = psycopg2.connect(
                            host="database-twitter.cdh6c6zr5mbr.us-east-1.rds.amazonaws.com",
                            database="postgres",
                            user="postgres",
                            password="DrorMai531"
                        )
                        new_cursor = new_conn.cursor()
                        
                        # Try a simpler approach - insert directly into the table
                        # This bypasses any ON CONFLICT issues
                        cols = ', '.join(valid_columns)
                        vals = ', '.join(['%s'] * len(valid_columns))
                        simple_sql = f"INSERT INTO tweets ({cols}) VALUES ({vals})"
                        
                        new_cursor.execute(simple_sql, record)
                        new_conn.commit()
                        successful_inserts += 1
                        new_cursor.close()
                        new_conn.close()
                    except Exception as e2:
                        if 'new_conn' in locals() and new_conn:
                            try:
                                new_conn.rollback()
                                new_cursor.close()
                                new_conn.close()
                            except:
                                pass
                        print(f"  Error on record {j+1} in batch {batch_num}: {e2}")
        
        # Get the total count of records in the database
        try:
            # Create a fresh connection to get the count
            count_conn = psycopg2.connect(
                host="database-twitter.cdh6c6zr5mbr.us-east-1.rds.amazonaws.com",
                database="postgres",
                user="postgres",
                password="DrorMai531"
            )
            count_cursor = count_conn.cursor()
            count_cursor.execute("SELECT COUNT(*) FROM tweets")
            total_in_db = count_cursor.fetchone()[0]
            count_cursor.close()
            count_conn.close()
            print(f"Total records in database after upload: {total_in_db}")
        except Exception as e:
            print(f"Error getting total count: {e}")
            print("Unable to determine total records in database")
        
        # Close cursor and connection if they're still open
        if 'cursor' in locals() and cursor and not cursor.closed:
            cursor.close()
        if 'conn' in locals() and conn and not conn.closed:
            conn.close()
        
        print(f"Database upload complete: {successful_inserts}/{total_records} records inserted")
        
    except Exception as e:
        print(f"Error connecting to database: {e}")
        import traceback
        traceback.print_exc()

def test_yfinance_accuracy():
    """
    Test yfinance accuracy with 5 well-known tickers.
    """
    import yfinance as yf
    from datetime import datetime, timedelta
    
    print("\n=== Testing YFinance Accuracy with Well-Known Tickers ===")
    
    # Define 5 well-known tickers to test
    test_tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'BTC-USD']
    
    # Get date range for the last week
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)
    
    print(f"Testing price data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    
    for ticker in test_tickers:
        try:
            # Download data for this ticker
            data = yf.download(ticker, start=start_date.strftime('%Y-%m-%d'), 
                              end=end_date.strftime('%Y-%m-%d'), progress=False)
            
            if data.empty:
                print(f"{ticker}: No data found")
                continue
                
            # Get first and last closing prices
            first_price = float(data['Close'].iloc[0].item())
            last_price = float(data['Close'].iloc[-1].item())
            price_change = ((last_price - first_price) / first_price) * 100
            
            # Print results
            print(f"{ticker}:")
            print(f"  First day ({data.index[0].strftime('%Y-%m-%d')}): ${first_price:.2f}")
            print(f"  Last day ({data.index[-1].strftime('%Y-%m-%d')}): ${last_price:.2f}")
            print(f"  Change: {price_change:.2f}%")
            print(f"  Available dates: {len(data)}")
            print()
            
        except Exception as e:
            print(f"{ticker}: Error - {e}")
    
    print("=== YFinance Test Complete ===\n")

def test_database_connection():
    """
    Test the connection to the PostgreSQL database.
    """
    try:
        conn = psycopg2.connect(
            host="database-twitter.cdh6c6zr5mbr.us-east-1.rds.amazonaws.com",
            database="postgres",
            user="postgres",
            password="DrorMai531"
        )
        conn.close()
        print("Database connection successful.")
        return True
    except Exception as e:
        print(f"Database connection failed: {e}")
        return False

def analyze_tweet_filtering(df):
    """
    Analyze why tweets are being filtered out of the actionable set.
    """
    print("\n=== TWEET FILTERING ANALYSIS ===")
    
    # Count total tweets by type
    tweet_types = df['tweet_type'].value_counts()
    print(f"Total tweets by type:")
    for tweet_type, count in tweet_types.items():
        print(f"  - {tweet_type}: {count} ({count/len(df)*100:.1f}%)")
    
    # Count tweets with/without tickers
    has_tickers = df['tickers_mentioned'].notna() & (df['tickers_mentioned'] != '')
    ticker_counts = has_tickers.value_counts()
    print(f"\nTweets with tickers: {ticker_counts.get(True, 0)} ({ticker_counts.get(True, 0)/len(df)*100:.1f}%)")
    print(f"Tweets without tickers: {ticker_counts.get(False, 0)} ({ticker_counts.get(False, 0)/len(df)*100:.1f}%)")
    
    # Count tweets by sentiment
    sentiment_counts = df['sentiment'].value_counts()
    print(f"\nTweets by sentiment:")
    for sentiment, count in sentiment_counts.items():
        print(f"  - {sentiment}: {count} ({count/len(df)*100:.1f}%)")
    
    # Count parent tweets with/without tickers
    parent_tweets = df[df['tweet_type'] == 'parent']
    parent_has_tickers = parent_tweets['tickers_mentioned'].notna() & (parent_tweets['tickers_mentioned'] != '')
    parent_ticker_counts = parent_has_tickers.value_counts()
    print(f"\nParent tweets with tickers: {parent_ticker_counts.get(True, 0)} ({parent_ticker_counts.get(True, 0)/len(parent_tweets)*100:.1f}%)")
    print(f"Parent tweets without tickers: {parent_ticker_counts.get(False, 0)} ({parent_ticker_counts.get(False, 0)/len(parent_tweets)*100:.1f}%)")
    
    # Count parent tweets by sentiment
    parent_sentiment_counts = parent_tweets['sentiment'].value_counts()
    print(f"\nParent tweets by sentiment:")
    for sentiment, count in parent_sentiment_counts.items():
        print(f"  - {sentiment}: {count} ({count/len(parent_tweets)*100:.1f}%)")
    
    # Count tweets by author
    author_counts = df['author'].value_counts()
    print(f"\nTop 10 authors by tweet count:")
    for author, count in author_counts.nlargest(10).items():
        print(f"  - {author}: {count} ({count/len(df)*100:.1f}%)")
    
    # Count parent tweets by author
    parent_author_counts = parent_tweets['author'].value_counts()
    print(f"\nTop 10 authors by parent tweet count:")
    for author, count in parent_author_counts.nlargest(10).items():
        print(f"  - {author}: {count} ({count/len(parent_tweets)*100:.1f}%)")
    
    # Count authors with no valid parent tweets
    authors_with_parents = set(parent_tweets[parent_has_tickers & parent_tweets['sentiment'].isin(['bullish', 'bearish'])]['author'].unique())
    all_authors = set(df['author'].unique())
    authors_without_valid_parents = all_authors - authors_with_parents
    print(f"\nAuthors with no valid parent tweets: {len(authors_without_valid_parents)} ({len(authors_without_valid_parents)/len(all_authors)*100:.1f}%)")
    print(f"Sample of authors without valid parent tweets: {list(authors_without_valid_parents)[:5]}")

def update_database_schema():
    """
    Update the database schema to accept neutral sentiment.
    """
    print("\nUpdating database schema to accept neutral sentiment...")
    
    try:
        # Connect to the database
        conn = psycopg2.connect(
            host="database-twitter.cdh6c6zr5mbr.us-east-1.rds.amazonaws.com",
            database="postgres",
            user="postgres",
            password="DrorMai531"
        )
        
        # Create a cursor
        cursor = conn.cursor()
        
        # Modify the check constraint to include 'neutral' sentiment
        cursor.execute("""
        ALTER TABLE tweets DROP CONSTRAINT IF EXISTS check_has_sentiment;
        ALTER TABLE tweets ADD CONSTRAINT check_has_sentiment 
        CHECK (sentiment IN ('bullish', 'bearish', 'neutral', 'unknown', 'mixed', 'bulllish', 'bulish'));
        """)
        
        # Commit the changes
        conn.commit()
        
        # Close cursor and connection
        cursor.close()
        conn.close()
        
        print("Database schema updated successfully.")
        return True
        
    except Exception as e:
        print(f"Error updating database schema: {e}")
        import traceback
        traceback.print_exc()
        return False

def ensure_database_table_exists():
    """
    Check if the tweets table exists and create it if it doesn't.
    """
    print("\nChecking if tweets table exists...")
    
    try:
        # Connect to the database
        conn = psycopg2.connect(
            host="database-twitter.cdh6c6zr5mbr.us-east-1.rds.amazonaws.com",
            database="postgres",
            user="postgres",
            password="DrorMai531"
        )
        
        # Create a cursor
        cursor = conn.cursor()
        
        # Check if the tweets table exists
        cursor.execute("""
        SELECT EXISTS (
            SELECT FROM information_schema.tables 
            WHERE table_name = 'tweets'
        )
        """)
        table_exists = cursor.fetchone()[0]
        
        if not table_exists:
            print("Tweets table does not exist. Creating it...")
            
            # Create the tweets table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS tweets (
                tweet_id VARCHAR(255) PRIMARY KEY,
                conversation_id VARCHAR(255),
                tweet_type VARCHAR(50),
                author VARCHAR(255),
                text TEXT,
                created_at TIMESTAMP,
                tickers_mentioned TEXT,
                company_names TEXT,
                stocks TEXT,
                time_horizon VARCHAR(50),
                trade_type VARCHAR(50),
                sentiment VARCHAR(50),
                reply_to_tweet_id VARCHAR(255),
                reply_to_user VARCHAR(255),
                likes INTEGER,
                retweets INTEGER,
                replies_count INTEGER,
                views INTEGER,
                author_followers INTEGER,
                author_following INTEGER,
                author_verified BOOLEAN,
                author_blue_verified BOOLEAN,
                created_date DATE,
                prediction_date DATE,
                start_price FLOAT,
                end_price FLOAT,
                start_date DATE,
                end_date DATE,
                prediction_correct BOOLEAN,
                price_change_pct FLOAT,
                actual_return FLOAT,
                validated_ticker VARCHAR(50)
            );
            
            ALTER TABLE tweets ADD CONSTRAINT check_has_sentiment 
            CHECK (sentiment IN ('bullish', 'bearish', 'neutral', 'unknown', 'mixed', 'bulllish', 'bulish'));
            """)
            
            conn.commit()
            print("Tweets table created successfully.")
        else:
            print("Tweets table already exists.")
            
            # Update the sentiment constraint to include all values
            try:
                cursor.execute("""
                ALTER TABLE tweets DROP CONSTRAINT IF EXISTS check_has_sentiment;
                ALTER TABLE tweets ADD CONSTRAINT check_has_sentiment 
                CHECK (sentiment IN ('bullish', 'bearish', 'neutral', 'unknown', 'mixed', 'bulllish', 'bulish'));
                """)
                conn.commit()
                print("Updated sentiment constraint to include all values.")
            except Exception as e:
                conn.rollback()
                print(f"Error updating sentiment constraint: {e}")
        
        # Close cursor and connection
        cursor.close()
        conn.close()
        
        return True
        
    except Exception as e:
        print(f"Error checking/creating database table: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """
    Main function to run the analysis pipeline.
    """
    print("=== Twitter Financial Analysis Pipeline ===")
    
    # Test database connection
    print("\nTesting database connection...")
    if not test_database_connection():
        print("WARNING: Database connection test failed. Data may not be saved properly.")
    else:
        # Ensure the database table exists
        ensure_database_table_exists()
    
    # Test yfinance accuracy with well-known tickers
    test_yfinance_accuracy()
    
    # Load and process data
    df = load_processed_data()
    
    if df is not None:
        # Print original data size
        print(f"\nOriginal dataset size: {len(df)} tweets")
        
        # Standardize data values
        df = standardize_data(df)
        
        # Analyze tweet filtering
        analyze_tweet_filtering(df)
        
        # Explore data columns
        explore_data_columns(df)
        
        # Filter actionable tweets
        filtered_data = filter_actionable_tweets(df)
        
        # Analyze user accuracy
        user_metrics = analyze_user_accuracy(filtered_data['actionable_tweets'])
        
        # Analyze ticker sentiment
        ticker_sentiment = analyze_ticker_sentiment(filtered_data['actionable_tweets'])
        
        # Generate trading signals
        trading_signals = generate_trading_signals(ticker_sentiment, user_metrics)
        
        # Analyze conversation threads
        conversation_analysis = analyze_conversation_threads(filtered_data['actionable_tweets'])
        
        # Add market validation columns to actionable tweets
        validated_df = add_market_validation_columns(
            filtered_data['actionable_tweets']
        )
        
        # Upload to database
        upload_to_database(validated_df)
        
        # Visualize results
        visualize_results(ticker_sentiment, user_metrics, validated_df)
        
        # Save results
        save_results(
            filtered_data, 
            user_metrics, 
            ticker_sentiment, 
            trading_signals, 
            conversation_analysis
        )
        
        print("\nAnalysis pipeline completed successfully!")
    else:
        print("\nAnalysis pipeline failed: No data to process.")

if __name__ == "__main__":
    main()