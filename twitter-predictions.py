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
import sqlite3

# Create results directory if it doesn't exist
results_dir = 'results'
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

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
    Filter tweets to find those with actionable trading signals and include replies
    for context.
    
    Parent tweets must have:
    1. Tickers mentioned
    2. Bullish or bearish sentiment
    3. Known time horizon
    
    All replies to actionable parent tweets are included for context,
    regardless of their content.
    """
    print("\nFiltering tweets for actionable trading signals...")
    
    # Count original parent tweets
    original_parents = df[df['tweet_type'] == 'parent']
    print(f"Original parent tweets: {len(original_parents)} ({len(original_parents)/len(df)*100:.1f}%)")
    
    # Apply the strict filters to parents only
    actionable_parents = apply_strict_filters_corrected(df)
    
    print(f"Actionable parent tweets: {len(actionable_parents)} ({len(actionable_parents)/len(original_parents)*100:.1f}% of original parents)")
    print(f"Parents lost in filtering: {len(original_parents) - len(actionable_parents)}")
    
    # Check if conversation_id exists, if not try to use tweet_id or any alternative
    conversation_id_col = None
    possible_columns = ['conversation_id', 'conversationId', 'in_reply_to', 'thread_id']
    
    for col in possible_columns:
        if col in df.columns:
            conversation_id_col = col
            print(f"Using '{col}' as conversation identifier")
            break
    
    # If no conversation id column is found, use tweet_id as a fallback
    if conversation_id_col is None:
        print("No conversation identifier column found. Using tweet_id as fallback.")
        # Add a conversation_id column that's the same as tweet_id
        df['conversation_id'] = df['tweet_id']
        conversation_id_col = 'conversation_id'
    
    # Get all action-relevant tweet IDs to track their replies
    actionable_tweet_ids = set(actionable_parents['tweet_id'].astype(str))
    
    # Find ALL replies to actionable parent tweets
    all_replies = df[
        (df['tweet_type'] == 'reply') & 
        (df['in_reply_to'].astype(str).isin(actionable_tweet_ids))
    ]
    
    if len(all_replies) > 0:
        print(f"All reply tweets to actionable parents: {len(all_replies)} ({len(all_replies)/len(df)*100:.1f}%)")
    else:
        print("No reply tweets found to actionable parent tweets.")
    
    # Get ticker column name
    ticker_column = 'tickers_mentioned' if 'tickers_mentioned' in df.columns else 'tickers'
    
    # Count replies with tickers for informational purposes
    if len(all_replies) > 0:
        ticker_replies = all_replies[all_replies[ticker_column].notna() & (all_replies[ticker_column] != '')]
        print(f"Reply tweets with tickers: {len(ticker_replies)} ({len(ticker_replies)/len(all_replies)*100:.1f}% of replies)")
    else:
        ticker_replies = pd.DataFrame(columns=df.columns)
        print("No reply tweets with tickers.")
    
    # Create database set with ALL replies + actionable parents
    # This ensures we include ALL replies for context, regardless of their content
    database_tweets = pd.concat([actionable_parents, all_replies]).drop_duplicates()
    
    # For analysis, use the same set
    analysis_tweets = database_tweets.copy()
    
    print(f"\nTotal tweets for database upload: {len(database_tweets)}")
    print(f"Total tweets for analysis: {len(analysis_tweets)}")
    
    # Final summary
    print("\n=== SUMMARY OF TWEET FILTERING ===")
    print(f"Original parent tweets: {len(original_parents)}")
    print(f"Actionable parents: {len(actionable_parents)}")
    print(f"Percentage retained: {len(actionable_parents)/len(original_parents)*100:.1f}%")
    print(f"Total replies included: {len(all_replies)}")
    print(f"Total database tweets: {len(database_tweets)}")
    
    # Count tweet types in database set
    database_tweet_types = database_tweets['tweet_type'].value_counts()
    print("\nTweet types in database set:")
    for tweet_type, count in database_tweet_types.items():
        print(f"  - {tweet_type}: {count} ({count/len(database_tweets)*100:.1f}%)")
    
    # Return both sets in a dictionary as expected by main()
    return {
        'actionable_tweets': database_tweets,  # For database upload (includes all replies)
        'analysis_tweets': analysis_tweets     # For analysis (same as database set)
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
    
    # Check if conversation_id exists, if not try to use tweet_id or any alternative
    conversation_id_col = None
    possible_columns = ['conversation_id', 'conversationId', 'in_reply_to', 'thread_id']
    
    for col in possible_columns:
        if col in df.columns:
            conversation_id_col = col
            print(f"Using '{col}' as conversation identifier")
            break
    
    # If no conversation id column is found, use tweet_id as a fallback
    if conversation_id_col is None:
        print("No conversation identifier column found. Using tweet_id as fallback.")
        # Add a conversation_id column that's the same as tweet_id
        df['conversation_id'] = df['tweet_id']
        conversation_id_col = 'conversation_id'
    
    # Get ticker column name
    ticker_column = 'tickers_mentioned' if 'tickers_mentioned' in df.columns else 'tickers'
    
    # Group by conversation_id
    conversation_groups = df.groupby(conversation_id_col)
    
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
        
        # Try to get likes and retweets
        likes_col = next((col for col in ['likes', 'like_count', 'likeCount'] if col in conv_tweets.columns), None)
        retweets_col = next((col for col in ['retweets', 'retweet_count', 'retweetCount'] if col in conv_tweets.columns), None)
        
        # Calculate engagement metrics
        total_likes = conv_tweets[likes_col].sum() if likes_col else 0
        total_retweets = conv_tweets[retweets_col].sum() if retweets_col else 0
        
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
        has_ticker = pd.notna(parent_tweet[ticker_column]) and isinstance(parent_tweet[ticker_column], str) and parent_tweet[ticker_column].strip() != ''
        
        actionable_trade_types = ['trade_suggestion', 'portfolio_update', 'analysis']
        is_actionable = parent_tweet['trade_type'] in actionable_trade_types
        
        # Store metrics
        conversation_metrics.append({
            'conversation_id': conv_id,
            'parent_author': parent_tweet['author'],
            'parent_text': parent_tweet['text'][:100] + '...' if len(parent_tweet['text']) > 100 else parent_tweet['text'],
            'parent_sentiment': parent_tweet['sentiment'],
            'parent_trade_type': parent_tweet['trade_type'],
            'parent_tickers': parent_tweet[ticker_column],
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
    Ensures proper date calculations based on the specific time horizon:
    
    - Intraday: +1 day (next trading day)
    - Daily: +5 trading days
    - Weekly: +4 weeks
    - Short-term: +1 month
    - Medium-term: +3 months
    - Long-term: +6 months
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
    
    # Create a new column for prediction date initialized to NaT (Not a Time)
    df['prediction_date'] = pd.NaT
    
    # Process each time horizon with the correct offset
    time_horizon_offsets = {
        'intraday': pd.DateOffset(days=1),      # Check next day's close
        'daily': pd.DateOffset(days=5),         # 5 trading days (~1 week)
        'weekly': pd.DateOffset(weeks=4),       # 4 weeks
        'short_term': pd.DateOffset(months=1),  # 1 month
        'medium_term': pd.DateOffset(months=3), # 3 months
        'long_term': pd.DateOffset(months=6)    # 6 months
    }
    
    # Apply offsets based on time horizon
    for horizon, offset in time_horizon_offsets.items():
        # Create a mask for this time horizon
        mask = df['time_horizon'] == horizon
        if mask.any():
            count = mask.sum()
            print(f"  Calculating {count} {horizon} predictions (+{offset})")
            df.loc[mask, 'prediction_date'] = pd.to_datetime(df.loc[mask, 'created_at']) + offset
    
    # Handle unknown time horizons with a default of +2 weeks
    unknown_mask = df['prediction_date'].isna()
    if unknown_mask.any():
        unknown_count = unknown_mask.sum()
        print(f"  Found {unknown_count} tweets with unknown/invalid time horizon - using default +2 weeks")
        df.loc[unknown_mask, 'prediction_date'] = pd.to_datetime(df.loc[unknown_mask, 'created_at']) + pd.DateOffset(weeks=2)
    
    # Convert prediction_date to date only (no time)
    df['prediction_date'] = pd.to_datetime(df['prediction_date']).dt.date
    
    # Sanity check: ensure no prediction is more than 1 year in the future
    max_date = pd.Timestamp.now().date() + pd.DateOffset(years=1)
    future_mask = pd.to_datetime(df['prediction_date']) > pd.to_datetime(max_date)
    if future_mask.any():
        too_far_count = future_mask.sum()
        print(f"  Capping {too_far_count} predictions that are more than 1 year in the future")
        df.loc[future_mask, 'prediction_date'] = max_date
    
    # Count predictions by time horizon
    horizon_counts = df['time_horizon'].value_counts()
    print("\nPrediction counts by time horizon:")
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
    # Create a has_ticker flag based on tickers column
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
    Validates predictions based on the specific time horizon of each tweet.
    """
    if df is None or df.empty:
        print("No data to validate.")
        return df
    
    print("\nAdding market validation columns...")
    print(f"Processing {len(df)} tweets for market validation")
    
    # Print count by tweet type
    tweet_types = df['tweet_type'].value_counts()
    print(f"Tweet types to validate: {tweet_types.to_dict()}")
    
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
    
    # Check that both created_date and prediction_date exist
    if 'created_date' not in df.columns or 'prediction_date' not in df.columns:
        print("ERROR: Missing required date columns. Please run standardize_data() first.")
        return df
        
    # Collect tickers and date ranges
    print("Collecting ticker data ranges...")
    for _, row in df.iterrows():
        if pd.notna(row['tickers_mentioned']) and row['tweet_type'] == 'parent':
            tickers_mentioned = row['tickers_mentioned'].split(',')
            
            # Get the relevant dates for this tweet
            start_date = row['created_date']
            end_date = row['prediction_date']
            
            # Ensure they're different (should be fixed by calculate_prediction_dates)
            if start_date == end_date and row['time_horizon'] != 'unknown':
                print(f"Warning: Start and end dates are the same for tweet {row['tweet_id']} with horizon {row['time_horizon']}")
                # Force end date to be at least 1 day after start date for intraday
                if row['time_horizon'] == 'intraday':
                    end_date = start_date + timedelta(days=1)
                    print(f"  - Adjusting intraday end date to {end_date}")
            
            for ticker in tickers_mentioned:
                tickers.add(ticker)
                if ticker not in date_ranges:
                    date_ranges[ticker] = {'start': start_date, 'end': end_date}
                else:
                    date_ranges[ticker]['start'] = min(date_ranges[ticker]['start'], start_date)
                    date_ranges[ticker]['end'] = max(date_ranges[ticker]['end'], end_date)
    
    # Download data for each unique ticker
    print(f"\nDownloading data for {len(tickers)} unique tickers...")
    for ticker in tickers:
        start_date = date_ranges[ticker]['start']
        end_date = date_ranges[ticker]['end']
        
        # Ensure end_date is not in the future
        end_date = min(end_date, datetime.now().date())
        
        # Add buffer days for potential trading day adjustments
        buffered_start = start_date - timedelta(days=7)  # 7 days before to account for weekends/holidays
        buffered_end = end_date + timedelta(days=7)      # 7 days after
        
        # Download data for the required date range
        data = download_stock_data(ticker, buffered_start, buffered_end)
        
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
    
    # Track successfully validated
    validated_count = 0
    parent_count = 0
    
    # Only validate parent tweets - replies are for context only
    # --- MODIFICATION: Also filter out tweets already marked as deleted ---
    parent_tweets = df[(df['tweet_type'] == 'parent') & (df['is_deleted'] == False)]
    parent_count = len(parent_tweets)
    
    if parent_count == 0:
        print("No active parent tweets found to validate.")
        # Ensure summary file is still created but shows zero counts if needed
        summary_df = pd.DataFrame({
            'Metric': ['Parent Tweets', 'Validated', 'Correct', 'Incorrect', 'Unknown'],
            'Count': [0, 0, 0, 0, 0],
            'Percentage': [100.0, 0, 0, 0, 0] 
        })
        if output_dir:
             os.makedirs(output_dir, exist_ok=True)
             summary_df.to_csv(f'{output_dir}/prediction_summary.csv', index=False)
        return df # Return df as validation columns are already initialized

    print(f"\nValidating {parent_count} active parent tweets...")
    
    # Process actionable parent tweets only
    for idx, row in parent_tweets.iterrows():
        if pd.notna(row['tickers_mentioned']):
            tickers_mentioned = row['tickers_mentioned'].split(',')
            for ticker in tickers_mentioned:
                if ticker in ticker_data:
                    data = ticker_data[ticker]
                    start_date = row['created_date']
                    end_date = row['prediction_date']
                    
                    # Ensure different dates for validation
                    if start_date == end_date:
                        # For intraday specifically, use next day
                        if row['time_horizon'] == 'intraday':
                            end_date = pd.to_datetime(start_date) + pd.DateOffset(days=1)
                            end_date = end_date.date()
                    
                    # Find the closest trading days
                    start_date = find_closest_trading_day(data, start_date)
                    end_date = find_closest_trading_day(data, end_date)
                    
                    if start_date and end_date and start_date != end_date:
                        # Get the prices as scalar values, not Series
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
                        
                        validated_count += 1
                        # Break the loop since we've found a valid ticker
                        break
    
    print(f"Successfully validated {validated_count} out of {parent_count} parent tweets ({validated_count/parent_count*100:.1f}%)")
    
    # Convert prediction_correct to boolean where possible
    df['prediction_correct'] = df['prediction_correct'].astype('object')
    
    # Count validated predictions
    validated_predictions = df[df['prediction_correct'].notna()]
    validated_count = len(validated_predictions)
    validated_pct = (validated_count / parent_count) * 100 if parent_count > 0 else 0
    
    # Count correct predictions
    correct_predictions = validated_predictions[validated_predictions['prediction_correct'] == True]
    correct_count = len(correct_predictions)
    correct_pct = (correct_count / validated_count) * 100 if validated_count > 0 else 0
    
    # Count incorrect predictions
    incorrect_predictions = validated_predictions[validated_predictions['prediction_correct'] == False]
    incorrect_count = len(incorrect_predictions)
    incorrect_pct = (incorrect_count / validated_count) * 100 if validated_count > 0 else 0
    
    # Count unknown predictions (parent tweets only)
    unknown_predictions = parent_tweets[parent_tweets['prediction_correct'].isna()]
    unknown_count = len(unknown_predictions)
    unknown_pct = (unknown_count / parent_count) * 100 if parent_count > 0 else 0
    
    # Save summary statistics
    summary_df = pd.DataFrame({
        'Metric': ['Parent Tweets', 'Validated', 'Correct', 'Incorrect', 'Unknown'],
        'Count': [parent_count, validated_count, correct_count, incorrect_count, unknown_count],
        'Percentage': [100.0, validated_pct, correct_pct, incorrect_pct, unknown_pct]
    })
    summary_df.to_csv(f'{output_dir}/prediction_summary.csv', index=False)
    
    print("\nValidation Summary:")
    print(f"Parent tweets: {parent_count}")
    print(f"Validated: {validated_count} ({validated_pct:.1f}%)")
    print(f"Correct predictions: {correct_count} ({correct_pct:.1f}% of validated)")
    print(f"Incorrect predictions: {incorrect_count} ({incorrect_pct:.1f}% of validated)")
    print(f"Unable to validate: {unknown_count} ({unknown_pct:.1f}%)")
    
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
    Upload the actionable tweets and their replies to the database.
    Parent tweets are validated for predictions, replies are stored for context only.
    """
    if df is None or df.empty:
        print("No data to upload.")
        return
    
    print(f"\nUploading {len(df)} tweets to database (including replies)")
    
    # First, ensure the database schema accepts all sentiment types
    # if not update_database_schema(): # Temporarily commented out as ensure_database_table_exists handles this
    #     print("WARNING: Failed to update database schema. Non-standard sentiment tweets may not be uploaded.")
    
    # Print tweet types distribution
    tweet_types = df['tweet_type'].value_counts().to_dict()
    print(f"Tweet types: {tweet_types}")
    
    # Print sentiment distribution before filtering
    sentiment_counts = df['sentiment'].value_counts()
    print(f"Sentiment distribution before filtering: {sentiment_counts.to_dict()}")
    
    # Create a copy of the dataframe to avoid SettingWithCopyWarning
    upload_df = df.copy()
    
    # Add missing columns that the database expects but might not be in our data
    required_columns = [
        'conversation_id', 'reply_to_tweet_id', 'reply_to_user', 'company_names', 
        'stocks', 'author_followers', 'author_following', 'author_verified', 
        'author_blue_verified'
    ]
    
    for col in required_columns:
        if col not in upload_df.columns:
            print(f"Adding missing column: {col}")
            upload_df[col] = None
    
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
    
    # Mark reply tweets for special handling - we don't validate predictions on replies
    upload_df['is_reply'] = upload_df['tweet_type'] == 'reply'
    
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
        'validated_ticker', 'is_deleted', 'is_analytically_actionable'  # Added is_deleted and is_analytically_actionable
    ]
    
    valid_columns = [col for col in valid_columns if col in upload_df.columns]
    print(f"Valid columns for upload: {valid_columns}")
    
    # Filter to only include valid columns
    upload_df = upload_df[valid_columns]
    
    # Create a copy of the original upload dataframe
    original_upload_df = upload_df.copy()
    
    # Handle the sentiment constraint for the database upload
    try:
        # Connect to the database
        print("\nConnecting to database...")
        conn = psycopg2.connect(
            host="database-twitter.cdh6c6zr5mbr.us-east-1.rds.amazonaws.com",
            database="postgres",
            user="postgres",
            password="DrorMai531"
        )
        print("Database connection established.")
        
        # Create a cursor
        cursor = conn.cursor()
        
        # Check if the schema update was successful by querying the constraint
        try:
            cursor.execute("""
            SELECT pg_get_constraintdef(oid) 
            FROM pg_constraint 
            WHERE conname = 'check_has_sentiment'
            """)
            constraint_def = cursor.fetchone()
            
            if constraint_def:
                constraint_def = constraint_def[0]
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
            else:
                print("No sentiment constraint found. Proceeding with all sentiment values.")
        except Exception as e:
            print(f"Could not check sentiment constraint: {e}")
        
        # Check for the ticker constraint - and handle it differently for parent vs. reply tweets
        try:
            cursor.execute("""
            SELECT pg_get_constraintdef(oid) 
            FROM pg_constraint 
            WHERE conname = 'check_has_ticker'
            """)
            ticker_constraint = cursor.fetchone()
            
            if ticker_constraint:
                ticker_constraint_def = ticker_constraint[0]
                print(f"Ticker constraint exists: {ticker_constraint_def}")
                
                # For parent tweets: ensure they have tickers
                parent_tweets = upload_df[upload_df['tweet_type'] == 'parent']
                ticker_column = 'tickers_mentioned' if 'tickers_mentioned' in parent_tweets.columns else 'tickers'
                
                # Filter parent tweets to ensure they have tickers
                original_parents_count = len(parent_tweets)
                valid_parents = parent_tweets[parent_tweets[ticker_column].notna() & (parent_tweets[ticker_column] != '')]
                
                # For reply tweets: we keep all of them regardless of tickers
                reply_tweets = upload_df[upload_df['tweet_type'] == 'reply']
                
                # Try to modify the ticker constraint temporarily to allow empty tickers for replies
                try:
                    cursor.execute("""
                    ALTER TABLE tweets DROP CONSTRAINT IF EXISTS check_has_ticker;
                    """)
                    conn.commit()
                    print("Temporarily removed ticker constraint to allow reply tweets without tickers")
                except Exception as e:
                    conn.rollback()
                    print(f"Could not modify ticker constraint: {e}")
                    print("Will attempt to proceed by explicitly handling replies")
                
                # Combine filtered parents and replies
                if len(valid_parents) < original_parents_count:
                    print(f"Filtered out {original_parents_count - len(valid_parents)} parent tweets without tickers")
                    
                upload_df = pd.concat([valid_parents, reply_tweets])
                print(f"Final upload set: {len(upload_df)} tweets ({len(valid_parents)} parents, {len(reply_tweets)} replies)")
        except Exception as e:
            print(f"Could not check ticker constraint: {e}")
            print("Proceeding with all tweets")
        
        # Divide into two sets: parent tweets (for prediction) and reply tweets (for context)
        parent_tweets = upload_df[upload_df['tweet_type'] == 'parent']
        reply_tweets = upload_df[upload_df['tweet_type'] == 'reply']

        print(f"\nSplit data: {len(parent_tweets)} parent tweets, {len(reply_tweets)} reply tweets")
        
        # Convert NumPy types to Python native types
        for col in upload_df.columns:
            if upload_df[col].dtype.name.startswith('int') or upload_df[col].dtype.name.startswith('float'):
                upload_df[col] = upload_df[col].astype(float)
                
        # Check if there's any data left to upload after all the filtering
        if upload_df.empty:
            print("\nERROR: No tweets left to upload after filtering!")
            return
            
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
            # --- MODIFICATION: Use DO UPDATE SET --- 
            update_columns = [col for col in valid_columns if col not in conflict_columns]
            if not update_columns: # Should not happen if tweet_id is the only conflict col
                 print("Warning: No columns left to update on conflict. Using DO NOTHING.")
                 update_clause = "DO NOTHING"
            else:
                set_statements = ", ".join([f"{col} = EXCLUDED.{col}" for col in update_columns])
                update_clause = f"DO UPDATE SET {set_statements}"

            sql = f"""
            INSERT INTO tweets ({columns})
            VALUES ({placeholders})
            ON CONFLICT ({conflict_clause}) {update_clause}
            """
            print(f"Using ON CONFLICT ({conflict_clause}) {update_clause}")
        else:
            # Simple insert without conflict handling
            sql = f"""
            INSERT INTO tweets ({columns})
            VALUES ({placeholders})
            """
            print("Using simple INSERT without conflict handling")
        
        # Try a more reliable approach for small datasets - just do a single insert per tweet
        if len(data) <= 10:
            print(f"\nUsing direct insert approach for {len(data)} records...")
            
            # Print each record for debugging
            for i, record in enumerate(data):
                print(f"Record {i+1}: {record[:5]}...") # Only show first 5 fields to keep output manageable
                
                try:
                    cursor.execute(sql, record)
                    conn.commit()
                    print(f"Successfully inserted record {i+1}")
                except Exception as e:
                    conn.rollback()
                    print(f"Error inserting record {i+1}: {e}")
                    
                    # Try with a bare minimum set of columns if needed
                    try:
                        min_columns = ['tweet_id', 'author', 'text', 'sentiment', 'tweet_type']
                        min_columns = [col for col in min_columns if col in valid_columns]
                        
                        if len(min_columns) >= 3:  # Need at least tweet_id, author, text
                            min_sql = f"""
                            INSERT INTO tweets ({', '.join(min_columns)})
                            VALUES ({', '.join(['%s'] * len(min_columns))})
                            """
                            
                            min_record = tuple(record[valid_columns.index(col)] for col in min_columns)
                            print(f"Trying minimal insert with columns: {min_columns}")
                            
                            cursor.execute(min_sql, min_record)
                            conn.commit()
                            print(f"Successfully inserted minimal record {i+1}")
                    except Exception as e2:
                        conn.rollback()
                        print(f"Error inserting minimal record {i+1}: {e2}")
        else:
            # Upload in batches to avoid memory issues
            batch_size = 50  # Smaller batch size to reduce errors
            total_records = len(data)
            successful_inserts = 0
            
            print(f"\nUploading {total_records} records in batches of {batch_size}...")
            
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
                            cursor.execute(sql, record)
                            conn.commit()
                            successful_inserts += 1
                            print(f"  Successfully inserted record {j+1} in batch {batch_num}")
                        except Exception as e2:
                            conn.rollback()
                            print(f"  Error on record {j+1} in batch {batch_num}: {e2}")
        
        # Restore the ticker constraint if we removed it
        try:
            cursor.execute("""
            ALTER TABLE tweets ADD CONSTRAINT check_has_ticker 
            CHECK (tickers_mentioned IS NOT NULL AND tickers_mentioned <> '');
            """)
            conn.commit()
            print("Restored ticker constraint after upload")
        except Exception as e:
            conn.rollback()
            print(f"Note: Could not restore ticker constraint: {e}")
            print("This is not critical if it was already in place")
        
        # Get the total count of records in the database
        try:
            cursor.execute("SELECT COUNT(*) FROM tweets")
            total_in_db = cursor.fetchone()[0]
            print(f"Total records in database after upload: {total_in_db}")
        except Exception as e:
            print(f"Error getting total count: {e}")
            print("Unable to determine total records in database")
        
        # Close cursor and connection
        cursor.close()
        conn.close()
        
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
    
    # Check which ticker column name is present
    ticker_column = 'tickers' if 'tickers' in df.columns else 'tickers_mentioned'
    print(f"Using ticker column: {ticker_column}")
    
    # Count tweets with/without tickers
    has_tickers = df[ticker_column].notna() & (df[ticker_column] != '')
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
    parent_has_tickers = parent_tweets[ticker_column].notna() & (parent_tweets[ticker_column] != '')
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
                validated_ticker VARCHAR(50),
                is_deleted BOOLEAN,
                is_analytically_actionable BOOLEAN
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

def apply_strict_filters_corrected(df):
    """
    Apply strict filtering criteria to identify actionable trading signals.
    Only keeps parent tweets with tickers, bullish/bearish sentiment, and known time horizon.
    No longer filters by trade_type - accepts all values by default.
    """
    # Determine which column name to use
    ticker_column = 'tickers_mentioned' if 'tickers_mentioned' in df.columns else 'tickers'
    
    return df[
        (df['tweet_type'] == 'parent') &
        (df[ticker_column].notna()) &
        (df[ticker_column] != '') &
        (df['sentiment'].isin(['bullish', 'bearish'])) &
        (df['time_horizon'] != 'unknown')
    ]

def enrich_text_with_context(df):
    """
    Enriches tweet text with contextual understanding by mapping code words to tickers.
    Special attention is given to specific traders like Jedi_ant who use code words.
    """
    print("\nEnriching tweet content with contextual understanding...")
    
    # Create a copy to avoid modifying the original
    df = df.copy()
    
    # Check if we need to enrich the data
    if 'text' not in df.columns or 'author' not in df.columns:
        print("Required columns for enrichment (text, author) not found. Skipping enrichment.")
        return df
    
    # Track enrichment stats
    enriched_count = 0
    
    # Define mappings for code words by user
    code_word_mappings = {
        'Jedi_ant': {
            'china': ['KTEC', 'FXI'],
            'China': ['KTEC', 'FXI'],
            # Add more code words as you discover them
        },
        # You can add more users with their specific code words
    }
    
    # Column to track added tickers
    df['enriched_tickers'] = ''
    
    # Process each row
    for idx, row in df.iterrows():
        author = row['author']
        text = row['text'] if pd.notna(row['text']) else ''
        
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
            ticker_column = 'tickers_mentioned' if 'tickers_mentioned' in df.columns else 'tickers'
            current_tickers = set()
            
            if pd.notna(row[ticker_column]) and row[ticker_column]:
                current_tickers = set(row[ticker_column].split(','))
            
            # Combine with enriched tickers
            all_tickers = current_tickers.union(enriched_tickers)
            
            # Update the tickers column
            df.at[idx, ticker_column] = ','.join(all_tickers)
    
    print(f"Enriched {enriched_count} tweets with contextual ticker information.")
    
    return df

def identify_and_flag_actionable_tweets(df):
    """
    Identifies actionable trading signals and flags relevant tweets.

    Actionable parent tweets must have:
    1. Tickers mentioned
    2. Bullish or bearish sentiment
    3. Known time horizon

    Flags actionable parents and all their replies with is_analytically_actionable = True.
    Marks non-actionable tweets as is_deleted = True.
    Returns the original dataframe with the added flag columns.
    """
    print("\nIdentifying and flagging actionable tweets...")

    # Ensure tweet_type column exists
    if 'tweet_type' not in df.columns:
        print("Warning: 'tweet_type' column missing. Assuming all are parents for initial filter.")
        if 'in_reply_to' in df.columns:
             df['tweet_type'] = df['in_reply_to'].apply(lambda x: 'reply' if pd.notna(x) else 'parent')
        else:
             df['tweet_type'] = 'parent' # Fallback

    # Ensure is_deleted column exists and defaulted to False
    if 'is_deleted' not in df.columns:
        df['is_deleted'] = False
    # else: # Ensure it defaults to False if it exists -- REMOVED THIS, default is handled by flag logic below
    #     df['is_deleted'] = False
    
    # Apply the strict filters to parents only to identify them
    actionable_parents = apply_strict_filters_corrected(df) # Uses internal logic for parent type
    print(f"Identified {len(actionable_parents)} actionable parent tweets meeting criteria.")

    # Handle conversation ID
    conversation_id_col = None
    possible_columns = ['conversation_id', 'conversationId', 'in_reply_to', 'thread_id', 'tweet_id'] # Add tweet_id as last resort
    for col in possible_columns:
        if col in df.columns:
            conversation_id_col = col
            print(f"Using '{col}' as conversation identifier for linking replies.")
            break
    if conversation_id_col is None:
        print("ERROR: Cannot identify conversations to link replies. Aborting flagging.")
        df['is_analytically_actionable'] = False # Ensure column exists even if we abort
        df['is_deleted'] = True # Mark all as deleted if we can't link
        return df # Return original df without flag
    
    # Ensure conversation ID is string for comparison/lookup
    df[conversation_id_col] = df[conversation_id_col].astype(str)
    
    # Ensure the column exists in actionable_parents before converting
    if conversation_id_col in actionable_parents.columns:
         actionable_parents[conversation_id_col] = actionable_parents[conversation_id_col].astype(str)
    else:
        # Handle case where conversation_id_col might not be in actionable_parents (e.g., if empty)
         print(f"Warning: Conversation ID column '{conversation_id_col}' not found in actionable_parents DataFrame.")
         actionable_conv_ids = set() # No conversations to flag if actionable_parents is empty or lacks the ID

    # Get the conversation IDs of the actionable parent tweets if possible
    if conversation_id_col in actionable_parents.columns and not actionable_parents.empty:
        actionable_conv_ids = set(actionable_parents[conversation_id_col].unique())
    else:
        actionable_conv_ids = set() # Initialize as empty if column missing or df empty

    print(f"Found {len(actionable_conv_ids)} conversation threads started by actionable parents.")

    # Initialize the flag columns
    df['is_analytically_actionable'] = False
    df['is_deleted'] = False # Default to False initially

    # Flag all tweets belonging to these actionable conversation threads
    if actionable_conv_ids:
        actionable_mask = df[conversation_id_col].isin(actionable_conv_ids)
        
        # Mark actionable tweets
        df.loc[actionable_mask, 'is_analytically_actionable'] = True
        df.loc[actionable_mask, 'is_deleted'] = False # Explicitly False for actionable
        
        # Mark non-actionable tweets as deleted --- RESTORED THIS LOGIC ---
        df.loc[~actionable_mask, 'is_deleted'] = True 
        df.loc[~actionable_mask, 'is_analytically_actionable'] = False # Explicitly False for non-actionable
        
        flagged_count = actionable_mask.sum()
        deleted_count = (~actionable_mask).sum() # Calculate deleted count
        # non_actionable_count = (~actionable_mask).sum()
        
        print(f"Flagged {flagged_count} total tweets (parents and replies) as analytically actionable.")
        print(f"Marked {deleted_count} tweets as deleted (is_deleted = True).") # Restored message
        # print(f"Flagged {non_actionable_count} tweets as NOT analytically actionable.")

    else:
        print("No actionable conversations found to flag.")
        # If no actionable conversations, mark all as deleted --- RESTORED THIS LOGIC ---
        df['is_deleted'] = True
        df['is_analytically_actionable'] = False
        print(f"Marked all {len(df)} tweets as deleted since no actionable conversations were found.") # Restored message
        # print(f"Flagged all {len(df)} tweets as NOT analytically actionable since no actionable conversations were found.")

    # Return the full dataframe with the new flags
    return df

def main():
    """
    Main function to run the analysis pipeline.
    """
    print("=== Twitter Financial Analysis Pipeline ===")
    
    # Define the input file
    input_file = 'all_tweets_llm_analysis.csv'
    print(f"Using input file: {input_file}")
    
    # Test database connection
    print("\nTesting database connection...")
    if not test_database_connection():
        print("WARNING: Database connection test failed. Data may not be saved properly.")
    else:
        # Ensure the database table exists
        ensure_database_table_exists()
    
    # Test yfinance accuracy with well-known tickers
    test_yfinance_accuracy()
    
    # Load and process data using the specified file
    df = load_processed_data(file_path=input_file)
    
    if df is not None:
        # Print original data size
        print(f"\nOriginal dataset size: {len(df)} tweets")
        
        # Standardize column names - map 'tickers' to 'tickers_mentioned' if needed
        if 'tickers' in df.columns and 'tickers_mentioned' not in df.columns:
            print("Renaming 'tickers' column to 'tickers_mentioned' for compatibility")
            df['tickers_mentioned'] = df['tickers']
        
        # Standardize data values
        df = standardize_data(df)
        
        # NEW: Enrich text with contextual understanding (code words → tickers)
        df = enrich_text_with_context(df)
        
        # Analyze tweet filtering
        analyze_tweet_filtering(df)
        
        # Explore data columns
        explore_data_columns(df)
        
        # Identify and flag actionable tweets (instead of filtering)
        df = identify_and_flag_actionable_tweets(df)

        # --- Perform Analysis only on Actionable Subset ---
        actionable_df = df[df['is_analytically_actionable'] == True].copy()
        print(f"\nPerforming analysis on {len(actionable_df)} actionable tweets...")

        if not actionable_df.empty:
            # Calls below now use actionable_df
            user_metrics = analyze_user_accuracy(actionable_df)
            ticker_sentiment = analyze_ticker_sentiment(actionable_df)
            trading_signals = generate_trading_signals(ticker_sentiment, user_metrics)
            conversation_analysis = analyze_conversation_threads(actionable_df)
            visualize_results(ticker_sentiment, user_metrics, actionable_df) # Visualize based on actionable
        else:
            print("No actionable tweets found for detailed analysis.")
            user_metrics = pd.DataFrame()
            ticker_sentiment = pd.DataFrame()
            trading_signals = pd.DataFrame()
            conversation_analysis = pd.DataFrame()

        # --- Add Market Validation (operates on actionable subset internally) ---
        df = add_market_validation_columns(df)

        # --- Ensure required columns for upload exist ---
        if 'is_deleted' not in df.columns:
            print("Adding 'is_deleted' column with default False.")
            df['is_deleted'] = False
        # Ensure is_analytically_actionable exists (should be added earlier)
        if 'is_analytically_actionable' not in df.columns:
            print("Warning: 'is_analytically_actionable' column missing before upload.")
            # Handle potentially by adding it or erroring, depending on requirements
            # For now, let's add it defaulted to False if missing, though it should exist
            df['is_analytically_actionable'] = False 

        # --- Upload FULL DataFrame with flags ---
        upload_to_database(df)

        # --- Save Results ---
        print("\nSaving full results to CSV files...")
        os.makedirs(results_dir, exist_ok=True)
        df.to_csv(f'{results_dir}/all_processed_tweets_with_flags.csv', index=False)
        print(f"Saved full dataframe with flags to {results_dir}/all_processed_tweets_with_flags.csv")

        # Optionally save the analysis subsets as before
        if not user_metrics.empty: user_metrics.to_csv(f'{results_dir}/user_metrics.csv', index=False)
        if not ticker_sentiment.empty: ticker_sentiment.to_csv(f'{results_dir}/ticker_sentiment.csv', index=False)
        if not trading_signals.empty: trading_signals.to_csv(f'{results_dir}/trading_signals.csv', index=False)
        if not conversation_analysis.empty: conversation_analysis.to_csv(f'{results_dir}/conversation_analysis.csv', index=False)

        print("\nAnalysis pipeline completed successfully!")
    else:
        print("\nAnalysis pipeline failed: No data to process.")

if __name__ == "__main__":
    main()
