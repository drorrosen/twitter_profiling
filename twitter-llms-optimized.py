import pandas as pd
import openai
import ast
import json
import re
from collections import Counter
import csv
from openai import OpenAI
import os
import time
from datetime import timedelta
import concurrent.futures

# Set the API key

#load the data
df = pd.read_csv('twitter_data.csv', low_memory=False)
cols_to_drop = [col for col in df.columns if 'Unnamed' in col]
df = df.drop(cols_to_drop, axis=1)

# Convert string representations of dictionaries to actual dictionaries
# and expand them into separate columns
def expand_column(df, column_name):
    # Create a copy to avoid modifying the original
    df = df.copy()
    
    # Function to safely evaluate string to dict
    def safe_eval(x):
        if pd.isna(x):
            return {}  # Return empty dict for NaN values
        try:
            return ast.literal_eval(x)
        except:
            return {}  # Return empty dict for any parsing errors
    
    # Convert strings to dictionaries, handling NaN values
    df[column_name] = df[column_name].apply(safe_eval)
    
    # Expand the dictionary into separate columns
    expanded = pd.json_normalize(df[column_name])
    
    # Rename columns to include the original column name as prefix
    expanded = expanded.add_prefix(f'{column_name}_')
    
    # Join with original dataframe
    return pd.concat([df.drop(columns=[column_name]), expanded], axis=1)

# Expand both columns
df = expand_column(df, 'author')
df = expand_column(df, 'entities')

print("Expanded columns:")
print(df.columns.tolist())
print("\nFirst few rows:")
print(df.head())

# Function to extract tickers using regex
def extract_tickers_regex(text):
    if pd.isna(text):
        return []
    
    # Pattern for $TICKER or #TICKER format
    pattern = r'[\$#][A-Z]{1,5}'  # 1-5 uppercase letters after $ or #
    
    # Find all matches and clean them
    tickers = re.findall(pattern, str(text).upper())
    # Remove $ and # from tickers
    tickers = [ticker[1:] for ticker in tickers]
    return tickers

# Function to extract tickers using OpenAI
def extract_tickers_llm(text, model="gpt-4o-mini"):
    if pd.isna(text):
        return []
    
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a financial analyst. Extract only real stock tickers from the text. Return them as a comma-separated list. If none, return 'None'."},
                {"role": "user", "content": text}
            ],
            temperature=0,
            max_tokens=50
        )
        
        result = response.choices[0].message.content.strip()
        if result.lower() == 'none':
            return []
        
        # Split and clean the tickers
        tickers = [ticker.strip().upper() for ticker in result.split(',')]
        return tickers
    except Exception as e:
        print(f"Error in LLM extraction: {e}")
        return []

# Organize tweets with their replies
def organize_tweets(df):
    # Create a dictionary to store conversation threads
    conversations = {}
    
    print(f"Total tweets before organizing: {len(df)}")
    
    # First pass: Identify parent tweets and create conversation structure
    for _, tweet in df.iterrows():
        try:
            conversation_id = str(tweet['conversationId'])
            tweet_id = str(tweet['id'])
            in_reply_to_id = str(tweet['inReplyToId']) if not pd.isna(tweet['inReplyToId']) else None
            
            # A tweet is a parent if:
            # 1. It's not a reply (inReplyToId is None/NaN) OR
            # 2. It's the start of the conversation (tweet_id equals conversation_id)
            is_parent = (in_reply_to_id is None) or (tweet_id == conversation_id)
            
            if conversation_id not in conversations:
                conversations[conversation_id] = {
                    'parent_tweet': None,
                    'replies': []
                }
            
            tweet_dict = tweet.to_dict()
            tweet_dict['tickers_regex'] = extract_tickers_regex(tweet_dict.get('text', ''))
            
            if is_parent and conversations[conversation_id]['parent_tweet'] is None:
                conversations[conversation_id]['parent_tweet'] = tweet_dict
            else:
                conversations[conversation_id]['replies'].append(tweet_dict)
                
        except Exception as e:
            print(f"Error processing tweet: {e}")
            continue
    
    print(f"\nNumber of conversations found: {len(conversations)}")
    
    # Convert to a more structured format
    organized_tweets = []
    for conv_id, conv_data in conversations.items():
        if conv_data['parent_tweet'] is not None:
            organized_tweets.append({
                'tweet': conv_data['parent_tweet'],
                'replies': sorted(conv_data['replies'], 
                                key=lambda x: str(x.get('createdAt', '')))
            })
    
    print(f"Number of organized threads: {len(organized_tweets)}")
    return organized_tweets

# Organize the tweets
organized_tweets = organize_tweets(df)

if not organized_tweets:
    print("Warning: No organized tweets were generated!")
else:
    # Save organized tweets to JSON file
    with open('organized_tweets.json', 'w', encoding='utf-8') as f:
        json.dump(organized_tweets, f, ensure_ascii=False, indent=2)

    # Save as CSV (flattened format)
    flattened_data = []
    for thread in organized_tweets:
        try:
            parent = thread['tweet']
            parent['tweet_type'] = 'parent'
            flattened_data.append(parent)
            
            for reply in thread['replies']:
                reply['tweet_type'] = 'reply'
                flattened_data.append(reply)
        except Exception as e:
            print(f"Error flattening thread: {e}")
            continue

    print(f"Number of flattened tweets: {len(flattened_data)}")  # Debug print

    if flattened_data:
        df_organized = pd.DataFrame(flattened_data)
        print(f"Columns in organized data: {df_organized.columns.tolist()}")  # Debug print
        df_organized.to_csv('organized_tweets.csv', index=False)
        print("\nData has been saved to:")
        print("1. organized_tweets.json - Contains nested structure with parent tweets and their replies")
        print("2. organized_tweets.csv - Flattened format with all tweets marked as parent/reply")
    else:
        print("Warning: No data to save to CSV!")

# After organizing tweets, find and display the longest conversation
if organized_tweets:
    # Find the conversation with most replies
    longest_thread = max(organized_tweets, key=lambda x: len(x['replies']))
    
    print("\nLongest conversation details:")
    parent = longest_thread['tweet']
    replies = longest_thread['replies']
    
    print(f"\nParent Tweet:")
    print(f"Date: {parent.get('createdAt', 'unknown date')}")
    print(f"By @{parent.get('author_userName', 'unknown')}")
    print(f"Text: {str(parent.get('text', ''))}")
    print(f"\nNumber of replies: {len(replies)}")
    
    print("\nReplies in chronological order:")
    for i, reply in enumerate(replies, 1):
        print(f"\n{i}. Reply Date: {reply.get('createdAt', 'unknown date')}")
        print(f"By @{reply.get('author_userName', 'unknown')}")
        print(f"Text: {str(reply.get('text', ''))[:200]}...")  # Show first 200 chars
        
    print("\n" + "-"*80)

# Print example output
if organized_tweets:
    print("\nExample of organized tweets:")
    for i, thread in enumerate(organized_tweets[:3]):
        parent = thread['tweet']
        replies = thread['replies']
        print(f"\nThread {i+1}:")
        print(f"Parent Tweet (by @{parent.get('author_userName', 'unknown')}):")
        print(f"Text: {str(parent.get('text', ''))[:100]}...")
        print(f"Number of replies in thread: {len(replies)}")
        if replies:
            print("First reply:")
            print(f"By @{replies[0].get('author_userName', 'unknown')}: {str(replies[0].get('text', ''))[:100]}...")
        print("-" * 80)

# After organizing tweets, analyze ticker mentions
if organized_tweets:
    print("\nAnalyzing ticker mentions...")
    
    # Collect all tickers
    all_tickers_regex = []
    
    for thread in organized_tweets:
        # Get tickers from parent tweet
        parent_tickers = thread['tweet'].get('tickers_regex', [])
        all_tickers_regex.extend(parent_tickers)
        
        # Get tickers from replies
        for reply in thread['replies']:
            reply_tickers = reply.get('tickers_regex', [])
            all_tickers_regex.extend(reply_tickers)
    
    # Count ticker mentions
    ticker_counts = Counter(all_tickers_regex)
    
    print("\nMost mentioned tickers (Regex approach):")
    for ticker, count in ticker_counts.most_common(10):
        print(f"${ticker}: {count} mentions")
    
    # Print example of a conversation with tickers
    print("\nExample of a conversation with ticker mentions:")
    for thread in organized_tweets:
        parent = thread['tweet']
        parent_tickers = parent.get('tickers_regex', [])
        
        if parent_tickers:
            print(f"\nParent Tweet (by @{parent.get('author_userName', 'unknown')}):")
            print(f"Text: {str(parent.get('text', ''))[:100]}...")
            print(f"Tickers mentioned: ${', $'.join(parent_tickers)}")
            
            if thread['replies']:
                print("\nReplies with tickers:")
                for reply in thread['replies']:
                    reply_tickers = reply.get('tickers_regex', [])
                    if reply_tickers:
                        print(f"By @{reply.get('author_userName', 'unknown')}: ${', $'.join(reply_tickers)}")
            
            print("-" * 80)
            break  # Show just one example

# Function to flatten conversations
def flatten_conversations(organized_tweets):
    flattened_conversations = []
    
    for thread in organized_tweets:
        parent = thread['tweet']
        replies = thread['replies']
        
        # Initialize conversation data
        conversation = {
            'conversation_id': parent.get('conversationId'),
            'parent_tweet_id': parent.get('id'),
            'author': parent.get('author_userName'),
            'start_date': parent.get('createdAt'),
            'end_date': parent.get('createdAt'),  # Will update if there are replies
            'num_replies': len(replies),
            'participants': {parent.get('author_userName')},  # Using set for unique participants
            'tickers_mentioned': set(parent.get('tickers_regex', [])),
            'full_text': f"[{parent.get('createdAt')}] @{parent.get('author_userName')}: {parent.get('text', '')}"
        }
        
        # Add replies information
        if replies:
            # Sort replies by date
            sorted_replies = sorted(replies, key=lambda x: str(x.get('createdAt', '')))
            
            # Update end date to last reply
            conversation['end_date'] = sorted_replies[-1].get('createdAt')
            
            # Add each reply to the conversation text and collect participants
            for reply in sorted_replies:
                conversation['full_text'] += f"\n[{reply.get('createdAt')}] @{reply.get('author_userName')}: {reply.get('text', '')}"
                conversation['participants'].add(reply.get('author_userName'))
                conversation['tickers_mentioned'].update(reply.get('tickers_regex', []))
        
        # Convert sets to lists for JSON serialization
        conversation['participants'] = list(conversation['participants'])
        conversation['tickers_mentioned'] = list(conversation['tickers_mentioned'])
        
        # Add additional metrics
        conversation['num_participants'] = len(conversation['participants'])
        conversation['num_tickers'] = len(conversation['tickers_mentioned'])
        
        flattened_conversations.append(conversation)
    
    return pd.DataFrame(flattened_conversations)

def get_valid_tickers():
    """Collect tickers from multiple sources"""
    all_tickers = set()
    
    # Try multiple sources
    try:
        # Get NASDAQ listings
        nasdaq = pd.read_csv('https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqtraded.txt', sep='|')
        nasdaq_tickers = set(nasdaq[nasdaq['Test Issue'] == 'N']['Symbol'].tolist())
        all_tickers.update(nasdaq_tickers)
    except:
        print("Could not fetch NASDAQ tickers")
    
    # Add common ETFs and indices
    common_symbols = {['A', 'AA', 'AAA', 'AAAU', 'AACBU', 'AACG', 'AACT', 'AADI', 'AADR', 'AAL', 'AAM', 'AAM.U', 'AAM.W', 'AAME', 'AAMI', 'AAOI', 'AAON', 'AAP', 'AAPB', 'AAPD']
    }
    all_tickers.update(common_symbols)
    
    # Save tickers to file for future use
    with open('valid_tickers.txt', 'w') as f:
        f.write('\n'.join(sorted(all_tickers)))
    
    return all_tickers

# Modify the extract_tickers_and_names_regex function
def extract_tickers_and_names_regex(text):
    if pd.isna(text):
        return [], [], []
    
    text = str(text)
    
    # Load the list of valid tickers
    try:
        with open('valid_tickers.txt', 'r') as f:
            VALID_TICKERS = set(line.strip() for line in f if line.strip())
    except FileNotFoundError:
        print("Warning: valid_tickers.txt not found. Using limited ticker validation.")
        VALID_TICKERS = {'SPY', 'QQQ', 'AAPL', 'MSFT', 'GOOGL'}  # fallback to basic tickers
    
    # Multiple patterns to catch different stock mention formats
    patterns = [
        # $TICKER or #TICKER format (most reliable)
        r'[\$#]([A-Z]{1,5})',
        
        # TICKER: format
        r'([A-Z]{1,5}):\s',
        
        # (TICKER) format
        r'\(([A-Z]{1,5})\)',
        
        # stock TICKER or TICKER stock
        r'stock\s([A-Z]{1,5})|([A-Z]{1,5})\sstock',
        
        # shares of TICKER or TICKER shares
        r'shares of\s([A-Z]{1,5})|([A-Z]{1,5})\sshares',
        
        # TICKER.N (Reuters format)
        r'([A-Z]{1,5})\.N',
        
        # Common formats like "AAPL +2.3%" or "AAPL -1.5%"
        r'([A-Z]{1,5})\s*[\+\-]\d+\.?\d*%?',
        
        # TICKER/USD or TICKER/USDT (crypto format)
        r'([A-Z]{1,5})/(?:USD|USDT)',
        
        # Context-aware patterns
        r'CEO of ([A-Z]{1,5})',
        r'([A-Z]{1,5}) CEO',
        r'([A-Z]{1,5}) earnings',
        r'([A-Z]{1,5}) reported',
        r'([A-Z]{1,5}) Q[1-4]',
        r'guidance for ([A-Z]{1,5})',
        r'price target for ([A-Z]{1,5})',
        r'([A-Z]{1,5}) price target',
    ]
    
    # Collect all matches
    all_tickers = []
    for pattern in patterns:
        matches = re.finditer(pattern, text)
        for match in matches:
            # Get all groups and filter out None values
            groups = [g for g in match.groups() if g is not None]
            if groups:
                all_tickers.extend(groups)
    
    # Filter tickers based on context and validity
    filtered_tickers = []
    for ticker in all_tickers:
        # Clean the ticker (remove any remaining special characters)
        clean_ticker = ticker.strip().upper()
        
        # Accept if it's in our valid tickers list
        if clean_ticker in VALID_TICKERS:
            filtered_tickers.append(clean_ticker)
        # Accept if it has a $ or # prefix in original text (these are explicit stock mentions)
        elif f"${clean_ticker}" in text or f"#{clean_ticker}" in text:
            filtered_tickers.append(clean_ticker)
        # Accept if it appears with strong contextual indicators
        elif any(indicator in text.upper() for indicator in [
            f"{clean_ticker} EARNINGS",
            f"{clean_ticker} REPORTED",
            f"{clean_ticker} CEO",
            f"CEO OF {clean_ticker}",
            f"{clean_ticker} Q1", f"{clean_ticker} Q2", f"{clean_ticker} Q3", f"{clean_ticker} Q4",
            f"{clean_ticker} GUIDANCE",
            f"{clean_ticker} PRICE TARGET"
        ]):
            filtered_tickers.append(clean_ticker)
    
    # Remove duplicates while preserving order
    tickers = list(dict.fromkeys(filtered_tickers))
    
    # For compatibility with existing code
    names = [''] * len(tickers)  # Empty names
    stocks = tickers.copy()  # Just use tickers as stocks
    
    return tickers, names, stocks

# Function to classify tweet with LLM - optimized version with improved prompt
def classify_tweet_with_llm(text: str) -> dict:
    """
    Use GPT-4o mini to analyze tweets and extract structured information about:
    - Time horizon (intraday/daily/weekly/short/medium/long term)
    - Trade type (suggestion, analysis, news, etc)
    - Sentiment (bullish/bearish/neutral)
    """
    # Skip empty or NaN text
    if pd.isna(text) or not text or text.strip() == '':
        return {
            "time_horizon": "unknown",
            "trade_type": "unknown",
            "sentiment": "neutral"
        }
    
    try:
        client = OpenAI()
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
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
            response_format={"type": "json_object"},
            max_tokens=150  # Reduce token count for faster response
        )
        
        # Parse the response
        content = json.loads(response.choices[0].message.content)
        
        # Validate and normalize the response
        return {
            "time_horizon": content.get("time_horizon", "unknown"),
            "trade_type": content.get("trade_type", "unknown"),
            "sentiment": content.get("sentiment", "neutral")
        }
        
    except Exception as e:
        print(f"Error in LLM classification: {e}")
        return {
            "time_horizon": "unknown",
            "trade_type": "unknown",
            "sentiment": "neutral"
        }

# Modify create_tweet_level_dataset to use the optimized LLM analysis
def create_tweet_level_dataset(organized_tweets):
    """Create a dataset with one row per tweet, including parent tweets and replies"""
    tweet_level_data = []
    
    # Process tweets in batches to improve efficiency
    batch_size = 10  # Adjust based on your needs
    all_tweets = []
    
    # Collect all tweets (parents and replies) into a single list
    for thread in organized_tweets:
        parent = thread['tweet']
        all_tweets.append(parent)
        all_tweets.extend(thread['replies'])
    
    print(f"Processing {len(all_tweets)} tweets in batches of {batch_size}...")
    
    # Process tweets in batches
    for i in range(0, len(all_tweets), batch_size):
        batch = all_tweets[i:i+batch_size]
        
        # Show progress
        if i % 100 == 0:
            print(f"Processing tweets {i} to {min(i+batch_size, len(all_tweets))} of {len(all_tweets)}...")
        
        for tweet in batch:
            # Extract tickers and company names
            tickers, names, stocks = extract_tickers_and_names_regex(tweet.get('text', ''))
            
            # Add LLM analysis
            llm_analysis = classify_tweet_with_llm(tweet.get('text', ''))
            
            # Determine if this is a parent tweet or reply
            is_parent = tweet.get('inReplyToId') is None
            
            tweet_data = {
                'tweet_id': str(tweet.get('id')),
                'conversation_id': str(tweet.get('conversationId')),
                'tweet_type': 'parent' if is_parent else 'reply',
                'author': tweet.get('author_userName'),
                'text': tweet.get('text', ''),
                'created_at': tweet.get('createdAt'),
                'tickers_mentioned': tickers,
                'company_names': names,
                'stocks': stocks,
                # Add LLM analysis fields
                'time_horizon': llm_analysis.get('time_horizon'),
                'trade_type': llm_analysis.get('trade_type'),
                'sentiment': llm_analysis.get('sentiment'),
                'reply_to_tweet_id': str(tweet.get('inReplyToId')) if tweet.get('inReplyToId') else None,
                'reply_to_user': tweet.get('inReplyToUsername'),
                'likes': tweet.get('likeCount'),
                'retweets': tweet.get('retweetCount'),
                'replies_count': tweet.get('replyCount'),
                'views': tweet.get('viewCount'),
                'author_followers': tweet.get('author_followers'),
                'author_following': tweet.get('author_following'),
                'author_verified': tweet.get('author_isVerified'),
                'author_blue_verified': tweet.get('author_isBlueVerified'),
            }
            tweet_level_data.append(tweet_data)
    
    df = pd.DataFrame(tweet_level_data)
    
    # Convert lists to strings for CSV storage
    df['tickers_mentioned'] = df['tickers_mentioned'].apply(lambda x: ','.join(x) if x else '')
    df['company_names'] = df['company_names'].apply(lambda x: ','.join(filter(None, x)) if x else '')
    df['stocks'] = df['stocks'].apply(lambda x: ','.join(x) if x else '')
    
    # Ensure IDs are stored as strings
    id_columns = ['tweet_id', 'conversation_id', 'reply_to_tweet_id']
    for col in id_columns:
        if col in df.columns:
            df[col] = df[col].astype(str)
    
    return df

# Create both conversation-level and tweet-level datasets
if organized_tweets:
    print("\nCreating datasets...")
    
    # Find the user with the least conversations (excluding NaN users)
    user_conversation_counts = {}
    for thread in organized_tweets:
        parent = thread['tweet']
        author = parent.get('author_userName')
        if pd.isna(author) or not author:  # Skip NaN or empty authors
            continue
        if author not in user_conversation_counts:
            user_conversation_counts[author] = 0
        user_conversation_counts[author] += 1
    
    # Make sure we have at least one valid user
    if not user_conversation_counts:
        print("No valid users found with conversations. Using all tweets instead.")
        min_user_tweets = organized_tweets
    else:
        # Get the user with the least conversations (but at least 1)
        min_user = min(user_conversation_counts.items(), key=lambda x: x[1] if x[1] > 0 else float('inf'))[0]
        print(f"\nProcessing only tweets from user: @{min_user} (has {user_conversation_counts[min_user]} conversations)")
        
        # Filter organized_tweets to only include this user's tweets
        min_user_tweets = [thread for thread in organized_tweets if thread['tweet'].get('author_userName') == min_user]
    
    # Create and save conversation-level dataset for this user only
    df_conversations = flatten_conversations(min_user_tweets)
    
    # Check if dataframe is empty or missing expected columns
    if df_conversations.empty:
        print("Warning: No conversation data found for the selected user.")
    else:
        # Only sort if the column exists
        if 'num_replies' in df_conversations.columns:
            df_conversations = df_conversations.sort_values('num_replies', ascending=False)
        
        df_conversations.to_csv('conversation_level_data_sample.csv', index=False, encoding='utf-8')
        print(f"Saved {len(df_conversations)} conversations to conversation_level_data_sample.csv")
    
    # Create and save tweet-level dataset for this user only
    df_tweets = create_tweet_level_dataset(min_user_tweets)
    
    # Check if dataframe is empty
    if df_tweets.empty:
        print("Warning: No tweet data found for the selected user.")
    else:
        # Only sort if the columns exist
        if 'conversation_id' in df_tweets.columns and 'created_at' in df_tweets.columns:
            df_tweets = df_tweets.sort_values(['conversation_id', 'created_at'])
        
        # Save with specific CSV options to preserve long numbers and all stock information
        df_tweets.to_csv('tweet_level_data_sample.csv', 
                         index=False, 
                         encoding='utf-8',
                         quoting=csv.QUOTE_NONNUMERIC)  # Quote all non-numeric fields
        
        print(f"Saved {len(df_tweets)} tweets to tweet_level_data_sample.csv")

# Add timing utilities
def log_time(start_time, message):
    """Log elapsed time with a message"""
    elapsed = time.time() - start_time
    elapsed_str = str(timedelta(seconds=int(elapsed)))
    print(f"{message}: {elapsed_str} (HH:MM:SS)")
    return time.time()

# Update the process_all_data function to include timing logs
def process_all_data():
    """
    Process all tweets in the dataset and generate comprehensive CSV files.
    """
    print("\n=== Processing All Data ===\n")
    
    start_time = time.time()
    
    if not organized_tweets:
        print("No organized tweets available to process.")
        return
    
    print(f"Processing all {len(organized_tweets)} conversations in the dataset")
    
    # Create tweet-level dataset for all tweets using parallel processing
    print("Creating tweet-level dataset for all tweets using parallel processing...")
    processing_start = time.time()
    df_all_tweets = process_tweets_in_parallel(organized_tweets, max_workers=8)  # Adjust workers as needed
    processing_time = time.time() - processing_start
    print(f"LLM processing completed in {str(timedelta(seconds=int(processing_time)))} (HH:MM:SS)")
    
    # Save to CSV
    save_start = time.time()
    output_file = "all_tweets_analysis.csv"
    df_all_tweets.to_csv(output_file, 
                         index=False, 
                         encoding='utf-8',
                         quoting=csv.QUOTE_NONNUMERIC)
    
    log_time(save_start, "CSV file saved in")
    print(f"\nFile created: {output_file}")
    print(f"Contains {len(df_all_tweets)} tweets with LLM analysis")
    
    # Print sample of the data
    if not df_all_tweets.empty:
        print("\nSample of analyzed tweets:")
        sample_cols = ['author', 'created_at', 'text', 'time_horizon', 'trade_type', 'sentiment']
        print(df_all_tweets[sample_cols].head(3))
    
    # Get statistics on users
    stats_start = time.time()
    print("\nUser statistics:")
    user_counts = df_all_tweets['author'].value_counts()
    print(f"Total unique users: {len(user_counts)}")
    print("\nTop 10 users by tweet count:")
    for user, count in user_counts.head(10).items():
        print(f"@{user}: {count} tweets")
    
    # Get statistics on sentiment
    print("\nSentiment distribution:")
    sentiment_counts = df_all_tweets['sentiment'].value_counts()
    for sentiment, count in sentiment_counts.items():
        percentage = (count / len(df_all_tweets)) * 100
        print(f"{sentiment}: {count} tweets ({percentage:.1f}%)")
    
    # Get statistics on time horizons
    print("\nTime horizon distribution:")
    horizon_counts = df_all_tweets['time_horizon'].value_counts()
    for horizon, count in horizon_counts.items():
        percentage = (count / len(df_all_tweets)) * 100
        print(f"{horizon}: {count} tweets ({percentage:.1f}%)")
    
    log_time(stats_start, "Statistics generated in")
    log_time(start_time, "Total processing time")
    
    print("\n=== Processing Complete ===\n")
    
    return df_all_tweets

# Update process_tweets_in_parallel to include more detailed timing logs
def process_tweets_in_parallel(organized_tweets, max_workers=4, progress_callback=None):
    """
    Process tweets using parallel execution to speed up LLM analysis.
    """
    print("\n=== Processing Tweets in Parallel ===\n")
    
    start_time = time.time()
    
    if not organized_tweets:
        print("No organized tweets available to process.")
        return pd.DataFrame()
    
    # Flatten the tweets structure
    flatten_start = time.time()
    
    # First, create a dictionary to track which tweets are parents
    parent_tweet_ids = set()
    
    # Collect parent tweet IDs
    for thread in organized_tweets:
        parent = thread['tweet']
        parent_tweet_ids.add(str(parent.get('id')))
    
    # Now prepare all tweets for processing
    all_tweets = []
    parent_tweets = []
    reply_tweets = []
    
    # Separate parents and replies
    for thread in organized_tweets:
        parent = thread['tweet']
        parent['_is_parent'] = True  # Mark as parent
        parent_tweets.append(parent)
        
        for reply in thread['replies']:
            reply['_is_parent'] = False  # Mark as reply
            reply_tweets.append(reply)
    
    # Combine all tweets
    all_tweets = parent_tweets + reply_tweets
    
    log_time(flatten_start, "Flattened tweet structure in")
    print(f"Processing {len(all_tweets)} tweets using {max_workers} parallel workers...")
    print(f"Identified {len(parent_tweets)} parent tweets and {len(reply_tweets)} reply tweets")
    
    # Function to process a single tweet
    def process_tweet(tweet):
        tickers, names, stocks = extract_tickers_and_names_regex(tweet.get('text', ''))
        llm_analysis = classify_tweet_with_llm(tweet.get('text', ''))
        
        # Use the pre-assigned parent/reply flag
        is_parent = tweet.get('_is_parent', False)
        
        return {
            'tweet_id': str(tweet.get('id')),
            'conversation_id': str(tweet.get('conversationId')),
            'tweet_type': 'parent' if is_parent else 'reply',
            'author': tweet.get('author_userName'),
            'text': tweet.get('text', ''),
            'created_at': tweet.get('createdAt'),
            'tickers_mentioned': tickers,
            'company_names': names,
            'stocks': stocks,
            'time_horizon': llm_analysis.get('time_horizon'),
            'trade_type': llm_analysis.get('trade_type'),
            'sentiment': llm_analysis.get('sentiment'),
            'reply_to_tweet_id': str(tweet.get('inReplyToId')) if tweet.get('inReplyToId') else None,
            'reply_to_user': tweet.get('inReplyToUsername'),
            'likes': tweet.get('likeCount'),
            'retweets': tweet.get('retweetCount'),
            'replies_count': tweet.get('replyCount'),
            'views': tweet.get('viewCount'),
            'author_followers': tweet.get('author_followers'),
            'author_following': tweet.get('author_following'),
            'author_verified': tweet.get('author_isVerified'),
            'author_blue_verified': tweet.get('author_isBlueVerified'),
        }
    
    # Process tweets in parallel
    parallel_start = time.time()
    tweet_level_data = []
    processed_count = 0
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tweets for processing
        print("Submitting tweets to processing queue...")
        future_to_tweet = {executor.submit(process_tweet, tweet): tweet for tweet in all_tweets}
        
        # Process as they complete
        for future in concurrent.futures.as_completed(future_to_tweet):
            tweet = future_to_tweet[future]
            try:
                result = future.result()
                tweet_level_data.append(result)
                
                # Update progress
                processed_count += 1
                print(f"Processed tweet {processed_count}/{len(all_tweets)}")
                if progress_callback:
                    progress_callback(processed_count, len(all_tweets))
                
            except Exception as e:
                print(f"Error processing tweet: {e}")
    
    log_time(parallel_start, "Parallel processing completed in")
    print(f"Completed processing {processed_count} tweets")
    
    # Convert to DataFrame
    df_start = time.time()
    df = pd.DataFrame(tweet_level_data)
    
    # Convert lists to strings for CSV storage
    df['tickers_mentioned'] = df['tickers_mentioned'].apply(lambda x: ','.join(x) if x else '')
    df['company_names'] = df['company_names'].apply(lambda x: ','.join(filter(None, x)) if x else '')
    df['stocks'] = df['stocks'].apply(lambda x: ','.join(x) if x else '')
    
    # Ensure IDs are stored as strings
    id_columns = ['tweet_id', 'conversation_id', 'reply_to_tweet_id']
    for col in id_columns:
        if col in df.columns:
            df[col] = df[col].astype(str)
    
    # Verify parent/reply classification
    parent_count = len(df[df['tweet_type'] == 'parent'])
    reply_count = len(df[df['tweet_type'] == 'reply'])
    
    print(f"\nParent/Reply verification:")
    print(f"Parent tweets: {parent_count} (expected {len(parent_tweets)})")
    print(f"Reply tweets: {reply_count} (expected {len(reply_tweets)})")
    
    log_time(df_start, "DataFrame creation and formatting in")
    log_time(start_time, "Total parallel processing time")
    
    return df

# Modify the process_single_user_test function to use a specific user
def process_single_user_test(specific_user=None):
    """
    Process tweets from a single user as a test before running on the full dataset.
    If specific_user is provided, use that user; otherwise find a suitable test user.
    """
    print("\n=== Processing Single User Test ===\n")
    
    if not organized_tweets:
        print("No organized tweets available to process.")
        return
    
    if specific_user:
        test_user = specific_user
        print(f"Using specified test user: @{test_user}")
    else:
        # Find a user with a reasonable number of tweets (not too many, not too few)
        user_counts = {}
        for thread in organized_tweets:
            parent = thread['tweet']
            author = parent.get('author_userName')
            if author and not pd.isna(author):
                user_counts[author] = user_counts.get(author, 0) + 1
        
        # Find users with 5-20 conversations (good test size)
        test_users = [user for user, count in user_counts.items() if 5 <= count <= 20]
        
        if not test_users:
            # If no users in ideal range, just take the first user with any tweets
            test_users = list(user_counts.keys())[:1]
        
        if not test_users:
            print("No suitable test users found.")
            return
        
        test_user = test_users[0]
        print(f"Selected test user: @{test_user}")
    
    # Filter threads for this user
    user_threads = []
    for thread in organized_tweets:
        if thread['tweet'].get('author_userName') == test_user:
            user_threads.append(thread)
    
    print(f"Found {len(user_threads)} conversations by @{test_user}")
    
    # Process these threads
    if user_threads:
        start_time = time.time()
        df_user_tweets = process_tweets_in_parallel(user_threads, max_workers=4)
        
        # Verify parent/reply classification
        parent_count = len(df_user_tweets[df_user_tweets['tweet_type'] == 'parent'])
        reply_count = len(df_user_tweets[df_user_tweets['tweet_type'] == 'reply'])
        
        print(f"\nTweet type verification:")
        print(f"Parent tweets: {parent_count} ({parent_count/len(df_user_tweets)*100:.1f}%)")
        print(f"Reply tweets: {reply_count} ({reply_count/len(df_user_tweets)*100:.1f}%)")
        
        # Save to CSV
        output_file = f"test_user_{test_user}_tweets.csv"
        df_user_tweets.to_csv(output_file, 
                             index=False, 
                             encoding='utf-8',
                             quoting=csv.QUOTE_NONNUMERIC)
        
        print(f"File created: {output_file}")
        log_time(start_time, "Test processing completed in")
        
        # Print sample of the data
        if not df_user_tweets.empty:
            print("\nSample of analyzed tweets:")
            sample_cols = ['tweet_type', 'created_at', 'text', 'time_horizon', 'trade_type', 'sentiment']
            print(df_user_tweets[sample_cols].head(3))
        
        return df_user_tweets
    else:
        print(f"No tweets found for user @{test_user}")
        return None

# Run the test to make sure everything is working
def test_llm_classification():
    """
    Test the LLM classification on a few example tweets before processing the entire dataset.
    """
    print("\n=== Testing LLM Classification ===\n")
    
    # Sample tweets with different time horizons
    test_tweets = [
        # Intraday
        """$SPY looking weak in the next hour. Watching 450 level for a breakdown. Might scalp some puts.""",
        
        # Daily
        """$AAPL should bounce tomorrow after the oversold conditions. Looking for entry around $190.""",
        
        # Weekly
        """$NVDA earnings next week. IV is high, considering selling some put spreads for next Friday expiration.""",
        
        # Short term
        """$TSLA looks bullish on the daily chart. Breaking out of resistance at $190, targeting $210 by end of quarter."""
    ]
    
    for i, tweet in enumerate(test_tweets):
        print(f"Test Tweet {i+1}: {tweet}\n")
        
        # Run classification
        print(f"Running LLM classification for tweet {i+1}...")
        result = classify_tweet_with_llm(tweet)
        
        # Print results in a readable format
        print(f"\nClassification Results for tweet {i+1}:")
        print(f"Time Horizon: {result['time_horizon']}")
        print(f"Trade Type: {result['trade_type']}")
        print(f"Sentiment: {result['sentiment']}")
        print("\n" + "-"*50 + "\n")
    
    print("\n=== Test Complete ===\n")
    
    return result

# Modify the main execution section to use the specific user
# Run the test to make sure everything is working
test_result = test_llm_classification()

# # Process a specific user as a test
# print("\nProcessing specific user test...")
# test_user_df = process_single_user_test("BarrySchwartzBW")  # Specify the user here

# Ask for confirmation before processing all data
user_input = input("\nDo you want to process all data? This may take several hours. (y/n): ")
if user_input.lower() == 'y':
    # Process all data with timing information
    print("\nProcessing all data...")
    start_time = time.time()
    all_tweets_df = process_all_data()
    log_time(start_time, "Total script execution time")
else:
    print("Full processing skipped. Test completed successfully.")