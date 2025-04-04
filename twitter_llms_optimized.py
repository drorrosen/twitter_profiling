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

API_KEY = st.secrets["OPENAI_API_KEY"]
# Set the API key
client = OpenAI(api_key=API_KEY)  # This will use the OPENAI_API_KEY environment variable

# Convert string representations of dictionaries to actual dictionaries
# and expand them into separate columns
def expand_column(df, column_name):
    # Create a copy to avoid modifying the original
    df = df.copy()
    
    # Check if the column exists
    if column_name not in df.columns:
        print(f"Warning: Column {column_name} not found in the dataframe")
        return df
    
    # Check first value to see if conversion is needed
    first_val = df[column_name].iloc[0] if not df.empty else None
    
    # If it's already a dict, no need to convert
    if isinstance(first_val, dict):
        print(f"Column {column_name} already contains dictionaries, no conversion needed")
    else:
        # Function to safely evaluate string to dict
        def safe_eval(x):
            if pd.isna(x):
                return {}  # Return empty dict for NaN values
            try:
                if isinstance(x, str):
                    return ast.literal_eval(x)
                elif isinstance(x, dict):
                    return x
                else:
                    return {}
            except:
                return {}  # Return empty dict for any parsing errors
        
        # Convert strings to dictionaries, handling NaN values
        print(f"Converting {column_name} from strings to dictionaries...")
        df[column_name] = df[column_name].apply(safe_eval)
    
    # Expand the dictionary into separate columns
    print(f"Expanding {column_name} into separate columns...")
    
    try:
        expanded = pd.json_normalize(df[column_name])
        
        # Rename columns to include the original column name as prefix
        expanded = expanded.add_prefix(f'{column_name}_')
        
        # Join with original dataframe
        return pd.concat([df.drop(columns=[column_name]), expanded], axis=1)
    except Exception as e:
        print(f"Error expanding {column_name}: {e}")
        return df

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
def extract_tickers_llm(text, model="gpt-4o"):
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

# Add this function to the script, after the extract_tickers functions
def enrich_tickers_with_context(tweet_data):
    """
    Enriches tweet content with contextual understanding by mapping code words to tickers.
    Special attention is given to specific traders like Jedi_ant who use code words.
    """
    # Skip if there's no text or author
    if 'text' not in tweet_data or 'author' not in tweet_data or not tweet_data.get('text'):
        return tweet_data
    
    # Define mappings for code words by user
    code_word_mappings = {
        'Jedi_ant': {  # Using correct username format
            'china': ['KTEC', 'FXI'],
            'China': ['KTEC', 'FXI'],
            # Add more code words as you discover them
        },
        # You can add more users with their specific code words
    }
    
    author = tweet_data.get('author', '')
    text = tweet_data.get('text', '')
    
    # Skip if not an author we're tracking
    if not author or author not in code_word_mappings:
        return tweet_data
    
    # Get the code word mapping for this author
    mappings = code_word_mappings[author]
    
    # Check for each code word
    enriched_tickers = set()
    for code_word, tickers in mappings.items():
        if code_word in text:
            # Add these tickers to the enriched set
            enriched_tickers.update(tickers)
    
    # If we found code words, update the tickers
    if enriched_tickers:
        # Get current tickers (if any)
        current_tickers = set()
        if 'tickers' in tweet_data and tweet_data['tickers']:
            current_tickers = set(tweet_data['tickers'].split(','))
        
        # Combine with enriched tickers
        all_tickers = current_tickers.union(enriched_tickers)
        
        # Update the tickers field
        tweet_data['tickers'] = ','.join(all_tickers)
        
        # Add a field to track this was enriched
        tweet_data['enriched_context'] = True
    
    return tweet_data

# Organize tweets with their replies
def organize_tweets(df):
    # Create a dictionary to store conversation threads
    conversations = {}
    
    print(f"Total tweets before organizing: {len(df)}")
    
    # Check if the required columns exist
    required_cols = ['conversationId', 'id', 'inReplyToId', 'text', 'createdAt']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        print(f"Warning: Missing required columns: {missing_cols}")
        print("Using alternative column names if available...")
        
        # Try to map to alternative column names
        col_mapping = {
            'conversationId': ['conversation_id', 'conversationid'],
            'id': ['tweet_id', 'tweetid'],
            'inReplyToId': ['in_reply_to_id', 'inreplyto_id'],
            'text': ['tweet_text', 'content'],
            'createdAt': ['created_at', 'timestamp']
        }
        
        for missing in missing_cols:
            alternatives = col_mapping.get(missing, [])
            for alt in alternatives:
                if alt in df.columns:
                    print(f"Using '{alt}' instead of '{missing}'")
                    df[missing] = df[alt]
                    break
    
    # First pass: Identify parent tweets and create conversation structure
    for _, tweet in df.iterrows():
        try:
            conversation_id = str(tweet.get('conversationId', tweet.get('conversation_id', '')))
            tweet_id = str(tweet.get('id', tweet.get('tweet_id', '')))
            in_reply_to_id = str(tweet.get('inReplyToId', tweet.get('in_reply_to_id', ''))) if not pd.isna(tweet.get('inReplyToId', tweet.get('in_reply_to_id', ''))) else None
            
            # Skip if missing critical information
            if not conversation_id or not tweet_id:
                continue
                
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
                                key=lambda x: str(x.get('createdAt', x.get('created_at', ''))))
            })
    
    print(f"Number of organized threads: {len(organized_tweets)}")
    return organized_tweets

# Function to run a simplified analysis on our sample data
def analyze_sample_tweets(organized_tweets):
    print("\nRunning basic analysis on all tweets...")
    
    if not organized_tweets:
        print("No organized tweets to analyze!")
        return
    
    # Count total tweets
    total_tweets = 0
    parent_tweets = 0
    
    # Count tickers mentioned
    all_tickers = []
    
    for thread in organized_tweets:
        parent = thread['tweet']
        replies = thread['replies']
        
        parent_tweets += 1
        total_tweets += 1 + len(replies)
        
        # Extract tickers from parent tweet
        parent_tickers = parent.get('tickers_regex', [])
        all_tickers.extend(parent_tickers)
        
        # Extract tickers from replies
        for reply in replies:
            reply_tickers = reply.get('tickers_regex', [])
            all_tickers.extend(reply_tickers)
    
    print(f"\nBasic Analysis Results:")
    print(f"Total tweets: {total_tweets}")
    print(f"Parent tweets: {parent_tweets}")
    print(f"Reply tweets: {total_tweets - parent_tweets}")
    
    # Count ticker mentions
    ticker_counter = Counter(all_tickers)
    
    print("\nTop tickers mentioned:")
    for ticker, count in ticker_counter.most_common(10):
        print(f"${ticker}: {count} mentions")
    
    # Save a simplified analysis to CSV
    results = []
    
    for thread in organized_tweets:
        parent = thread['tweet']
        
        # Get basic tweet info
        tweet_data = {
            'tweet_id': parent.get('id', ''),
            'author': parent.get('author_userName', parent.get('authorUsername', '')),
            'text': parent.get('text', ''),
            'created_at': parent.get('createdAt', parent.get('created_at', '')),
            'tickers': ','.join(parent.get('tickers_regex', [])),
            'reply_count': len(thread['replies']),
            'like_count': parent.get('likeCount', parent.get('like_count', 0)),
            'retweet_count': parent.get('retweetCount', parent.get('retweet_count', 0))
        }
        
        results.append(tweet_data)
    
    # Save to CSV
    if results:
        df_results = pd.DataFrame(results)
        output_file = 'basic_tweet_analysis.csv'
        df_results.to_csv(output_file, index=False)
        print(f"\nSaved basic analysis to {output_file}")
    
    return results

# Function to classify tweet with LLM
def classify_tweet_with_llm(text: str) -> dict:
    """
    Use GPT-4o to analyze tweets and extract structured information about:
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
        response = client.chat.completions.create(
            model="gpt-4o",
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
            max_tokens=150
        )
        
        # Parse the response
        content = json.loads(response.choices[0].message.content)
        
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

# Function to process tweets in parallel
def process_tweets_in_parallel(tweet_list, max_workers=8):
    """Process a list of tweets in parallel using multiple workers"""
    start_time = time.time()
    total_tweets = len(tweet_list)
    processed = 0
    results = []
    
    print(f"\nProcessing {total_tweets} tweets with {max_workers} workers...")
    
    # Function to process a single tweet
    def process_single_tweet(tweet):
        try:
            # Extract tickers using regex
            tickers = tweet.get('tickers_regex', [])
            
            # Analyze with LLM
            tweet_text = tweet.get('text', '')
            analysis = classify_tweet_with_llm(tweet_text)
            
            # Get the author
            author = tweet.get('author_userName', tweet.get('authorUsername', ''))
            
            # Create the initial tweet data dict
            tweet_data = {
                'tweet_id': tweet.get('id', ''),
                'author': author,
                'text': tweet_text,
                'created_at': tweet.get('createdAt', tweet.get('created_at', '')),
                'tickers': ','.join(tickers),
                'is_parent': tweet.get('is_parent', True),
                'in_reply_to': tweet.get('in_reply_to', ''),
                'time_horizon': analysis.get('time_horizon', 'unknown'),
                'trade_type': analysis.get('trade_type', 'unknown'),
                'sentiment': analysis.get('sentiment', 'neutral'),
                'like_count': tweet.get('likeCount', tweet.get('like_count', 0)),
                'retweet_count': tweet.get('retweetCount', tweet.get('retweet_count', 0)),
            }
            
            # Enrich with code word mapping for specific users like Jedi_ant
            if author == 'Jedi_ant':
                code_word_mappings = {
                    'china': ['KTEC', 'FXI'],
                    'China': ['KTEC', 'FXI'],
                    # Add more code words as needed
                }
                
                # Check for code words in the tweet text
                enriched_tickers = set()
                for code_word, mapped_tickers in code_word_mappings.items():
                    if code_word in tweet_text:
                        enriched_tickers.update(mapped_tickers)
                
                # Add enriched tickers to the existing ones
                if enriched_tickers:
                    current_tickers = set(tickers)
                    all_tickers = current_tickers.union(enriched_tickers)
                    tweet_data['tickers'] = ','.join(all_tickers)
            
            return tweet_data
            
        except Exception as e:
            print(f"Error processing tweet: {e}")
            return None
    
    # Use ThreadPoolExecutor to process tweets in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_tweet = {executor.submit(process_single_tweet, tweet): i 
                          for i, tweet in enumerate(tweet_list)}
        
        for future in concurrent.futures.as_completed(future_to_tweet):
            processed += 1
            
            # Print progress every 25 tweets
            if processed % 25 == 0 or processed == total_tweets:
                elapsed = time.time() - start_time
                tweets_per_sec = processed / elapsed if elapsed > 0 else 0
                est_remaining = (total_tweets - processed) / tweets_per_sec if tweets_per_sec > 0 else "unknown"
                
                if isinstance(est_remaining, float):
                    est_remaining_str = str(timedelta(seconds=int(est_remaining)))
                else:
                    est_remaining_str = est_remaining
                    
                print(f"Processed {processed}/{total_tweets} tweets - " 
                      f"{processed/total_tweets*100:.1f}% complete - "
                      f"Est. remaining: {est_remaining_str}")
            
            # Get the result
            result = future.result()
            if result:
                results.append(result)
    
    # Create DataFrame from results
    df_results = pd.DataFrame(results)
    
    # Add tweet_type column based on is_parent
    if 'is_parent' in df_results.columns:
        df_results['tweet_type'] = df_results['is_parent'].apply(lambda x: 'parent' if x else 'reply')
        df_results = df_results.drop('is_parent', axis=1)
    
    return df_results

# Function for enhanced analysis with LLM using parallel processing
def analyze_all_tweets_with_parallel_llm(organized_tweets, max_workers=8):
    print("\nRunning parallel LLM analysis on ALL tweets...")
    
    if not organized_tweets:
        print("No organized tweets to analyze!")
        return pd.DataFrame()
    
    # Prepare list of all tweets (both parents and replies)
    all_tweets = []
    
    for thread in organized_tweets:
        # Add parent tweet with a type indicator
        parent = thread['tweet']
        parent['is_parent'] = True
        all_tweets.append(parent)
        
        # Add replies with a type indicator
        for reply in thread['replies']:
            reply['is_parent'] = False
            reply['in_reply_to'] = parent.get('id', '')
            all_tweets.append(reply)
    
    print(f"Prepared {len(all_tweets)} tweets for parallel processing")
    
    # Process all tweets in parallel
    start_time = time.time()
    results_df = process_tweets_in_parallel(all_tweets, max_workers=max_workers)
    end_time = time.time()
    
    # Save to CSV
    if not results_df.empty:
        output_file = 'all_tweets_llm_analysis.csv'
        results_df.to_csv(output_file, index=False)
        print(f"\nSaved complete LLM analysis to {output_file}")
        
        # Print completion time
        total_time = end_time - start_time
        print(f"\nCompleted LLM analysis on {len(results_df)} tweets in {str(timedelta(seconds=int(total_time)))}")
        print(f"Average processing time: {total_time/len(results_df):.2f} seconds per tweet")
        
        # Print sentiment distribution
        if 'sentiment' in results_df.columns:
            sentiment_counts = results_df['sentiment'].value_counts()
            print("\nSentiment distribution:")
            for sentiment, count in sentiment_counts.items():
                percentage = (count / len(results_df)) * 100
                print(f"{sentiment}: {count} tweets ({percentage:.1f}%)")
    
    return results_df

# Only execute this code when the script is run directly, not when imported
if __name__ == "__main__":
    # Load the data from our example file instead of twitter_data.csv
    print("Loading data from example_tweets.csv...")
    df = pd.read_csv('example_tweets.csv', low_memory=False)
    cols_to_drop = [col for col in df.columns if 'Unnamed' in col]
    df = df.drop(cols_to_drop, axis=1)

    print(f"Loaded {len(df)} tweets from example_tweets.csv")
    print("Column names:", df.columns.tolist())

    # Check if author and entities columns exist and are in the right format
    author_col = 'author' if 'author' in df.columns else None
    entities_col = 'entities' if 'entities' in df.columns else None

    print(f"Author column: {author_col}")
    print(f"Entities column: {entities_col}")

    # Expand both columns if they exist
    if author_col:
        df = expand_column(df, author_col)
    if entities_col:
        df = expand_column(df, entities_col)

    print("\nExpanded columns:")
    print(df.columns.tolist())
    print("\nFirst few rows:")
    print(df.head())

    # Organize the tweets
    print("\nOrganizing tweets into conversation threads...")
    organized_tweets = organize_tweets(df)

    # Run the basic analysis
    print("\nRunning basic analysis without LLM...")
    basic_results = analyze_sample_tweets(organized_tweets)

    # Now run the parallel processing on ALL tweets
    print("\nNow running parallel LLM analysis on ALL tweets...")
    # Use 8 workers as requested for optimal performance
    MAX_WORKERS = 8

    all_results = analyze_all_tweets_with_parallel_llm(organized_tweets, max_workers=MAX_WORKERS)

    print("\nAnalysis complete!")
