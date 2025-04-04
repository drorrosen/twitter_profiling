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

# Function to analyze sample tweets
def analyze_sample_tweets(organized_tweets):
    """
    Perform basic analysis on the sample tweets.
    """
    # Create a DataFrame to store the results
    results = []
    
    for conversation_id, conversation in organized_tweets:
        # Analyze parent tweet
        parent = conversation['tweet']
        parent_text = parent.get('text', '')
        parent_id = parent.get('id', '')
        
        # Extract tickers using regex
        tickers = extract_tickers_regex(parent_text)
        
        # Basic sentiment analysis - just a placeholder for now
        sentiment = 'neutral'
        if any(bullish_word in parent_text.lower() for bullish_word in ['bullish', 'long', 'call', 'buy', 'moon']):
            sentiment = 'bullish'
        elif any(bearish_word in parent_text.lower() for bearish_word in ['bearish', 'short', 'put', 'sell', 'crash']):
            sentiment = 'bearish'
        
        # Add parent tweet analysis
        results.append({
            'id': parent_id,
            'text': parent_text,
            'is_parent': True,
            'tickers': ','.join(tickers),
            'sentiment': sentiment,
            'has_tickers': len(tickers) > 0
        })
        
        # Analyze replies
        for reply in conversation['replies']:
            reply_text = reply.get('text', '')
            reply_id = reply.get('id', '')
            
            # Extract tickers
            reply_tickers = extract_tickers_regex(reply_text)
            
            # Basic sentiment analysis
            reply_sentiment = 'neutral'
            if any(bullish_word in reply_text.lower() for bullish_word in ['bullish', 'long', 'call', 'buy', 'moon']):
                reply_sentiment = 'bullish'
            elif any(bearish_word in reply_text.lower() for bearish_word in ['bearish', 'short', 'put', 'sell', 'crash']):
                reply_sentiment = 'bearish'
            
            # Add reply analysis
            results.append({
                'id': reply_id,
                'text': reply_text,
                'is_parent': False,
                'in_reply_to': parent_id,
                'tickers': ','.join(reply_tickers),
                'sentiment': reply_sentiment,
                'has_tickers': len(reply_tickers) > 0
            })
    
    # Convert to DataFrame
    df_results = pd.DataFrame(results)
    
    # Print summary statistics
    print(f"\nAnalyzed {len(df_results)} tweets")
    print(f"Parent tweets: {df_results['is_parent'].sum()}")
    print(f"Reply tweets: {len(df_results) - df_results['is_parent'].sum()}")
    print(f"Tweets with tickers: {df_results['has_tickers'].sum()}")
    
    # Sentiment distribution
    sentiment_counts = df_results['sentiment'].value_counts()
    print("\nSentiment distribution:")
    for sentiment, count in sentiment_counts.items():
        percentage = (count / len(df_results)) * 100
        print(f"{sentiment}: {count} tweets ({percentage:.1f}%)")
    
    # Save the results to CSV
    df_results.to_csv('basic_tweet_analysis.csv', index=False)
    print("\nBasic analysis saved to basic_tweet_analysis.csv")
    
    return df_results

# Function to classify a single tweet with LLM
def classify_tweet_with_llm(text: str, client: OpenAI) -> dict:
    """
    Use OpenAI's GPT model to classify a tweet for financial analysis.
    Returns a dictionary with sentiment, tickers, time horizon, trade type, and price targets.
    """
    if pd.isna(text) or not text:
        return {
            'sentiment': 'unknown',
            'tickers': '',
            'time_horizon': 'unknown',
            'trade_type': 'unknown',
            'entry_price': None,
            'target_price': None,
            'stop_loss': None
        }
    
    # Create a prompt for the GPT model
    prompt = f"""
    Analyze this tweet for financial trading context. Extract the following information:
    
    Tweet: "{text}"
    
    1. Overall sentiment: [bullish/bearish/neutral]
    2. Stock tickers mentioned: [comma-separated list of tickers without $ symbol, or "none"]
    3. Time horizon: [day_trade/swing_trade/long_term/unknown]
    4. Type of post: [trade_suggestion/analysis/general_comment/question]
    5. Price targets (if any):
       - Entry price: [price or "none"]
       - Target price: [price or "none"]
       - Stop loss: [price or "none"]
    
    Response format: JSON only with these keys: sentiment, tickers, time_horizon, trade_type, entry_price, target_price, stop_loss
    """
    
    try:
        # Call GPT with retry mechanism
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": "You are a financial analyst specializing in extracting structured data from trading tweets. Respond ONLY with the requested JSON."},
                        {"role": "user", "content": prompt}
                    ],
                    response_format={"type": "json_object"},
                    temperature=0,
                    max_tokens=150
                )
                
                # Extract the JSON response
                result_text = response.choices[0].message.content
                result = json.loads(result_text)
                
                # Ensure we have all the expected keys with default values
                expected_keys = ['sentiment', 'tickers', 'time_horizon', 'trade_type', 'entry_price', 'target_price', 'stop_loss']
                for key in expected_keys:
                    if key not in result:
                        result[key] = 'unknown' if key != 'tickers' else ''
                
                return result
                
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"Attempt {attempt+1} failed: {e}. Retrying...")
                    time.sleep(2)  # Wait before retrying
                else:
                    print(f"Failed after {max_retries} attempts: {e}")
                    raise
        
    except Exception as e:
        print(f"Error in LLM classification: {e}")
        return {
            'sentiment': 'unknown',
            'tickers': '',
            'time_horizon': 'unknown',
            'trade_type': 'unknown',
            'entry_price': None,
            'target_price': None,
            'stop_loss': None
        }

# Function to process tweets in parallel
def process_tweets_in_parallel(tweet_list, client: OpenAI, max_workers=8):
    """
    Process a list of tweets in parallel using the LLM classification function.
    """
    # Function to process a single tweet
    def process_single_tweet(tweet):
        try:
            # Extract the text and id
            tweet_text = tweet.get('text', '')
            tweet_id = tweet.get('id', '')
            
            # Classify with LLM
            llm_result = classify_tweet_with_llm(tweet_text, client)
            
            # Combine the tweet data with the LLM results
            result = {
                'id': tweet_id,
                'text': tweet_text,
                'author': tweet.get('author', ''),
                'created_at': tweet.get('createdAt', ''),
                'is_parent': tweet.get('is_parent', True),
                'in_reply_to': tweet.get('in_reply_to', ''),
                **llm_result
            }
            
            return result
        except Exception as e:
            print(f"Error processing tweet {tweet.get('id', 'unknown')}: {e}")
            return None
    
    # Process tweets in parallel
    all_results = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks and create a map of future to tweet id
        future_to_id = {executor.submit(process_single_tweet, tweet): tweet.get('id', i) 
                        for i, tweet in enumerate(tweet_list)}
        
        # Process results as they complete
        for i, future in enumerate(concurrent.futures.as_completed(future_to_id)):
            tweet_id = future_to_id[future]
            try:
                result = future.result()
                if result:
                    all_results.append(result)
                
                # Print progress update every 10 tweets
                if (i+1) % 10 == 0 or i+1 == len(tweet_list):
                    print(f"Processed {i+1}/{len(tweet_list)} tweets ({(i+1)/len(tweet_list)*100:.1f}%)")
            except Exception as e:
                print(f"Error processing tweet {tweet_id}: {e}")
    
    # Convert to DataFrame
    if all_results:
        return pd.DataFrame(all_results)
    else:
        return pd.DataFrame()

# Function to analyze all tweets with parallel processing
def analyze_all_tweets_with_parallel_llm(organized_tweets, client: OpenAI, max_workers=8):
    """
    Analyze all organized tweets with parallel LLM processing for improved efficiency.
    """
    # Create a flat list of all tweets for processing
    all_tweets = []
    
    for conversation_id, conversation in organized_tweets:
        # Add the parent tweet
        parent = conversation['tweet']
        all_tweets.append(parent)
        
        # Add all replies
        for reply in conversation['replies']:
            all_tweets.append(reply)
    
    print(f"Prepared {len(all_tweets)} tweets for parallel processing")
    
    # Process all tweets in parallel
    start_time = time.time()
    results_df = process_tweets_in_parallel(all_tweets, client, max_workers=max_workers)
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

# This code will only run when the script is executed directly, not when imported
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
    
    # Organize the tweets into conversation threads
    organized_tweets = organize_tweets(df)
    
    # Run the basic analysis
    print("\nRunning basic analysis without LLM...")
    basic_results = analyze_sample_tweets(organized_tweets)
    
    # Now run the parallel processing on ALL tweets
    print("\nNow running parallel LLM analysis on ALL tweets...")
    # Use 8 workers as requested for optimal performance
    MAX_WORKERS = 8
    
    # Initialize client here for standalone execution
    # You might want to load API_KEY from env var or a local file for standalone runs
    try:
        standalone_api_key = os.environ.get("OPENAI_API_KEY") # Example: Load from environment variable
        if not standalone_api_key:
             # Fallback or error if key not found for standalone run
             # For testing, you could hardcode temporarily, but avoid committing it
             print("Warning: OPENAI_API_KEY environment variable not set for standalone run.")
             # standalone_api_key = "your_temp_key_for_local_test_only" 
             # For now, let's raise an error if not set
             raise ValueError("API key needed for standalone execution")
             
        standalone_client = OpenAI(api_key=standalone_api_key)
        all_results = analyze_all_tweets_with_parallel_llm(organized_tweets, standalone_client, max_workers=MAX_WORKERS)
    except Exception as main_exec_err:
         print(f"Error during standalone execution: {main_exec_err}")
         all_results = pd.DataFrame() # Ensure all_results is defined

    print("\nAnalysis complete!")
