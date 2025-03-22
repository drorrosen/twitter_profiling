from apify_client import ApifyClient
import pandas as pd
from typing import Dict, List
import json
from datetime import datetime, timedelta
import os
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import psycopg2

# Sample mode flag - set to True to fetch only a small number of tweets for testing
SAMPLE_MODE = True
# Number of tweets to fetch in sample mode
SAMPLE_SIZE = 100

# Database connection parameters
DB_HOST = "database-twitter.cdh6c6zr5mbr.us-east-1.rds.amazonaws.com"
DB_NAME = "postgres"
DB_USER = "postgres"
DB_PASSWORD = "DrorMai531"

class TwitterScraper:
    def __init__(self, api_token: str):
        """Initialize the Twitter scraper with Apify API token."""
        self.client = ApifyClient(api_token)
        
    def prepare_input(self, 
                     start_urls: List[str] = None,
                     search_terms: List[str] = None,
                     twitter_handles: List[str] = None,
                     conversation_ids: List[str] = None,
                     max_items: int = 1000,
                     **kwargs) -> Dict:
        """Prepare the input parameters for the Apify Actor."""
        run_input = {
            "startUrls": start_urls or [],
            "searchTerms": search_terms or [],
            "twitterHandles": twitter_handles or [],
            "conversationIds": conversation_ids or [],
            "maxItems": max_items,
            "sort": "Latest",
            "tweetLanguage": "en",
        }
        # Add any additional parameters
        run_input.update(kwargs)
        return run_input

    def extract_tweets(self, run_input: Dict) -> List[Dict]:
        """Run the Apify Actor and extract tweets."""
        # Run the Actor and wait for it to finish
        run = self.client.actor("61RPP7dywgiy0JPD0").call(run_input=run_input)
        
        # Fetch results from the run's dataset
        tweets = list(self.client.dataset(run["defaultDatasetId"]).iterate_items())
        return tweets

    def save_to_csv(self, tweets: List[Dict], output_file: str = 'tweets.csv'):
        """Save tweets to a CSV file."""
        df = pd.DataFrame(tweets)
        df.to_csv(output_file, index=False)
        print(f"Saved {len(tweets)} tweets to {output_file}")

    def save_to_json(self, tweets: List[Dict], output_file: str = 'tweets.json'):
        """Save tweets to a JSON file."""
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(tweets, f, ensure_ascii=False, indent=2)
        print(f"Saved {len(tweets)} tweets to {output_file}")

# Google Sheets integration functions
def connect_to_sheets():
    """Connect to Google Sheets API."""
    try:
        # Path to your service account credentials JSON file
        SERVICE_ACCOUNT_FILE = 'tweet-individuals.json'
        SCOPES = ['https://www.googleapis.com/auth/spreadsheets']  # Full access scope
        
        # Create credentials using service account
        creds = service_account.Credentials.from_service_account_file(
            SERVICE_ACCOUNT_FILE, scopes=SCOPES)

        # Build the Sheets API service
        service = build('sheets', 'v4', credentials=creds)
        
        return service

    except HttpError as err:
        print(f"An error occurred connecting to Google Sheets: {err}")
        return None
    except FileNotFoundError:
        print(f"Google Sheets credentials file not found: {SERVICE_ACCOUNT_FILE}")
        print("Please check that the credentials file exists in the correct location")
        return None

def get_sheet_data(spreadsheet_id, range_name):
    """Retrieves data from specified Google Sheet."""
    try:
        service = connect_to_sheets()
        
        # Call the Sheets API
        sheet = service.spreadsheets()
        result = sheet.values().get(
            spreadsheetId=spreadsheet_id,
            range=range_name
        ).execute()
        
        values = result.get('values', [])
        return values

    except HttpError as err:
        print(f"An error occurred retrieving sheet data: {err}")
        return None

def extract_username(url):
    """Extract username from Twitter/X URL."""
    if not url:
        return ""
    
    # If it's already just a username without URL, return it
    if not url.startswith('http') and not url.startswith('@'):
        return url.strip()
    
    # Handle @username format
    if url.startswith('@'):
        return url[1:].strip()
    
    # Remove any trailing status part of the URL
    url = url.split('/status/')[0]
    # Remove query parameters if present
    url = url.split('?')[0]
    # Get the last part of the URL which is typically the username
    username = url.rstrip('/').split('/')[-1]
    
    # Additional validation
    if username in ['twitter.com', 'x.com']:
        # In case URL format is just the domain
        return ""
    
    return username.strip()

def get_twitter_handles_from_sheet(verbose=True):
    """Get Twitter handles from Google Sheet."""
    try:
        # Google Sheet ID and range
        SPREADSHEET_ID = '1Ksv8ETBvjqBKRbdqfkT9yEsRwHGn98oBu7dVNsC5Bsc'
        RANGE_NAME = 'Twitter Handles!A2:B'
        
        # Get data from sheet
        values = get_sheet_data(SPREADSHEET_ID, RANGE_NAME)
        
        if not values:
            print("❌ ERROR: No data returned from Google Sheet")
            print("⚠️ Check that the spreadsheet ID and range are correct")
            return [], []
        
        if verbose:
            print("✅ Data retrieved successfully from Google Sheet")
            print(f"✅ Retrieved {len(values)} rows from Google Sheet")
        
        # Extract handles from the sheet
        twitter_handles = []
        twitter_lists = []
        
        for row in values:
            if len(row) > 0 and row[0].strip():
                handle = row[0].strip()
                
                # Check if it's a Twitter list or a handle
                if '/i/lists/' in handle or '/lists/' in handle:
                    twitter_lists.append(handle)
                else:
                    # Remove @ if present
                    if handle.startswith('@'):
                        handle = handle[1:]
                    
                    twitter_handles.append(handle)
                    if verbose:
                        print(f"Added handle from sheet: {handle}")
        
        if verbose:
            print(f"Found {len(twitter_handles)} Twitter handles and {len(twitter_lists)} Twitter lists")
        
        # Check if we found any handles
        if not twitter_handles:
            print("❌ ERROR: No handles found in the Google Sheet")
            return [], []
        
        return twitter_handles, twitter_lists
    
    except Exception as e:
        print(f"❌ Error getting Twitter handles from sheet: {e}")
        return [], []

def get_common_params():
    """Return common parameters for both processes."""
    return {
        "sort": "Latest",
        "tweetLanguage": "en",
        "includeSearchTerms": False,
        "onlyImage": False,
        "onlyQuote": False,
        "onlyTwitterBlue": False,
        "onlyVerifiedUsers": False,
        "onlyVideo": False,
        "customMapFunction": "(object) => { return {...object} }"
    }

def connect_to_database():
    """Connect to the PostgreSQL database."""
    try:
        conn = psycopg2.connect(
            host=DB_HOST,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD
        )
        return conn
    except Exception as e:
        print(f"Error connecting to database: {e}")
        return None

def get_existing_authors_from_db():
    """Get list of Twitter handles that already have data in the database."""
    try:
        conn = connect_to_database()
        if not conn:
            print("⚠️ Could not connect to database - cannot check for existing authors")
            return []
        
        cursor = conn.cursor()
        # Query to get distinct authors from the database
        cursor.execute("SELECT DISTINCT author FROM tweets")
        authors = [row[0].lower() for row in cursor.fetchall() if row[0]]
        
        cursor.close()
        conn.close()
        
        print(f"📊 Found {len(authors)} existing authors in database")
        return authors
    except Exception as e:
        print(f"❌ Error getting existing authors: {e}")
        return []

def get_author_date_ranges_from_db():
    """Get date ranges of tweets for each author in the database."""
    try:
        conn = connect_to_database()
        if not conn:
            return {}
        
        cursor = conn.cursor()
        # Query to get min and max dates for each author
        cursor.execute("""
            SELECT author, MIN(created_at), MAX(created_at)
            FROM tweets
            GROUP BY author
        """)
        
        author_dates = {}
        for row in cursor.fetchall():
            if row[0]:  # Ensure author is not None
                author_dates[row[0].lower()] = {
                    'earliest_date': row[1],
                    'latest_date': row[2]
                }
        
        cursor.close()
        conn.close()
        
        print(f"Got date ranges for {len(author_dates)} authors from database")
        return author_dates
    except Exception as e:
        print(f"Error getting author date ranges: {e}")
        return {}

def process_current_tweets(scraper, twitter_handles, period="daily"):
    """Process current tweets (daily or weekly)."""
    today = datetime.now().strftime("%Y-%m-%d")
    
    if period == "daily":
        # For daily, get tweets from yesterday to today
        yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
        start_date = yesterday
        file_suffix = "daily"
    else:
        # For weekly, get tweets from 7 days ago to today
        week_ago = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
        start_date = week_ago
        file_suffix = "weekly"
    
    print(f"📅 Fetching {period} tweets from {start_date} to {today}")
    
    # Get the date ranges for each author in the database
    author_date_ranges = get_author_date_ranges_from_db()
    
    # Filter handles that need updates
    handles_to_update = []
    known_handles = set()  # Track handles we've already added to prevent duplicates
    
    for handle in twitter_handles:
        handle_lower = handle.lower()
        
        # Skip if we've already processed this handle (case-insensitive)
        if handle_lower in known_handles:
            continue
            
        # If author exists in database, check if we have recent data
        if handle_lower in author_date_ranges:
            latest_date = author_date_ranges[handle_lower]['latest_date']
            # If latest data is older than our start date, we need to update
            if latest_date.strftime("%Y-%m-%d") < start_date:
                handles_to_update.append(handle)
                known_handles.add(handle_lower)
                print(f"🔄 Will update {handle} - Last data from {latest_date.strftime('%Y-%m-%d')}")
            else:
                print(f"✅ Skip {handle} - Already has data from {latest_date.strftime('%Y-%m-%d')}")
        else:
            # If author is not in database, add to update list
            handles_to_update.append(handle)
            known_handles.add(handle_lower)
            print(f"🆕 Will fetch current tweets for new handle: {handle}")
    
    if not handles_to_update:
        print(f"No handles need {period} updates")
        return []
    
    print(f"Updating {period} tweets for {len(handles_to_update)} handles: {', '.join(handles_to_update)}")
    
    # Configure the run input with specific parameters
    common_params = get_common_params()
    
    # Use sample size if in sample mode, otherwise use original size
    max_items = SAMPLE_SIZE if SAMPLE_MODE else 10000
    
    run_input = scraper.prepare_input(
        twitter_handles=handles_to_update,
        max_items=max_items,
        start=start_date,
        end=today,
        **common_params
    )
    
    # Extract tweets
    tweets = scraper.extract_tweets(run_input)
    
    return tweets

def load_processed_handles():
    """Load the list of handles that have already been processed historically."""
    try:
        if os.path.exists('processed_handles.json'):
            with open('processed_handles.json', 'r') as f:
                return json.load(f)
        return []
    except Exception as e:
        print(f"Error loading processed handles: {e}")
        return []

def save_processed_handles(handles):
    """Save the list of handles that have been processed historically."""
    try:
        with open('processed_handles.json', 'w') as f:
            json.dump(handles, f)
    except Exception as e:
        print(f"Error saving processed handles: {e}")

def process_historical_tweets_for_new_handles(scraper, all_handles, processed_handles):
    """Process historical tweets only for handles that haven't been processed before."""
    # First check which handles are already in the database
    existing_authors = get_existing_authors_from_db()
    
    # Create lowercase set of processed handles for efficient lookup
    processed_handles_lower = {h.lower() for h in processed_handles}
    
    # Filter for handles not in the database and not already processed
    new_handles = []
    known_handles = set()
    
    # Track handles that were marked as processed but aren't in the database
    # (these might need to be reprocessed)
    handles_to_reset = []
    
    for handle in all_handles:
        handle_lower = handle.lower()
        
        if handle_lower not in existing_authors:
            print(f"📝 Handle not in database: {handle}")
            
            if handle_lower not in processed_handles_lower:
                print(f"🆕 NEW HANDLE DETECTED: {handle} - Will process historical data")
                if handle_lower not in known_handles:
                    new_handles.append(handle)
                    known_handles.add(handle_lower)
            else:
                print(f"⚠️ WARNING: {handle} was previously marked as processed but is not in the database!")
                print(f"🔄 Will reset and reprocess {handle}")
                # Add to handles to be reset
                handles_to_reset.append(handle)
                # Also add to new handles to process
                if handle_lower not in known_handles:
                    new_handles.append(handle)
                    known_handles.add(handle_lower)
        else:
            print(f"✅ Handle already exists in database: {handle}")
    
    # Reset the previously processed handles that aren't in database
    if handles_to_reset:
        print(f"Resetting {len(handles_to_reset)} handles that were marked as processed but not in database")
        # Remove from processed_handles list
        processed_handles = [h for h in processed_handles if h.lower() not in {handle.lower() for handle in handles_to_reset}]
        # Save the updated list
        save_processed_handles(processed_handles)
    
    if not new_handles:
        print("No new handles to process historically")
        return []
    
    print(f"Processing historical tweets for {len(new_handles)} new handles: {', '.join(new_handles)}")
    
    today = datetime.now().strftime("%Y-%m-%d")
    start_date = "2023-01-01"
    
    print(f"Fetching historical tweets from {start_date} to {today}")
    
    # Configure the run input with specific parameters
    common_params = get_common_params()
    
    # Use sample size if in sample mode, otherwise use original size
    max_items = SAMPLE_SIZE if SAMPLE_MODE else 150000
    
    run_input = scraper.prepare_input(
        twitter_handles=new_handles,
        max_items=max_items,  # Modified to respect sample mode
        start=start_date,
        end=today,
        **common_params
    )
    
    # Extract tweets
    tweets = scraper.extract_tweets(run_input)
    
    # Add the new handles to the processed list
    processed_handles.extend(new_handles)
    save_processed_handles(processed_handles)
    
    return tweets

def process_historical_tweets(scraper, twitter_handles):
    """Process historical tweets from 2023 until now."""
    today = datetime.now().strftime("%Y-%m-%d")
    start_date = "2023-01-01"
    
    print(f"Fetching historical tweets from {start_date} to {today}")
    
    # Configure the run input with specific parameters
    common_params = get_common_params()
    
    # Use sample size if in sample mode, otherwise use original size
    max_items = SAMPLE_SIZE if SAMPLE_MODE else 150000
    
    run_input = scraper.prepare_input(
        twitter_handles=twitter_handles,
        max_items=max_items,  # Modified to respect sample mode
        start=start_date,
        end=today,
        **common_params
    )
    
    # Extract tweets
    tweets = scraper.extract_tweets(run_input)
    
    return tweets

def print_processed_vs_database_status():
    """Print a debug report comparing processed handles to database handles."""
    processed_handles = load_processed_handles()
    existing_authors = get_existing_authors_from_db()
    
    print("\n🔍 PROCESSED VS DATABASE STATUS:")
    print(f"✅ Handles in processed_handles.json: {len(processed_handles)}")
    print(f"📊 Handles in database: {len(existing_authors)}")
    
    # Find handles in processed list but not in database
    missing_from_db = [h for h in processed_handles if h.lower() not in existing_authors]
    if missing_from_db:
        print(f"⚠️ {len(missing_from_db)} handles are marked as processed but not in database:")
        for handle in missing_from_db:
            print(f"   - {handle}")
    else:
        print("✅ All processed handles are in the database")
    
    # Find handles in database but not in processed list
    missing_from_processed = [h for h in existing_authors if not any(p.lower() == h for p in processed_handles)]
    if missing_from_processed:
        print(f"ℹ️ {len(missing_from_processed)} handles are in database but not marked as processed:")
        for handle in missing_from_processed[:10]:  # Limit to first 10 to avoid huge output
            print(f"   - {handle}")
        if len(missing_from_processed) > 10:
            print(f"   - ... and {len(missing_from_processed) - 10} more")

def main():
    # API token
    api_token = "apify_api_kdevcdwOVQ5K4HugmeLYlaNgaxeOGG3dkcwc"
    
    # Initialize scraper
    scraper = TwitterScraper(api_token)
    
    # Get Twitter handles from Google Sheet
    twitter_handles, twitter_lists = get_twitter_handles_from_sheet()
    
    if not twitter_handles:
        print("No Twitter handles found. Exiting.")
        return
    
    # Print debug information comparing processed handles to database
    print_processed_vs_database_status()
    
    # Load the list of handles that have already been processed historically
    processed_handles = load_processed_handles()
    print(f"Previously processed {len(processed_handles)} handles historically")
    
    # Process current tweets (daily) for all handles that need updates
    current_tweets = process_current_tweets(scraper, twitter_handles, period="daily")
    print(f"Processed {len(current_tweets)} current tweets")
    
    # Process historical tweets only for new handles not in the database
    new_historical_tweets = process_historical_tweets_for_new_handles(
        scraper, twitter_handles, processed_handles)
    print(f"Processed {len(new_historical_tweets)} historical tweets for new handles")
    
    # Combine all tweets and save to organized_tweets.csv
    all_tweets = current_tweets + new_historical_tweets
    if all_tweets:
        print(f"Total tweets processed: {len(all_tweets)}")
    else:
        print("No new tweets processed")

if __name__ == "__main__":
    main()
