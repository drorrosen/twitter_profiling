import psycopg2
import time
import sys
import pandas as pd
from tabulate import tabulate

def test_connection(host, database, user, password, max_attempts=3):
    """Test connection to PostgreSQL database with multiple attempts"""
    
    print(f"Testing connection to {host}...")
    print(f"Database: {database}")
    print(f"User: {user}")
    
    for attempt in range(1, max_attempts + 1):
        try:
            print(f"\nAttempt {attempt}/{max_attempts}...")
            
            # Connect to the database
            start_time = time.time()
            conn = psycopg2.connect(
                host=host,
                database=database,
                user=user,
                password=password,
                connect_timeout=10
            )
            
            # Calculate connection time
            connection_time = time.time() - start_time
            print(f"✅ Connection successful! (took {connection_time:.2f} seconds)")
            
            # Get server version
            cursor = conn.cursor()
            cursor.execute("SELECT version();")
            version = cursor.fetchone()[0]
            print(f"📊 Server version: {version}")
            
            # List all tables in the database
            cursor.execute("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public'
                ORDER BY table_name;
            """)
            tables = cursor.fetchall()
            
            if tables:
                print(f"📋 Tables in database:")
                for table in tables:
                    print(f"  - {table[0]}")
            else:
                print("📋 No tables found in the database.")
            
            # Close connection
            cursor.close()
            conn.close()
            print("✅ Connection closed properly")
            
            return True, conn, cursor
            
        except Exception as e:
            print(f"❌ Connection failed: {e}")
            
            if attempt < max_attempts:
                wait_time = 2 * attempt  # Exponential backoff
                print(f"⏳ Waiting {wait_time} seconds before retrying...")
                time.sleep(wait_time)
            else:
                print("❌ All connection attempts failed.")
                return False, None, None

def fetch_tweets_data(conn, limit=10):
    """Fetch tweets data from the database"""
    try:
        cursor = conn.cursor()
        
        # Get total count of tweets
        cursor.execute("SELECT COUNT(*) FROM tweets")
        total_count = cursor.fetchone()[0]
        print(f"\n📊 Total tweets in database: {total_count}")
        
        # If there are no tweets, return early
        if total_count == 0:
            print("❗ No tweets found in the database. Run the twitter-predictions.py script first to populate the database.")
            cursor.close()
            return None
        
        # Get count of validated predictions
        cursor.execute("SELECT COUNT(*) FROM tweets WHERE prediction_correct IS NOT NULL")
        validated_count = cursor.fetchone()[0]
        print(f"📊 Validated predictions: {validated_count} ({validated_count/total_count*100:.1f}% of total)")
        
        # Get count of correct predictions
        cursor.execute("SELECT COUNT(*) FROM tweets WHERE prediction_correct = 'true'")
        correct_count = cursor.fetchone()[0]
        if validated_count > 0:
            print(f"📊 Correct predictions: {correct_count} ({correct_count/validated_count*100:.1f}% of validated)")
        
        # Get most recent tweets
        print(f"\n📋 Most recent {limit} tweets:")
        cursor.execute("""
            SELECT tweet_id, author, created_at, tickers_mentioned, sentiment, 
                   prediction_correct, start_price, end_price
            FROM tweets
            ORDER BY created_at DESC
            LIMIT %s
        """, (limit,))
        
        recent_tweets = cursor.fetchall()
        
        # Convert to DataFrame for better display
        columns = ["Tweet ID", "Author", "Created At", "Tickers", "Sentiment", 
                   "Prediction Correct", "Start Price", "End Price"]
        df = pd.DataFrame(recent_tweets, columns=columns)
        
        # Display the data
        print(tabulate(df, headers='keys', tablefmt='psql', showindex=False))
        
        # Get top authors by tweet count
        print(f"\n📊 Top 5 authors by tweet count:")
        cursor.execute("""
            SELECT author, COUNT(*) as tweet_count
            FROM tweets
            GROUP BY author
            ORDER BY tweet_count DESC
            LIMIT 5
        """)
        
        top_authors = cursor.fetchall()
        author_df = pd.DataFrame(top_authors, columns=["Author", "Tweet Count"])
        print(tabulate(author_df, headers='keys', tablefmt='psql', showindex=False))
        
        # Get top tickers mentioned
        print(f"\n📊 Top 5 tickers mentioned:")
        cursor.execute("""
            SELECT tickers_mentioned, COUNT(*) as mention_count
            FROM tweets
            WHERE tickers_mentioned IS NOT NULL AND tickers_mentioned != ''
            GROUP BY tickers_mentioned
            ORDER BY mention_count DESC
            LIMIT 5
        """)
        
        top_tickers = cursor.fetchall()
        ticker_df = pd.DataFrame(top_tickers, columns=["Tickers", "Mention Count"])
        print(tabulate(ticker_df, headers='keys', tablefmt='psql', showindex=False))
        
        cursor.close()
        return df
        
    except Exception as e:
        print(f"❌ Error fetching tweets data: {e}")
        return None

if __name__ == "__main__":
    # RDS connection details
    HOST = "database-twitter.cdh6c6zr5mbr.us-east-1.rds.amazonaws.com"
    DATABASE = "postgres"
    USER = "postgres"
    PASSWORD = "DrorMai531"
    
    # Test the connection
    success, _, _ = test_connection(HOST, DATABASE, USER, PASSWORD)
    
    if success:
        # Create a new connection for additional operations
        try:
            conn = psycopg2.connect(
                host=HOST,
                database=DATABASE,
                user=USER,
                password=PASSWORD
            )
            
            # Fetch and display tweets data
            tweets_df = fetch_tweets_data(conn, limit=15)
            
            # Close the connection
            conn.close()
            print("\n✅ Database operations completed successfully")
            
        except Exception as e:
            print(f"\n❌ Error during database operations: {e}")

