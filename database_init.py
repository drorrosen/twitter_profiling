import psycopg2

# Connect to the RDS instance with the correct database name
# Note: The database name should be "postgres" (default) or the actual database name
# The database identifier in AWS is different from the database name
conn = psycopg2.connect(
    host="database-twitter.cdh6c6zr5mbr.us-east-1.rds.amazonaws.com",
    database="postgres",  # Default database name
    user="postgres",
    password="DrorMai531"
)

# Create a cursor object
cursor = conn.cursor()

# First, drop the existing table if it exists
drop_table_query = '''
DROP TABLE IF EXISTS tweets;
'''

# SQL to create the tweets table with a more flexible structure
create_table_query = '''
CREATE TABLE IF NOT EXISTS tweets (
    id SERIAL PRIMARY KEY,
    tweet_id FLOAT,
    author VARCHAR(100),
    text TEXT,
    created_at TIMESTAMP,
    likes FLOAT,
    retweets FLOAT,
    replies_count FLOAT,
    views FLOAT,
    author_followers FLOAT,
    author_following FLOAT,
    sentiment VARCHAR(50),
    trade_type VARCHAR(50),
    time_horizon VARCHAR(50),
    prediction_date TIMESTAMP,
    tickers_mentioned TEXT,
    conversation_id FLOAT,
    tweet_type VARCHAR(50),
    prediction_correct BOOLEAN,
    start_price FLOAT,
    end_price FLOAT,
    start_date TIMESTAMP,
    end_date TIMESTAMP,
    company_names TEXT,
    stocks TEXT,
    reply_to_tweet_id FLOAT,
    reply_to_user VARCHAR(100),
    author_verified BOOLEAN,
    author_blue_verified BOOLEAN,
    created_date DATE,
    has_ticker BOOLEAN,
    price_change_pct FLOAT,
    actual_return FLOAT,
    validated_ticker VARCHAR(50),
    prediction_score FLOAT,
    is_deleted BOOLEAN DEFAULT FALSE,
    is_analytically_actionable BOOLEAN DEFAULT FALSE,
    processing_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Add a unique constraint after table creation
ALTER TABLE tweets ADD CONSTRAINT unique_tweet_analysis 
UNIQUE (tweet_id, created_at, sentiment);

-- Add check constraints to ensure only high-quality tweets are stored
ALTER TABLE tweets ADD CONSTRAINT check_has_ticker 
CHECK (tickers_mentioned IS NOT NULL AND tickers_mentioned != '');

ALTER TABLE tweets ADD CONSTRAINT check_has_sentiment
CHECK (sentiment IN ('bullish', 'bearish'));

-- Remove the time horizon constraint
ALTER TABLE tweets DROP CONSTRAINT IF EXISTS check_has_time_horizon;

-- Remove any constraint on trade_type if it exists
ALTER TABLE tweets DROP CONSTRAINT IF EXISTS check_trade_type;

-- Create indexes for faster queries
CREATE INDEX IF NOT EXISTS idx_tweet_id ON tweets(tweet_id);
CREATE INDEX IF NOT EXISTS idx_author ON tweets(author);
CREATE INDEX IF NOT EXISTS idx_tickers ON tweets(tickers_mentioned);
CREATE INDEX IF NOT EXISTS idx_created_at ON tweets(created_at);
CREATE INDEX IF NOT EXISTS idx_processing_date ON tweets(processing_date);
'''

try:
    # Execute the drop table query
    cursor.execute(drop_table_query)
    print("✅ Existing 'tweets' table dropped")
    
    # Execute the create table query
    cursor.execute(create_table_query)
    
    # Commit the changes
    conn.commit()
    print("✅ New 'tweets' table created with proper unique constraint!")
    
    # Show table structure
    cursor.execute("SELECT column_name, data_type, is_nullable FROM information_schema.columns WHERE table_name = 'tweets';")
    columns = cursor.fetchall()
    print("\n📋 Table Structure:")
    for column in columns:
        print(f"  - {column[0]}: {column[1]} (Nullable: {column[2]})")
    
    # Show constraints information
    cursor.execute("""
        SELECT conname, pg_get_constraintdef(oid)
        FROM pg_constraint
        WHERE conrelid = 'tweets'::regclass;
    """)
    constraints = cursor.fetchall()
    print("\n🔒 Table Constraints:")
    for constraint in constraints:
        print(f"  - {constraint[0]}: {constraint[1]}")
    
    # Show indexes information
    cursor.execute("""
        SELECT indexname, indexdef
        FROM pg_indexes
        WHERE tablename = 'tweets';
    """)
    indexes = cursor.fetchall()
    print("\n📊 Table Indexes:")
    for index in indexes:
        print(f"  - {index[0]}: {index[1]}")
    
except Exception as e:
    print(f"❌ Error updating table: {e}")
    import traceback
    traceback.print_exc()
finally:
    # Close the cursor and connection
    cursor.close()
    conn.close()
    print("\n✅ Database connection closed")

