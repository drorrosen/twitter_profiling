import os
from google.oauth2.credentials import Credentials
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# If modifying these scopes, delete the file token.json.
SCOPES = ['https://www.googleapis.com/auth/spreadsheets.readonly']

# Path to your service account credentials JSON file
SERVICE_ACCOUNT_FILE = 'tweet-individuals.json'

def connect_to_sheets():
    try:
        # Try to get credentials from Streamlit secrets first
        try:
            import streamlit as st
            creds_info = st.secrets["gcp_service_account"]
            creds = service_account.Credentials.from_service_account_info(
                creds_info, scopes=SCOPES)
        except:
            # Fall back to file if available
            creds = service_account.Credentials.from_service_account_file(
                SERVICE_ACCOUNT_FILE, scopes=SCOPES)

        # Build the Sheets API service
        service = build('sheets', 'v4', credentials=creds)
        
        return service

    except Exception as err:
        print(f"An error occurred connecting to Google Sheets: {err}")
        return None

def get_sheet_data(spreadsheet_id, range_name):
    """
    Retrieves data from specified Google Sheet
    
    Args:
        spreadsheet_id: The ID of the spreadsheet (found in the URL)
        range_name: The A1 notation of the range to retrieve
    
    Returns:
        The values from the specified range
    """
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
        print(f"An error occurred: {err}")
        return None

def extract_username(url):
    """Extract username from Twitter/X URL"""
    if not url:
        return ""
    # Remove any trailing status part of the URL
    url = url.split('/status/')[0]
    # Get the last part of the URL which is typically the username
    return url.rstrip('/').split('/')[-1]

if __name__ == "__main__":
    # Update with your specific spreadsheet ID
    SAMPLE_SPREADSHEET_ID = '1e7qzNwQv7NCuT9coefCy7v1WtSV5b_FKW6TLy-3b25o'
    # Update range to capture the Twitter accounts in columns A and B
    SAMPLE_RANGE_NAME = 'Sheet1!A1:B54'  # Adjusted to cover all rows with data
    
    data = get_sheet_data(SAMPLE_SPREADSHEET_ID, SAMPLE_RANGE_NAME)
    if data:
        print('Data retrieved successfully:')
        
        # Print Lists
        print('\nTwitter Lists:')
        print('-' * 50)
        for row in data[1:]:  # Skip header row
            if row[0]:  # If there's a value in the first column
                url = row[0]
                list_id = extract_username(row[0])
                print(f"List ID: {list_id:<20} URL: {url}")
        
        # Print Individual Accounts
        print('\nIndividual Accounts:')
        print('-' * 50)
        for row in data[1:]:  # Skip header row
            if len(row) > 1 and row[1]:  # If there's a value in the second column
                url = row[1]
                username = extract_username(row[1])
                print(f"Username: {username:<20} URL: {url}")
