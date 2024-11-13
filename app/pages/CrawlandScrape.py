import streamlit as st
import pandas as pd
from scrape.news_scrapper import NewsScrapper
import io
from utils.logging_config import setup_cloudwatch_logging

logger = setup_cloudwatch_logging('crawl_page')

# Cache file reading for uploaded files
@st.cache_data
def read_uploaded_file(uploaded_file):
    try:
        if uploaded_file.name.endswith('.csv'):
            return pd.read_csv(uploaded_file)
        else:
            return pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"Error reading file: {str(e)}")
        return None

def render_crawl_page():
    logger.info("Rendering crawl page")
    st.title("Website Crawler")
    
    # Split the screen into two columns
    left_col, right_col = st.columns(2)
    
    # Left column - Manual input
    with left_col:
        st.header("Manual Input")
        
        # Initialize session state for manual entries if not exists
        if 'manual_entries' not in st.session_state:
            st.session_state.manual_entries = []
        
        # Form for manual entry
        with st.form("manual_entry_form"):
            website_url = st.text_input("Website URL")
            login_url = st.text_input("Login URL (optional)")
            username = st.text_input("Username (optional)")
            password = st.text_input("Password (optional)", type="password")
            crawl = st.checkbox("Enable Crawling", value=False)
            max_pages = st.number_input("Max Pages", min_value=1, value=3)
            
            submitted = st.form_submit_button("Add Website")
            if submitted and website_url:
                new_entry = {
                    'website_url': website_url,
                    'login_url': login_url,
                    'username': username,
                    'password': password,
                    'crawl': crawl,
                    'max_pages': max_pages
                }
                st.session_state.manual_entries.append(new_entry)
        
        # Display manual entries
        if st.session_state.manual_entries:
            st.subheader("Added Websites")
            for i, entry in enumerate(st.session_state.manual_entries):
                st.text(f"{i+1}. {entry['website_url']}")
            
            if st.button("Clear All"):
                st.session_state.manual_entries = []
    
    # Right column - File upload
    with right_col:
        st.header("File Upload")
        uploaded_file = st.file_uploader("Upload CSV or Excel file", type=['csv', 'xlsx'])
        
        # Initialize file_entries as empty list
        file_entries = []
        
        if uploaded_file is not None:
            try:
                df = read_uploaded_file(uploaded_file)
                
                st.dataframe(df)
                
                # Convert DataFrame to list of dictionaries
                file_entries = df.to_dict('records')
                
                # Validate required columns
                required_cols = ['website_url']
                if not all(col in df.columns for col in required_cols):
                    st.error("File must contain 'website_url' column")
                    return
            except Exception as e:
                st.error(f"Error reading file: {str(e)}")
                return
    
    # Start crawling button
    if st.button("Start Crawling"):
        logger.info(f"Starting crawl for {len(all_entries)} websites")
        
        # Combine manual and file entries
        all_entries = st.session_state.manual_entries + file_entries
        
        if not all_entries:
            st.warning("No websites to crawl. Please add websites manually or upload a file.")
            return
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, entry in enumerate(all_entries):
            try:
                logger.info(f"Crawling website: {entry['website_url']}")
                status_text.text(f"Crawling {entry['website_url']}...")
                
                # Initialize scraper
                scraper = NewsScrapper(**entry)
                
                # Perform scraping
                articles = scraper.scrape()
                
                # Store website info in database
                scraper.add_website()
                
                # Close the scraper
                scraper.close()
                
                # Update progress
                progress = (i + 1) / len(all_entries)
                progress_bar.progress(progress)
                
                logger.info(f"Successfully crawled: {entry['website_url']}")
                st.success(f"Successfully crawled {entry['website_url']}")
                
            except Exception as e:
                logger.error(f"Crawl failed for {entry['website_url']}: {e}", exc_info=True)
                st.error(f"Error crawling {entry['website_url']}: {str(e)}")
                continue
        
        status_text.text("Crawling completed!")
        st.session_state.manual_entries = []  # Clear manual entries after crawling 

if __name__ == "__main__":
    render_crawl_page()