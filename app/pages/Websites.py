import streamlit as st
import sqlite3
import pandas as pd
from pathlib import Path
import logging
from datetime import datetime

# Set up logging
def setup_logging():
    base_dir = Path(__file__).resolve().parent.parent
    logs_dir = base_dir / "logs"
    logs_dir.mkdir(exist_ok=True)
    
    log_file = logs_dir / f"database_page_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

def get_available_databases():
    """Get list of SQLite databases in the Data/dbs directory"""
    try:
        db_dir = Path(__file__).resolve().parent.parent / "Data" / "dbs"
        # For now, only return news_scrapper.db
        all_dbs = list(db_dir.glob("*.db"))
        return [db for db in all_dbs if db.name == "news_scrapper.db"]
    except Exception as e:
        logger.error(f"Error getting databases: {e}")
        return []

def get_tables_in_db(db_path):
    """Get list of tables in the selected database"""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        conn.close()
        return [table[0] for table in tables]
    except Exception as e:
        logger.error(f"Error getting tables from {db_path}: {e}")
        return []

def get_table_data(db_path, table_name):
    """Get data from selected table as a pandas DataFrame"""
    try:
        conn = sqlite3.connect(db_path)
        query = f"SELECT * FROM {table_name}"
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df
    except Exception as e:
        logger.error(f"Error getting data from {table_name}: {e}")
        return pd.DataFrame()

def update_table_data(db_path, table_name, df):
    """Update table with modified DataFrame data"""
    try:
        conn = sqlite3.connect(db_path)
        # Drop the existing table and replace with new data
        df.to_sql(table_name, conn, if_exists='replace', index=False)
        conn.close()
        logger.info(f"Successfully updated table {table_name}")
        return True
    except Exception as e:
        logger.error(f"Error updating table {table_name}: {e}")
        return False

def render_databases_page():
    st.title("Database Manager")

    # Add custom CSS
    st.markdown("""
        <style>
        .stButton button {
            background: linear-gradient(120deg, #000000, #1e3a8a);
            color: white;
            border: none;
            padding: 0.6rem 1.2rem;
            border-radius: 5px;
            font-weight: 600;
            transition: all 0.3s ease;
        }
        
        .stSelectbox {
            background: rgba(255, 255, 255, 0.05);
            border-radius: 5px;
        }
        
        .dataframe {
            background: rgba(255, 255, 255, 0.05);
            border-radius: 5px;
            padding: 10px;
        }
        
        .info-box {
            padding: 1rem;
            background-color: rgba(255, 255, 255, 0.05);
            border-radius: 5px;
            margin-bottom: 1rem;
        }
        </style>
    """, unsafe_allow_html=True)

    # Add info box about news_scrapper.db
    st.markdown("""
        <div class="info-box">
        This page allows you to view and edit the news scrapper database tables. 
        The database contains information about scraped articles and website configurations.
        </div>
    """, unsafe_allow_html=True)

    # Get available databases
    databases = get_available_databases()
    
    if not databases:
        st.warning("No databases found in Data/dbs directory")
        return

    # Database selection
    selected_db = st.selectbox(
        "Select Database",
        options=databases,
        format_func=lambda x: x.name
    )

    if selected_db:
        # Get tables in selected database
        tables = get_tables_in_db(selected_db)
        
        if not tables:
            st.warning(f"No tables found in {selected_db.name}")
            return

        # Table selection
        selected_table = st.selectbox(
            "Select Table",
            options=tables,
            help="Select a table to view and edit its contents"
        )

        if selected_table:
            # Get and display table data
            df = get_table_data(selected_db, selected_table)
            
            if df.empty:
                st.warning(f"No data found in table {selected_table}")
                return

            # Create an editable data editor with improved styling
            edited_df = st.data_editor(
                df,
                num_rows="dynamic",
                use_container_width=True,
                hide_index=True,
                key=f"editor_{selected_table}"
            )

            # Add save button with confirmation
            if st.button("Save Changes", help="Save your modifications to the database"):
                if st.session_state.get("username_id"):  # Check if user is logged in
                    confirm = st.checkbox("Confirm changes? This action cannot be undone.")
                    if confirm:
                        if update_table_data(selected_db, selected_table, edited_df):
                            st.success("Changes saved successfully!")
                            # Log the change
                            logger.info(f"User {st.session_state.username_id} updated table {selected_table}")
                        else:
                            st.error("Failed to save changes. Please check the logs.")
                else:
                    st.error("Please log in to make changes to the database.")

if __name__ == "__main__":
    render_databases_page() 