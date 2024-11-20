import streamlit as st
from agents.agents_graph_V2 import StateMachines
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import PyPDF2
import json
from typing import Union, List, Dict, Any, Tuple
import time
from pathlib import Path
import traceback
import logging
import os
from datetime import datetime
import sqlite3
import uuid
from utils.logging_config import setup_cloudwatch_logging
import re
import streamlit.components.v1 as components

# Set up logging
logger = setup_cloudwatch_logging('simulation_page')

def initialize_database():
    """Ensures that the SQLite database and user_threads table exist."""
    try:
        logger.info("Initializing database")
        conn = sqlite3.connect('Data/dbs/users_data.db')
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_threads (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL,
                thread_id TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
        conn.close()
        logger.info("Database initialized successfully.")
    except Exception as e:
        logger.error(f"Database initialization failed: {e}", exc_info=True)

@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_user_threads(username: str) -> List[tuple]:
    """Retrieves all thread_ids and their creation dates for a given user."""
    try:
        conn = sqlite3.connect('Data/dbs/users_data.db')
        cursor = conn.cursor()
        cursor.execute("""
            SELECT thread_id, timestamp FROM user_threads
            WHERE username = ?
            ORDER BY timestamp DESC
        """, (username,))
        threads = cursor.fetchall()
        conn.close()
        return threads
    except Exception as e:
        logger.error(f"Error fetching threads for user {username}: {e}")
        return []

def add_new_thread(username: str, thread_id: str):
    """Adds a new thread record for a user."""
    if not username or not thread_id:
        raise ValueError("Username and thread_id must be provided")
        
    try:
        conn = sqlite3.connect('Data/dbs/users_data.db')
        cursor = conn.cursor()
        
        # Check if thread_id already exists
        cursor.execute("""
            SELECT COUNT(*) FROM user_threads 
            WHERE thread_id = ?
        """, (thread_id,))
        
        if cursor.fetchone()[0] > 0:
            raise ValueError(f"Thread ID {thread_id} already exists")
        
        # Insert new thread
        cursor.execute("""
            INSERT INTO user_threads (username, thread_id)
            VALUES (?, ?)
        """, (username, thread_id))
        
        conn.commit()
        logger.info(f"New thread {thread_id} added for user {username}")
        
    except sqlite3.Error as e:
        logger.error(f"Database error while adding thread: {e}")
        raise
    except Exception as e:
        logger.error(f"Error adding new thread for user {username}: {e}")
        raise
    finally:
        if 'conn' in locals():
            conn.close()

@st.cache_data
def read_pdf(uploaded_file) -> str:
    try:
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {str(e)}")
        st.error(f"Full traceback:\n{traceback.format_exc()}")
        return ""


def display_conversation_flow(current_state: dict):
    st.markdown("### Conversation Flow")
    
    conversations = [
        ("Level 2-3", "level2_3_conversation"),
        ("Level 1-3", "level1_3_conversation"),
        ("Level 1-2", "level1_2_conversation"),
        ("CEO Messages", "ceo_messages"),
        ("Digest", "digest")
    ]
    
    for title, key in conversations:
        st.markdown(f"#### {title}")
        if key in current_state:
            if key == "digest":
                # Display digest as a list of strings
                for item in current_state[key]:
                    st.text(item)
            else:
                # Display conversations
                for msg in current_state[key]:
                    if isinstance(msg, (HumanMessage, AIMessage, SystemMessage)):
                        content = msg.content
                    elif isinstance(msg, dict):
                        content = json.dumps(msg, indent=2)
                    else:
                        content = str(msg)
                    
                    if isinstance(content, str):
                        try:
                            # Try to parse the content as JSON for pretty printing
                            parsed_content = json.loads(content)
                            content = json.dumps(parsed_content, indent=2)
                        except json.JSONDecodeError:
                            # If it's not valid JSON, use the original string
                            pass
                    
                    st.text_area(f"{type(msg).__name__}", value=content, height=150, key=f"{key}_{id(msg)}")
        else:
            st.info(f"No {title} available.")
        
        st.markdown("---")  # Add a separator between sections

def add_start_page_css():
    st.markdown("""
        <style>
        /* Existing CSS */

        /* Compact control panel */
        .control-panel {
            display: flex;
            flex-direction: row;
            justify-content: center; /* Center the buttons */
            align-items: center;
            padding: 10px;
            margin-bottom: 10px;
            background: rgba(255, 255, 255, 0.05);
            border-radius: 10px;
        }

        /* Existing button styles */
        .stButton button {
            background: linear-gradient(120deg, #000000, #1e3a8a);
            color: white;
            border: none;
            padding: 0.6rem 1.2rem !important;    /* Adjusted padding for buttons */
            border-radius: 5px;
            font-weight: 600;
            transition: all 0.3s ease;
            width: 140px !important;              /* Adjusted width to accommodate three buttons */
            margin: 0 5px !important;             /* Reduce space between buttons */
            font-size: 0.8rem !important;         /* Optional: reduce font size */
            text-align: center;
        }

        /* New styles for control-panel buttons to make them 20% smaller */
        .control-panel > div.stButton > button {
            width: 160px !important;              /* 20% smaller than 200px */
            padding: 0.6rem 1.2rem !important;    /* 20% smaller padding */
            margin: 0 5px !important;             /* Reduce space between buttons */
            font-size: 0.8rem !important;         /* Optional: reduce font size */
        }

        /* Optional: Adjust spacing within the control panel */
        .control-panel > div.stButton {
            margin: 0 !important;
            padding: 0 !important;
        }

        /* Additional existing styles */
        /* ... (other existing styles) ... */

        </style>
    """, unsafe_allow_html=True)

def render_logo():
    st.markdown("""
        <div class="logo-container">
            <div class="logo-text">
                SEVEN<span style="font-family: 'Roboto'; transform: rotate(90deg); display: inline-block;">🤖</span>OTS
            </div>
        </div>
    """, unsafe_allow_html=True)

def render_start_page():
    # Initialize the database
    initialize_database()
    
    # Add all styles
    add_start_page_css()
    add_custom_styles()
    
    # Render logo at the very top
    render_logo()
    
    # Initialize state machine if needed
    if "state_machine" not in st.session_state or st.session_state.state_machine is None:
        initialize_state_machine()
    
    if st.session_state.state_machine is None:
        st.error("State machine is not initialized. Please check the logs.")
        return
    
    # Render main layout
    if not st.session_state.get("conversation_started", False):
        render_main_layout()
    else:
        render_conversation_state()

def render_main_layout():
    """Displays the three-column layout for thread selection and new conversation."""
    # Show current user and logout option
    col_user, col_logout = st.columns([3, 1])
    with col_user:
        st.markdown(f"**Current User:** {st.session_state.username_id}")
    with col_logout:
        if st.button("Logout"):
            st.session_state.username_id = None
            st.rerun()
    
    # Main layout columns - Previous conversations takes 1/3, New Simulation takes 2/3
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### Previous Conversations")
        username = st.session_state.username_id
        threads = get_user_threads(username)
        
        for thread_id, timestamp in threads:
            if st.button(f"📅 {timestamp}\n🔗 {thread_id[:8]}...", key=f"thread_{thread_id}"):
                logger.info(f"Loading existing thread {thread_id} for user {username}")
                initialize_conversation(
                    content="",
                    interrupt_before=True,
                    thread_id=thread_id,
                    is_new_thread=False
                )
    
    with col2:
        st.markdown("### Start New Simulation")
        uploaded_files = st.file_uploader("Upload PDF files", type=['pdf'], accept_multiple_files=True)
        
        if uploaded_files:
            combined_content = ""
            for uploaded_file in uploaded_files:
                content = read_pdf(uploaded_file)
                if content:
                    combined_content += f"\n\nFile: {uploaded_file.name}\n{content}"
            
            if combined_content:
                if st.button("Start New Simulation"):
                    try:
                        # Generate new thread ID
                        new_thread_id = str(uuid.uuid4())
                        
                        # Store in database first
                        add_new_thread(st.session_state.username_id, new_thread_id)
                        logger.info(f"Created new thread {new_thread_id} for user {st.session_state.username_id}")
                        
                        # Then initialize conversation with combined content
                        initialize_conversation(
                            content=combined_content,
                            interrupt_before=True,
                            thread_id=new_thread_id,
                            is_new_thread=True
                        )
                    except Exception as e:
                        logger.error(f"Failed to create new thread: {e}")
                        st.error("Failed to create new simulation. Please try again.")

def render_conversation_state():
    # Wrap the buttons in a control panel at the top using HTML
    st.markdown('<div class="control-panel">', unsafe_allow_html=True)

    # Render the Continue, Retry, and Reset buttons
    col1, col2, col3 = st.columns(3, gap="small")

    with col1:
        if st.button("Continue", key="btn_continue"):
            handle_continue()
    with col2:
        if st.button("Retry", key="btn_retry"):
            handle_retry()
    with col3:
        if st.button("Reset", key="btn_reset"):
            handle_reset()

    st.markdown('</div>', unsafe_allow_html=True)

    # Render columns for conversation messages
    cols = st.columns(2)

    with cols[0]:
        st.markdown('<div class="column-header">Meeting Simulation</div>', unsafe_allow_html=True)
        render_conversation_messages('meeting_simulation', only_content=True)  # Show only message content

    with cols[1]:
        # Dynamically choose from state elements that have 'conversation' in their names
        current_state_keys = st.session_state.current_state.keys()
        conversation_keys = [key for key in current_state_keys if 'conversation' in key]
        if conversation_keys:
            selected_state = st.selectbox("Select conversation to display", conversation_keys)
            st.markdown(f'<div class="column-header">{selected_state}</div>', unsafe_allow_html=True)
            render_conversation_messages(selected_state)
        else:
            st.info("No conversation elements available.")

def format_message_content(content: str) -> str:
    """
    Uses regex to find and clean JSON-like content within strings by removing brackets,
    quotes, and formatting with bullet points and new lines.

    Adjustments:
    - No empty line before the dict content.
    - Adds an empty line between each element of the dict.
    """
    import re

    # Pattern to find JSON-like structures: matches content between { and }
    json_pattern = r'{([^{}]*?)}'

    def clean_json_content(match):
        json_content = match.group(1)
        # Split by commas not within quotes
        parts = re.split(r',\s*(?=(?:[^"]*"[^"]*")*[^"]*$)', json_content)

        # Clean each key-value pair
        cleaned_parts = []
        for part in parts:
            # Remove quotes and extra whitespace
            cleaned = re.sub(r'"', '', part.strip())
            # Replace ':' with ': ' for better readability
            cleaned = re.sub(r'\s*:\s*', ': ', cleaned)
            cleaned_parts.append(f"- {cleaned}")

        # Join with double newlines between elements
        return '\n' + '\n\n'.join(cleaned_parts)

    # Replace JSON-like structures with cleaned format
    # Ensure there's no extra empty line before the dict
    cleaned_content = re.sub(json_pattern, clean_json_content, content)
    return cleaned_content

def render_conversation_messages(key, only_content=False):
    if key in st.session_state.current_state:
        messages = st.session_state.current_state[key]
        for i, msg in enumerate(messages):
            # Extract the original content
            if isinstance(msg, (HumanMessage, AIMessage, SystemMessage)):
                original_content = msg.content
            elif isinstance(msg, dict):
                original_content = msg.get('content', str(msg))
            elif hasattr(msg, 'content'):
                original_content = msg.content
            else:
                original_content = str(msg)

            # Format the content using regex
            formatted_content = format_message_content(original_content)

            # Create a unique key for the text area
            key_prefix = f"{key}_{i}"

            # Calculate the height based on the content
            textarea_height = get_text_area_height(formatted_content)

            # Display the editable text area with adjusted height
            edited_content = st.text_area(
                label="",
                value=formatted_content,
                key=f"{key_prefix}_edit",
                height=textarea_height
            )

            # If the content has changed, update the message
            if edited_content != formatted_content:
                # Update the message content in the session state
                if hasattr(msg, 'content'):
                    msg.content = edited_content
                elif isinstance(msg, dict):
                    msg['content'] = edited_content
                # Update the message in the session state
                st.session_state.current_state[key][i] = msg

            # Reduce spacing between messages
            st.markdown("<div style='margin-bottom: -10px;'></div>", unsafe_allow_html=True)

# Cache the state machine creation
@st.cache_resource
def get_shared_state_machine(interrupt_before: bool = True):
    """Create a single StateMachine instance shared across all sessions"""
    logger.info(f"Creating new shared StateMachine instance with interrupt_before={interrupt_before}")
    prompts_dir = os.path.join("app", "Data", "Prompts")
    return StateMachines(str(prompts_dir).strip(), interrupt_before)

def initialize_state_machine():
    with st.spinner("Initializing state machine..."):
        try:
            # Get interrupt_before value from session state, default to True if not set
            interrupt_before = st.session_state.get('interrupt_before', True)
            # Use the shared cached instance with interrupt_before parameter
            st.session_state.state_machine = get_shared_state_machine(interrupt_before)
            logger.info(f"Using shared state machine instance with interrupt_before={interrupt_before}")
            st.success("State machine initialized successfully!")
        except Exception as e:
            handle_error("Failed to initialize state machine", e)

def initialize_conversation(content: str, interrupt_before: bool, thread_id: str, is_new_thread: bool):
    logger.info(f"Initializing conversation: thread_id={thread_id}, is_new={is_new_thread}")
    """Initialize conversation with either new content or existing thread."""
    with st.spinner("Initializing conversation..."):
        try:
            st.session_state.interrupt_before = interrupt_before
            st.session_state.thread_id = thread_id
            
            if is_new_thread:
                initial_state = {
                    "news_insights": [content],
                    "digest": [""],
                    "ceo_messages": [],
                    "ceo_mode": ["research_information"]
                }
            else:
                initial_state = {
                    "news_insights": [""],
                    "digest": [""],
                    "ceo_messages": [],
                    "ceo_mode": ["research_information"]
                }
            
            # Pass thread_id to the start method
            result = st.session_state.state_machine.start(initial_state, thread_id=thread_id)
            if result is None:
                st.error("State machine returned None. Please check the implementation.")
                return
            
            st.session_state.current_state = result
            st.session_state.conversation_started = True
            
            if 'file_content' in st.session_state:
                del st.session_state.file_content
            
            st.success("Conversation started successfully!")
            time.sleep(1)
            st.rerun()
            logger.info(f"Conversation initialized successfully: thread_id={thread_id}")
        except Exception as e:
            logger.error(f"Conversation initialization failed: {e}", exc_info=True)
            handle_error("Error starting conversation", e)

def handle_continue():
    """Handles the Continue button click."""
    with st.spinner("Processing next step..."):
        try:
            username = st.session_state.get("username")
            thread_id = st.session_state.get("thread_id")
            logger.info(f"User '{username}' continuing thread '{thread_id}'.")
    
            result = st.session_state.state_machine.resume(st.session_state.current_state, thread_id=thread_id)
            
            if result is None:
                st.error("State machine returned None. The conversation may have ended.")
                logger.error("State machine returned None result.")
                return
                
            st.session_state.current_state = result
            st.success("Step completed successfully!")
            logger.info("State machine step completed successfully.")
            time.sleep(1)
            st.rerun()
        except Exception as e:
            if "Recursion limit" in str(e):
                st.error("The conversation appears to be stuck in a loop. Try resetting or adjusting the state.")
                logger.error(f"Recursion limit reached: {e}")
            else:
                handle_error("Error processing step", e)

def handle_retry():
    """Handles the Retry button click."""
    with st.spinner("Retrying last step..."):
        try:
            username = st.session_state.get("username")
            thread_id = st.session_state.get("thread_id")
            logger.info(f"User '{username}' retrying thread '{thread_id}'.")
    
            result = st.session_state.state_machine.resume(st.session_state.current_state, thread_id=thread_id)
            
            if result is None:
                st.error("State machine returned None. The conversation may have ended.")
                logger.error("State machine returned None result during retry.")
                return
                
            st.session_state.current_state = result
            st.success("Step retried successfully!")
            logger.info("State machine retry completed successfully.")
            time.sleep(1)
            st.rerun()
        except Exception as e:
            handle_error("Error retrying step", e)

def handle_reset():
    if st.button("Confirm Reset", key="confirm_reset"):
        st.session_state.clear()
        st.rerun()

def handle_error(message: str, error: Exception):
    error_msg = f"{message}: {str(error)}"
    logger.error(error_msg)
    logger.error(f"Full traceback:\n{traceback.format_exc()}")
    st.error(error_msg)
    st.error(f"Full traceback:\n{traceback.format_exc()}")

# Additional styling elements
def add_custom_styles():
    st.markdown("""
        <style>
        /* Existing styles */
        .stTextArea textarea {
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(0, 0, 0, 0.1);
            border-radius: 5px;
            font-size: 0.9rem;
            resize: vertical;
        }
        
        /* Button styling */
        .stButton button {
            background: linear-gradient(120deg, #000000, #1e3a8a);
            color: white;
            border: none;
            padding: 0.75rem 1.5rem;
            border-radius: 5px;
            font-weight: 600;
            transition: all 0.3s ease;
            width: 200px;
            text-align: center;
        }
        .stButton button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);
        }
        
        /* New chat message styles */
        .stChatMessage {
            background-color: rgba(240, 242, 246, 0.05) !important;
            border: 1px solid rgba(0, 0, 0, 0.1) !important;
            padding: 15px !important;
            margin-bottom: 5px !important;
            border-radius: 10px !important;
        }
        
        /* Adjust chat message container spacing */
        .stChatMessageContent {
            margin: 0 !important;
            padding: 0 !important;
        }
        
        /* Style the chat message avatar */
        .stChatMessageAvatar {
            margin-right: 10px !important;
        }
        
        /* Adjust chat flow container */
        .element-container.stChatFlow {
            margin-bottom: -15px !important;
            padding-bottom: 0 !important;
        }
        
        /* Existing styles continued... */
        .stMultiSelect > div {
            background: rgba(255, 255, 255, 0.05);
            border-radius: 5px;
        }
        
        .stMultiSelect [data-baseweb="tag"] {
            background: linear-gradient(120deg, #000000, #1e3a8a);
            color: white;
        }
        
        /* Success/Error message styling */
        .stSuccess, .stError {
            padding: 1rem;
            border-radius: 5px;
            margin: 1rem 0;
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(5px);
        }
        
        /* Hide default Streamlit elements */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        
        /* Spinner styling */
        .stSpinner > div {
            border-top-color: #1e3a8a !important;
        }
        </style>
    """, unsafe_allow_html=True)

# Logic for new buttons
def quit_and_save():
    # Logic to save the current state and quit
    st.session_state.saved_state = st.session_state.current_state
    st.success("State saved successfully!")
    st.stop()

def delete_all():
    # Logic to delete all state data
    st.session_state.clear()
    st.success("All state data deleted!")
    st.rerun()

def get_text_area_height(content, min_height=50, max_height=500, line_height=42):
    """Calculate the height of the text area based on content."""
    lines = content.count('\n') + 1
    height = min_height + lines * line_height
    height = max(min_height, min(height, max_height))
    return height

if __name__ == "__main__":
    render_start_page()
