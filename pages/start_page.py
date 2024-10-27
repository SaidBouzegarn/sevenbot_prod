import streamlit as st
from agents.agents_graph_V2 import StateMachines
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import PyPDF2
import json
from typing import Union, List
import time
from pathlib import Path
import traceback
import logging
import os
from datetime import datetime

# Set up logging
def setup_logging():
    base_dir = Path(__file__).resolve().parent.parent
    logs_dir = base_dir / "logs"
    logs_dir.mkdir(exist_ok=True)
    
    log_file = logs_dir / f"app_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
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

def preserve_message_type(message: Union[str, dict, HumanMessage, AIMessage, SystemMessage]) -> Union[HumanMessage, AIMessage, SystemMessage]:
    if isinstance(message, (HumanMessage, AIMessage, SystemMessage)):
        message_type = type(message)
        return message_type(content=message.content)
    elif isinstance(message, dict):
        if message.get('type') == 'human':
            return HumanMessage(content=message.get('content', ''))
        elif message.get('type') == 'ai':
            return AIMessage(content=message.get('content', ''))
        elif message.get('type') == 'system':
            return SystemMessage(content=message.get('content', ''))
    return HumanMessage(content=str(message))

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
        /* Compact control panel */
        .control-panel {
            display: flex;
            flex-direction: row;
            justify-content: space-between;
            align-items: center;
            padding: 10px;
            margin-bottom: 10px;
            background: rgba(255, 255, 255, 0.05);
            border-radius: 10px;
        }

        /* Action buttons container - compact */
        .action-buttons {
            display: flex;
            flex-direction: column;
            gap: 5px;
            z-index: 1000;
            padding: 5px;
        }
        
        /* State selector container - compact */
        .state-selector {
            flex-grow: 1;
            margin-right: 10px;
            padding: 5px;
        }
        
        /* Button styling - compact */
        .stButton button {
            width: 100px;  /* Reduced width */
            padding: 0.25rem 0.5rem;  /* Reduced padding */
        }

        /* New button styles */
        .quit-button {
            background-color: grey;
            color: white;
        }
        
        .delete-button {
            background-color: red;
            color: white;
        }

        /* Column layout */
        .state-columns {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 20px;
            padding: 20px;
            height: calc(100vh - 200px);
            margin-top: 20px;
        }
        
        .state-column {
            background: rgba(255, 255, 255, 0.05);
            border-radius: 10px;
            padding: 15px;
            height: 100%;
            border: 1px solid rgba(0, 0, 0, 0.1);
            overflow-y: auto;
            backdrop-filter: blur(5px);
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        
        /* Multiselect styling */
        .stMultiSelect {
            background: rgba(255, 255, 255, 0.1);
            border-radius: 5px;
            padding: 10px;
        }
        
        /* Column headers */
        .column-header {
            font-size: 1.2rem;
            font-weight: 600;
            margin-bottom: 15px;
            padding-bottom: 10px;
            border-bottom: 2px solid rgba(0, 0, 0, 0.1);
            text-align: center;
        }
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
    # Add all styles
    add_start_page_css()
    add_custom_styles()
    
    # Add logo
    render_logo()
    
    # Initialize state machine if needed
    if "state_machine" not in st.session_state or st.session_state.state_machine is None:
        initialize_state_machine()
    
    if st.session_state.state_machine is None:
        st.error("State machine is not initialized. Please check the logs.")
        return

    # Render appropriate view based on conversation state
    if not st.session_state.get("conversation_started", False):
        render_upload_section()
    else:
        render_conversation_state()

def render_upload_section():
    st.markdown('<div class="upload-container">', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Upload your documents", type=['txt', 'pdf'])
    if uploaded_file:
        try:
            with st.spinner("Processing file..."):
                if uploaded_file.type == "application/pdf":
                    content = read_pdf(uploaded_file)
                else:
                    content = uploaded_file.read().decode('utf-8')
                st.session_state.file_content = content
                st.success("File processed successfully!")
                
                # Show the Start button only after successful upload
                if st.button("Start Conversation", use_container_width=True):
                    initialize_conversation(content)
        except Exception as e:
            handle_error("Error processing file", e)
    
    st.markdown('</div>', unsafe_allow_html=True)

def render_conversation_state():
    # Wrap controls in a single container
    st.markdown('<div class="control-panel">', unsafe_allow_html=True)
    
    # State selector (now on the left)
    st.markdown('<div class="state-selector">', unsafe_allow_html=True)
    available_states = list(st.session_state.current_state.keys())
    if 'selected_states' not in st.session_state:
        st.session_state.selected_states = available_states[:4]
    
    selected_states = st.multiselect(
        "Select state elements to display (max 4)",
        available_states,
        default=st.session_state.selected_states,
        max_selections=4
    )
    
    if len(selected_states) <= 4:
        st.session_state.selected_states = selected_states
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Action buttons (now on the right)
    st.markdown('<div class="action-buttons">', unsafe_allow_html=True)
    if st.button("Continue", key="btn_continue"):
        handle_continue()
    if st.button("Retry", key="btn_retry"):
        handle_retry()
    if st.button("Reset", key="btn_reset"):
        handle_reset()
    if st.button("Quit and Save", key="btn_quit"):  # Removed css_class
        quit_and_save()
    if st.button("Delete All", key="btn_delete"):  # Removed css_class
        delete_all()
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Render columns for selected states
    if selected_states:
        cols = st.columns(4)
        for i, col in enumerate(cols):
            with col:
                if i < len(selected_states):
                    state_key = selected_states[i]
                    st.markdown(f'<div class="column-header">{state_key}</div>', 
                              unsafe_allow_html=True)
                    render_conversation_messages(state_key)

def render_conversation_messages(key):
    if key in st.session_state.current_state:
        messages = st.session_state.current_state[key]
        for i, msg in enumerate(messages):
            content = msg.content if hasattr(msg, 'content') else str(msg)
            edited = st.text_area(
                f"{type(msg).__name__ if hasattr(msg, '__class__') else 'Message'} {i+1}",
                content,
                key=f"{key}_{i}",
                height=150
            )
            # Store edited content back to state
            if edited != content:
                st.session_state.current_state[key][i] = preserve_message_type(edited)

# Helper functions for state management and error handling
def initialize_state_machine():
    with st.spinner("Initializing state machine..."):
        try:
            base_dir = Path(__file__).resolve().parent.parent
            prompts_dir = os.path.join("Data", "Prompts")
            logger.info(f"Attempting to initialize StateMachines with prompts_dir: {prompts_dir}")
            st.session_state.state_machine = StateMachines(str(prompts_dir).strip())
            logger.info("State machine initialized successfully!")
            st.success("State machine initialized successfully!")
        except Exception as e:
            handle_error("Failed to initialize state machine", e)

def initialize_conversation(content):
    with st.spinner("Initializing conversation..."):
        try:
            initial_state = {
                "news_insights": [content],
                "digest": [""],
                "ceo_messages": [],
                "ceo_mode": ["research_information"]
            }
            
            result = st.session_state.state_machine.start(initial_state)
            if result is None:
                st.error("State machine returned None. Please check the implementation.")
                return
            
            st.session_state.current_state = result
            st.session_state.conversation_started = True
            st.success("Conversation started successfully!")
            time.sleep(1)
            st.rerun()
        except Exception as e:
            handle_error("Error starting conversation", e)

def handle_continue():
    with st.spinner("Processing next step..."):
        try:
            logger.info("Attempting to resume state machine with current state")
            result = st.session_state.state_machine.resume(st.session_state.current_state)
            
            if result is None:
                st.error("State machine returned None. The conversation may have ended.")
                logger.error("State machine returned None result")
                return
                
            st.session_state.current_state = result
            st.success("Step completed successfully!")
            logger.info("State machine step completed successfully")
            time.sleep(1)
            st.rerun()
        except Exception as e:
            handle_error("Error processing step", e)

def handle_retry():
    with st.spinner("Retrying last step..."):
        try:
            logger.info("Attempting to retry state machine with current state")
            result = st.session_state.state_machine.resume(st.session_state.current_state)
            
            if result is None:
                st.error("State machine returned None. The conversation may have ended.")
                logger.error("State machine returned None result during retry")
                return
                
            st.session_state.current_state = result
            st.success("Step retried successfully!")
            logger.info("State machine retry completed successfully")
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
        /* Message box styling */
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
        
        /* Multiselect styling */
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

if __name__ == "__main__":
    render_start_page()
