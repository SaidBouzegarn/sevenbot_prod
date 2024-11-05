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

    # Render logo at the very top
    render_logo()

    # Initialize state machine if needed
    if "state_machine" not in st.session_state or st.session_state.state_machine is None:
        initialize_state_machine()

    if st.session_state.state_machine is None:
        st.error("State machine is not initialized. Please check the logs.")
        return

    # Render control panel and conversation state based on conversation status
    if not st.session_state.get("conversation_started", False):
        render_upload_section()
    else:
        # Render control panel (buttons) and conversation state
        render_conversation_state()

def render_upload_section():
    st.markdown('<div class="upload-container">', unsafe_allow_html=True)
    
    # Add checkbox for interrupt_before
    interrupt_before = st.checkbox("Interrupt before state transitions", 
                                 value=True,
                                 help="If checked, the conversation will pause before major state transitions")
    
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
                    initialize_conversation(content, interrupt_before)
        except Exception as e:
            handle_error("Error processing file", e)
    
    st.markdown('</div>', unsafe_allow_html=True)

def render_conversation_state():
    # Wrap the buttons in a control panel at the top
    st.markdown('<div class="control-panel">', unsafe_allow_html=True)

    # Place buttons horizontally and centered
    col1, col2, col3 = st.columns([1, 1, 1])
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

def render_conversation_messages(key, only_content=False):
    if key in st.session_state.current_state:
        messages = st.session_state.current_state[key]
        for i, msg in enumerate(messages):
            # Extract content based on message type
            if isinstance(msg, (HumanMessage, AIMessage, SystemMessage)):
                content = msg.content
            elif isinstance(msg, dict):
                content = msg.get('content', str(msg))
            elif hasattr(msg, 'content'):
                content = msg.content
            else:
                content = str(msg)

            # Display the message content without any extra styling or borders
            st.markdown(f"{content}")

            # Optionally, add a horizontal line between messages
            st.markdown("---")

# Add caching for the state machine

#@st.cache_resource

def get_shared_state_machine(interrupt_before: bool = True):
    """Create a single StateMachine instance shared across all sessions"""
    logger.info(f"Creating new shared StateMachine instance with interrupt_before={interrupt_before}")
    prompts_dir = os.path.join("Data", "Prompts")
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

def initialize_conversation(content, interrupt_before: bool):
    with st.spinner("Initializing conversation..."):
        try:
            # Store interrupt_before in session state
            st.session_state.interrupt_before = interrupt_before
            
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
            
            # Clean up the file content from session state
            if 'file_content' in st.session_state:
                del st.session_state.file_content
            
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
            if "Recursion limit" in str(e):
                st.error("The conversation appears to be stuck in a loop. Try resetting or adjusting the state.")
                logger.error(f"Recursion limit reached: {e}")
            else:
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
