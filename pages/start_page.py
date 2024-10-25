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

def render_start_page():
    st.title("7bot DAssist Interaction")
    
    if "state_machine" not in st.session_state or st.session_state.state_machine is None:
        with st.spinner("Initializing state machine..."):
            base_dir = Path(__file__).resolve().parent.parent
            #prompts_dir = os.path.join(base_dir, "Data", "Prompts")
            prompts_dir = os.path.join("Data", "Prompts")
            logger.info(f"Attempting to initialize StateMachines with prompts_dir: {prompts_dir}")
            st.session_state.state_machine = StateMachines(str(prompts_dir).strip())
            logger.info("State machine initialized successfully!")
            st.success("State machine initialized successfully!")

    
    if st.session_state.state_machine is None:
        st.error("State machine is not initialized. Please check the logs for detailed error information.")
        return

    if "conversation_started" not in st.session_state:
        st.session_state.conversation_started = False
        
    if "current_state" not in st.session_state:
        st.session_state.current_state = None
    
    if not st.session_state.conversation_started:
        st.subheader("Initial State Configuration")
        
        uploaded_file = st.file_uploader("Upload a file", type=['txt', 'pdf'])
        if uploaded_file:
            try:
                with st.spinner("Processing file..."):
                    if uploaded_file.type == "application/pdf":
                        content = read_pdf(uploaded_file)
                    else:
                        content = uploaded_file.read().decode('utf-8')
                    st.session_state.file_content = content
                    st.success("File processed successfully!")
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
                st.session_state.file_content = ""
        
        news = st.text_area(
            "News content", 
            value=st.session_state.get('file_content', ''),
            height=200
        )
        
        digest = st.text_area(
            "Initial digest", 
            value="",
            height=200
        )
        
        ceo_message = st.text_area(
            "CEO initial message",
            value="",
            height=100
        )
        
        if st.button("Start Conversation"):
            with st.spinner("Initializing conversation..."):
                try:
                    initial_state = {
                        "news_insights": [news],
                        "digest": [digest],
                        "ceo_messages": [HumanMessage(content=ceo_message)] if ceo_message else [],
                        "ceo_mode": ["research_information"]
                    }
                    
                    result = st.session_state.state_machine.start(initial_state)
                    if result is None:
                        st.error("State machine returned None. Please check the StateMachines implementation.")
                        return
                    
                    st.session_state.current_state = result
                    st.session_state.conversation_started = True
                    st.success("Conversation started successfully!")
                    time.sleep(1)
                    st.rerun()
                except Exception as e:
                    st.error(f"Error starting conversation: {str(e)}")
    
    else:
        st.subheader("Current Conversation State")
        
        display_conversation_flow(st.session_state.current_state)
        
        tabs = st.tabs(["Level 1-2", "Level 1-3", "Level 2-3", "Assistant", "CEO Messages"])
        
        conversations = {
            "level1_2_conversation": "Level 1-2 Conversation",
            "level1_3_conversation": "Level 1-3 Conversation",
            "level2_3_conversation": "Level 2-3 Conversation",
            "ceo_assistant_conversation": "Assistant Conversation",
            "ceo_messages": "CEO Messages"
        }
        
        edited_state = st.session_state.current_state.copy()
        
        for i, (key, title) in enumerate(conversations.items()):
            with tabs[i]:
                if key in st.session_state.current_state:
                    messages = st.session_state.current_state[key]
                    edited_messages = []
                    
                    for j, msg in enumerate(messages):
                        content = msg.content if hasattr(msg, 'content') else str(msg)
                        message_type = type(msg).__name__ if hasattr(msg, '__class__') else "Message"
                        edited_content = st.text_area(
                            f"{message_type} {j+1}",
                            content,
                            key=f"{key}_{j}"
                        )
                        edited_messages.append(preserve_message_type(msg))
                    
                    edited_state[key] = edited_messages
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("Continue"):
                with st.spinner("Processing next step..."):
                    try:
                        result = st.session_state.state_machine.resume(edited_state)
                        st.session_state.current_state = result
                        st.success("Step completed successfully!")
                        time.sleep(1)
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error processing step: {str(e)}")
        
        with col2:
            if st.button("Retry"):
                with st.spinner("Retrying last step..."):
                    try:
                        result = st.session_state.state_machine.resume(st.session_state.current_state)
                        st.session_state.current_state = result
                        st.success("Step retried successfully!")
                        time.sleep(1)
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error retrying step: {str(e)}")
        
        with col3:
            if st.button("Reset Conversation"):
                st.session_state.clear()
                st.rerun()

if __name__ == "__main__":
    render_start_page()
