import streamlit as st
from agents.agents_graph_V2 import StateMachines
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import PyPDF2
import json
from typing import Union, List
import time

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
    """Preserve message type when converting edited content."""
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
    """Display a visual representation of the conversation flow."""
    st.markdown("### Conversation Flow")
    
    # Create columns for each agent
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("**CEO**")
        if "ceo_messages" in current_state:
            for msg in current_state["ceo_messages"]:
                st.info(msg.content[:100] + "..." if len(msg.content) > 100 else msg.content)
    
    with col2:
        st.markdown("**Assistant**")
        if "ceo_assistant_conversation" in current_state:
            for msg in current_state["ceo_assistant_conversation"]:
                if isinstance(msg, AIMessage):
                    st.success(msg.content[:100] + "..." if len(msg.content) > 100 else msg.content)
                else:
                    st.info(msg.content[:100] + "..." if len(msg.content) > 100 else msg.content)
    
    with col3:
        st.markdown("**Level 1-2**")
        if "level1_2_conversation" in current_state:
            for msg in current_state["level1_2_conversation"]:
                st.info(msg.content[:100] + "..." if len(msg.content) > 100 else msg.content)
    
    with col4:
        st.markdown("**Level 1-3**")
        if "level1_3_conversation" in current_state:
            for msg in current_state["level1_3_conversation"]:
                st.info(msg.content[:100] + "..." if len(msg.content) > 100 else msg.content)

def render_start_page():
    st.title("7bot DAssist Interaction")
    
    # Initialize session state variables
    if "state_machine" not in st.session_state:
        with st.spinner("Initializing state machine..."):
            try:
                st.session_state.state_machine = StateMachines("Data/Prompts")
                st.success("State machine initialized successfully!")
            except Exception as e:
                st.error(f"Failed to initialize state machine: {str(e)}")
                return  # Exit if initialization fails
    
    if "conversation_started" not in st.session_state:
        st.session_state.conversation_started = False
        
    if "current_state" not in st.session_state:
        st.session_state.current_state = None
    
    # Initial state configuration
    if not st.session_state.conversation_started:
        st.subheader("Initial State Configuration")
        
        # File upload with error handling
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
        
        # Display and allow editing of news from file
        news = st.text_area(
            "News content", 
            value=st.session_state.get('file_content', ''),
            height=200
        )
        
        # Display and allow editing of digest
        digest = st.text_area(
            "Initial digest", 
            value="",
            height=200
        )
        
        # CEO initial message
        ceo_message = st.text_area(
            "CEO initial message",
            value="",
            height=100
        )
        
        # Start button with progress indication
        if st.button("Start Conversation"):
            with st.spinner("Initializing conversation..."):
                try:
                    # Create initial state with user inputs
                    initial_state = {
                        "news_insights": [news],
                        "digest": [digest],
                        "ceo_messages": [HumanMessage(content=ceo_message)] if ceo_message else [],
                        "ceo_mode": ["research_information"]
                    }
                    
                    # Start the state machine with the initial state
                    result = st.session_state.state_machine.start(initial_state)  # Pass the initial_state here
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
        # Display current conversation state
        st.subheader("Current Conversation State")
        
        # Display visual conversation flow
        display_conversation_flow(st.session_state.current_state)
        
        # Create tabs for detailed message editing
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
        
        col1, col2 = st.columns(2)
        
        # Continue button with progress indication
        with col1:
            if st.button("Continue"):
                with st.spinner("Processing next step..."):
                    try:
                        # Resume the state machine with edited state
                        result = st.session_state.state_machine.resume(edited_state)
                        st.session_state.current_state = result
                        st.success("Step completed successfully!")
                        time.sleep(1)  # Give user time to see success message
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error processing step: {str(e)}")
        
        # Reset button
        if st.button("Reset Conversation"):
            st.session_state.clear()
            st.rerun()

if __name__ == "__main__":
    render_start_page()
