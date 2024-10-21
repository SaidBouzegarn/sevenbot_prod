import streamlit as st
from agents.agents_graph import create_agents_graph, Level3State, HumanMessage
from langchain.schema import AIMessage

def render_start_page():
    st.title("7bot DAssist Interaction")
    
    if "initial_state" not in st.session_state:
        st.session_state.initial_state = None
        st.session_state.graph = None
        st.session_state.interaction_complete = False

    # Input fields for initial state (only shown if initial_state is not set)
    if st.session_state.initial_state is None:
        st.subheader("Initial State Configuration")
        company_knowledge = st.text_area("Company Knowledge", "Our company is a tech startup focusing on AI solutions.")
        news_insights = st.text_area("News Insights", "AI market is growing rapidly\nNew regulations on data privacy")
        ceo_message = st.text_input("CEO Message", "What's our strategy for the next quarter?")
        
        if st.button("Start Interaction"):
            # Create initial state
            st.session_state.initial_state = Level3State(
                level2_3_conversation=[],
                level1_3_conversation=[],
                company_knowledge=company_knowledge,
                news_insights=news_insights.split('\n'),
                digest=[],
                action=[],
                ceo_messages=[HumanMessage(content=ceo_message, type="human")],
                ceo_assistant_conversation=[],
                ceo_mode="research_information",
                level2_agents=["Director1", "Director2"],
                level1_agents=["Agent1", "Agent2", "Agent3", "Agent4"],
            )
            
            # Create the graph
            st.session_state.graph = create_agents_graph()
            
            # Trigger the first interaction
            st.session_state.trigger_interaction = True
            st.rerun()

    if st.session_state.initial_state is not None:
        # Display chat messages and handle interactions
        for message in st.session_state.initial_state.ceo_messages:
            with st.chat_message(message.type):
                st.markdown(message.content)

        # Check if we need to trigger the first interaction
        if getattr(st.session_state, 'trigger_interaction', False):
            st.session_state.trigger_interaction = False
            run_interaction(st.session_state.initial_state)

        # Chat input for user
        user_input = st.chat_input("Your message:")
        if user_input:
            # Update the initial state with the new user message
            st.session_state.initial_state.ceo_messages.append(HumanMessage(content=user_input, type="human"))
            run_interaction(st.session_state.initial_state)

        # Show "See Elements" button only when interaction is complete
        if st.session_state.interaction_complete:
            if st.button("See Elements"):
                show_state_elements()

def run_interaction(state):
    # Run the graph
    config = {
        "configurable": {"thread_id": "test_thread3"},
        "recursion_limit": 150  # Set a higher recursion limit
    }
    
    # Stream the results
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        for s in st.session_state.graph.stream(state, config, stream_mode="updates"):
            if isinstance(s, AIMessage):
                full_response += s.content + "\n\n"
                message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)
    
    # Update the initial state for the next iteration
    state.ceo_messages.append(AIMessage(content=full_response))
    state.ceo_assistant_conversation.append(AIMessage(content=full_response))
    
    # Set interaction_complete to True
    st.session_state.interaction_complete = True
    
    st.rerun()

def show_state_elements():
    st.sidebar.subheader("Edit State Elements")
    
    # Create a selectbox to choose which element to edit
    element_to_edit = st.sidebar.selectbox(
        "Choose element to edit",
        ["company_knowledge", "news_insights", "ceo_mode", "level2_agents", "level1_agents"]
    )
    
    # Display the chosen element for editing
    if element_to_edit == "company_knowledge":
        new_value = st.sidebar.text_area("Company Knowledge", st.session_state.initial_state.company_knowledge)
        if st.sidebar.button("Save Company Knowledge"):
            st.session_state.initial_state.company_knowledge = new_value
    elif element_to_edit == "news_insights":
        new_value = st.sidebar.text_area("News Insights", "\n".join(st.session_state.initial_state.news_insights))
        if st.sidebar.button("Save News Insights"):
            st.session_state.initial_state.news_insights = new_value.split('\n')
    elif element_to_edit == "ceo_mode":
        new_value = st.sidebar.text_input("CEO Mode", st.session_state.initial_state.ceo_mode)
        if st.sidebar.button("Save CEO Mode"):
            st.session_state.initial_state.ceo_mode = new_value
    elif element_to_edit == "level2_agents":
        new_value = st.sidebar.text_input("Level 2 Agents", ", ".join(st.session_state.initial_state.level2_agents))
        if st.sidebar.button("Save Level 2 Agents"):
            st.session_state.initial_state.level2_agents = [agent.strip() for agent in new_value.split(",")]
    elif element_to_edit == "level1_agents":
        new_value = st.sidebar.text_input("Level 1 Agents", ", ".join(st.session_state.initial_state.level1_agents))
        if st.sidebar.button("Save Level 1 Agents"):
            st.session_state.initial_state.level1_agents = [agent.strip() for agent in new_value.split(",")]
    
    if st.sidebar.button("Run with Updated State"):
        st.session_state.trigger_interaction = True
        st.session_state.interaction_complete = False
        st.rerun()

# ... rest of the code ...
