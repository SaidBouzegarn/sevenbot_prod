import streamlit as st
import os
from dotenv import load_dotenv
from pages.start_page import render_start_page

# Load environment variables
load_dotenv()

# Streamlit app
st.set_page_config(page_title="7bot", layout="wide")

# Initialize session state
if 'page' not in st.session_state:
    st.session_state.page = "Home"

# Home page
if st.session_state.page == "Home":
    st.markdown(
        "<h1 style='text-align: center; font-size: 72px;'>7bot <span style='font-size: 36px;'>DAssist</span></h1>", 
        unsafe_allow_html=True
    )
    
    # Center the Start button
    col1, col2, col3 = st.columns([1,1,1])
    with col2:
        if st.button("Start", use_container_width=True):
            st.session_state.page = "Start"
            st.rerun()

# Start page
elif st.session_state.page == "Start":
    render_start_page()

# CSS to inject contained in a string
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 
