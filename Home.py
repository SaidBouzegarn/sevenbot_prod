import streamlit as st
import os
from dotenv import load_dotenv
from pages.start_page import render_start_page
import PyPDF2
from pathlib import Path

# Load environment variables
load_dotenv()

# Set up base path
BASE_DIR = Path(__file__).resolve().parent

# Streamlit app
st.set_page_config(page_title="7bot", layout="wide")

# Initialize session state
if 'page' not in st.session_state:
    st.session_state.page = "Home"
    st.session_state.state_machine = None
    st.session_state.base_dir = BASE_DIR

# Home page
if st.session_state.page == "Home":
    st.markdown(
        "<h1 style='text-align: center; font-size: 100px; font-family: Arial, sans-serif;'>"
        "<span style='background: linear-gradient(45deg, #FE6B8B 30%, #FF8E53 90%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>"
        "7bot</span></h1>", 
        unsafe_allow_html=True
    )
    
    # Center the Start button
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        if st.button("Start", use_container_width=True):
            st.session_state.page = "Start"
            st.rerun()

# Start page
elif st.session_state.page == "Start":
    render_start_page()

# Hide Streamlit elements
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 
