import streamlit as st
import os
from dotenv import load_dotenv
from pages.start_page import render_start_page
import PyPDF2

# Load environment variables
load_dotenv()

# Streamlit app
st.set_page_config(page_title="7bot", layout="wide")

# Initialize session state
if 'page' not in st.session_state:
    st.session_state.page = "Home"
    st.session_state.state_machine = None

# Home page
if st.session_state.page == "Home":
    st.markdown(
        "<h1 style='text-align: center; font-size: 72px;'>7bot <span style='font-size: 36px;'>DAssist</span></h1>", 
        unsafe_allow_html=True
    )
    
    # File uploader in the center
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        uploaded_file = st.file_uploader("Upload PDF", type="pdf")
        if uploaded_file is not None:
            # Read PDF content
            pdf_reader = PyPDF2.PdfReader(uploaded_file)
            text_content = ""
            for page in pdf_reader.pages:
                text_content += page.extract_text()
            
            # Store PDF content in session state
            st.session_state.pdf_content = text_content
            
            # Show Start button after file is uploaded
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
