import streamlit as st
import os
from dotenv import load_dotenv
from pages.start_page import render_start_page
import PyPDF2
from pathlib import Path
import hmac

# Load environment variables
load_dotenv()

# Set up base path
BASE_DIR = Path(__file__).resolve().parent

# Add this near the top of your file, after imports
DEBUG_MODE = True  # Set to False to enable authentication

# Add CSS for the stylish landing page
def add_custom_css():
    st.markdown("""
        <style>
        .main {
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
        }
        
        .title-container {
            text-align: center;
            margin-bottom: 2rem;
        }
        
        .main-title {
            font-family: 'Arial Black', sans-serif;
            font-size: 4.5rem;
            font-weight: 900;
            background: linear-gradient(120deg, #000000, #1e3a8a);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
        }
        
        /* Form container styling */
        [data-testid="stForm"] {
            max-width: 400px;
            padding: 2rem;
            background: rgba(255, 255, 255, 0.05);
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        
        /* Submit button styling */
        [data-testid="stForm"] button {
            background: linear-gradient(120deg, #000000, #1e3a8a);
            width: 100%;
        }
        </style>
    """, unsafe_allow_html=True)

def check_password():
    # Skip authentication if in debug mode
    if DEBUG_MODE:
        st.session_state["password_correct"] = True
        return True
        
    """Returns `True` if the user had a correct password."""

    def login_form():
        """Form with widgets to collect user information"""
        # Add the title before the form
        st.markdown('<div class="title-container">'
                   '<h1 class="main-title">SevenBlue</h1>'
                   '</div>', unsafe_allow_html=True)
        
        with st.form("Credentials"):
            st.text_input("Username", key="username")
            st.text_input("Password", type="password", key="password")
            st.form_submit_button("Log in", on_click=password_entered)

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if st.session_state["username"] in st.secrets[
            "passwords"
        ] and hmac.compare_digest(
            st.session_state["password"],
            st.secrets.passwords[st.session_state["username"]],
        ):
            st.session_state["password_correct"] = True
            st.session_state.page = "Start"  # Automatically route to Start page
            del st.session_state["password"]
            del st.session_state["username"]
        else:
            st.session_state["password_correct"] = False

    if st.session_state.get("password_correct", False):
        return True

    # Add custom CSS before showing the login form
    add_custom_css()
    
    login_form()
    if "password_correct" in st.session_state:
        st.error("😕 User not known or password incorrect")
    return False

# Streamlit app
st.set_page_config(
    page_title="SevenBlue", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Initialize session state
if 'page' not in st.session_state:
    st.session_state.page = "Home"
    st.session_state.state_machine = None
    st.session_state.base_dir = BASE_DIR

# Modify the authentication check
if DEBUG_MODE or check_password():
    # Your main app code here
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
else:
    st.stop()
