import streamlit as st
import os
from dotenv import load_dotenv
from pages.Simulation import render_start_page
import PyPDF2
from pathlib import Path
import hmac
from pages.CrawlandScrape import render_crawl_page

# Set page config first, before any other Streamlit commands
st.set_page_config(
    page_title="SevenBlue", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Load environment variables
load_dotenv()

# Set up base path
BASE_DIR = Path(__file__).resolve().parent

# Initialize session state
if 'state_machine' not in st.session_state:
    st.session_state.state_machine = None
    st.session_state.base_dir = BASE_DIR

if 'username_id' not in st.session_state:
    st.session_state.username_id = None

if 'password_correct' not in st.session_state:
    st.session_state.password_correct = None

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
            background: linear-gradient(120deg, 
                rgba(0, 0, 0, 1),  /* Pure black */
                rgba(30, 58, 138, 1)  /* Pure deep blue */
            );
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
            text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.1);  /* Subtle shadow for depth */
        }
        
        /* Form container styling */
        [data-testid="stForm"] {
            max-width: 400px;
            padding: 2rem;
            background: rgba(255, 255, 255, 0.15);  /* More visible white background */
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);  /* Stronger shadow */
            border: 1px solid rgba(0, 0, 0, 0.1);  /* Subtle border */
        }
        
        /* Submit button styling */
        [data-testid="stForm"] button {
            background: linear-gradient(120deg, 
                rgba(0, 0, 0, 1),  /* Pure black */
                rgba(30, 58, 138, 1)  /* Pure deep blue */
            );
            width: 100%;
            color: white !important;  /* Ensure text is white */
            font-weight: 600;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);  /* Button shadow */
        }

        /* Input field styling */
        [data-testid="stForm"] input {
            border: 1px solid rgba(0, 0, 0, 0.2);
            background: rgba(255, 255, 255, 0.9);  /* Nearly white background */
        }
        </style>
    """, unsafe_allow_html=True)

def check_password():
        
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
        if st.session_state["username"] in st.secrets["passwords"] and hmac.compare_digest(
            st.session_state["password"],
            st.secrets.passwords[st.session_state["username"]],
        ):
            st.session_state["password_correct"] = True
            st.session_state["username_id"] = st.session_state["username"]  # Ensure username is in session
            del st.session_state["password"]
            # After successful login, the user will navigate to the start page
        else:
            st.session_state["password_correct"] = False

    if st.session_state.get("password_correct", False):
        return True

    # Add custom CSS before showing the login form
    add_custom_css()
    
    login_form()
    # Only show error if a login attempt was made and failed
    if st.session_state.password_correct is False:
        st.error("User not known or password incorrect")
    return False

# Main app logic
if check_password():
    # Add custom CSS for the landing page
    st.markdown("""
        <style>
        .landing-container {
            text-align: center;
            padding: 4rem 2rem;
            max-width: 800px;
            margin: 0 auto;
        }
        
        .app-title {
            font-family: 'Arial Black', sans-serif;
            font-size: 4.5rem;
            font-weight: 900;
            background: linear-gradient(120deg, 
                rgba(0, 0, 0, 1),
                rgba(30, 58, 138, 1)
            );
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 2rem;
        }
        
        .app-description {
            font-size: 1.2rem;
            color: #4a5568;
            margin: 0 auto 3rem auto;
            line-height: 1.6;
        }
        
        .button-container {
            display: flex;
            justify-content: center;
            gap: 2rem;
            margin-top: 3rem;
        }
        
        .nav-button {
            background: linear-gradient(120deg, 
                rgba(0, 0, 0, 1),
                rgba(30, 58, 138, 1)
            );
            color: white !important;
            padding: 1rem 2rem !important;
            border-radius: 8px !important;
            text-decoration: none;
            font-weight: 600 !important;
            transition: all 0.3s ease;
            border: none;
            cursor: pointer;
            width: 200px;
        }
        
        .nav-button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Landing page content with st.page_link
    st.markdown('<div class="landing-container">', unsafe_allow_html=True)
    st.markdown('<h1 class="app-title">Seven Blue Bot</h1>', unsafe_allow_html=True)
    st.markdown('''
        <p class="app-description">
            Welcome to Seven Blue Bot - your AI-powered news analysis assistant. Our advanced system 
            scrapes news articles from any website and simulates executive meetings to extract valuable 
            insights. By leveraging cutting-edge AI technology, we help you stay ahead of market trends 
            and make informed decisions based on comprehensive news analysis.
        </p>
    ''', unsafe_allow_html=True)
    
    # Replace HTML buttons with Streamlit columns and buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Start", key="start_btn", use_container_width=True):
            st.session_state.page = "start"
            st.switch_page("pages/start_page.py")
    
    with col2:
        if st.button("Crawl", key="crawl_btn", use_container_width=True):
            st.session_state.page = "crawl"
            st.switch_page("pages/crawl_page.py")

    st.markdown('</div>', unsafe_allow_html=True)

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

custom_sidebar_style = """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] {
        min-width: 200px;
        max-width: 200px;
        background-color: #f0f2f6;
    }
    [data-testid="stSidebar"][aria-expanded="false"] {
        min-width: 0px;
    }
    </style>
"""
st.markdown(custom_sidebar_style, unsafe_allow_html=True)


