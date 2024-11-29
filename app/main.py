import streamlit as st

# Set page config - THIS MUST BE THE FIRST ST COMMAND
st.set_page_config(
    page_title="Heart Disease Analysis",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

from page1 import page1
from page2 import page2

# Custom CSS for modern, clean UI
st.markdown("""
<style>
    /* Main content styling */
    .main {
        padding: 0 1rem;
    }
    
    /* Header styling */
    .header {
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        z-index: 999;
        background: linear-gradient(90deg, #FF6B6B 0%, #FF8E8E 100%);
        color: white;
        padding: 1rem 1rem 1rem 4rem;
        display: flex;
        align-items: center;
        gap: 15px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .header-title {
        font-size: 24px;
        font-weight: 600;
        margin: 0;
        color: white;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #f8f9fa;
        padding: 5rem 0.5rem 1rem 0.5rem;
        width: 250px !important;
    }
    
    [data-testid="stSidebar"] > div {
        width: 250px !important;
    }
    
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h1 {
        color: #FF6B6B;
        font-size: 20px;
        font-weight: 600;
        margin: 0 0 1rem 0;
        padding: 0 1rem;
    }
    
    /* Navigation buttons styling */
    .nav-button {
        width: 100%;
        padding: 0.5rem 1rem;
        margin: 0.25rem 0;
        border: none;
        border-radius: 5px;
        background-color: #f8f9fa;
        color: #444;
        text-align: left;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    
    .nav-button:hover {
        background-color: #FF8E8E;
        color: white;
    }
    
    .nav-button.active {
        background-color: #FF6B6B;
        color: white;
    }
    
    /* Content area styling */
    .content-section {
        background-color: white;
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        margin-top: 1rem;
    }
    
    .section-header {
        color: #333;
        font-size: 20px;
        font-weight: 600;
        margin-bottom: 1.5rem;
    }
    
    /* Theme toggle styling */
    .theme-toggle {
        position: absolute;
        top: 1rem;
        right: 2rem;
    }
    
    /* Adjust main content area to account for fixed header */
    [data-testid="stAppViewContainer"] > section:nth-child(2) {
        padding-top: 3rem;
    }
    
    /* Custom button styling */
    div[data-testid="stHorizontalBlock"] button {
        background-color: #FF6B6B;
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    
    div[data-testid="stHorizontalBlock"] button:hover {
        background-color: #FF8E8E;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Header with logo and title
st.markdown("""
    <div class="header">
        <span style="font-size: 32px;">❤️</span>
        <h1 class="header-title">Heart Disease Analysis Dashboard</h1>
    </div>
""", unsafe_allow_html=True)

# Theme toggle in header
with st.container():
    col1, col2 = st.columns([6, 1])
    with col2:
        theme = st.radio(
            "Theme",
            options=["Light", "Dark"],
            horizontal=True,
            label_visibility="collapsed",
        )

# Apply dark theme if selected
if theme == "Dark":
    st.markdown("""
        <style>
            [data-testid="stAppViewContainer"] {
                background-color: #1E1E1E;
                color: #FFFFFF;
            }
            .content-section {
                background-color: #2D2D2D;
            }
            .section-header {
                color: #FFFFFF;
            }
            [data-testid="stSidebar"] {
                background-color: #252526;
            }
            .nav-button {
                background-color: #2D2D2D;
                color: #FFFFFF;
            }
            .nav-button:hover {
                background-color: #FF8E8E;
            }
        </style>
    """, unsafe_allow_html=True)

# Sidebar navigation with custom buttons
with st.sidebar:
    st.title("Navigation")
    
    # Create buttons using markdown and JavaScript
    st.markdown("""
        <div style="display: flex; flex-direction: column; gap: 0.5rem;">
            <button class="nav-button" id="eda-btn" onclick="
                document.querySelectorAll('.nav-button').forEach(btn => btn.classList.remove('active'));
                this.classList.add('active');
                window.location.href='#eda'">
                📊 EDA
            </button>
            <button class="nav-button" id="cnn-btn" onclick="
                document.querySelectorAll('.nav-button').forEach(btn => btn.classList.remove('active'));
                this.classList.add('active');
                window.location.href='#cnn'">
                🧠 CNN Analysis
            </button>
        </div>
    """, unsafe_allow_html=True)
    
    # Hidden radio for state management
    selected_page = st.radio(
        "Navigation",
        ["EDA", "CNN Analysis"],
        label_visibility="collapsed",
        key="navigation",
    )

# Main content area
main_container = st.container()
with main_container:
    if selected_page == "EDA":
        with st.container():
            st.markdown('<div class="content-section" id="eda">', unsafe_allow_html=True)
            st.markdown('<h2 class="section-header">Exploratory Data Analysis</h2>', unsafe_allow_html=True)
            page1()
            st.markdown('</div>', unsafe_allow_html=True)
    
    elif selected_page == "CNN Analysis":
        with st.container():
            st.markdown('<div class="content-section" id="cnn">', unsafe_allow_html=True)
            st.markdown('<h2 class="section-header">CNN Model Analysis</h2>', unsafe_allow_html=True)
            page2()
            st.markdown('</div>', unsafe_allow_html=True)