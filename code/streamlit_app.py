"""
Streamlit Web UI for RAG Research Assistant
Clean Minimalist Theme - Educational Product Design
"""

import streamlit as st
from pathlib import Path
import sys, traceback
import os
import warnings
import uuid
from PyPDF2 import PdfReader
from docx import Document


# Suppressing warnings
os.environ["GLOG_minloglevel"] = "2"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# Just to ensure code is in the path just once
project_root = (Path(__file__).resolve().parent.parent)
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))



# Page configuration
st.set_page_config(
    page_title="Aloysia",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Clean Minimalist CSS - Inspired by paperreview.ai
st.markdown("""
<style>
    /* Import fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
            
    /* Root variables - Light Grey + Dark Green Theme */
    :root {
        --primary-color: #2d5a3d;
        --primary-hover: #1e4a2d;
        --primary-light: #3d7a4d;
        --text-primary: #1a2e1a;
        --text-secondary: #4a5f4a;
        --text-light: #f0f4f0;
        --bg-primary: #d0d8d0;
        --bg-secondary: #c0c8c0;
        --bg-dark: #2d5a3d;
        --border-color: #a5b0a5;
        --success-color: #2d5a3d;
        --accent-light: #b8c8b8;
    }


            
    /* Global font */
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Main container */
    .main {
        background-color: var(--bg-secondary);
    }
    
    /* Header styling - Clean and minimal */
    .header-container {
        background: var(--bg-primary);
        padding: 1.5rem 2rem;
        border-radius: 0.75rem;
        margin-bottom: 1.5rem;
        border: 1px solid var(--border-color);
        display: flex;
        align-items: center;
        justify-content: space-between;
    }

    .header-left {
        display: flex;
        align-items: center;
        gap: 1rem;
    }

    .header-logo {
        width: 40px;
        height: 40px;
        background: var(--primary-color);
        border-radius: 0.5rem;
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-weight: 700;
        font-size: 1.25rem;
    }

    .header-title {
        font-size: 1.5rem;
        font-weight: 600;
        color: var(--text-primary);
        margin: 0;
    }

    .header-subtitle {
        font-size: 0.875rem;
        color: var(--text-secondary);
        margin: 0;
    }

    .workspace-badge {
        background: var(--accent-light);
        color: var(--primary-color);
        padding: 0.375rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 500;
    }

    /* Stats cards - Minimal */
    .stats-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 1rem;
        margin-bottom: 1.5rem;
    }
            
    .stat-card {
        background: var(--bg-primary);
        padding: 1.25rem;
        border-radius: 0.75rem;
        border: 1px solid var(--border-color);
        text-align: center;
    }
            
    .stat-number {
        font-size: 1.75rem;
        font-weight: 700;
        color: var(--text-primary);
        margin-bottom: 0.25rem;
    }
    
    .stat-label {
        font-size: 0.8rem;
        color: var(--text-secondary);
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* Chat messages - Clean bubbles */
    .user-message {
        background: var(--primary-color);
        color: white;
        padding: 1rem 1.25rem;
        border-radius: 1rem;
        border-bottom-right-radius: 0.25rem;
        margin: 0.75rem 0;
        margin-left: 20%;
        font-size: 0.9375rem;
        line-height: 1.5;
    }
    
    .assistant-message {
        background: var(--bg-primary);
        color: var(--text-primary);
        padding: 1rem 1.25rem;
        border-radius: 1rem;
        border-bottom-left-radius: 0.25rem;
        margin: 0.75rem 0;
        margin-right: 20%;
        border: 1px solid var(--border-color);
        font-size: 0.9375rem;
        line-height: 1.6;
    }
    
    /* Buttons - Clean primary style */
    .stButton>button {
        background: var(--primary-color);
        color: white;
        border: none;
        border-radius: 0.5rem;
        padding: 0.5rem 1.25rem;
        font-weight: 500;
        font-size: 0.875rem;
        transition: all 0.2s;
    }
    
    .stButton>button:hover {
        background: var(--primary-hover);
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.25);
    }
    
    /* Sidebar - Clean */
    [data-testid="stSidebar"] {
        background-color: var(--bg-primary);
        border-right: 1px solid var(--border-color);
    }
    
    [data-testid="stSidebar"] .stMarkdown {
        color: var(--text-primary);
    }

    /* Sidebar - ALL text elements dark forest green */
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] .stCheckbox label,
    [data-testid="stSidebar"] .stRadio label,
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stTextInput label,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] .stCaption,
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
        color: var(--text-primary) !important;
    }

    /* Sidebar checkbox and radio text */
    [data-testid="stSidebar"] .stCheckbox > label > div[data-testid="stMarkdownContainer"] > p,
    [data-testid="stSidebar"] .stCheckbox span {
        color: var(--text-primary) !important;
    }

    /* Sidebar headers */
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        font-size: 1rem;
        font-weight: 600;
        color: var(--text-primary) !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.75rem;
    }

    
    /* File uploader - Clean border */
    [data-testid="stFileUploader"] {
        border: 2px dashed var(--border-color);
        border-radius: 0.75rem;
        padding: 1rem;
        background: var(--bg-secondary);
    }
    
    [data-testid="stFileUploader"]:hover {
        border-color: var(--primary-color);
    }
    
    /* Tool cards - Subtle */
    .tool-card {
        background: var(--bg-primary);
        border: 1px solid var(--border-color);
        border-radius: 0.75rem;
        padding: 1.25rem;
        margin-bottom: 1rem;
        transition: all 0.2s;
    }
    
    .tool-card:hover {
        border-color: var(--primary-color);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
    }
    
    .tool-title {
        color: var(--text-primary);
        font-size: 1rem;
        font-weight: 600;
        margin-bottom: 0.375rem;
    }
    
    .tool-description {
        color: var(--text-secondary);
        font-size: 0.875rem;
    }

    /* Workspace selector in sidebar */
    .workspace-selector {
        background: var(--accent-light);
        border-radius: 0.5rem;
        padding: 0.75rem;
        margin-bottom: 1rem;
    }

    .workspace-label {
        font-size: 0.75rem;
        color: var(--text-secondary);
        margin-bottom: 0.25rem;
    }

    .workspace-name {
        font-size: 0.875rem;
        font-weight: 600;
        color: var(--primary-color);
    }

    /* Tabs - Clean underline style */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
        border-bottom: 1px solid var(--border-color);
    }

    .stTabs [data-baseweb="tab"] {
        color: var(--text-secondary);
        font-weight: 500;
    }

    .stTabs [data-baseweb="tab"]:hover {
        color: var(--text-primary);
    }

    .stTabs [aria-selected="true"] {
        color: var(--primary-color);
    }

    /* Footer */
    .footer {
        text-align: center;
        color: var(--text-secondary);
        font-size: 0.8rem;
        padding: 1rem 0;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)



# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'agent' not in st.session_state:
    st.session_state.agent = None
if 'rag_initialized' not in st.session_state:
    st.session_state.rag_initialized = False
if 'documents' not in st.session_state:
    st.session_state.documents = []
if 'stats' not in st.session_state:
    st.session_state.stats = {
        'documents': 0,
        'pages': 0,
        'queries': 0
    }
if 'selected_model' not in st.session_state:
    st.session_state.selected_model = "Groq"

# Multi-user: Email-based identification
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'user_email' not in st.session_state:
    st.session_state.user_email = ""
if 'user_id' not in st.session_state:
    st.session_state.user_id = "default_user"
if 'workspace_name' not in st.session_state:
    st.session_state.workspace_name = "My Research"


# ============================================================================
# LOGIN SCREEN
# ============================================================================
def show_login_screen():
    """Display the login screen for email-based identification"""
    
    # Center the login form
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("""
        <div style="text-align: center; padding: 2rem 0;">
            <div style="font-size: 3rem; font-weight: 700; color: var(--primary-color); margin-bottom: 0.5rem;">Aloysia</div>
            <div style="font-size: 1rem; color: var(--text-secondary); margin-bottom: 2rem;">Agentic Research Assistant</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### Sign in to your workspace")
        st.caption("Enter your email to access your personal research workspace.")
        
        with st.form("login_form"):
            email = st.text_input(
                "Email Address",
                placeholder="you@example.com",
                help="Your email will be used to identify your workspace"
            )
            
            workspace = st.text_input(
                "Workspace Name (optional)",
                placeholder="My Research",
                help="Give your workspace a name"
            )
            
            submit = st.form_submit_button("Sign In", use_container_width=True)
            
            if submit:
                if email and "@" in email:
                    st.session_state.logged_in = True
                    st.session_state.user_email = email
                    st.session_state.user_id = email  # Use email as user_id
                    st.session_state.workspace_name = workspace if workspace else "My Research"
                    st.rerun()
                else:
                    st.error("Please enter a valid email address")
        
        st.divider()
        st.caption("Your documents and conversations are stored privately in your workspace.")


# Check if user is logged in - if not, show login screen
if not st.session_state.logged_in:
    show_login_screen()
    st.stop()  # Stop execution here until logged in



def initialize_agent():
    """Initialize the RAG agent with appropriate caching"""
    if st.session_state.agent is None:
        with st.spinner("Initializing Aloysia..."):
            try:
                from code.rag_init import initialize_rag
                initialize_rag()


                from code.agent import create_agentic_rag
                st.session_state.agent = create_agentic_rag()
                st.session_state.rag_initialized = True

                st.success("Aloysia initialized successfully!")
                return True

            except Exception as e:
                st.error(f"Error initializing agent: {str(e)}")
                traceback.print_exc()
                return False
    return True




def load_document_from_folder(folder_path: Path):
    """Load documents from a folder"""
    try:
        from code.app import load_publication


        docs = load_publication(pub_dir=folder_path)
        st.session_state.documents = docs

        unique_sources = set([d['metadata']['source'] for d in docs])

        # Update stats
        st.session_state.stats['documents'] = len(unique_sources)
        st.session_state.stats['pages'] = sum([d['metadata'].get('page_count', 0) for d in docs])

        return docs
    except Exception as e:
        st.error(f"Error loading documents: {str(e)}")
        traceback.print_exc()
        return []
    

def process_query(query: str):
    """Process user query through the agent"""

    if not st.session_state.agent:
        if not initialize_agent():
            return "Failed to initialize agent."

    try:
        from langchain_core.messages import HumanMessage, AIMessage

        # Build history from existing messages BEFORE adding the new one
        history = []
        max_history = 10
        prior_messages = st.session_state.messages[-max_history:] if len(st.session_state.messages) > max_history else st.session_state.messages
        
        for msg in prior_messages:
            if msg["role"] == "user":
                history.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                history.append(AIMessage(content=msg["content"]))
        
        # Add the current message to session state for UI persistence
        st.session_state.messages.append({"role": "user", "content": query})
        
        # Also add to agent history (not duplicated — session state append above is for UI only)
        history.append(HumanMessage(content=query))

        initial_state = {
            "messages": history,
            "quality_passed": True,
            "loop_count": 0,
            "original_query": query,
            "selected_model": st.session_state.get("selected_model", "Groq").lower(),
            "user_id": st.session_state.get("user_id", "default_user")  # Pass user_id to agent
        }
        
        print(f"Invoking agent with {len(history)} messages...")

        # Invoke agent
        from code.agent import user_id_var
        token = user_id_var.set(st.session_state.get("user_id", "default_user"))
        try:
            result = st.session_state.agent.invoke(initial_state)
        finally:
            user_id_var.reset(token)

        if not result.get("messages"):
            return "I am sorry, but I couldn't generate a response. Please try again"

        # Get response
        final_message = result["messages"][-1]
        response = final_message.content

        print(f"Preview: {response[:100]}...")

        if not response or len(response.strip()) < 5:
            response = "I couldn't find enough information to answer your question. Could you rephrase it or ask something else?"


        # Add assistant response
        st.session_state.messages.append({
            "role": "assistant",
            "content": response
        })

        # Update query count
        st.session_state.stats['queries'] += 1

        return response
    
    except Exception as e:
        error_trace = traceback.format_exc()
        print(f"Process Query Error:\n{error_trace}")

        if "tool_use_failed" in str(e):
            return "I encountered an issue with tool execution. Please try:\n1. Rephrasing your question\n2. Making it more specific\n3. Breaking it into smaller questions"
        else:
            return f"An error occurred: {str(e)}\n\nPlease try again or rephrase your question."
    

# Header - Clean minimal design
st.markdown("""
<div class="header-container">
    <div class="header-left">
        <div class="header-logo">A</div>
        <div>
            <div class="header-title">Aloysia</div>
            <div class="header-subtitle">Your Agentic Research Assistant</div>
        </div>
    </div>
    <div class="workspace-badge">Research Workspace</div>
</div>
""", unsafe_allow_html=True)


# Sidebar
with st.sidebar:
    # User info and workspace
    st.markdown("### Account")
    st.caption(f"Signed in as: **{st.session_state.user_email}**")
    
    workspace_name = st.text_input(
        "Workspace",
        value=st.session_state.workspace_name,
        help="Name your research workspace",
        label_visibility="collapsed"
    )
    if workspace_name != st.session_state.workspace_name:
        st.session_state.workspace_name = workspace_name
    
    if st.button("Sign Out", use_container_width=True, type="secondary"):
        st.session_state.logged_in = False
        st.session_state.user_email = ""
        st.session_state.user_id = "default_user"
        st.session_state.messages = []
        st.rerun()
    
    st.divider()


    # Document upload
    st.markdown("### Documents")
    uploaded_files = st.file_uploader(
        "Upload research papers",
        type=["pdf", "docx", "txt", "md"],
        accept_multiple_files=True,
        key="main_uploader",
        help="PDF, DOCX, TXT, or Markdown files",
        label_visibility="collapsed"
    )

    if uploaded_files and st.button("Process Files", use_container_width=True):
        with st.spinner("Processing..."):
            # Save uploaded files temporarily
            upload_dir = Path("./uploaded_docs")
            upload_dir.mkdir(exist_ok=True)

            for uploaded_file in uploaded_files:
                file_path = upload_dir / uploaded_file.name
                with open(file_path, "wb") as f:
                    try:
                        f.write(uploaded_file.getbuffer())
                    except Exception:
                        f.write(uploaded_file.read())

            # Load documents
            docs = load_document_from_folder(upload_dir)

            if docs:
                st.success(f"Processed {len(docs)} chunks from {len(uploaded_files)} file(s)...")
                try:
                    if initialize_agent():
                        from code.rag_init import get_rag
                        rag = get_rag()
                        user_id = st.session_state.get("user_id", "default_user")
                        # Set source filename on each chunk before adding to Supabase
                        for doc in docs:
                            if "metadata" in doc and "source" not in doc:
                                doc["source"] = doc["metadata"].get("source", "uploaded_file")
                        rag.db.add_doc(docs, user_id=user_id)
                        st.session_state.pop('cached_sources', None)  # Force library refresh
                        st.success(f"Added to your knowledge base!")
                except Exception as e:
                    st.error(f"Error adding to knowledge base: {str(e)}")
                    traceback.print_exc()

    # Show loaded documents (Cached list to avoid DB lag)
    if st.session_state.get("rag_initialized", False):
        try:
            if 'cached_sources' not in st.session_state or st.button("🔄", help="Refresh Library"):
                from code.rag_init import get_rag
                rag = get_rag()
                
                # Fetch only for this user
                all_metadatas = rag.db.list_all_metadata(user_id=st.session_state.user_id)
                unique_sources = sorted(set([m.get("source", "Unknown") for m in all_metadatas]))
                st.session_state.cached_sources = unique_sources
                
            if st.session_state.cached_sources:
                with st.expander(f"{len(st.session_state.cached_sources)} Documents", expanded=False):
                    for source in st.session_state.cached_sources:
                        st.caption(f"• {source}")
        except Exception:
            pass

    st.divider()

    # Settings - Compact
    st.markdown("### Settings")
    st.session_state.selected_model = st.selectbox(
        "Model", 
        ["Groq", "Gemini"], 
        index=0 if st.session_state.get("selected_model", "Groq") == "Groq" else 1,
        label_visibility="collapsed"
    )
    
    col1, col2 = st.columns(2)
    with col1:
        st.checkbox("Re-ranking", value=True, key="rerank")
    with col2:
        st.checkbox("Web Search", value=True, key="websearch")

    st.divider()

    # Quick actions
    st.markdown("### Quick Actions")
    if st.button("Compare Docs", use_container_width=True):
        st.session_state.page = "compare"
    if st.button("Bibliography", use_container_width=True):
        st.session_state.page = "bibliography"
    if st.button("Literature Review", use_container_width=True):
        st.session_state.page = "review"
    
    # Clear Chat
    if st.button("Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.divider()

    # Sync to Mobile
    st.markdown("### Sync to Mobile")
    st.markdown(f"""
    Access your documents on Telegram:
    1. Open [**@Aloysia_telegram_bot**](https://t.me/Aloysia_telegram_bot)
    2. Send this command:
    ```
    /link {st.session_state.user_email}
    ```
    """)



# Main content area - Stats
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-number">{st.session_state.stats['documents']}</div>
        <div class="stat-label">Documents</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-number">{st.session_state.stats['pages']}</div>
        <div class="stat-label">Pages</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-number">{st.session_state.stats['queries']}</div>
        <div class="stat-label">Queries</div>
    </div>
    """, unsafe_allow_html=True)

# Tabs for different functionalities
tab1, tab2, tab3 = st.tabs(["Chat", "Tools", "Library"])

with tab1:
    # Chat container
    chat_container = st.container()

    with chat_container:
        # Display chat history
        if not st.session_state.messages:
            st.markdown("""
            <div style="text-align: center; padding: 3rem; color: #6b7280;">
                <div style="font-size: 1.125rem; font-weight: 500; margin-bottom: 0.5rem;">Welcome to Aloysia</div>
                <div style="font-size: 0.875rem;">Upload papers and start asking questions about your research.</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            for message in st.session_state.messages:
                if message["role"] == "user":
                    st.markdown(f"""
                    <div class="user-message">
                        {message["content"]}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="assistant-message">
                        {message["content"]}
                    </div>
                    """, unsafe_allow_html=True)

    # Chat input
    with st.form(key="chat-form", clear_on_submit=True):
        col1, col2 = st.columns([6, 1])

        with col1:
            user_input = st.text_input(
                "Ask anything about your documents...",
                placeholder="e.g., What is the main finding of this paper?",
                label_visibility="collapsed"
            )
            
        with col2:
            submit_button = st.form_submit_button("Send", use_container_width=True)

        if submit_button and user_input:
            with st.spinner("Thinking..."):
                response = process_query(user_input)
            st.rerun()

with tab2:
    st.markdown("#### Research Tools")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="tool-card">
            <div class="tool-title">Compare Documents</div>
            <div class="tool-description">Analyze similarities and differences between papers</div>
        </div>
        """, unsafe_allow_html=True)

        with st.expander("Compare"):
            doc1 = st.text_input("Document 1", placeholder="paper1.pdf")
            doc2 = st.text_input("Document 2", placeholder="paper2.pdf")
            topic = st.text_input("Topic", placeholder="methodology")

            if st.button("Compare", key="compare_btn"):
                if doc1 and doc2 and topic:
                    query = f"Compare {doc1} and {doc2} on topic: {topic}"
                    with st.spinner("Comparing..."):
                        response = process_query(query)
                        st.markdown(response)

    with col2:
        st.markdown("""
        <div class="tool-card">
            <div class="tool-title">Semantic Search</div>
            <div class="tool-description">Find relevant passages across all documents</div>
        </div>
        """, unsafe_allow_html=True)

        with st.expander("Search"):
            search_query = st.text_input("Search Query", placeholder="treatment outcomes")

            if st.button("Search", key="search_btn"):
                if search_query:
                    with st.spinner("Searching..."):
                        response = process_query(search_query)
                        st.markdown(response)

with tab3:
    st.markdown("#### Document Library")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        topic = st.text_input("Research Topic", placeholder="e.g., Antimicrobial Resistance")
    
    with col2:
        export_format = st.selectbox("Format", ["Markdown", "LaTeX", "Word"])
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Generate Bibliography", use_container_width=True):
            with st.spinner("Generating..."):
                response = process_query("Generate bibliography for all documents")
                st.markdown(response)
    
    with col2:
        if st.button("Generate Literature Review", use_container_width=True):
            if topic:
                with st.spinner("Generating..."):
                    response = process_query(f"Generate literature review on: {topic}")
                    st.markdown(response)
            else:
                st.warning("Enter a topic first")


# Footer
st.markdown("""
<div class="footer">
    Aloysia • Built by Nago
</div>
""", unsafe_allow_html=True)
