"""
Streamlit Web UI for RAG Research Assistant
Army Green Theme with Dark Mode Support
"""

import streamlit as st
from pathlib import Path
import sys, traceback
import os, chromadb
import warnings
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

# Custom CSS for Army Green Theme with Dark Mode Support
st.markdown("""
<style>
    /* Import fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
            
    /* Root variables for theming */
    :root {
        --primary-green: #4a5f3a;
        --secondary-green: #6b7f5a;
        --light-green: #8b9f7a;
        --accent-green: #3a4f2a;
    }
            
    /* Main container */
    .main {
        background-color: var(--background-color);
    }
    
    /* Header styling */
    .header-container {
        background: linear-gradient(135deg, #4a5f3a 0%, #3a4f2a 100%);
        padding: 2rem;
        border-radius: 1rem;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
    }

    .header-title {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }

    .header-subtitle {
        font-size: 1.1rem;
        opacity: 0.9;
    }

    /* Stats cards */
    .stats-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin-bottom: 2rem;
    }
            
    .stat-card {
        background: linear-gradient(135deg, #4a5f3a 0%, #6b7f5a 100%);
        padding: 1.5rem;
        border-radius: 0.75rem;
        color: white;
        text-align: center;
        box-shadow: 0 4px 12px rgba(74, 95, 58, 0.2)
    }
            
    .stat-number {
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    .stat-label {
        font-size: 0.9rem;
        opacity: 0.9;
    }
    
    /* Chat messages */
    .user-message {
        background: linear-gradient(135deg, #4a5f3a 0%, #6b7f5a 100%);
        color: white;
        padding: 1rem 1.25rem;
        border-radius: 1rem;
        border-bottom-right-radius: 0.25rem;
        margin: 0.5rem 0;
        margin-left: 20%;
    }
    
    .assistant-message {
        background: #f5f7f5;
        color: #2d3a2d;
        padding: 1rem 1.25rem;
        border-radius: 1rem;
        border-bottom-left-radius: 0.25rem;
        margin: 0.5rem 0;
        margin-right: 20%;
        border: 1px solid #d0d8d0;
    }
    
    .citation {
        font-size: 0.85rem;
        color: rgba(255,255,255,0.8);
        margin-top: 0.5rem;
        padding-top: 0.5rem;
        border-top: 1px solid rgba(255,255,255,0.2);
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #4a5f3a 0%, #6b7f5a 100%);
        color: white;
        border: none;
        border-radius: 0.5rem;
        padding: 0.5rem 1.5rem;
        font-weight: 600;
        transition: transform 0.2s;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(74, 95, 58, 0.3);
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #000000;
    }
    
    [data-testid="stSidebar"] .stMarkdown {
        color: #2d3a2d;
    }
    
    /* File uploader */
    [data-testid="stFileUploader"] {
        border: 2px dashed #4a5f3a;
        border-radius: 0.75rem;
        padding: 1rem;
        background: white;
    }
    
    /* Tool cards */
    .tool-card {
        background: white;
        border: 2px solid #d0d8d0;
        border-radius: 0.75rem;
        padding: 1.5rem;
        margin-bottom: 1rem;
        transition: all 0.3s;
    }
    
    .tool-card:hover {
        border-color: #4a5f3a;
        transform: translateY(-4px);
        box-shadow: 0 8px 20px rgba(74, 95, 58, 0.1);
    }
    
    .tool-title {
        color: #4a5f3a;
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    
    .tool-description {
        color: #5a6b5a;
        font-size: 0.9rem;
    }
    
    /* Badge */
    .badge {
        display: inline-block;
        background: #4a5f3a;
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 1rem;
        font-size: 0.75rem;
        font-weight: 600;
        margin-left: 0.5rem;
    }
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

    print(f"Processing query: {query}")
    print(f"{'='*50}\n")

    if not st.session_state.agent:
        if not initialize_agent():
            return "Failed to initialize agent."

    try:
        from langchain_core.messages import HumanMessage, AIMessage

        # Add user message to conversation
        st.session_state.messages.append({
            "role": "user",
            "content": query
        })
        
        # Build history from session state
        history = []
        for msg in st.session_state.messages[:-1]: # Skip the one we just added (will be added as current input)
            if msg["role"] == "user":
                history.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                history.append(AIMessage(content=msg["content"]))
        
        # Current message
        history.append(HumanMessage(content=query))

        initial_state = {
            "messages": history,
            "quality_passed": True,
            "loop_count": 0,
            "original_query": query
        }
        
        print(f"Invoking agent with {len(history)} messages...")

        # Invoke agent
        result = st.session_state.agent.invoke(initial_state)

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
    

# Header
st.markdown("""
<div class="header-container">
    <div class="header-title">Aloysia</div>
    <div class="header-subtitle">
        Your AI-powered document analyst and research companion
    </div>
</div>
""", unsafe_allow_html=True)


# Sidebar
with st.sidebar:
    st.title("Document Management")

    # Document upload
    st.subheader("Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload PDF, DOCX, TXT, or MD files",
        type=["pdf", "docx", "txt", "md"],
        accept_multiple_files=True,
        key="main_uploader",
        help="Upload research papers, documents, or text files"
    )

    if uploaded_files and st.button("Process Uploaded Files"):
        with st.spinner("Processing uploaded files..."):
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
                st.success(f"Loaded {len(docs)} document chunks")

                
                try:
                    if initialize_agent():
                        from code.rag_init import initialize_rag
                        rag = initialize_rag()
                        st.success("Documents added to knowledge base")
                except Exception as e:
                    st.error(f"Error initializing agent: {str(e)}")
                    traceback.print_exc()

    # Show loaded documents
    st.subheader("Loaded Documents")

    if st.session_state.get("rag_initialized", False):
        try:
            from code.rag_init import get_rag
            rag = get_rag()
            
            all_results = rag.db.collection.get()
            unique_sources = set()

            for metadata in all_results.get("metadatas", [])[:10]:
                source = metadata.get("source")
                if source:
                    unique_sources.add(source)
            if unique_sources:
                for source in sorted(unique_sources):
                    st.text(f"{source}")
            else:
                st.info("No documents loaded yet")
        except Exception as e:
            st.info("Upload files to get started")
    else: 
        st.info("No documents loaded. Upload files to get started.")

    st.divider()

    # Setting up features
    st.title("Features")

    if st.button("Compare Documents", use_container_width=True):
        st.session_state.page = "compare"
    
    if st.button("Generate Bibliography", use_container_width=True, key="sidebar_bibliography_btn"):
        st.session_state.page = "bibliography"
    
    if st.button("Literature Review", use_container_width=True):
        st.session_state.page = "review"
    
    if st.button("Export Options", use_container_width=True):
        st.session_state.page = "export"
    
    st.divider()


    #Integrations
    st.title("Integrations")
    st.markdown("""
    **Coming Soon:**
    - WhatsApp Bot
    - Telegram Bot
    - API Access
    """)

    st.divider()

    # Settings
    with st.expander("Settings"):
        st.selectbox("LLM Provider", ["Gemini", "Groq"])
        st.slider("Search Results", 1, 10, 15)
        st.checkbox("Enable Re-ranking", value=True)
        st.checkbox("Enable Web Search", value=True, help="Allow agent to search the web when needed")



# Main content area
stats_col1, stats_col2, stats_col3 = st.columns(3)

with stats_col1:
    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-number">{st.session_state.stats['documents']}</div>
        <div class="stat-label">Documents</div>
    </div>
    """, unsafe_allow_html=True)

with stats_col2:
    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-number">{st.session_state.stats['pages']}</div>
        <div class="stat-label">Pages</div>
    </div>
    """, unsafe_allow_html=True)

with stats_col3:
    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-number">{st.session_state.stats['queries']}</div>
        <div class="stat-label">Queries</div>
    </div>
    """, unsafe_allow_html=True)

# Tabs for different functionalities
tab1, tab2, tab3, tab4 = st.tabs(["Chat", "Tools", "Bibliography", "Literature Review"])

with tab1:
    st.subheader("Chat with your Documents")

    # Chat container
    chat_container = st.container()

    with chat_container:
        # Display chat history
        for message in st.session_state.messages:
            if message["role"] == "user":
                st.markdown(f"""
                <div class="user-message">
                    <strong>You:</strong><br>
                    {message["content"]}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="assistant-message">
                    <strong>Aloysia:</strong><br>
                    {message["content"]}
                </div>
                """, unsafe_allow_html=True)

    # Chat input
    with st.form(key="chat-form", clear_on_submit=True):
        col1, col2 = st.columns([5, 1])

        with col1:
            user_input = st.text_input(
                "Ask anything about your documents...",
                placeholder="E.g., What is Aritificial Intelligence?",
                label_visibility="collapsed"
            )
            
        with col2:
            submit_button = st.form_submit_button("Send", use_container_width=True)

        if submit_button and user_input:
            with st.spinner("Processing..."):
                response = process_query(user_input)
            st.rerun()

with tab2:
    st.subheader("Research Tools")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="tool-card">
            <div class="tool-title">Compare Documents</div>
            <div class="tool-description">Compare two documents side-by-side on specific topics</div>
        </div>
        """, unsafe_allow_html=True)

        with st.expander("Compare two documents"):
            doc1 = st.text_input("Document 1(e.g., xyz.pdf)")
            doc2 = st.text_input("Document 2(e.g., med.pdf)")
            topic = st.text_input("Topic to compare (e.g., treatment)")

            if st.button("Compare", key="compare_btn"):
                if doc1 and doc2 and topic:
                    query = f"Compare {doc1} and {doc2} on topic: {topic}"
                    with st.spinner("Comparing documents..."):
                        response = process_query(query)
                        st.markdown(response)

    with col2:
        st.markdown("""
        <div class="tool-card">
            <div class="tool-title">Advanced Search</div>
            <div class="tool-description">Search with filters and advanced options</div>
        </div>
        """, unsafe_allow_html=True)

        with st.expander("Advanced Search Options"):
            search_query = st.text_input("Search Query")
            num_results = st.slider("Number of Results", 1, 10, 5)
            use_reranking = st.checkbox("Enable re-ranking", value=True)

            if st.button("Search", key="search_btn"):
                if search_query:
                    with st.spinner("Searching..."):
                        response = process_query(search_query)
                        st.markdown(response)

with tab3:
    st.subheader("Generate Bibliography")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("Generate and export bibliographies in multiple formats")
    
    with col2:
        export_format = st.selectbox("Format", ["Word", "LaTeX", "Markdown"])
    
    if st.button("Generate Bibliography", use_container_width=True, key="tab_bibliography_btn"):
        with st.spinner("Generating bibliography..."):
            response = process_query("Generate bibliography")
            st.markdown(response)
    
    if st.button(f"Export to {export_format}", use_container_width=True):
        with st.spinner(f"Exporting to {export_format}..."):
            response = process_query(f"Export bibliography to {export_format}")
            st.success(response)

with tab4:
    st.subheader("Generate Literature Review")

    col1, col2 = st.columns([2, 1])
    
    topic = st.text_input("Research topic", placeholder="E.g., Antimicrobial Resistance")
    # max_sources = st.slider("Maximum sources", 1, 20, 10)
    
    with col1:
        st.markdown("Generate and export literature reviews in multiple formats")
   
    with col2:
        export_format = st.selectbox("Export format", ["Word", "LaTeX", "Markdown"], key="review_format")
        
    if st.button(f"Export Review as {export_format}", use_container_width=True):
        if topic:
            with st.spinner(f"Exporting to {export_format}..."):
                response = process_query(f"Export literature review on {topic} as {export_format}")
                st.success(response)
        else:
            st.warning("Please enter a research topic")

    
    if st.button("Generate Review", use_container_width=True):
        if topic:
            with st.spinner("Generating literature review..."):
                response = process_query(f"Generate literature review on: {topic}")
                st.markdown(response)
        else:
            st.warning("Please enter a research topic")


# Footer
st.divider()
st.markdown("""
<div style="text-align: center; color: #5a6b5a; font-size: 0.9rem;">
    <p>Aloysia | Built by Nago</p>
</div>
""", unsafe_allow_html=True)

                        
