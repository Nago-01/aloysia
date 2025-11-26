"""RAG Initialization Script with Streamlit interface"""

from pathlib import Path
import streamlit as st



def initialize_rag():
    """Initialize RAG assistant with Streamlit state session as primary cache"""

    if 'rag_assistant' in st.session_state and st.session_state.rag_assistant is not None:
        print(f"\nUsing cached RAG from Streamlit session state")
        return st.session_state.rag_assistant

    print("\nFIRST-TIME RAG INITIALIZATION...")
    print("="*50)


    from .app import QAAssistant

    assistant = QAAssistant()
    print("QA Assistant created")
        
    # Trying multiple paths if document is not loaded yet
    possible_paths = [
        Path("./data"),
        Path(__file__).resolve().parent/"data",
        Path(__file__).resolve().parent.parent/"data"
    ]

    docs = None
    for data_path in possible_paths:
        if data_path.exists():
            try:
                print(f"Loading documents from {data_path}")
                from .app import load_publication


                docs = load_publication(pub_dir=data_path)
                print(f"Loaded {len(docs)} document chunks")
                break
            except Exception as e:
                print(f"Error loading from {data_path}: {e}")
                continue

    if docs is None:
        raise FileNotFoundError("Could not find data folder")

    # Adding documents to Vector DB
    print(f"Adding {len(docs)} document chunks to the Vector DB...")
    assistant.add_doc(docs)

    st.session_state.rag_assistant = assistant
    st.session_state.rag_initialized = True
        
    print("RAG INITIALIZATION COMPLETE")
    return assistant



def get_rag():
    """Get RAG assistant always from Streamlit session state"""

    if 'rag_assistant' in st.session_state and st.session_state.rag_assistant is not None:
        return st.session_state.rag_assistant

    # If it is not in session state, initialize it
    print("RAG not found in session state, initializing...")
    return initialize_rag()



def reset_rag():
    """Reset the RAG assistant (clear session state)"""
    if 'rag_assistant' in st.session_state:
        del st.session_state.rag_assistant
    if 'rag_initialized' in st.session_state:
        del st.session_state.rag_initialized

    print("RAG reset from session state")