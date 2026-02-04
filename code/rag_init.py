"""RAG Initialization Script with Streamlit interface"""

from pathlib import Path
import sys

# Global singleton cache for Bot/Script usage
_rag_cache = {"instance": None}




def initialize_rag():
    """Initialize RAG assistant - Supports both Streamlit and Bot execution"""

    # 1. Check Streamlit Session State (if running in Streamlit)
    try:
        if "streamlit" in sys.modules:
            import streamlit as st
            if st.runtime.exists():
                if 'rag_assistant' in st.session_state and st.session_state.rag_assistant is not None:
                    print(f"\nUsing cached RAG from Streamlit session state")
                    return st.session_state.rag_assistant
    except (ImportError, AttributeError, RuntimeError):
        pass

    # 2. Check Global Cache (for Bot/Script)
    if _rag_cache["instance"] is not None:
        print(f"\nUsing cached RAG from global memory")
        return _rag_cache["instance"]

    print("\nFIRST-TIME RAG INITIALIZATION...")
    print("="*50)

    from .app import QAAssistant
    assistant = QAAssistant()
    print("QA Assistant created")

    # OPTIONAL SEEDING: 
    # Only load from ./data if explicitly requested via environment variable.
    # In production (Render), we already have the data in Supabase, so we skip this.
    if os.getenv("SEED_DATA", "false").lower() == "true":
        print("SEED_DATA=true: Loading initial documents from disk...")
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

        if docs:
            # Adding documents to Vector DB
            print(f"Adding {len(docs)} document chunks to the Vector DB...")
            assistant.add_doc(docs)
    else:
        print("SEED_DATA=false: Skipping initial document loading (using Supabase library).")

    # 3. Store in Caches
    _rag_cache["instance"] = assistant
    
    try:
        if "streamlit" in sys.modules:
            import streamlit as st
            # heuristics to check if running in streamlit
            if st.runtime.exists():
                st.session_state.rag_assistant = assistant
                st.session_state.rag_initialized = True
    except (ImportError, AttributeError, RuntimeError):
        pass
        
    print("RAG INITIALIZATION COMPLETE")
    return assistant



def get_rag():
    """Get RAG assistant - robustly handles both envs"""

    # Try Streamlit first
    try:
        if "streamlit" in sys.modules:
            import streamlit as st
            if st.runtime.exists():
                if 'rag_assistant' in st.session_state and st.session_state.rag_assistant is not None:
                    return st.session_state.rag_assistant
    except (ImportError, AttributeError, RuntimeError):
        pass

    # Try Global Cache
    if _rag_cache["instance"] is not None:
        return _rag_cache["instance"]

    # Not found -> Init
    print("RAG not found in cache, initializing...")
    return initialize_rag()



def reset_rag():
    """Reset the RAG assistant (clear session state)"""
    # Clear Streamlit Cache
    try:
        if "streamlit" in sys.modules:
            import streamlit as st
            if st.runtime.exists():
                if 'rag_assistant' in st.session_state:
                    del st.session_state.rag_assistant
                if 'rag_initialized' in st.session_state:
                    del st.session_state.rag_initialized
    except (ImportError, AttributeError, RuntimeError):
        pass
        
    # Clear Global Cache
    _rag_cache["instance"] = None

    print("RAG reset from session state")