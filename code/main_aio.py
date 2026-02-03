import os
import sys
import threading
import asyncio
from pathlib import Path

# Add project root to path
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def run_streamlit():
    """Run the Streamlit app using its CLI module."""
    print("🚀 Starting Streamlit UI...")
    import streamlit.web.cli as stcli
    
    # Get port from environment (Render provides this)
    port = os.getenv("PORT", "10000")
    
    # Prepare arguments
    sys.argv = [
        "streamlit", 
        "run", 
        "code/streamlit_app.py", 
        "--server.port", port, 
        "--server.address", "0.0.0.0"
    ]
    stcli.main()

def run_telegram_bot():
    """Run the Telegram bot in its own event loop."""
    print("🤖 Starting Telegram Bot...")
    try:
        from code.telegram_bot import main as bot_main
        # telegram-bot v20+ uses its own internal loop and is blocking
        bot_main()
    except Exception as e:
        print(f"❌ Telegram Bot error: {e}")

if __name__ == "__main__":
    # 1. Start Telegram Bot in a background thread
    # Note: Bot main() is blocking, so we put it in a thread.
    bot_thread = threading.Thread(target=run_telegram_bot, daemon=True)
    bot_thread.start()
    
    # 2. Start Streamlit in the main thread (blocking)
    run_streamlit()
