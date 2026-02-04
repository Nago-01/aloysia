import os
import sys
import threading
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler

# Add project root to path
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

class RedirectHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        # Redirect all traffic to the Telegram bot
        self.send_response(301)
        self.send_header('Location', 'https://t.me/Aloysia_telegram_bot')
        self.end_headers()

def check_environment():
    """Verify that all required environment variables are present."""
    required_vars = [
        "TELEGRAM_BOT_TOKEN",
        "SUPABASE_URL",
        "SUPABASE_SERVICE_ROLE_KEY",
        "GEMINI_API_KEY",
        "GROQ_API_KEY"
    ]
    missing = [var for var in required_vars if not os.getenv(var)]
    
    if missing:
        print("❌ CRITICAL ERROR: Missing environment variables!")
        for var in missing:
            print(f"   - {var} is NOT SET")
        print("\nPlease check your Render Dashboard -> Environment tab.")
        return False
    
    print("✅ Environment Check Passed. All required keys found.")
    return True

def run_gateway():
    """Run a tiny HTTP gateway to redirect to Telegram and satisfy Render."""
    if not check_environment():
        print("🛑 Gateway will still run to keep Render happy, but Bot will likely fail.")
        
    port = int(os.getenv("PORT", "10000"))
    print(f"🚀 Telegram Gateway starting on port {port}...")
    server = HTTPServer(('0.0.0.0', port), RedirectHandler)
    server.serve_forever()

def run_telegram_bot():
    """Run the Telegram bot in its own event loop with a cooldown."""
    import time
    print("🤖 Telegram Bot: Entering 30s cooldown to prioritize Gateway startup...")
    time.sleep(30)
    
    print("🤖 Starting Telegram Bot...")
    try:
        from code.telegram_bot import main as bot_main
        bot_main()
    except Exception as e:
        print(f"❌ Telegram Bot error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    import signal
    
    # 1. Start Telegram Bot in a background thread
    bot_thread = threading.Thread(target=run_telegram_bot, daemon=True)
    bot_thread.start()
    
    # 2. Start Redirect Gateway in the main thread
    port = int(os.getenv("PORT", "10000"))
    print(f"🚀 Telegram Gateway starting on port {port}...")
    server = HTTPServer(('0.0.0.0', port), RedirectHandler)
    
    def signal_handler(sig, frame):
        print(f"🛑 Received signal {sig}, shutting down...")
        # Closing the server will break the serve_forever loop
        threading.Thread(target=server.shutdown).start()
        sys.exit(0)

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    
    server.serve_forever()
