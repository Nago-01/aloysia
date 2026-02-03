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

def run_gateway():
    """Run a tiny HTTP gateway to redirect to Telegram and satisfy Render."""
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
    # 1. Start Telegram Bot in a background thread
    bot_thread = threading.Thread(target=run_telegram_bot, daemon=True)
    bot_thread.start()
    
    # 2. Start Redirect Gateway in the main thread (blocking)
    run_gateway()
