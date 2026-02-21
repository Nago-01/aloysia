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
    def do_HEAD(self):
        """Handle HEAD requests for Render health checks."""
        if self.path in ("/", "/status"):
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
        else:
            self.send_response(301)
            self.send_header('Location', 'https://t.me/Aloysia_telegram_bot')
            self.end_headers()

    def do_GET(self):
        if self.path in ("/", "/status"):
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            
            # Check thread status
            is_alive = any(t.name == "BotThread" and t.is_alive() for t in threading.enumerate())
            status_color = "green" if is_alive else "red"
            status_text = "RUNNING" if is_alive else "CRASHED/STOPPED"
            
            bot_username = os.getenv("BOT_USERNAME", "Aloysia_telegram_bot")
            html = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="utf-8">
                <meta name="viewport" content="width=device-width, initial-scale=1">
                <title>Aloysia Status</title>
                <style>
                    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
                           text-align: center; padding: 60px 20px; background: #f5f5f5; }}
                    .card {{ background: white; border-radius: 16px; padding: 40px;
                             max-width: 400px; margin: 0 auto; box-shadow: 0 2px 12px rgba(0,0,0,0.08); }}
                    h1 {{ font-size: 1.5rem; margin-bottom: 8px; color: #1a1a1a; }}
                    .status {{ font-size: 2rem; font-weight: 700; color: {status_color}; margin: 16px 0; }}
                    .btn {{ display: inline-block; margin-top: 24px; padding: 14px 32px;
                            background: #0088cc; color: white; border-radius: 12px;
                            text-decoration: none; font-weight: 600; font-size: 1rem; }}
                    .btn:hover {{ background: #006faa; }}
                </style>
            </head>
            <body>
                <div class="card">
                    <h1>Aloysia</h1>
                    <p style="color:#666;">Bot Status</p>
                    <div class="status">{status_text}</div>
                    <!-- tg:// opens the Telegram app on mobile; JS fallback covers browsers that ignore it -->
                    <a class="btn"
                       href="tg://resolve?domain={bot_username}"
                       onclick="setTimeout(function(){{ window.location='https://t.me/{bot_username}' }}, 600); return true;">
                       💬 Open in Telegram
                    </a>
                </div>
            </body>
            </html>
            """
            self.wfile.write(html.encode())
            return

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
        print("CRITICAL ERROR: Missing environment variables!")
        for var in missing:
            print(f"   - {var} is NOT SET")
        print("\nPlease check your Render Dashboard -> Environment tab.")
        return False
    
    print("Environment Check Passed. All required keys found.")
    return True

def run_gateway():
    """Run a tiny HTTP gateway to redirect to Telegram and satisfy Render."""
    if not check_environment():
        print("Gateway will still run to keep Render happy, but Bot will likely fail.")
        
    port = int(os.getenv("PORT", "10000"))
    print(f"Telegram Gateway starting on port {port}...")
    server = HTTPServer(('0.0.0.0', port), RedirectHandler)
    server.serve_forever()

def run_telegram_bot():
    """Run the Telegram bot in its own event loop with a cooldown."""
    import time
    print("Telegram Bot: Entering 30s cooldown to prioritize Gateway startup...")
    time.sleep(30)
    
    print("Starting Telegram Bot...")
    try:
        from code.telegram_bot import main as bot_main
        bot_main()
    except Exception as e:
        print(f"Telegram Bot error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    import signal
    
    # 1. Start Telegram Bot in a background thread
    bot_thread = threading.Thread(target=run_telegram_bot, daemon=True, name="BotThread")
    bot_thread.start()
    
    # 2. Start Redirect Gateway in the main thread
    port = int(os.getenv("PORT", "10000"))
    print(f"Telegram Gateway starting on port {port}...")
    server = HTTPServer(('0.0.0.0', port), RedirectHandler)
    
    def signal_handler(sig, frame):
        print(f"Received signal {sig}, shutting down...")
        # Closing the server will break the serve_forever loop
        threading.Thread(target=server.shutdown).start()
        sys.exit(0)

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    
    server.serve_forever()
