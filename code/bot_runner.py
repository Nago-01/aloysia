"""
Standalone Telegram Bot Runner for Fly.io Deployment
Runs bot polling with a lightweight HTTP health endpoint.
"""

import os
import sys
import asyncio
import logging
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
import threading

project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)


class HealthHandler(BaseHTTPRequestHandler):
    """Lightweight health check endpoint for Fly.io"""
    
    def log_message(self, format, *args):
        pass
    
    def do_GET(self):
        if self.path == "/health":
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"status": "healthy", "service": "aloysia-bot"}')
        else:
            self.send_response(404)
            self.end_headers()
    
    def do_HEAD(self):
        if self.path == "/health":
            self.send_response(200)
            self.end_headers()
        else:
            self.send_response(404)
            self.end_headers()


def run_health_server():
    """Run health check server in background thread"""
    port = int(os.getenv("PORT", "8080"))
    server = HTTPServer(('0.0.0.0', port), HealthHandler)
    logger.info(f"Health server started on port {port}")
    server.serve_forever()


def check_environment():
    """Verify required environment variables"""
    required_vars = [
        "TELEGRAM_BOT_TOKEN",
        "SUPABASE_URL",
        "SUPABASE_SERVICE_ROLE_KEY",
    ]
    
    missing = [var for var in required_vars if not os.getenv(var)]
    
    if missing:
        logger.error("Missing required environment variables:")
        for var in missing:
            logger.error(f"  - {var}")
        return False
    
    logger.info("Environment check passed")
    return True


def main():
    """Main entry point for Fly.io deployment"""
    logger.info("=" * 50)
    logger.info("Aloysia Telegram Bot - Fly.io Deployment")
    logger.info("=" * 50)
    
    if not check_environment():
        logger.error("Environment check failed. Exiting.")
        sys.exit(1)
    
    health_thread = threading.Thread(
        target=run_health_server,
        daemon=True,
        name="HealthServer"
    )
    health_thread.start()
    logger.info("Health check server started")
    
    logger.info("Starting Telegram bot...")
    
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    from code.telegram_bot import main as bot_main
    bot_main()


if __name__ == "__main__":
    main()
