import os
import sys
import asyncio
import logging
import traceback
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add project root to path to allow imports from 'code' package
# This fixes "ModuleNotFoundError: No module named 'code'"
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from telegram import Update, constants
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

# Constants
TEMP_DIR = Path("./temp_telegram_uploads")
TEMP_DIR.mkdir(exist_ok=True)

class AloysiaBot:
    def __init__(self):
        self._agent = None
        self._db = None

    @property
    def agent(self):
        """Lazy-initialize agent only when needed."""
        if self._agent is None:
            from code.agent import create_agentic_rag
            self._agent = create_agentic_rag()
        return self._agent

    @property
    def db(self):
        """Lazy-initialize DB only when needed."""
        if self._db is None:
            from code.db import VectorDB
            self._db = VectorDB()
        return self._db

    def _resolve_user_id(self, chat_id: str) -> str:
        """Helper to get the mapped email for a chat_id, or return chat_id if not linked."""
        return self.db.get_mapped_user(chat_id)

    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Send a message when the command /start is issued."""
        user = update.effective_user
        await update.message.reply_html(
            f"Hi {user.mention_html()}! I'm <b>Aloysia</b>, your Agentic Research Assistant.\n\n"
            f"I help you research, summarize, and cite papers directly from Telegram.\n\n"
            f"<b>Quick Start:</b>\n"
            f"1. <b>Upload a paper</b> - I'll read and index it for your personal library.\n"
            f"2. <b>Ask a question</b> - I'll search your papers and provide cited answers.\n"
            f"3. <b>Sync Accounts</b> - Use <code>/link your@email.com</code> to sync your knowledge base with the Aloysia Web Portal.\n\n"
            f"📚 Use /library to see your papers.\n"
            f"❓ Use /help for all commands."
        )

    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Send a message when the command /help is issued."""
        help_text = (
            "<b>Available Commands:</b>\n\n"
            "/start - Restart and see instructions\n"
            "/library - List your uploaded papers\n"
            "/link [email] - Link your account to the Web App\n"
            "/clear - Clear conversation history context\n"
            "/help - Show this help message\n\n"
            "<b>Features:</b>\n"
            "• Send any PDF file to add it to your knowledge base\n"
            "• Send text to ask questions about your papers\n"
            "• Linked accounts share the same library on Web and Mobile!\n\n"
            f"<i>Aloysia Bot v1.2 (Build: 2026-02-03)</i>"
        )
        await update.message.reply_html(help_text)

    async def link_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Link Telegram account to an email for cross-platform sync."""
        chat_id = str(update.effective_chat.id)
        if not context.args:
            await update.message.reply_html(
                "❌ <b>Missing Email</b>\n\nUsage: <code>/link user@example.com</code>\n\n"
                "Once linked, your Telegram bot will share the same library as your Web App."
            )
            return

        email = context.args[0].lower().strip()
        if "@" not in email or "." not in email:
            await update.message.reply_text("❌ Please provide a valid email address.")
            return

        success = self.db.link_user(chat_id, email)
        if success:
            await update.message.reply_html(
                f"<b>Account Linked!</b>\n\nYour Telegram bot is now synced with <b>{email}</b>.\n"
                "All papers you upload here will appear on the Web App, and vice versa."
            )
        else:
            await update.message.reply_text("❌ Failed to link account. Please try again later.")

    async def clear_state(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Clear conversation history (UI level for now)."""
        await update.message.reply_text("Conversation history cleared (context reset).")

    async def list_library(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """List documents uploaded by the user."""
        chat_id = str(update.effective_chat.id)
        user_id = self._resolve_user_id(chat_id)
        
        await update.message.reply_text(f"Checking library for {user_id if '@' in user_id else 'your account'}...")
        
        try:
            # Fetch all metadata but filter by resolved user_id
            results = self.db.list_all_metadata(user_id=user_id) 
            
            user_docs = set()
            for meta in results:
                user_docs.add(meta.get("source", "Unknown"))
            
            if user_docs:
                doc_list = "\n".join([f"• {doc}" for doc in sorted(user_docs)])
                await update.message.reply_text(f"<b>Your Papers ({len(user_docs)}):</b>\n\n{doc_list}", parse_mode="HTML")
            else:
                await update.message.reply_text("Your library is empty.\n\nUpload a PDF to get started!")
                
        except Exception as e:
            logger.error(f"Library error: {e}")
            await update.message.reply_text("❌ creating library list failed.")

    async def handle_document(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle file uploads."""
        document = update.message.document
        chat_id = str(update.effective_chat.id)
        
        # Check extensions
        file_ext = Path(document.file_name).suffix.lower()
        if file_ext not in ['.pdf', '.docx', '.txt']:
            await update.message.reply_text("Please upload PDF, DOCX or TXT files only.")
            return

        file = await context.bot.get_file(document.file_id)
        file_path = TEMP_DIR / document.file_name
        
        status_msg = await update.message.reply_text(f"Downloading <b>{document.file_name}</b>...", parse_mode="HTML")
        
        try:
            # Download file
            await file.download_to_drive(file_path)
            
            await context.bot.edit_message_text(
                f"Processing <b>{document.file_name}</b>...\n<i>Reading and chunking...</i>",
                chat_id=chat_id,
                message_id=status_msg.message_id,
                parse_mode="HTML"
            )
            
            # Process using app.py extraction logic
            from code.app import extract_text_with_page_numbers
            chunks_data = extract_text_with_page_numbers(file_path, file_ext)
            
            # Format for VectorDB
            from langchain_core.documents import Document
            
            user_id = self._resolve_user_id(chat_id)
            
            docs = []
            for chunk in chunks_data:
                docs.append(Document(
                    page_content=chunk["content"],
                    metadata={
                        "source": document.file_name,
                        "page": chunk["page_number"],
                        "user_id": user_id, # RLS Isolation (Linked or chat_id)
                    }
                ))
            
            # Add to DB
            if docs:
                self.db.add_doc(docs, user_id=user_id)
                msg = f"<b>{document.file_name}</b> added to your library!\n\nYou can now ask questions about it."
            else:
                msg = f"Could not extract text from <b>{document.file_name}</b>."
            
            # Clean up
            if file_path.exists():
                os.remove(file_path)
            
            await context.bot.edit_message_text(
                msg,
                chat_id=chat_id,
                message_id=status_msg.message_id,
                parse_mode="HTML"
            )
            
        except Exception as e:
            logger.error(f"Upload error: {e}")
            traceback.print_exc()
            await context.bot.edit_message_text(
                f"❌ Error processing file: {str(e)}",
                chat_id=chat_id,
                message_id=status_msg.message_id
            )

    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle text messages (RAG queries)."""
        chat_id = str(update.effective_chat.id)
        query = update.message.text
        
        if len(query) < 3:
            return

        # Send typing action
        await context.bot.send_chat_action(chat_id=chat_id, action=constants.ChatAction.TYPING)
        
        # Placeholder message
        thinking_msg = await update.message.reply_text("🤔 Thinking...")
        
        try:
            # Construct Agent State
            # We maintain a minimalist history here for now (stateless per turn for MVP)
            # Future: store history in a dict keyed by chat_id
            
            from langchain_core.messages import HumanMessage
            from code.agent import user_id_var
            
            user_id = self._resolve_user_id(chat_id)
            
            initial_state = {
                "messages": [HumanMessage(content=query)],
                "quality_passed": True,
                "loop_count": 0,
                "original_query": query,
                "selected_model": "groq",  # Default to fast model for Bot
                "user_id": user_id  # RLS Enforced Isolation!
            }
            
            # Run Agent in a separate thread to avoid blocking the event loop
            # and add a timeout to prevent indefinite hangs
            token = user_id_var.set(user_id)
            try:
                # Local function to ensure 'self.agent' lazy-init happens in the thread
                def _run_agent_sync():
                    return self.agent.invoke(initial_state)

                # 3 minute timeout for complex RAG tasks
                result = await asyncio.wait_for(
                    asyncio.to_thread(_run_agent_sync), 
                    timeout=180.0
                )
            except asyncio.TimeoutError:
                logger.error("Agent execution timed out after 180s")
                await context.bot.edit_message_text(
                    "❌ I'm sorry, the research task took too long (Timeout). Please try a simpler question.",
                    chat_id=chat_id,
                    message_id=thinking_msg.message_id
                )
                return
            finally:
                user_id_var.reset(token)
            
            if result.get("messages"):
                final_response = result["messages"][-1].content
                
                # Format response (Markdown cleaning if needed)
                # Telegram supports a subset of HTML/Markdown
                
                await context.bot.edit_message_text(
                    final_response,
                    chat_id=chat_id,
                    message_id=thinking_msg.message_id,
                    parse_mode=None # Let Telegram handle auto-linking, avoid parsing errors
                )
            else:
                await context.bot.edit_message_text(
                    "I couldn't generate a response. Please try rephrasing.",
                    chat_id=chat_id,
                    message_id=thinking_msg.message_id
                )
                
        except Exception as e:
            logger.error(f"Query error: {e}")
            traceback.print_exc()
            await context.bot.edit_message_text(
                f"❌ Error: {str(e)}",
                chat_id=chat_id,
                message_id=thinking_msg.message_id
            )

    async def create_app(self):
        """Create and configure the bot application."""
        token = os.getenv("TELEGRAM_BOT_TOKEN")
        if not token:
            print("Error: TELEGRAM_BOT_TOKEN environment variable not set.")
            return None
            
        application = Application.builder().token(token).build()

        # Add handlers
        application.add_handler(CommandHandler("start", self.start))
        application.add_handler(CommandHandler("help", self.help_command))
        application.add_handler(CommandHandler("library", self.list_library))
        application.add_handler(CommandHandler("link", self.link_command))
        application.add_handler(CommandHandler("clear", self.clear_state))
        
        # File uploads
        application.add_handler(MessageHandler(filters.Document.PDF, self.handle_document))
        
        # Text messages
        application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_message))

        return application

def main():
    """Start the bot."""
    bot = AloysiaBot()
    
    # Load environment variables if needed
    # from dotenv import load_dotenv
    # load_dotenv()
    
    # Ensure event loop policy for Windows
    if sys.platform == 'win32':
        import asyncio
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    # Build app
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        print("❌ TELEGRAM_BOT_TOKEN not found. Set it in your environment variables.")
        return

    print("🤖 Aloysia Telegram Bot Starting...")
    
    # We use a helper or just define it here. 
    # To be safe and clean, let's use the bot's own registration logic if it had one, 
    # but since main is the entry point for Render, let's just fix it here.
    application = Application.builder().token(token).build()

    # Add handlers
    application.add_handler(CommandHandler("start", bot.start))
    application.add_handler(CommandHandler("help", bot.help_command))
    application.add_handler(CommandHandler("library", bot.list_library))
    application.add_handler(CommandHandler("link", bot.link_command))
    application.add_handler(CommandHandler("clear", bot.clear_state))
    
    # File handlers
    application.add_handler(MessageHandler(filters.Document.PDF, bot.handle_document))
    
    # Message handlers
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, bot.handle_message))

    # Run
    print("✅ Bot is polling...")
    # NOTE: When running in a thread (via gateway_runner.py), 
    # we MUST disable signal handling (stop_signals=None) 
    # and not close the loop in this thread if handled elsewhere.
    # We set drop_pending_updates=True to clear any legacy webhook/conflict state on start.
    application.run_polling(
        allowed_updates=Update.ALL_TYPES,
        drop_pending_updates=True,
        stop_signals=None,
        close_loop=False
    )

if __name__ == "__main__":
    main()
