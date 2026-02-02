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

# Import Aloysia components
try:
    from code.agent import create_agentic_rag, user_id_var
    from code.db import VectorDB
    from code.app import extract_text_with_page_numbers
except ImportError:
    # Fallback for running directly inside code/ folder
    sys.path.append(str(Path(__file__).resolve().parent))
    from agent import create_agentic_rag, user_id_var
    from db import VectorDB
    from app import extract_text_with_page_numbers

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
        self.agent = create_agentic_rag()
        self.db = VectorDB()  # Initialize DB connection

    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Send a message when the command /start is issued."""
        user = update.effective_user
        await update.message.reply_html(
            f"👋 Hi {user.mention_html()}! I'm <b>Aloysia</b>, your Agentic Research Assistant.\n\n"
            f"I can help you review documents and answer research questions.\n\n"
            f"<b>How to use me:</b>\n"
            f"1. 📤 <b>Upload a paper in PDF/docs/txt format</b> directly in this chat to add it to your library.\n"
            f"2. 💬 <b>Ask a question</b> and I'll search your papers for answers.\n"
            f"3. 📚 Use /library to see what you've uploaded.\n\n"
            f"<i>Note: Your papers are private and isolated to this chat.</i>"
        )

    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Send a message when the command /help is issued."""
        help_text = (
            "<b>Available Commands:</b>\n\n"
            "/start - Restart and see instructions\n"
            "/library - List your uploaded papers\n"
            "/clear - Clear conversation history context\n"
            "/help - Show this help message\n\n"
            "<b>Features:</b>\n"
            "• Send any PDF file to add it to your knowledge base\n"
            "• Send text to ask questions about your papers"
        )
        await update.message.reply_html(help_text)

    async def list_library(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """List documents uploaded by the user."""
        chat_id = str(update.effective_chat.id)
        
        await update.message.reply_text("📚 Checking your library...")
        
        try:
            # We can't efficiently query by metadata in Supabase via simple list
            # So we use a dummy search to get unique sources for this user
            # Ideally we'd add a specialized method in db.py, but this works for MVP
            results = self.db.list_all_metadata() 
            
            # Filter manually for this user (since list_all_metadata gets everything)
            # Optimization: In production, add get_user_docs(user_id) to db.py
            user_docs = set()
            for meta in results:
                if str(meta.get("user_id")) == chat_id:
                     user_docs.add(meta.get("source", "Unknown"))
            
            if user_docs:
                doc_list = "\n".join([f"• {doc}" for doc in sorted(user_docs)])
                await update.message.reply_text(f"<b>Your Papers ({len(user_docs)}):</b>\n\n{doc_list}", parse_mode="HTML")
            else:
                await update.message.reply_text("📂 Your library is empty.\n\nUpload a PDF to get started!")
                
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
            await update.message.reply_text("⚠️ Please upload PDF, DOCX or TXT files only.")
            return

        file = await context.bot.get_file(document.file_id)
        file_path = TEMP_DIR / document.file_name
        
        status_msg = await update.message.reply_text(f"📥 Downloading <b>{document.file_name}</b>...", parse_mode="HTML")
        
        try:
            # Download file
            await file.download_to_drive(file_path)
            
            await context.bot.edit_message_text(
                f"⚙️ Processing <b>{document.file_name}</b>...\n<i>Reading and chunking...</i>",
                chat_id=chat_id,
                message_id=status_msg.message_id,
                parse_mode="HTML"
            )
            
            # Process using app.py extraction logic
            chunks_data = extract_text_with_page_numbers(file_path, file_ext)
            
            # Format for VectorDB
            from langchain_core.documents import Document
            
            docs = []
            for chunk in chunks_data:
                docs.append(Document(
                    page_content=chunk["content"],
                    metadata={
                        "source": document.file_name,
                        "page": chunk["page_number"],
                        "user_id": chat_id, # RLS Isolation
                    }
                ))
            
            # Add to DB
            if docs:
                self.db.add_doc(docs, user_id=chat_id)
                msg = f"✅ <b>{document.file_name}</b> added to your library!\n\nYou can now ask questions about it."
            else:
                msg = f"⚠️ Could not extract text from <b>{document.file_name}</b>."
            
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
            
            initial_state = {
                "messages": [HumanMessage(content=query)],
                "quality_passed": True,
                "loop_count": 0,
                "original_query": query,
                "selected_model": "groq",  # Default to fast model for Bot
                "user_id": chat_id  # RLS Enforced Isolation!
            }
            
            # Run Agent
            # Set context variable for RLS
            token = user_id_var.set(chat_id)
            try:
                result = self.agent.invoke(initial_state)
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
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    # Build app
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        print("❌ TELEGRAM_BOT_TOKEN not found. Set it in your environment variables.")
        return

    print("🤖 Aloysia Telegram Bot Starting...")
    application = Application.builder().token(token).build()

    # Add handlers
    application.add_handler(CommandHandler("start", bot.start))
    application.add_handler(CommandHandler("help", bot.help_command))
    application.add_handler(CommandHandler("library", bot.list_library))
    application.add_handler(MessageHandler(filters.Document.PDF, bot.handle_document))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, bot.handle_message))

    # Run
    print("✅ Bot is polling...")
    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()
