"""QA Bot — entry point. Run with: python -m qa_bot.bot"""

import logging
import os
import sys

from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
)

from .handlers import (
    cmd_start, cmd_help, cmd_testcase, cmd_bugreport,
    cmd_quiz, cmd_tools, cmd_ask, handle_message,
)

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


def build_app(token: str) -> Application:
    """Build and configure the Telegram application."""
    app = Application.builder().token(token).build()

    # Register command handlers
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(CommandHandler("testcase", cmd_testcase))
    app.add_handler(CommandHandler("bugreport", cmd_bugreport))
    app.add_handler(CommandHandler("quiz", cmd_quiz))
    app.add_handler(CommandHandler("tools", cmd_tools))
    app.add_handler(CommandHandler("ask", cmd_ask))

    # Handle all text messages (fallback)
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    return app


def main() -> None:
    """Start the bot."""
    token = os.environ.get("QA_BOT_TOKEN")
    if not token:
        logger.error("QA_BOT_TOKEN environment variable is required!")
        logger.error("Get your token from @BotFather on Telegram")
        sys.exit(1)

    openrouter_key = os.environ.get("OPENROUTER_API_KEY")
    if not openrouter_key:
        logger.warning("OPENROUTER_API_KEY not set — LLM responses will not work!")
        logger.warning("Get a free key at https://openrouter.ai/")

    logger.info("Starting QA Bot...")
    app = build_app(token)
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
