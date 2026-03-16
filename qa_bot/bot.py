"""QA Bot — entry point. Run with: python -m qa_bot.bot"""

from __future__ import annotations

import logging
import os
import sys

from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    MessageHandler,
    filters,
)

from .handlers import (
    cmd_start, cmd_help, cmd_testcase, cmd_bugreport,
    cmd_quiz, cmd_interview, interview_level_callback,
    cmd_roadmap, cmd_tools, cmd_ask, cmd_reset,
    handle_message,
)
from .storage import init_db

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


def build_app(token: str) -> Application:
    """Build and configure the Telegram application."""
    app = Application.builder().token(token).build()

    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(CommandHandler("testcase", cmd_testcase))
    app.add_handler(CommandHandler("bugreport", cmd_bugreport))
    app.add_handler(CommandHandler("quiz", cmd_quiz))
    app.add_handler(CommandHandler("interview", cmd_interview))
    app.add_handler(CommandHandler("roadmap", cmd_roadmap))
    app.add_handler(CommandHandler("tools", cmd_tools))
    app.add_handler(CommandHandler("ask", cmd_ask))
    app.add_handler(CommandHandler("reset", cmd_reset))

    # Inline keyboard callbacks
    app.add_handler(CallbackQueryHandler(
        interview_level_callback, pattern="^interview_"
    ))

    # Free-form text messages (fallback)
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    return app


def main() -> None:
    """Start the bot."""
    token = os.environ.get("QA_BOT_TOKEN")
    if not token:
        logger.error("QA_BOT_TOKEN environment variable is required!")
        sys.exit(1)

    if not os.environ.get("ANTHROPIC_API_KEY"):
        if not os.environ.get("OPENROUTER_API_KEY") and not os.environ.get("ZHIPUAI_API_KEY"):
            logger.warning("No LLM API key set — responses will not work!")
        else:
            logger.info("Using OpenRouter/Zhipu as LLM provider")
    else:
        logger.info("Using Anthropic as primary LLM provider")

    # Initialize persistent storage
    init_db()

    logger.info("Starting QA Bot...")
    app = build_app(token)
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
