"""QA Bot — Telegram message and command handlers."""

import logging
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ContextTypes

from .llm_client import ask_with_context
from .quiz import QuizSession

logger = logging.getLogger(__name__)

# --- Tools reference (compact, for /tools command) ---
TOOLS_TEXT = """🧰 *Инструменты QA-тестировщика*

*🌐 WEB — Ручное тестирование*
• [Chrome DevTools](https://developer.chrome.com/docs/devtools/) — инспектор, сеть, консоль
• [Burp Suite Community](https://portswigger.net/burp/communitydownload) — HTTP-перехват, анализ API

*🤖 WEB — Автоматизация UI*
• [Playwright](https://playwright.dev/) — современный, быстрый, мультибраузерный
• [Selenium](https://www.selenium.dev/) — классика, огромное сообщество
• [Cypress](https://www.cypress.io/) — только браузер, но очень удобный

*📡 API-тестирование*
• [Postman](https://www.postman.com/) — GUI для REST/GraphQL/gRPC
• [Insomnia](https://insomnia.rest/) — лёгкая альтернатива Postman
• [k6](https://k6.io/) — нагрузочное тестирование через JS

*📱 Mobile*
• [Appium](https://appium.io/) — автоматизация iOS и Android
• [Charles Proxy](https://www.charlesproxy.com/) — перехват трафика на мобиле
• [Android Studio Emulator](https://developer.android.com/studio) — эмулятор Android

*📋 Тест-менеджмент*
• [TestRail](https://www.testrail.com/) — профессиональный трекер тест-кейсов
• [Allure](https://allurereport.org/) — красивые отчёты для автотестов
• [Jira](https://www.atlassian.com/software/jira) — баг-трекер и таск-менеджер

Нужны инструкции по конкретному инструменту? Просто спроси! 💬"""


# --- Generate test cases prompt ---
TESTCASE_SYSTEM = """Ты — QA-инженер. Получаешь описание функциональности и пишешь тест-кейсы.

Формат каждого тест-кейса:
**TC-N: Название**
- Предусловие: ...
- Шаги:
  1. ...
  2. ...
- Ожидаемый результат: ...
- Тип: (positive/negative/boundary)

Пиши на русском. Покрой позитивные, негативные сценарии и граничные значения."""

BUG_REPORT_SYSTEM = """Ты — QA-инженер. Помогаешь правильно оформить баг-репорт.

Формат:
**Заголовок:** [краткое, точное описание]
**Severity:** Critical / High / Medium / Low
**Priority:** High / Medium / Low
**Шаги воспроизведения:**
1. ...
**Фактический результат:** ...
**Ожидаемый результат:** ...
**Окружение:** [OS, браузер, версия приложения]
**Вложения:** [скриншот/лог — если есть]"""


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /start command."""
    text = (
        "👋 Привет! Я QA-бот — твой помощник в тестировании.\n\n"
        "Что я умею:\n"
        "🧪 /testcase — сгенерировать тест-кейсы\n"
        "🐛 /bugreport — помочь с баг-репортом\n"
        "❓ /quiz — квиз по теории QA\n"
        "🧰 /tools — инструменты тестировщика\n"
        "💬 /ask — задать вопрос по QA\n"
        "ℹ️ /help — все команды\n\n"
        "Просто напиши мне вопрос — и я отвечу!"
    )
    await update.message.reply_text(text)


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /help command."""
    text = (
        "📖 *Команды бота:*\n\n"
        "/testcase — генерация тест-кейсов\n"
        "   _Пришли описание функции — получи тест-кейсы_\n\n"
        "/bugreport — баг-репорт\n"
        "   _Опиши баг — получи правильно оформленный репорт_\n\n"
        "/quiz — теория QA\n"
        "   _5 вопросов с вариантами ответов_\n\n"
        "/tools — инструменты\n"
        "   _Selenium, Playwright, Postman, Appium и другие_\n\n"
        "/ask — любой вопрос по QA\n"
        "   _Например: \"Чем regression отличается от retesting?\"_\n\n"
        "Или просто пиши — без команды 💬"
    )
    await update.message.reply_text(text, parse_mode="Markdown")


async def cmd_testcase(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /testcase command."""
    args = " ".join(context.args) if context.args else ""
    if args:
        await update.message.reply_text("⏳ Генерирую тест-кейсы...")
        response = ask_with_context(
            f"Напиши тест-кейсы для следующей функциональности:\n\n{args}",
            history=[]
        )
        await update.message.reply_text(response)
    else:
        await update.message.reply_text(
            "✍️ Опиши функциональность, для которой нужны тест-кейсы.\n\n"
            "Например:\n"
            "/testcase Форма регистрации: поля email, пароль, подтверждение пароля. "
            "Email должен быть уникальным. Пароль — минимум 8 символов."
        )
    # Store mode
    context.user_data["mode"] = "testcase"


async def cmd_bugreport(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /bugreport command."""
    args = " ".join(context.args) if context.args else ""
    if args:
        await update.message.reply_text("⏳ Оформляю баг-репорт...")
        response = ask_with_context(
            f"Оформи баг-репорт по описанию:\n\n{args}",
        )
        await update.message.reply_text(response)
    else:
        context.user_data["mode"] = "bugreport"
        await update.message.reply_text(
            "🐛 Опиши баг — что произошло, при каких условиях, что ожидалось.\n\n"
            "Пример:\n"
            "При регистрации с email 'test@test.com' и паролем '123' система "
            "показывает ошибку 500 вместо сообщения о слишком коротком пароле."
        )


async def cmd_quiz(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /quiz command — start a new quiz session."""
    session = QuizSession()
    context.user_data["quiz"] = session
    context.user_data["mode"] = "quiz"
    question = session.current_question()
    await update.message.reply_text(
        "🎯 *Квиз по теории QA*\n\n" + question,
        parse_mode="Markdown"
    )


async def cmd_tools(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /tools command."""
    await update.message.reply_text(TOOLS_TEXT, parse_mode="Markdown",
                                    disable_web_page_preview=True)


async def cmd_ask(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /ask command."""
    args = " ".join(context.args) if context.args else ""
    if args:
        await update.message.reply_text("🤔 Думаю...")
        response = ask_with_context(args)
        await update.message.reply_text(response)
    else:
        context.user_data["mode"] = "ask"
        await update.message.reply_text(
            "💬 Задай вопрос по QA-тестированию.\n\n"
            "Например: «В чём разница между smoke и sanity тестами?»"
        )


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle free-form text messages."""
    text = update.message.text.strip()
    mode = context.user_data.get("mode", "ask")

    # Quiz mode — process answer
    if mode == "quiz":
        session: QuizSession | None = context.user_data.get("quiz")
        if session and not session.is_finished:
            response = session.answer(text)
            if session.is_finished:
                context.user_data.pop("mode", None)
                context.user_data.pop("quiz", None)
            await update.message.reply_text(response, parse_mode="Markdown")
            return

    # Testcase mode
    if mode == "testcase":
        await update.message.reply_text("⏳ Генерирую тест-кейсы...")
        response = ask_with_context(
            f"Напиши тест-кейсы для:\n\n{text}",
        )
        context.user_data.pop("mode", None)
        await update.message.reply_text(response)
        return

    # Bugreport mode
    if mode == "bugreport":
        await update.message.reply_text("⏳ Оформляю баг-репорт...")
        response = ask_with_context(
            f"Оформи баг-репорт:\n\n{text}",
        )
        context.user_data.pop("mode", None)
        await update.message.reply_text(response)
        return

    # Default: general QA question
    await update.message.reply_text("🤔 Думаю...")
    history = context.user_data.get("history", [])
    response = ask_with_context(text, history=history)

    # Update history (keep last 6 turns)
    history.append({"role": "user", "content": text})
    history.append({"role": "assistant", "content": response})
    context.user_data["history"] = history[-12:]

    await update.message.reply_text(response)
