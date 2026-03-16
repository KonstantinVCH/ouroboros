"""QA Bot — Telegram message and command handlers."""

from __future__ import annotations

import logging
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ContextTypes

from .llm_client import ask_with_context, ask_llm
from .quiz import QuizSession
from .interview import InterviewSession, LEVELS
from .storage import get_history, add_message, clear_history

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Static content
# ---------------------------------------------------------------------------
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

ROADMAP_PROMPT = """Составь детальный roadmap обучения QA-инженера с нуля.

Структура:
1. **Фаза 1 (1-2 мес): Основы** — теория тестирования, тест-дизайн
2. **Фаза 2 (2-3 мес): Практика** — ручное тестирование, баг-репорты, Jira
3. **Фаза 3 (3-4 мес): API** — Postman, REST API, HTTP
4. **Фаза 4 (4-6 мес): Автоматизация** — Python/JavaScript, Selenium или Playwright
5. **Фаза 5 (6-9 мес): Специализация** — выбор направления (web/mobile/load)

Для каждой фазы укажи:
- Конкретные темы для изучения
- Рекомендуемые ресурсы (книги, курсы, практика)
- Критерии завершения фазы (что должен уметь)

Отвечай на русском, структурированно."""


# ---------------------------------------------------------------------------
# Command handlers
# ---------------------------------------------------------------------------

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /start command."""
    text = (
        "👋 Привет! Я QA-бот — твой помощник в тестировании.\n\n"
        "Что я умею:\n"
        "🧪 /testcase — сгенерировать тест-кейсы\n"
        "🐛 /bugreport — помочь с баг-репортом\n"
        "❓ /quiz — квиз по теории QA\n"
        "🎤 /interview — симуляция технического интервью\n"
        "🗺 /roadmap — план обучения QA с нуля\n"
        "🧰 /tools — инструменты тестировщика\n"
        "💬 /ask — задать вопрос по QA\n"
        "🔄 /reset — сбросить историю диалога\n"
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
        "/interview — техническое интервью\n"
        "   _Выбери уровень (junior/middle/senior) и отвечай на вопросы_\n\n"
        "/roadmap — план обучения\n"
        "   _Детальный путь от нуля до QA-специалиста_\n\n"
        "/tools — инструменты\n"
        "   _Selenium, Playwright, Postman, Appium и другие_\n\n"
        "/ask — любой вопрос по QA\n"
        "   _Например: «Чем regression отличается от retesting?»_\n\n"
        "/reset — сбросить историю\n"
        "   _Начать диалог с чистого листа_\n\n"
        "Или просто пиши — без команды 💬"
    )
    await update.message.reply_text(text, parse_mode="Markdown")


async def cmd_testcase(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /testcase command."""
    args = " ".join(context.args) if context.args else ""
    if args:
        await update.message.reply_text("⏳ Генерирую тест-кейсы...")
        response = ask_llm(
            messages=[{"role": "user", "content": f"Напиши тест-кейсы для:\n\n{args}"}],
            system=TESTCASE_SYSTEM,
        )
        await update.message.reply_text(response)
    else:
        context.user_data["mode"] = "testcase"
        await update.message.reply_text(
            "✍️ Опиши функциональность, для которой нужны тест-кейсы.\n\n"
            "Например:\n"
            "/testcase Форма регистрации: поля email, пароль, подтверждение пароля. "
            "Email должен быть уникальным. Пароль — минимум 8 символов."
        )


async def cmd_bugreport(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /bugreport command."""
    args = " ".join(context.args) if context.args else ""
    if args:
        await update.message.reply_text("⏳ Оформляю баг-репорт...")
        response = ask_llm(
            messages=[{"role": "user", "content": f"Оформи баг-репорт:\n\n{args}"}],
            system=BUG_REPORT_SYSTEM,
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


async def cmd_interview(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /interview command — show level selection."""
    keyboard = [
        [
            InlineKeyboardButton("👶 Junior", callback_data="interview_junior"),
            InlineKeyboardButton("🧑‍💻 Middle", callback_data="interview_middle"),
            InlineKeyboardButton("🧙 Senior", callback_data="interview_senior"),
        ]
    ]
    markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text(
        "🎤 *Симуляция технического интервью*\n\n"
        "Я буду задавать вопросы как настоящий интервьюер.\n"
        "После 5 вопросов получишь оценку и рекомендации.\n\n"
        "Выбери уровень:",
        parse_mode="Markdown",
        reply_markup=markup,
    )


async def interview_level_callback(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    """Handle interview level selection."""
    query = update.callback_query
    await query.answer()

    level = query.data.replace("interview_", "")
    if level not in LEVELS:
        await query.edit_message_text("❌ Неизвестный уровень")
        return

    await query.edit_message_text(
        f"🎤 Начинаю интервью на уровень *{LEVELS[level]}*...\n\n"
        "_Готовься!_",
        parse_mode="Markdown",
    )

    session = InterviewSession(level=level)
    context.user_data["interview"] = session
    context.user_data["mode"] = "interview"

    first_message = session.start()
    await query.message.reply_text(first_message)


async def cmd_roadmap(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /roadmap command."""
    await update.message.reply_text("🗺 Составляю план обучения... (займёт ~30 сек)")
    response = ask_llm(
        messages=[{"role": "user", "content": ROADMAP_PROMPT}],
        max_tokens=2000,
    )
    await update.message.reply_text(response, parse_mode="Markdown")


async def cmd_tools(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /tools command."""
    await update.message.reply_text(
        TOOLS_TEXT, parse_mode="Markdown", disable_web_page_preview=True
    )


async def cmd_ask(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /ask command."""
    args = " ".join(context.args) if context.args else ""
    if args:
        await update.message.reply_text("🤔 Думаю...")
        user_id = update.effective_user.id
        history = get_history(user_id)
        response = ask_with_context(args, history=history)
        add_message(user_id, "user", args)
        add_message(user_id, "assistant", response)
        await update.message.reply_text(response)
    else:
        context.user_data["mode"] = "ask"
        await update.message.reply_text(
            "💬 Задай вопрос по QA-тестированию.\n\n"
            "Например: «В чём разница между smoke и sanity тестами?»"
        )


async def cmd_reset(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Clear conversation history."""
    user_id = update.effective_user.id
    clear_history(user_id)
    context.user_data.clear()
    await update.message.reply_text(
        "🔄 История диалога очищена. Начинаем с чистого листа!"
    )


# ---------------------------------------------------------------------------
# Message handler
# ---------------------------------------------------------------------------

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle free-form text messages."""
    text = update.message.text.strip()
    user_id = update.effective_user.id
    mode = context.user_data.get("mode", "ask")

    # Interview mode
    if mode == "interview":
        session: InterviewSession | None = context.user_data.get("interview")
        if session and not session.is_finished:
            response = session.answer(text)
            if session.is_finished:
                context.user_data.pop("mode", None)
                context.user_data.pop("interview", None)
            await update.message.reply_text(response, parse_mode="Markdown")
            return

    # Quiz mode
    if mode == "quiz":
        quiz_session: QuizSession | None = context.user_data.get("quiz")
        if quiz_session and not quiz_session.is_finished:
            response = quiz_session.answer(text)
            if quiz_session.is_finished:
                context.user_data.pop("mode", None)
                context.user_data.pop("quiz", None)
            await update.message.reply_text(response, parse_mode="Markdown")
            return

    # Testcase mode
    if mode == "testcase":
        await update.message.reply_text("⏳ Генерирую тест-кейсы...")
        response = ask_llm(
            messages=[{"role": "user", "content": f"Напиши тест-кейсы для:\n\n{text}"}],
            system=TESTCASE_SYSTEM,
        )
        context.user_data.pop("mode", None)
        await update.message.reply_text(response)
        return

    # Bugreport mode
    if mode == "bugreport":
        await update.message.reply_text("⏳ Оформляю баг-репорт...")
        response = ask_llm(
            messages=[{"role": "user", "content": f"Оформи баг-репорт:\n\n{text}"}],
            system=BUG_REPORT_SYSTEM,
        )
        context.user_data.pop("mode", None)
        await update.message.reply_text(response)
        return

    # Default: general QA question with persistent history
    await update.message.reply_text("🤔 Думаю...")
    history = get_history(user_id)
    response = ask_with_context(text, history=history)

    add_message(user_id, "user", text)
    add_message(user_id, "assistant", response)

    await update.message.reply_text(response)
