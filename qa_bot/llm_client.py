"""QA Bot — LLM client with fallback (OpenRouter -> Zhipu AI)."""

import os
import time
import logging
from typing import Optional

import requests

logger = logging.getLogger(__name__)

# Optional Zhipu AI SDK (ChatGLM)
try:
    from zhipuai import ZhipuAI  # type: ignore
except Exception:  # pragma: no cover
    ZhipuAI = None  # type: ignore

# Free models in priority order (OpenRouter free tier)
FREE_MODELS = [
    "google/gemini-2.0-flash-exp:free",
    "meta-llama/llama-3.1-8b-instruct:free",
    "microsoft/phi-3-mini-128k-instruct:free",
    "qwen/qwen-2-7b-instruct:free",
]

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_API_KEY = (
    os.environ.get("QA_BOT_OPENROUTER_KEY")
    or os.environ.get("OPENROUTER_API_KEY")
    or ""
)

ZHIPU_API_KEY = (
    os.environ.get("QA_BOT_ZHIPU_API_KEY")
    or os.environ.get("ZHIPUAI_API_KEY")
    or os.environ.get("ZHIPU_API_KEY")
    or ""
)
ZHIPU_MODELS = [
    m.strip()
    for m in (os.environ.get("QA_BOT_ZHIPU_MODELS") or "glm-4,glm-4-flash").split(",")
    if m.strip()
]

# QA Bot system prompt — short, focused, professional
QA_SYSTEM_PROMPT = """Ты — опытный QA-инженер и ментор. Отвечаешь на русском языке.

Твои специализации:
- Написание тест-кейсов (ручное и автоматизированное тестирование)
- Тест-планирование и стратегии тестирования
- API-тестирование (Postman, REST, GraphQL)
- UI-автоматизация (Selenium, Playwright, Cypress, Appium)
- Мобильное тестирование (iOS, Android)
- Баг-репорты и работа с Jira
- Теория тестирования (виды тестов, пирамида тестирования, SDLC)

Стиль ответов:
- Конкретно и по делу
- С примерами кода когда уместно
- Без лишней воды
- Используй markdown для форматирования"""


def ask_llm(messages: list[dict], system: str = QA_SYSTEM_PROMPT,
            max_tokens: int = 1500) -> str:
    """Send messages to LLM with automatic fallback (OpenRouter -> Zhipu AI)."""
    if not OPENROUTER_API_KEY and not ZHIPU_API_KEY:
        return (
            "❌ Не задан ни один ключ LLM.\n"
            "Добавьте `OPENROUTER_API_KEY` (или `QA_BOT_OPENROUTER_KEY`) "
            "и/или `ZHIPUAI_API_KEY` (или `QA_BOT_ZHIPU_API_KEY`) в переменные окружения."
        )

    full_messages = [{"role": "system", "content": system}] + messages

    # 1) OpenRouter free-tier models (if key provided)
    if OPENROUTER_API_KEY:
        for model in FREE_MODELS:
            try:
                resp = requests.post(
                    OPENROUTER_API_URL,
                    headers={
                        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                        "HTTP-Referer": "https://github.com/ouroboros-ai",
                        "X-Title": "QA-Bot",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": model,
                        "messages": full_messages,
                        "max_tokens": max_tokens,
                        "temperature": 0.7,
                    },
                    timeout=30,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    content = data["choices"][0]["message"]["content"]
                    if content and len(content.strip()) > 10:
                        logger.info("LLM response from OpenRouter model: %s", model)
                        return content.strip()

                # Non-200 is typically quota/rate/temporary failure — try next model/provider.
                try:
                    err_preview = resp.text[:300]
                except Exception:
                    err_preview = "<unreadable>"
                logger.warning(
                    "OpenRouter non-200 (%s) for model %s: %s",
                    resp.status_code,
                    model,
                    err_preview,
                )
            except requests.Timeout:
                logger.warning("OpenRouter timeout from model %s, trying next", model)
            except Exception as exc:
                logger.warning("OpenRouter error from model %s: %s", model, exc)
            time.sleep(0.5)

    # 2) Zhipu AI (ChatGLM) fallback chain (if key provided)
    if ZHIPU_API_KEY:
        if ZhipuAI is None:
            return (
                "❌ Задан Zhipu API key, но не установлен пакет `zhipuai`.\n"
                "Установите зависимости и перезапустите: `pip install zhipuai`."
            )
        try:
            client = ZhipuAI(api_key=ZHIPU_API_KEY)
        except Exception as exc:
            logger.warning("Failed to init ZhipuAI client: %s", exc)
            client = None

        if client is not None:
            for model in ZHIPU_MODELS:
                try:
                    resp = client.chat.completions.create(
                        model=model,
                        messages=full_messages,
                        max_tokens=max_tokens,
                        temperature=0.7,
                    )
                    content = resp.choices[0].message.content
                    if content and len(content.strip()) > 10:
                        logger.info("LLM response from Zhipu model: %s", model)
                        return content.strip()
                except Exception as exc:
                    logger.warning("Zhipu error from model %s: %s", model, exc)
                time.sleep(0.5)

    return "⚠️ Все модели временно недоступны. Попробуйте через минуту."


def ask_with_context(user_text: str, history: list[dict] | None = None) -> str:
    """Ask LLM with optional conversation history."""
    messages = history or []
    messages = messages[-6:]  # Keep last 6 turns for context
    messages.append({"role": "user", "content": user_text})
    return ask_llm(messages)
