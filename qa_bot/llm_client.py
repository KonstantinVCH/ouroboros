"""QA Bot — LLM client using OpenRouter free models with fallback."""

import os
import time
import logging
from typing import Optional

import requests

logger = logging.getLogger(__name__)

# Free models in priority order (OpenRouter free tier)
FREE_MODELS = [
    "google/gemini-2.0-flash-exp:free",
    "meta-llama/llama-3.1-8b-instruct:free",
    "microsoft/phi-3-mini-128k-instruct:free",
    "qwen/qwen-2-7b-instruct:free",
]

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")

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
    """Send messages to LLM with automatic fallback through free models."""
    if not OPENROUTER_API_KEY:
        return "❌ OPENROUTER_API_KEY не задан. Добавьте ключ в переменные окружения."

    full_messages = [{"role": "system", "content": system}] + messages

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
                    logger.info("LLM response from model: %s", model)
                    return content.strip()
        except requests.Timeout:
            logger.warning("Timeout from model %s, trying next", model)
        except Exception as exc:
            logger.warning("Error from model %s: %s", model, exc)
        time.sleep(0.5)

    return "⚠️ Все модели временно недоступны. Попробуйте через минуту."


def ask_with_context(user_text: str, history: list[dict] | None = None) -> str:
    """Ask LLM with optional conversation history."""
    messages = history or []
    messages = messages[-6:]  # Keep last 6 turns for context
    messages.append({"role": "user", "content": user_text})
    return ask_llm(messages)
