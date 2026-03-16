"""QA Bot — LLM client.

Priority:
  1. Anthropic (claude-3-5-haiku-20241022) — if ANTHROPIC_API_KEY is set
  2. OpenRouter free tier — if OPENROUTER_API_KEY is set
  3. Zhipu AI (ChatGLM) — if ZHIPUAI_API_KEY is set
"""

from __future__ import annotations

import logging
import os
import time
from typing import Optional

import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL = os.environ.get("QA_BOT_ANTHROPIC_MODEL", "claude-3-5-haiku-20241022")
ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"

OPENROUTER_API_KEY = (
    os.environ.get("QA_BOT_OPENROUTER_KEY")
    or os.environ.get("OPENROUTER_API_KEY")
    or ""
)
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
FREE_MODELS = [
    "google/gemini-2.0-flash-exp:free",
    "meta-llama/llama-3.1-8b-instruct:free",
    "microsoft/phi-3-mini-128k-instruct:free",
    "qwen/qwen-2-7b-instruct:free",
]

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

# Optional Zhipu AI SDK (ChatGLM)
try:
    from zhipuai import ZhipuAI  # type: ignore
except Exception:
    ZhipuAI = None  # type: ignore

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
QA_SYSTEM_PROMPT = """Ты — опытный QA-инженер и ментор. Отвечаешь на русском языке.

Твои специализации:
- Написание тест-кейсов (ручное и автоматизированное тестирование)
- Тест-планирование и стратегии тестирования
- API-тестирование (Postman, REST, GraphQL)
- UI-автоматизация (Selenium, Playwright, Cypress, Appium)
- Мобильное тестирование (iOS, Android)
- Баг-репорты и работа с Jira
- Теория тестирования (виды тестов, пирамида тестирования, SDLC)
- Подготовка к техническому интервью на позицию QA

Стиль ответов:
- Конкретно и по делу
- С примерами когда уместно
- Без лишней воды
- Используй markdown для форматирования"""


# ---------------------------------------------------------------------------
# Provider implementations
# ---------------------------------------------------------------------------

def _call_anthropic(messages: list[dict], system: str, max_tokens: int) -> Optional[str]:
    """Call Anthropic API directly."""
    if not ANTHROPIC_API_KEY:
        return None
    try:
        resp = requests.post(
            ANTHROPIC_API_URL,
            headers={
                "x-api-key": ANTHROPIC_API_KEY,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": ANTHROPIC_MODEL,
                "system": system,
                "messages": messages,
                "max_tokens": max_tokens,
            },
            timeout=60,
        )
        if resp.status_code == 200:
            data = resp.json()
            content = data["content"][0]["text"]
            if content and len(content.strip()) > 10:
                logger.info("LLM response from Anthropic: %s", ANTHROPIC_MODEL)
                return content.strip()
        logger.warning("Anthropic non-200: %s — %s", resp.status_code, resp.text[:300])
    except requests.Timeout:
        logger.warning("Anthropic timeout")
    except Exception as exc:
        logger.warning("Anthropic error: %s", exc)
    return None


def _call_openrouter(messages: list[dict], system: str, max_tokens: int) -> Optional[str]:
    """Try OpenRouter free-tier models in order."""
    if not OPENROUTER_API_KEY:
        return None
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
                    logger.info("LLM response from OpenRouter: %s", model)
                    return content.strip()
            logger.warning(
                "OpenRouter non-200 (%s) for %s: %s",
                resp.status_code, model, resp.text[:300],
            )
        except requests.Timeout:
            logger.warning("OpenRouter timeout for model %s", model)
        except Exception as exc:
            logger.warning("OpenRouter error for model %s: %s", model, exc)
        time.sleep(0.5)
    return None


def _call_zhipu(messages: list[dict], system: str, max_tokens: int) -> Optional[str]:
    """Try Zhipu AI (ChatGLM) models."""
    if not ZHIPU_API_KEY or ZhipuAI is None:
        return None
    full_messages = [{"role": "system", "content": system}] + messages
    try:
        client = ZhipuAI(api_key=ZHIPU_API_KEY)
    except Exception as exc:
        logger.warning("Failed to init ZhipuAI client: %s", exc)
        return None
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
                logger.info("LLM response from Zhipu: %s", model)
                return content.strip()
        except Exception as exc:
            logger.warning("Zhipu error for model %s: %s", model, exc)
        time.sleep(0.5)
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def ask_llm(
    messages: list[dict],
    system: str = QA_SYSTEM_PROMPT,
    max_tokens: int = 1500,
) -> str:
    """Call LLM with automatic provider fallback: Anthropic → OpenRouter → Zhipu."""
    if not ANTHROPIC_API_KEY and not OPENROUTER_API_KEY and not ZHIPU_API_KEY:
        return (
            "❌ Не задан ни один ключ LLM.\n"
            "Добавьте ANTHROPIC_API_KEY, OPENROUTER_API_KEY или ZHIPUAI_API_KEY."
        )

    result = (
        _call_anthropic(messages, system, max_tokens)
        or _call_openrouter(messages, system, max_tokens)
        or _call_zhipu(messages, system, max_tokens)
    )
    return result or "⚠️ Все модели временно недоступны. Попробуйте через минуту."


def ask_with_context(user_text: str, history: list[dict] | None = None) -> str:
    """Ask LLM with optional conversation history (last 6 turns)."""
    messages = (history or [])[-12:]
    messages = messages + [{"role": "user", "content": user_text}]
    return ask_llm(messages)
