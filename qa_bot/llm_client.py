"""QA Bot — thin LLM client using OpenRouter (free models)."""
from __future__ import annotations
import httpx
from .config import get_config

FREE_MODELS = [
    "meta-llama/llama-3.1-8b-instruct:free",
    "google/gemini-2.0-flash-exp:free",
    "mistralai/mistral-7b-instruct:free",
]

SYSTEM_PROMPT = """Ты — опытный QA-инженер и ментор. Ты помогаешь тестировщикам:
- генерировать тест-кейсы из требований
- объяснять концепции тестирования
- разбирать ошибки и баги
- готовиться к собеседованиям по QA

Отвечай чётко, структурировано, с примерами. Используй markdown для форматирования."""


async def ask_llm(prompt: str, system: str = SYSTEM_PROMPT, max_tokens: int = 1500) -> str:
    """Send prompt to LLM via OpenRouter, try free models in order."""
    cfg = get_config()
    if not cfg.openrouter_key:
        return "⚠️ LLM не настроен: задай QA_BOT_OPENROUTER_KEY."

    headers = {
        "Authorization": f"Bearer {cfg.openrouter_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/ouroboros-ai/ouroboros",
        "X-Title": "QA Mentor Bot",
    }
    payload: dict = {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.7,
    }
    models = [cfg.model] + [m for m in FREE_MODELS if m != cfg.model]

    async with httpx.AsyncClient(timeout=60.0) as client:
        for model in models:
            try:
                payload["model"] = model
                resp = await client.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers=headers,
                    json=payload,
                )
                data = resp.json()
                text = data["choices"][0]["message"]["content"].strip()
                if text:
                    return text
            except Exception:
                continue

    return "⚠️ Не удалось получить ответ от LLM. Попробуй позже."
