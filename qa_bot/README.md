# QA Bot — Telegram-бот для тестировщиков 🧪

Умный помощник для QA-инженеров: генерирует тест-кейсы, помогает с баг-репортами,
проводит теоретические квизы и отвечает на вопросы по тестированию.

## Возможности

| Команда | Что делает |
|---------|------------|
| `/testcase` | Генерирует тест-кейсы по описанию функциональности |
| `/bugreport` | Помогает правильно оформить баг-репорт |
| `/quiz` | Квиз по теории QA (5 вопросов с вариантами) |
| `/tools` | Список инструментов тестировщика (WEB + Mobile) |
| `/ask` | Любой вопрос по тестированию |

## Быстрый старт

### 1. Установка зависимостей

```bash
pip install python-telegram-bot requests zhipuai
```

### 2. Создание бота

1. Откройте [@BotFather](https://t.me/BotFather) в Telegram
2. Отправьте `/newbot`
3. Дайте боту имя и username (например, `@MyQABot`)
4. Скопируйте полученный токен

### 3. Получение ключа LLM (OpenRouter и/или Zhipu AI)

#### Вариант A: OpenRouter (как было)

1. Зарегистрируйтесь на [openrouter.ai](https://openrouter.ai/)
2. Создайте API-ключ (есть бесплатный лимит)
3. Скопируйте ключ

#### Вариант B: Zhipu AI (ChatGLM)

1. Зарегистрируйтесь на [open.bigmodel.cn](https://open.bigmodel.cn/)
2. Создайте API key
3. Скопируйте ключ

### 4. Запуск

```bash
# Установите переменные окружения
export QA_BOT_TOKEN="ваш_токен_от_BotFather"

# Можно использовать OpenRouter:
export OPENROUTER_API_KEY="ваш_ключ_от_openrouter"

# И/или Zhipu AI (ChatGLM):
export ZHIPUAI_API_KEY="ваш_ключ_от_zhipu"

# (опционально) цепочка fallback-моделей Zhipu, по умолчанию: glm-4,glm-4-flash
export QA_BOT_ZHIPU_MODELS="glm-4,glm-4-flash"

# Запустите бота
python -m qa_bot.bot
```

На Windows:
```cmd
set QA_BOT_TOKEN=ваш_токен_от_BotFather
set OPENROUTER_API_KEY=ваш_ключ_от_openrouter
set ZHIPUAI_API_KEY=ваш_ключ_от_zhipu
set QA_BOT_ZHIPU_MODELS=glm-4,glm-4-flash
python -m qa_bot\bot.py
```

### 5. Запуск в Google Colab

```python
import os
os.environ["QA_BOT_TOKEN"] = "ваш_токен"
os.environ["OPENROUTER_API_KEY"] = "ваш_ключ"

# Установите библиотеки
!pip install python-telegram-bot requests

# Запустите
import subprocess
subprocess.Popen(["python", "-m", "qa_bot.bot"])
```

## Структура проекта

```
qa_bot/
├── __init__.py        — пакет
├── bot.py             — точка входа, регистрация хендлеров
├── config.py          — конфигурация из env
├── handlers.py        — обработчики команд и сообщений
├── llm_client.py      — запросы к бесплатным LLM (OpenRouter)
├── quiz.py            — квиз-режим с банком вопросов
└── README.md          — документация
```

## Используемые модели (бесплатные)

Бот автоматически выбирает лучшую доступную модель:
1. `google/gemini-2.0-flash-exp:free`
2. `meta-llama/llama-3.1-8b-instruct:free`
3. `microsoft/phi-3-mini-128k-instruct:free`
4. `qwen/qwen-2-7b-instruct:free`

Если одна модель недоступна — автоматически переключается на следующую.

### Fallback на Zhipu AI (ChatGLM)

Если все бесплатные модели OpenRouter недоступны (лимит/квота/ошибка провайдера), бот автоматически попробует Zhipu AI,
если задан `ZHIPUAI_API_KEY` (или `QA_BOT_ZHIPU_API_KEY`).

Цепочку моделей Zhipu можно настроить через `QA_BOT_ZHIPU_MODELS` (по умолчанию: `glm-4,glm-4-flash`).

## Монетизация

Варианты для монетизации:

- **Freemium**: 5 бесплатных запросов в день, затем подписка
- **Telegram Stars**: встроенная оплата в Telegram
- **ЮКасса / LiqPay**: для аудитории в СНГ
- **Stripe**: для международной аудитории

Целевая аудитория: джуниор и мидл QA-инженеры в СНГ (~50,000+ активных специалистов).

## Развёртывание (production)

Для работы 24/7 рекомендуется:
- **Railway.app** — бесплатный тариф до 500 часов/месяц
- **Render.com** — бесплатный тариф
- **VPS** — ~$5/месяц на DigitalOcean или Hetzner

## Лицензия

MIT
