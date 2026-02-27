@echo off
REM Скрипт для запуска QA Fast Learning Bot на Windows
REM Запускать: run.bat

set QA_BOT_TOKEN=8413021697:AAE7fw3i11BpHXJTnXPFtR5YMTrTH7gk3c8
set OPENROUTER_API_KEY=твой_ключ_openrouter_сюда

REM Запуск бота
python -m qa_bot.bot
pause
