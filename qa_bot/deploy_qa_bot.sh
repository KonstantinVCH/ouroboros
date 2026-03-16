#!/usr/bin/env bash
# Deploy QA Bot as a systemd service on VPS.
# Usage: bash qa_bot/deploy_qa_bot.sh
# Run from repo root on the VPS server.

set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
SERVICE_NAME="qa-bot"
SERVICE_FILE="/etc/systemd/system/${SERVICE_NAME}.service"
ENV_FILE="/opt/ouroboros/.env.qa_bot"
PYTHON="python3"

echo "=== QA Bot Deploy ==="
echo "Repo: $REPO_DIR"

# ---- 1. Check env file ----
if [[ ! -f "$ENV_FILE" ]]; then
    echo "⚠️  Creating env template: $ENV_FILE"
    cat > "$ENV_FILE" << 'EOF'
QA_BOT_TOKEN=your_telegram_bot_token_here
ANTHROPIC_API_KEY=your_anthropic_key_here
# Optional fallbacks:
# OPENROUTER_API_KEY=...
# ZHIPUAI_API_KEY=...
QA_BOT_DB_PATH=/opt/ouroboros/qa_bot_history.db
EOF
    echo "   → Edit $ENV_FILE with your tokens, then re-run this script."
    exit 1
fi

# ---- 2. Install Python dependencies ----
echo "=== Installing dependencies ==="
$PYTHON -m pip install -q \
    "python-telegram-bot>=20.0" \
    "requests>=2.31" \
    "anthropic>=0.25" \
    "zhipuai>=2.0" 2>&1 | tail -5

# ---- 3. Write systemd service ----
echo "=== Writing systemd service ==="
cat > "$SERVICE_FILE" << EOF
[Unit]
Description=QA Bot — Telegram QA mentor bot
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=${REPO_DIR}
EnvironmentFile=${ENV_FILE}
ExecStart=${PYTHON} -m qa_bot.bot
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal
SyslogIdentifier=qa-bot

[Install]
WantedBy=multi-user.target
EOF

# ---- 4. Enable and start ----
echo "=== Enabling and starting qa-bot service ==="
systemctl daemon-reload
systemctl enable "$SERVICE_NAME"
systemctl restart "$SERVICE_NAME"
sleep 2
systemctl status "$SERVICE_NAME" --no-pager

echo ""
echo "✅ QA Bot deployed!"
echo "   Logs:    journalctl -u qa-bot -f"
echo "   Status:  systemctl status qa-bot"
echo "   Restart: systemctl restart qa-bot"
