#!/usr/bin/env bash
# deploy_vps.sh — деплой Ouroboros на Ubuntu VPS (64.188.72.89) как systemd-сервис
# Запуск: bash deploy_vps.sh

set -euo pipefail

VPS="root@64.188.72.89"
ENV_FILE="/c/Users/morco/repo/ouroboros/.env"

log() { echo "[deploy] $*"; }

# 1. Загружаем .env на сервер
log "Uploading .env..."
scp -o StrictHostKeyChecking=no "$ENV_FILE" "${VPS}:/tmp/ouroboros_deploy.env"

# 2. Запускаем установку на сервере
log "Running remote setup on ${VPS}..."
ssh -o StrictHostKeyChecking=no "$VPS" bash << 'REMOTE'
set -e

INSTALL_DIR="/opt/ouroboros"
REPO_URL="https://github.com/KonstantinVCH/ouroboros.git"
BRANCH="ouroboros"
SERVICE="ouroboros-bot"
LOG_DIR="/var/log/ouroboros"

echo "[vps] Installing system deps..."
apt-get install -y -q python3-pip git curl 2>/dev/null | tail -2

echo "[vps] Cloning/updating repo..."
if [ -d "${INSTALL_DIR}/.git" ]; then
    cd "${INSTALL_DIR}"
    git fetch origin
    git checkout "${BRANCH}"
    git reset --hard "origin/${BRANCH}"
else
    rm -rf "${INSTALL_DIR}"
    git clone -b "${BRANCH}" "${REPO_URL}" "${INSTALL_DIR}"
fi

echo "[vps] Preparing .env (без прокси — VPS имеет прямой доступ)..."
grep -v -i "proxy" /tmp/ouroboros_deploy.env > "${INSTALL_DIR}/.env" || true
echo "OUROBOROS_RUNTIME_ROOT=/opt/ouroboros/.runtime" >> "${INSTALL_DIR}/.env"

echo "[vps] Installing Python deps..."
cd "${INSTALL_DIR}"
pip3 install -q -r requirements.txt

echo "[vps] Creating log directory..."
mkdir -p "${LOG_DIR}"

echo "[vps] Writing systemd unit..."
cat > "/etc/systemd/system/${SERVICE}.service" << 'UNIT'
[Unit]
Description=Ouroboros Telegram Bot
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=root
WorkingDirectory=/opt/ouroboros
ExecStart=/usr/bin/python3 /opt/ouroboros/local_launcher.py
Restart=always
RestartSec=15
StandardOutput=append:/var/log/ouroboros/bot.log
StandardError=append:/var/log/ouroboros/bot.log
EnvironmentFile=/opt/ouroboros/.env

[Install]
WantedBy=multi-user.target
UNIT

systemctl daemon-reload
systemctl enable "${SERVICE}"
systemctl restart "${SERVICE}"

echo "[vps] Waiting for startup..."
sleep 8
systemctl status "${SERVICE}" --no-pager | head -15

echo ""
echo "=== Deploy complete ==="
echo "Logs:    tail -f /var/log/ouroboros/bot.log"
echo "Status:  systemctl status ouroboros-bot"
echo "Stop:    systemctl stop ouroboros-bot"
echo "Start:   systemctl start ouroboros-bot"
REMOTE

log "Done. Bot is running on VPS 24/7."
