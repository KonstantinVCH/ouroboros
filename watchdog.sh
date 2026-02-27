#!/usr/bin/env bash
# watchdog.sh — Ouroboros bot watchdog
# Checks proxy, detects duplicate/dead processes, auto-restarts the bot.
# Usage:
#   ./watchdog.sh          — single check + fix
#   ./watchdog.sh --loop   — run every 60 seconds (daemon mode)

set -euo pipefail

BOT_DIR="/c/Users/morco/repo/ouroboros"
LOG_FILE="/tmp/ouroboros.log"
SUPERVISOR_LOG="$HOME/.ouroboros/Ouroboros/logs/supervisor.jsonl"
PROXY="socks5://proxy_user:nmFZhByC9rNNOhz9@64.188.72.89:1080"
TELEGRAM_TOKEN="8771685756:AAHLXdRVZiCLqKlT5COHcUa4fbWwpvli2wc"
CHECK_INTERVAL=60  # seconds between checks in loop mode

# ── helpers ──────────────────────────────────────────────────────────────────

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

count_bot_processes() {
    wmic process where "name='python.exe'" get commandline 2>/dev/null \
        | grep -c "local_launcher" || echo 0
}

kill_all_bot_processes() {
    log "Killing all bot processes..."
    cmd.exe /c "wmic process where \"commandline like '%local_launcher%'\" call terminate" > /dev/null 2>&1 || true
    sleep 3
    # Double-check with wmic delete
    cmd.exe /c "wmic process where \"commandline like '%local_launcher%'\" delete" > /dev/null 2>&1 || true
    sleep 2
}

start_bot() {
    log "Starting bot..."
    cd "$BOT_DIR"
    nohup python local_launcher.py > "$LOG_FILE" 2>&1 &
    log "Bot started with PID $!"
    sleep 5
}

check_proxy() {
    local result
    result=$(curl -s -x "$PROXY" \
        "https://api.telegram.org/bot${TELEGRAM_TOKEN}/getMe" \
        --connect-timeout 10 --max-time 15 2>&1)
    if echo "$result" | grep -q '"ok":true'; then
        return 0
    else
        return 1
    fi
}

_recent_log_lines() {
    # Returns log lines from the last N minutes (default 2)
    local minutes="${1:-2}"
    if [ ! -f "$SUPERVISOR_LOG" ]; then return; fi
    local cutoff
    cutoff=$(date -u -d "-${minutes} minutes" '+%Y-%m-%dT%H:%M' 2>/dev/null \
        || date -u -v-${minutes}M '+%Y-%m-%dT%H:%M' 2>/dev/null \
        || echo "0000-00-00T00:00")
    tail -100 "$SUPERVISOR_LOG" 2>/dev/null | while IFS= read -r line; do
        local ts
        ts=$(echo "$line" | grep -o '"ts":"[^"]*"' | cut -d'"' -f4 | cut -c1-16)
        [ "$ts" \> "$cutoff" ] && echo "$line"
    done
}

check_recent_409() {
    # Returns 0 (true) if there are 409 errors in the last 2 minutes
    _recent_log_lines 2 | grep -q "409 Client Error"
}

check_recent_poll_errors() {
    # Returns count of telegram_poll_error in the last 2 minutes
    _recent_log_lines 2 | grep -c "telegram_poll_error" || echo 0
}

# ── main check logic ─────────────────────────────────────────────────────────

run_check() {
    log "====== Ouroboros watchdog check ======"

    # 1. Check proxy
    log "Checking proxy..."
    if check_proxy; then
        log "Proxy OK — Telegram API reachable"
    else
        log "WARNING: Proxy unavailable or Telegram unreachable"
    fi

    # 2. Count running bot processes
    local count
    count=$(count_bot_processes)
    log "Bot processes running: $count"

    if [ "$count" -eq 0 ]; then
        log "Bot is NOT running — starting..."
        start_bot
        log "Bot started."

    elif [ "$count" -gt 1 ]; then
        log "DUPLICATE processes detected ($count) — 409 Conflict possible"
        kill_all_bot_processes
        local remaining
        remaining=$(count_bot_processes)
        if [ "$remaining" -eq 0 ]; then
            log "All processes killed. Restarting single instance..."
            start_bot
            log "Bot restarted."
        else
            log "WARNING: Could not kill all processes ($remaining remaining)"
        fi

    else
        log "Bot is running (1 process) — OK"

        # 3. Check for 409 errors even with single process
        if check_recent_409; then
            log "409 Conflict errors detected in logs — restarting..."
            kill_all_bot_processes
            start_bot
            log "Bot restarted due to 409."
        else
            # 4. Check for excessive poll errors (proxy issues)
            local poll_errors
            poll_errors=$(check_recent_poll_errors)
            if [ "$poll_errors" -ge 3 ]; then
                log "WARNING: $poll_errors recent Telegram poll errors (proxy unstable?)"
            fi
        fi
    fi

    # 5. Summary from supervisor.jsonl
    if [ -f "$SUPERVISOR_LOG" ]; then
        local last_heartbeat
        last_heartbeat=$(tail -50 "$SUPERVISOR_LOG" 2>/dev/null \
            | grep "main_loop_heartbeat" | tail -1 \
            | grep -o '"ts":"[^"]*"' | cut -d'"' -f4 || echo "unknown")
        log "Last heartbeat: $last_heartbeat"
    fi

    log "====== Check complete ======"
}

# ── entry point ───────────────────────────────────────────────────────────────

if [ "${1:-}" = "--loop" ]; then
    log "Watchdog starting in loop mode (interval: ${CHECK_INTERVAL}s)"
    while true; do
        run_check
        log "Sleeping ${CHECK_INTERVAL}s..."
        sleep "$CHECK_INTERVAL"
    done
else
    run_check
fi
