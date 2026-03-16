"""
watchdog.py — Ouroboros bot watchdog

Проверяет:
  1. Доступность прокси / Telegram API (и авто-чинит если контейнер упал)
  2. Количество запущенных экземпляров бота
  3. Ошибки 409 Conflict и частые poll-ошибки в логах
  4. Автоматически убивает дубли и перезапускает бот

Запуск:
  python watchdog.py            — однократная проверка
  python watchdog.py --loop     — режим демона (проверка каждые 60 сек)
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import pathlib
import subprocess
import sys
import time
import urllib.request
import urllib.error

# ── конфигурация ──────────────────────────────────────────────────────────────

BOT_DIR = pathlib.Path(__file__).parent.resolve()
LOG_FILE = pathlib.Path("/tmp/ouroboros.log")
SUPERVISOR_LOG = pathlib.Path.home() / ".ouroboros" / "Ouroboros" / "logs" / "supervisor.jsonl"

PROXY = "socks5://proxy_user:nmFZhByC9rNNOhz9@64.188.72.89:1080"
TELEGRAM_TOKEN = "8771685756:AAHLXdRVZiCLqKlT5COHcUa4fbWwpvli2wc"

# SSH-параметры прокси-сервера для авто-починки
PROXY_SSH_HOST = "64.188.72.89"
PROXY_SSH_USER = "root"
PROXY_SSH_PASS = "6NH8EGE073Rb"
PROXY_CONTAINER = "amnezia-socks5proxy"

CHECK_INTERVAL_SEC = 60
RECENT_MINUTES = 2          # считать ошибки "свежими" если не старше 2 минут
POLL_ERROR_THRESHOLD = 3    # сколько poll-ошибок за 2 мин считать проблемой
RESTART_COOLDOWN_SEC = 90   # не перезапускать повторно раньше чем через N секунд

PYTHON_EXE = sys.executable  # тот же python, которым запущен watchdog

_last_restart_ts: float = 0.0  # время последнего рестарта


# ── логирование ───────────────────────────────────────────────────────────────

def log(msg: str) -> None:
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


# ── работа с процессами ───────────────────────────────────────────────────────

def _find_bot_processes() -> list[int]:
    """Возвращает список PID запущенных local_launcher.py."""
    try:
        import psutil
        pids = []
        for proc in psutil.process_iter(["pid", "name", "cmdline"]):
            try:
                cmdline = proc.info.get("cmdline") or []
                if any("local_launcher" in str(c) for c in cmdline):
                    pids.append(proc.info["pid"])
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        return pids
    except ImportError:
        return _find_bot_processes_fallback()


def _find_bot_processes_fallback() -> list[int]:
    """Fallback через tasklist если psutil не установлен."""
    try:
        result = subprocess.run(
            ["tasklist", "/FO", "CSV", "/NH"],
            capture_output=True, text=True, errors="replace"
        )
        # tasklist не показывает cmdline — используем wmic через shell
        result2 = subprocess.run(
            'wmic process where "name=\'python.exe\'" get processid,commandline',
            shell=True, capture_output=True, errors="replace"
        )
        raw = result2.stdout
        if isinstance(raw, bytes):
            for enc in ("utf-16-le", "cp1251", "utf-8"):
                try:
                    raw = raw.decode(enc)
                    break
                except Exception:
                    pass
        import re
        pids = []
        for line in str(raw).splitlines():
            if "local_launcher" in line:
                m = re.search(r"(\d+)\s*$", line.strip())
                if m:
                    pids.append(int(m.group(1)))
        return pids
    except Exception as e:
        log(f"WARNING: could not list processes: {e}")
        return []


def _kill_processes(pids: list[int]) -> None:
    """Завершает процессы по PID."""
    if not pids:
        return
    try:
        import psutil
        for pid in pids:
            try:
                proc = psutil.Process(pid)
                children = proc.children(recursive=True)
                proc.kill()
                for child in children:
                    try:
                        child.kill()
                    except Exception:
                        pass
                log(f"  Killed PID {pid}")
            except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
                log(f"  Could not kill PID {pid}: {e}")
    except ImportError:
        # Fallback через taskkill
        for pid in pids:
            try:
                subprocess.run(
                    ["taskkill", "/PID", str(pid), "/F", "/T"],
                    capture_output=True
                )
                log(f"  Killed PID {pid} (taskkill)")
            except Exception as e:
                log(f"  taskkill failed for PID {pid}: {e}")


def _start_bot() -> int:
    """Запускает бота в фоне, возвращает PID."""
    global _last_restart_ts
    proc = subprocess.Popen(
        [PYTHON_EXE, "local_launcher.py"],
        cwd=str(BOT_DIR),
        stdout=open(LOG_FILE, "a", encoding="utf-8"),
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    _last_restart_ts = time.time()
    return proc.pid


def _cooldown_active() -> bool:
    """Возвращает True если недавно был рестарт и нужно подождать."""
    elapsed = time.time() - _last_restart_ts
    if elapsed < RESTART_COOLDOWN_SEC:
        remaining = int(RESTART_COOLDOWN_SEC - elapsed)
        log(f"Cooldown active ({remaining}s remaining after last restart) — skipping restart")
        return True
    return False


# ── авто-починка прокси ───────────────────────────────────────────────────────

def _repair_proxy() -> bool:
    """
    Пытается починить прокси: подключается по SSH к серверу и перезапускает
    Docker-контейнер. Возвращает True если после починки прокси заработал.
    """
    log(f"Attempting proxy repair: SSH to {PROXY_SSH_HOST}, restarting {PROXY_CONTAINER}...")
    try:
        # sshpass нужен для передачи пароля; если нет — используем ключ
        ssh_cmd_prefix = []
        try:
            subprocess.run(["sshpass", "-V"], capture_output=True, check=True)
            ssh_cmd_prefix = ["sshpass", f"-p{PROXY_SSH_PASS}"]
        except (FileNotFoundError, subprocess.CalledProcessError):
            pass  # используем ssh без пароля (ключ уже добавлен)

        cmd = ssh_cmd_prefix + [
            "ssh", "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=15",
            f"{PROXY_SSH_USER}@{PROXY_SSH_HOST}",
            f"docker start {PROXY_CONTAINER} && docker update --restart unless-stopped {PROXY_CONTAINER}",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            log("SSH repair command OK — waiting 5s for container to start...")
            time.sleep(5)
            if check_proxy():
                log("Proxy is UP after repair.")
                return True
            else:
                log("Proxy still DOWN after repair attempt.")
                return False
        else:
            log(f"SSH repair failed (exit {result.returncode}): {result.stderr.strip()}")
            return False
    except Exception as e:
        log(f"Proxy repair error: {e}")
        return False


# ── проверка прокси ───────────────────────────────────────────────────────────

def check_proxy() -> bool:
    """Проверяет доступность Telegram API через прокси."""
    try:
        import urllib.request
        proxies = {
            "https": PROXY,
            "http": PROXY,
        }
        proxy_handler = urllib.request.ProxyHandler(proxies)
        opener = urllib.request.build_opener(proxy_handler)
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/getMe"
        with opener.open(url, timeout=10) as resp:
            data = json.loads(resp.read())
            return bool(data.get("ok"))
    except Exception:
        pass

    # Fallback через curl
    try:
        result = subprocess.run(
            ["curl", "-s", "-x", PROXY,
             f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/getMe",
             "--connect-timeout", "10", "--max-time", "15"],
            capture_output=True, text=True, timeout=20
        )
        return '"ok":true' in result.stdout
    except Exception:
        return False


# ── анализ логов ──────────────────────────────────────────────────────────────

def _recent_log_entries(minutes: int = 2) -> list[dict]:
    """Возвращает записи supervisor.jsonl за последние N минут."""
    if not SUPERVISOR_LOG.exists():
        return []
    cutoff = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(minutes=minutes)
    entries = []
    try:
        lines = SUPERVISOR_LOG.read_text(encoding="utf-8", errors="replace").splitlines()
        for line in lines[-200:]:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                ts_str = entry.get("ts", "")
                ts = datetime.datetime.fromisoformat(ts_str)
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=datetime.timezone.utc)
                if ts >= cutoff:
                    entries.append(entry)
            except Exception:
                pass
    except Exception:
        pass
    return entries


def check_recent_409(entries: list[dict]) -> bool:
    return any("409 Client Error" in str(e.get("error", "")) for e in entries)


def count_recent_poll_errors(entries: list[dict]) -> int:
    return sum(1 for e in entries if e.get("type") == "telegram_poll_error")


def last_heartbeat_ts() -> str:
    if not SUPERVISOR_LOG.exists():
        return "unknown"
    try:
        lines = SUPERVISOR_LOG.read_text(encoding="utf-8", errors="replace").splitlines()
        for line in reversed(lines[-100:]):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                if entry.get("type") == "main_loop_heartbeat":
                    return entry.get("ts", "unknown")[:19]
            except Exception:
                pass
    except Exception:
        pass
    return "unknown"


# ── основная логика ───────────────────────────────────────────────────────────

def run_check() -> None:
    log("====== Ouroboros watchdog check ======")

    # 1. Прокси
    log("Checking proxy...")
    proxy_ok = check_proxy()
    if proxy_ok:
        log("Proxy OK — Telegram API reachable")
    else:
        log("WARNING: Proxy unavailable — attempting auto-repair...")
        proxy_ok = _repair_proxy()
        if not proxy_ok:
            log("Proxy still unavailable after repair attempt")

    # 2. Процессы
    pids = _find_bot_processes()
    log(f"Bot processes running: {len(pids)}  {pids}")

    recent = _recent_log_entries(RECENT_MINUTES)

    if len(pids) == 0:
        if _cooldown_active():
            pass  # бот только что был запущен, ещё не успел появиться в списке
        else:
            log("Bot is NOT running — starting...")
            pid = _start_bot()
            log(f"Bot started (PID {pid})")

    elif len(pids) > 1:
        if _cooldown_active():
            log(f"Duplicate pids {pids} detected but cooldown active — waiting")
        else:
            log(f"DUPLICATE processes ({len(pids)}) — killing all and restarting...")
            _kill_processes(pids)
            time.sleep(3)
            remaining = _find_bot_processes()
            if remaining:
                log(f"WARNING: {len(remaining)} processes still alive after kill: {remaining}")
            else:
                pid = _start_bot()
                log(f"Bot restarted (PID {pid})")

    else:
        log(f"Bot is running (PID {pids[0]}) — OK")

        # 3. 409 Conflict (даже при одном процессе)
        if check_recent_409(recent):
            if _cooldown_active():
                pass  # 409 скорее всего от предыдущего рестарта, ждём
            else:
                log("409 Conflict detected in recent logs — restarting...")
                _kill_processes(pids)
                time.sleep(3)
                pid = _start_bot()
                log(f"Bot restarted due to 409 (PID {pid})")

        else:
            # 4. Частые poll-ошибки (нестабильный прокси)
            poll_errors = count_recent_poll_errors(recent)
            if poll_errors >= POLL_ERROR_THRESHOLD:
                log(f"WARNING: {poll_errors} Telegram poll errors in last {RECENT_MINUTES} min (proxy unstable?)")

    log(f"Last heartbeat: {last_heartbeat_ts()}")
    log("====== Check complete ======")


# ── точка входа ───────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="Ouroboros bot watchdog")
    ap.add_argument("--loop", action="store_true",
                    help=f"Run continuously every {CHECK_INTERVAL_SEC}s")
    ap.add_argument("--interval", type=int, default=CHECK_INTERVAL_SEC,
                    help="Check interval in seconds (loop mode)")
    args = ap.parse_args()

    if args.loop:
        interval = args.interval
        log(f"Watchdog starting in loop mode (interval: {interval}s)")
        while True:
            try:
                run_check()
            except Exception as e:
                log(f"ERROR in watchdog check: {e}")
            log(f"Sleeping {interval}s...")
            time.sleep(interval)
    else:
        run_check()


if __name__ == "__main__":
    main()
