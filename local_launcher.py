"""
Ouroboros — Local runtime launcher (no Google Colab).

This is a Windows/macOS/Linux-friendly entrypoint that mirrors `colab_launcher.py`,
but uses a local runtime directory instead of Google Drive and does not import
`google.colab`.

Required environment variables:
  - TELEGRAM_BOT_TOKEN
  - TOTAL_BUDGET
  - GITHUB_TOKEN
  - GITHUB_USER
  - GITHUB_REPO

Optional:
  - OPENROUTER_API_KEY (required only if using OpenRouter)
  - OPENAI_API_KEY
  - ANTHROPIC_API_KEY
  - OUROBOROS_LLM_BASE_URL (e.g. http://localhost:11434/v1 for Ollama OpenAI-compat)
  - OUROBOROS_LLM_API_KEY (optional for local endpoints; can be "ollama")
  - OUROBOROS_RUNTIME_ROOT (default: ~/.ouroboros/Ouroboros)
  - OUROBOROS_BOOT_BRANCH (default: ouroboros)
  - OUROBOROS_MAX_WORKERS (default: 5)
  - OUROBOROS_MODEL / OUROBOROS_MODEL_CODE / OUROBOROS_MODEL_LIGHT
"""

from __future__ import annotations

import argparse
import datetime
import logging
import os
import pathlib
import queue as _queue_mod
import subprocess
import sys
import threading
import time
import types
import uuid
from typing import Any, Dict, Optional, Set

log = logging.getLogger(__name__)


def _load_dotenv(path: pathlib.Path) -> None:
    """Minimal .env loader (no external deps). Existing env vars win."""
    try:
        if not path.exists() or not path.is_file():
            return
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[len("export ") :].strip()
            if "=" not in line:
                continue
            k, v = line.split("=", 1)
            k = k.strip()
            v = v.strip()
            if not k:
                continue
            if len(v) >= 2 and ((v[0] == v[-1] == '"') or (v[0] == v[-1] == "'")):
                v = v[1:-1]
            os.environ.setdefault(k, v)
    except Exception:
        log.debug("Failed to load dotenv file: %s", path, exc_info=True)


def _get_env(name: str, default: Optional[str] = None, required: bool = False) -> str:
    v = os.environ.get(name, default)
    if v is None:
        v = ""
    v = str(v)
    if required and not v.strip():
        raise RuntimeError(f"Missing required env var: {name}")
    return v


def _parse_budget(raw: str) -> float:
    try:
        import re
        _clean = re.sub(r"[^0-9.\-]", "", str(raw or ""))
        return float(_clean) if _clean else 0.0
    except Exception:
        return 0.0


def _transcribe_voice_local(audio_b64: str) -> str:
    """Transcribe voice message using local faster-whisper (offline, no API limits)."""
    import base64 as _b64
    import tempfile as _tmp
    import os as _os
    try:
        from faster_whisper import WhisperModel
        audio_bytes = _b64.b64decode(audio_b64)
        with _tmp.NamedTemporaryFile(suffix=".ogg", delete=False) as f:
            f.write(audio_bytes)
            tmp_path = f.name
        try:
            model = WhisperModel("tiny", device="cpu", compute_type="int8")
            segments, _ = model.transcribe(tmp_path, language=None, beam_size=3)
            return " ".join(s.text.strip() for s in segments).strip()
        finally:
            _os.unlink(tmp_path)
    except Exception as e:
        log.warning("Voice transcription failed: %s", e)
        return ""


def _git(cmd: list[str], cwd: pathlib.Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True)


def _ensure_boot_branches(repo_dir: pathlib.Path, boot_branch: str, stable_branch: str) -> None:
    """
    Ensure the fork has `boot_branch` and `stable_branch` on origin.
    Mirrors the logic in `colab_bootstrap_shim.py` (fresh forks only have main).
    """
    r = _git(["git", "fetch", "origin"], cwd=repo_dir)
    if r.returncode != 0:
        raise RuntimeError(f"git fetch failed: {(r.stderr or r.stdout).strip()}")

    rc = _git(["git", "rev-parse", "--verify", f"origin/{boot_branch}"], cwd=repo_dir).returncode
    if rc == 0:
        return

    print(f"[boot] branch {boot_branch} not found on fork — creating from origin/main")
    r2 = _git(["git", "checkout", "-b", boot_branch, "origin/main"], cwd=repo_dir)
    if r2.returncode != 0:
        raise RuntimeError(f"git checkout -b failed: {(r2.stderr or r2.stdout).strip()}")
    r3 = _git(["git", "push", "-u", "origin", boot_branch], cwd=repo_dir)
    if r3.returncode != 0:
        raise RuntimeError(f"git push boot branch failed: {(r3.stderr or r3.stdout).strip()}")

    r4 = _git(["git", "branch", stable_branch], cwd=repo_dir)
    if r4.returncode != 0:
        raise RuntimeError(f"git branch stable failed: {(r4.stderr or r4.stdout).strip()}")
    r5 = _git(["git", "push", "-u", "origin", stable_branch], cwd=repo_dir)
    if r5.returncode != 0:
        raise RuntimeError(f"git push stable branch failed: {(r5.stderr or r5.stdout).strip()}")


# ---------------------------------------------------------------------------
# Setup helpers (extracted from main to keep it under 200 lines)
# ---------------------------------------------------------------------------

def _load_and_validate_config(args: argparse.Namespace) -> dict:
    """Load env/config and return a settings dict. Raises on missing required vars."""
    if args.env_file:
        _load_dotenv(pathlib.Path(args.env_file).expanduser().resolve())
    else:
        _load_dotenv(pathlib.Path.cwd() / ".env")

    llm_base_url = str(os.environ.get("OUROBOROS_LLM_BASE_URL", "") or "").strip()
    using_custom_llm = bool(llm_base_url)
    core_required = ["TELEGRAM_BOT_TOKEN", "TOTAL_BUDGET", "GITHUB_TOKEN", "GITHUB_USER", "GITHUB_REPO"]
    missing_core = [n for n in core_required if not str(os.environ.get(n, "")).strip()]
    has_openrouter_key = bool(str(os.environ.get("OPENROUTER_API_KEY", "") or "").strip())
    if missing_core or (not using_custom_llm and not has_openrouter_key):
        bits = []
        if missing_core:
            bits.append(", ".join(missing_core))
        if not using_custom_llm and not has_openrouter_key:
            bits.append("OPENROUTER_API_KEY (or set OUROBOROS_LLM_BASE_URL for Ollama/local)")
        raise RuntimeError(
            "Missing required env var(s): " + ", ".join(bits) + "\n\n"
            "Tip: create a .env file in the current directory.\n"
            "Ollama example: OUROBOROS_LLM_BASE_URL=http://localhost:11434/v1\n"
        )

    openrouter_api_key = _get_env("OPENROUTER_API_KEY", required=False)
    telegram_bot_token = _get_env("TELEGRAM_BOT_TOKEN", required=True)
    total_budget_raw = _get_env("TOTAL_BUDGET", required=True)
    github_token = _get_env("GITHUB_TOKEN", required=True)
    github_user = _get_env("GITHUB_USER", required=True)
    github_repo = _get_env("GITHUB_REPO", required=True)

    from ouroboros.llm import DEFAULT_LIGHT_MODEL

    local_default_model = "llama3.1:8b" if using_custom_llm else "anthropic/claude-sonnet-4.6"
    model_main = _get_env("OUROBOROS_MODEL", default=local_default_model)
    model_code = _get_env("OUROBOROS_MODEL_CODE", default=model_main)
    model_light = _get_env("OUROBOROS_MODEL_LIGHT", default=DEFAULT_LIGHT_MODEL)

    runtime_root = pathlib.Path(
        _get_env("OUROBOROS_RUNTIME_ROOT",
                 default=str(pathlib.Path.home() / ".ouroboros" / "Ouroboros"))
    ).expanduser().resolve()

    cfg = {
        "openrouter_api_key": openrouter_api_key,
        "telegram_bot_token": telegram_bot_token,
        "total_budget_limit": _parse_budget(total_budget_raw),
        "github_token": github_token,
        "github_user": github_user,
        "github_repo": github_repo,
        "openai_api_key": _get_env("OPENAI_API_KEY", default=""),
        "anthropic_api_key": _get_env("ANTHROPIC_API_KEY", default=""),
        "model_main": model_main,
        "model_code": model_code,
        "model_light": model_light,
        "max_workers": int(_get_env("OUROBOROS_MAX_WORKERS", default="5") or "5"),
        "budget_report_every": 10,
        "soft_timeout_sec": max(60, int(_get_env("OUROBOROS_SOFT_TIMEOUT_SEC", default="600") or "600")),
        "hard_timeout_sec": max(120, int(_get_env("OUROBOROS_HARD_TIMEOUT_SEC", default="1800") or "1800")),
        "diag_heartbeat_sec": max(0, int(_get_env("OUROBOROS_DIAG_HEARTBEAT_SEC", default="30") or "30")),
        "diag_slow_cycle_sec": max(0, int(_get_env("OUROBOROS_DIAG_SLOW_CYCLE_SEC", default="20") or "20")),
        "llm_base_url": llm_base_url,
        "using_custom_llm": using_custom_llm,
        "boot_branch": _get_env("OUROBOROS_BOOT_BRANCH", default="ouroboros").strip() or "ouroboros",
        "runtime_root": runtime_root,
        "drive_root": runtime_root,
        "repo_dir": (runtime_root / "ouroboros_repo").resolve(),
    }
    cfg["branch_dev"] = cfg["boot_branch"]
    cfg["branch_stable"] = cfg["boot_branch"] + "-stable"
    cfg["remote_url"] = (
        f"https://{github_token}:x-oauth-basic@github.com/{github_user}/{github_repo}.git"
    )
    return cfg


def _export_env_vars(cfg: dict) -> None:
    """Write resolved config back to os.environ so tool subprocesses can read it."""
    if cfg["openrouter_api_key"]:
        os.environ["OPENROUTER_API_KEY"] = cfg["openrouter_api_key"]
    os.environ["OPENAI_API_KEY"] = cfg["openai_api_key"]
    os.environ["ANTHROPIC_API_KEY"] = cfg["anthropic_api_key"]
    os.environ["GITHUB_USER"] = cfg["github_user"]
    os.environ["GITHUB_REPO"] = cfg["github_repo"]
    os.environ["OUROBOROS_MODEL"] = cfg["model_main"]
    os.environ["OUROBOROS_MODEL_CODE"] = cfg["model_code"]
    os.environ["OUROBOROS_MODEL_LIGHT"] = cfg["model_light"]
    os.environ["TELEGRAM_BOT_TOKEN"] = cfg["telegram_bot_token"]
    os.environ["OUROBOROS_DIAG_HEARTBEAT_SEC"] = str(cfg["diag_heartbeat_sec"])
    os.environ["OUROBOROS_DIAG_SLOW_CYCLE_SEC"] = str(cfg["diag_slow_cycle_sec"])
    if cfg["using_custom_llm"]:
        os.environ["OUROBOROS_LLM_BASE_URL"] = cfg["llm_base_url"]
        if str(os.environ.get("OUROBOROS_LLM_API_KEY", "") or "").strip():
            os.environ["OUROBOROS_LLM_API_KEY"] = str(os.environ["OUROBOROS_LLM_API_KEY"])


def _init_supervisor_modules(cfg: dict):
    """Import and initialise all supervisor modules. Returns (tg, imports_ns)."""
    drive_root = cfg["drive_root"]
    repo_dir = cfg["repo_dir"]

    for sub in ["state", "logs", "memory", "index", "locks", "archive"]:
        (drive_root / sub).mkdir(parents=True, exist_ok=True)
    repo_dir.parent.mkdir(parents=True, exist_ok=True)

    from supervisor.state import (
        init as state_init,
        load_state,
        save_state,
        append_jsonl,
        update_budget_from_usage,
        status_text,
        rotate_chat_log_if_needed,
        init_state,
    )
    state_init(drive_root, cfg["total_budget_limit"])
    init_state()

    from supervisor.telegram import init as telegram_init, TelegramClient, send_with_budget, log_chat
    tg = TelegramClient(str(cfg["telegram_bot_token"]))
    telegram_init(
        drive_root=drive_root,
        total_budget_limit=cfg["total_budget_limit"],
        budget_report_every=cfg["budget_report_every"],
        tg_client=tg,
    )

    from supervisor.git_ops import init as git_ops_init, ensure_repo_present, safe_restart
    git_ops_init(
        repo_dir=repo_dir,
        drive_root=drive_root,
        remote_url=cfg["remote_url"],
        branch_dev=cfg["branch_dev"],
        branch_stable=cfg["branch_stable"],
    )

    from supervisor.queue import (
        enforce_task_timeouts,
        enqueue_evolution_task_if_needed,
        persist_queue_snapshot,
        restore_pending_from_snapshot,
        cancel_task_by_id,
        queue_review_task,
        sort_pending,
    )
    from supervisor.workers import (
        init as workers_init,
        get_event_q,
        WORKERS,
        PENDING,
        RUNNING,
        spawn_workers,
        kill_workers,
        assign_tasks,
        ensure_workers_healthy,
        handle_chat_direct,
        _get_chat_agent,
        auto_resume_after_restart,
    )
    workers_init(
        repo_dir=repo_dir,
        drive_root=drive_root,
        max_workers=cfg["max_workers"],
        soft_timeout=cfg["soft_timeout_sec"],
        hard_timeout=cfg["hard_timeout_sec"],
        total_budget_limit=cfg["total_budget_limit"],
        branch_dev=cfg["branch_dev"],
        branch_stable=cfg["branch_stable"],
    )

    from supervisor.events import dispatch_event

    ns = types.SimpleNamespace(
        load_state=load_state, save_state=save_state, append_jsonl=append_jsonl,
        update_budget_from_usage=update_budget_from_usage, status_text=status_text,
        rotate_chat_log_if_needed=rotate_chat_log_if_needed,
        ensure_repo_present=ensure_repo_present, safe_restart=safe_restart,
        enforce_task_timeouts=enforce_task_timeouts,
        enqueue_evolution_task_if_needed=enqueue_evolution_task_if_needed,
        persist_queue_snapshot=persist_queue_snapshot,
        restore_pending_from_snapshot=restore_pending_from_snapshot,
        cancel_task_by_id=cancel_task_by_id, queue_review_task=queue_review_task,
        sort_pending=sort_pending,
        get_event_q=get_event_q, WORKERS=WORKERS, PENDING=PENDING, RUNNING=RUNNING,
        spawn_workers=spawn_workers, kill_workers=kill_workers, assign_tasks=assign_tasks,
        ensure_workers_healthy=ensure_workers_healthy,
        handle_chat_direct=handle_chat_direct,
        _get_chat_agent=_get_chat_agent, auto_resume_after_restart=auto_resume_after_restart,
        dispatch_event=dispatch_event, log_chat=log_chat, send_with_budget=send_with_budget,
    )
    return tg, ns


def _bootstrap_repo_and_workers(cfg: dict, sv: types.SimpleNamespace) -> None:
    """Bootstrap repo, branches, workers. Logs startup event."""
    sv.ensure_repo_present()
    _ensure_boot_branches(cfg["repo_dir"], boot_branch=cfg["branch_dev"], stable_branch=cfg["branch_stable"])

    ok, msg = sv.safe_restart(reason="bootstrap", unsynced_policy="rescue_and_reset")
    if not ok:
        raise RuntimeError(f"Bootstrap failed: {msg}")

    sv.kill_workers()
    sv.spawn_workers(cfg["max_workers"])
    restored_pending = sv.restore_pending_from_snapshot()
    sv.persist_queue_snapshot(reason="startup")
    if restored_pending > 0:
        st = sv.load_state()
        if st.get("owner_chat_id"):
            sv.send_with_budget(int(st["owner_chat_id"]),
                                f"♻️ Restored pending queue from snapshot: {restored_pending} tasks.")

    sv.append_jsonl(
        cfg["drive_root"] / "logs" / "supervisor.jsonl",
        {
            "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "type": "launcher_start",
            "branch": sv.load_state().get("current_branch"),
            "sha": sv.load_state().get("current_sha"),
            "max_workers": cfg["max_workers"],
            "model_default": cfg["model_main"],
            "model_code": cfg["model_code"],
            "model_light": cfg["model_light"],
            "soft_timeout_sec": cfg["soft_timeout_sec"],
            "hard_timeout_sec": cfg["hard_timeout_sec"],
            "worker_start_method": str(os.environ.get("OUROBOROS_WORKER_START_METHOD") or ""),
            "diag_heartbeat_sec": cfg["diag_heartbeat_sec"],
            "diag_slow_cycle_sec": cfg["diag_slow_cycle_sec"],
            "runtime_root": str(cfg["runtime_root"]),
        },
    )
    sv.auto_resume_after_restart()


def main() -> None:
    ap = argparse.ArgumentParser(add_help=True)
    ap.add_argument("--env-file", default="",
                    help="Path to .env file (optional). If omitted, tries .env in current directory.")
    args = ap.parse_args()

    logging.basicConfig(
        level=os.environ.get("OUROBOROS_LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    cfg = _load_and_validate_config(args)
    _export_env_vars(cfg)
    tg, sv = _init_supervisor_modules(cfg)
    _bootstrap_repo_and_workers(cfg, sv)

    # Background consciousness
    from ouroboros.consciousness import BackgroundConsciousness

    def _get_owner_chat_id() -> Optional[int]:
        try:
            st = sv.load_state()
            cid = st.get("owner_chat_id")
            return int(cid) if cid else None
        except Exception:
            return None

    consciousness = BackgroundConsciousness(
        drive_root=cfg["drive_root"],
        repo_dir=cfg["repo_dir"],
        event_queue=sv.get_event_q(),
        owner_chat_id_fn=_get_owner_chat_id,
    )

    def reset_chat_agent() -> None:
        import supervisor.workers as _w
        _w._chat_agent = None

    # Watchdog for direct-mode chat agent
    def _chat_watchdog_loop() -> None:
        soft_warned = False
        while True:
            time.sleep(30)
            try:
                agent = sv._get_chat_agent()
                if not agent._busy:
                    soft_warned = False
                    continue
                now = time.time()
                idle_sec = now - agent._last_progress_ts
                total_sec = now - agent._task_started_ts
                if idle_sec >= cfg["hard_timeout_sec"]:
                    st = sv.load_state()
                    if st.get("owner_chat_id"):
                        sv.send_with_budget(int(st["owner_chat_id"]),
                                            f"⚠️ Task stuck ({int(total_sec)}s). Restarting agent.")
                    reset_chat_agent()
                    soft_warned = False
                elif idle_sec >= cfg["soft_timeout_sec"] and not soft_warned:
                    soft_warned = True
                    st = sv.load_state()
                    if st.get("owner_chat_id"):
                        sv.send_with_budget(int(st["owner_chat_id"]),
                                            f"⏱️ Task running {int(total_sec)}s, last progress {int(idle_sec)}s ago.")
            except Exception:
                log.debug("Failed to check/notify chat watchdog", exc_info=True)

    threading.Thread(target=_chat_watchdog_loop, daemon=True).start()

    # Supervisor command handler
    def _handle_supervisor_command(text: str, chat_id: int, tg_offset: int = 0):
        lowered = text.strip().lower()

        if lowered.startswith("/panic"):
            sv.send_with_budget(chat_id, "🛑 PANIC: stopping everything now.")
            sv.kill_workers()
            st2 = sv.load_state()
            st2["tg_offset"] = tg_offset
            sv.save_state(st2)
            raise SystemExit("PANIC")

        if lowered.startswith("/restart"):
            st2 = sv.load_state()
            st2["session_id"] = uuid.uuid4().hex
            st2["tg_offset"] = tg_offset
            sv.save_state(st2)
            sv.send_with_budget(chat_id, "♻️ Restarting (soft).")
            ok_r, msg_r = sv.safe_restart(reason="owner_restart", unsynced_policy="rescue_and_reset")
            if not ok_r:
                sv.send_with_budget(chat_id, f"⚠️ Restart cancelled: {msg_r}")
                return True
            sv.kill_workers()
            os.execv(sys.executable, [sys.executable, __file__])

        if lowered.startswith("/status"):
            status = sv.status_text(sv.WORKERS, sv.PENDING, sv.RUNNING,
                                    cfg["soft_timeout_sec"], cfg["hard_timeout_sec"])
            sv.send_with_budget(chat_id, status, force_budget=True)
            return "[Supervisor handled /status — status text already sent to chat]\n"

        if lowered.startswith("/review"):
            sv.queue_review_task(reason="owner:/review", force=True)
            return "[Supervisor handled /review — review task queued]\n"

        if lowered.startswith("/evolve"):
            parts = lowered.split()
            action = parts[1] if len(parts) > 1 else "on"
            turn_on = action not in ("off", "stop", "0")
            st2 = sv.load_state()
            st2["evolution_mode_enabled"] = bool(turn_on)
            sv.save_state(st2)
            if not turn_on:
                sv.PENDING[:] = [t for t in sv.PENDING if str(t.get("type")) != "evolution"]
                sv.sort_pending()
                sv.persist_queue_snapshot(reason="evolve_off")
            state_str = "ON" if turn_on else "OFF"
            sv.send_with_budget(chat_id, f"🧬 Evolution: {state_str}")
            return f"[Supervisor handled /evolve — evolution toggled {state_str}]\n"

        if lowered.startswith("/bg"):
            parts = lowered.split()
            action = parts[1] if len(parts) > 1 else "status"
            if action in ("start", "on", "1"):
                result = consciousness.start()
                sv.send_with_budget(chat_id, f"🧠 {result}")
            elif action in ("stop", "off", "0"):
                result = consciousness.stop()
                sv.send_with_budget(chat_id, f"🧠 {result}")
            else:
                bg_status = "running" if consciousness.is_running else "stopped"
                sv.send_with_budget(chat_id, f"🧠 Background consciousness: {bg_status}")
            return f"[Supervisor handled /bg {action}]\n"

        return ""

    # Start background consciousness if auto-start enabled
    event_ctx = types.SimpleNamespace(
        DRIVE_ROOT=cfg["drive_root"], REPO_DIR=cfg["repo_dir"],
        BRANCH_DEV=cfg["branch_dev"], BRANCH_STABLE=cfg["branch_stable"],
        TG=tg, WORKERS=sv.WORKERS, PENDING=sv.PENDING, RUNNING=sv.RUNNING,
        MAX_WORKERS=cfg["max_workers"],
        send_with_budget=sv.send_with_budget, load_state=sv.load_state, save_state=sv.save_state,
        update_budget_from_usage=sv.update_budget_from_usage, append_jsonl=sv.append_jsonl,
        cancel_task_by_id=sv.cancel_task_by_id, queue_review_task=sv.queue_review_task,
        persist_queue_snapshot=sv.persist_queue_snapshot, safe_restart=sv.safe_restart,
        kill_workers=sv.kill_workers, spawn_workers=sv.spawn_workers, sort_pending=sv.sort_pending,
        consciousness=consciousness,
    )

    _bg_auto = str(os.environ.get("OUROBOROS_BG_AUTO_START", "") or "").strip().lower() in {"1", "true", "yes", "on"}
    if _bg_auto:
        try:
            consciousness.start()
            log.info("🧠 Background consciousness auto-started (OUROBOROS_BG_AUTO_START=1)")
        except Exception as e:
            log.warning("consciousness auto-start failed: %s", e)
    else:
        log.info("🧠 Background consciousness is OFF by default (set OUROBOROS_BG_AUTO_START=1 to enable)")

    # ----------------------------
    # Main event loop
    # ----------------------------
    offset = int(sv.load_state().get("tg_offset") or 0)
    last_diag_heartbeat_ts = 0.0
    last_message_ts: float = time.time()
    active_mode_sec: int = 300

    while True:
        loop_started_ts = time.time()
        sv.rotate_chat_log_if_needed(cfg["drive_root"])
        sv.ensure_workers_healthy()

        event_q = sv.get_event_q()
        while True:
            try:
                evt = event_q.get_nowait()
            except _queue_mod.Empty:
                break
            sv.dispatch_event(evt, event_ctx)

        sv.enforce_task_timeouts()
        sv.enqueue_evolution_task_if_needed()
        sv.assign_tasks()
        sv.persist_queue_snapshot(reason="main_loop")

        now = time.time()
        active = (now - last_message_ts) < active_mode_sec
        poll_timeout = 0 if active else 10
        try:
            updates = tg.get_updates(offset=offset, timeout=poll_timeout)
        except Exception as e:
            sv.append_jsonl(
                cfg["drive_root"] / "logs" / "supervisor.jsonl",
                {"ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                 "type": "telegram_poll_error", "offset": offset, "error": repr(e)},
            )
            time.sleep(1.5)
            continue

        for upd in updates:
            offset = int(upd["update_id"]) + 1
            msg = upd.get("message") or upd.get("edited_message") or {}
            if not msg:
                continue

            chat_id = int(msg["chat"]["id"])
            from_user = msg.get("from") or {}
            user_id = int(from_user.get("id") or 0)
            text = str(msg.get("text") or "")
            caption = str(msg.get("caption") or "")
            now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()

            image_data = None
            if msg.get("photo"):
                best_photo = msg["photo"][-1]
                file_id = best_photo.get("file_id")
                if file_id:
                    b64, mime = tg.download_file_base64(file_id)
                    if b64:
                        image_data = (b64, mime, caption)
            elif msg.get("document"):
                doc = msg["document"]
                mime_type = str(doc.get("mime_type") or "")
                if mime_type.startswith("image/"):
                    file_id = doc.get("file_id")
                    if file_id:
                        b64, mime = tg.download_file_base64(file_id)
                        if b64:
                            image_data = (b64, mime, caption)
            elif msg.get("voice") or msg.get("audio"):
                voice_msg = msg.get("voice") or msg.get("audio")
                file_id = voice_msg.get("file_id")
                if file_id:
                    b64, _mime = tg.download_file_base64(file_id)
                    if b64:
                        transcribed = _transcribe_voice_local(b64)
                        if transcribed:
                            log.info("Voice transcribed: %s", transcribed[:100])
                            text = transcribed

            st = sv.load_state()
            if st.get("owner_id") is None:
                st["owner_id"] = user_id
                st["owner_chat_id"] = chat_id
                st["last_owner_message_at"] = now_iso
                sv.save_state(st)
                sv.log_chat("in", chat_id, user_id, text)
                sv.send_with_budget(chat_id, "✅ Owner registered. Ouroboros online.")
                continue

            if user_id != int(st.get("owner_id")):
                continue

            sv.log_chat("in", chat_id, user_id, text)
            st["last_owner_message_at"] = now_iso
            last_message_ts = time.time()
            sv.save_state(st)

            if text.strip().lower().startswith("/"):
                try:
                    result = _handle_supervisor_command(text, chat_id, tg_offset=offset)
                    if result is True:
                        continue
                    elif result:
                        text = result + text
                except SystemExit:
                    raise
                except Exception:
                    log.warning("Supervisor command handler error", exc_info=True)

            if not text and not image_data:
                continue

            consciousness.inject_observation(f"Owner message: {text[:100]}")

            agent = sv._get_chat_agent()
            if agent._busy:
                if image_data:
                    if text:
                        agent.inject_message(text)
                    sv.send_with_budget(chat_id, "📎 Photo received, but a task is in progress.")
                elif text:
                    agent.inject_message(text)
            else:
                consciousness.pause()

                def _run_task_and_resume(cid: int, txt: str, img):
                    try:
                        sv.handle_chat_direct(cid, txt, img)
                    finally:
                        consciousness.resume()

                t = threading.Thread(target=_run_task_and_resume,
                                     args=(chat_id, text, image_data), daemon=True)
                try:
                    t.start()
                except Exception as te:
                    log.error("Failed to start chat thread: %s", te)
                    consciousness.resume()

        st = sv.load_state()
        st["tg_offset"] = offset
        sv.save_state(st)

        now_epoch = time.time()
        loop_duration_sec = now_epoch - loop_started_ts

        if cfg["diag_slow_cycle_sec"] > 0 and loop_duration_sec >= float(cfg["diag_slow_cycle_sec"]):
            sv.append_jsonl(
                cfg["drive_root"] / "logs" / "supervisor.jsonl",
                {"ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                 "type": "main_loop_slow_cycle",
                 "duration_sec": round(loop_duration_sec, 3),
                 "pending_count": len(sv.PENDING), "running_count": len(sv.RUNNING)},
            )

        if cfg["diag_heartbeat_sec"] > 0 and (now_epoch - last_diag_heartbeat_ts) >= float(cfg["diag_heartbeat_sec"]):
            workers_total = len(sv.WORKERS)
            workers_alive = sum(1 for w in sv.WORKERS.values() if w.proc.is_alive())
            sv.append_jsonl(
                cfg["drive_root"] / "logs" / "supervisor.jsonl",
                {"ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                 "type": "main_loop_heartbeat",
                 "offset": offset, "workers_total": workers_total, "workers_alive": workers_alive,
                 "pending_count": len(sv.PENDING), "running_count": len(sv.RUNNING),
                 "event_q_size": _safe_qsize(event_q),
                 "running_task_ids": list(sv.RUNNING.keys())[:5],
                 "spent_usd": st.get("spent_usd")},
            )
            last_diag_heartbeat_ts = now_epoch

        loop_sleep = 0.1 if (now - last_message_ts) < active_mode_sec else 0.5
        time.sleep(loop_sleep)


def _safe_qsize(q: Any) -> int:
    try:
        return int(q.qsize())
    except Exception:
        return -1


if __name__ == "__main__":
    if sys.platform.startswith("win"):
        import multiprocessing as mp
        mp.freeze_support()
    main()
