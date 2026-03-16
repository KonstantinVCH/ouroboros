"""
Ouroboros — Text-based tool call protocol for models that don't support native tool_calls.

Used primarily for Zhipu and similar providers. Extracted from loop.py to keep it under 1000 lines.
"""
from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional


def is_zhipu_model(model: str) -> bool:
    m = str(model or "").strip().lower()
    return m.startswith("zhipu/") or m.startswith("zhipuai/")


def text_tool_protocol_enabled(active_model: str) -> bool:
    """
    Enable a JSON-based tool call protocol for providers that don't return native tool_calls.

    Controlled by env:
      - OUROBOROS_TEXT_TOOL_PROTOCOL=off|0|false  -> disabled
      - OUROBOROS_TEXT_TOOL_PROTOCOL=on|1|true    -> enabled
      - OUROBOROS_TEXT_TOOL_PROTOCOL=auto         -> enabled only for Zhipu models
    """
    mode = str(os.environ.get("OUROBOROS_TEXT_TOOL_PROTOCOL", "off") or "").strip().lower()
    if mode in {"0", "off", "false", "no"}:
        return False
    if mode in {"1", "on", "true", "yes"}:
        return True
    return is_zhipu_model(active_model)


def maybe_inject_text_tool_protocol(messages: List[Dict[str, Any]], active_model: str) -> None:
    """
    If the active model doesn't support native OpenAI tool_calls,
    inject a JSON-based protocol so the model can still request tool execution.
    """
    if not text_tool_protocol_enabled(active_model):
        return

    marker = "[TOOL_PROTOCOL_JSON_V1]"
    for m in messages:
        if m.get("role") == "system" and marker in str(m.get("content") or ""):
            return

    messages.append({
        "role": "system",
        "content": (
            f"{marker}\n"
            "You are running inside an agent loop with tools.\n"
            "When the user asks anything that requires reading/modifying the repo, running commands, web searching, or checking state,\n"
            "you MUST use tools (do not guess).\n\n"
            "If you need to use tools, DO NOT describe the action in prose.\n"
            "Instead, output ONLY a fenced JSON block in this exact shape:\n\n"
            "```tool_calls\n"
            "{\"tool_calls\": [{\"name\": \"repo_list\", \"arguments\": {}}]}\n"
            "```\n\n"
            "Rules:\n"
            "- Use only tools that are available in the tool list.\n"
            "- `arguments` must be a JSON object.\n"
            "- When you output a tool_calls block, do not include any other text outside the block.\n"
            "- If you do NOT need tools, respond normally.\n"
        ),
    })


def extract_tool_calls_from_text(text: Optional[str]) -> List[Dict[str, Any]]:
    """
    Parse JSON tool call protocol from model text.

    Accepts blocks like:
    ```tool_calls
    {"tool_calls":[{"name":"repo_list","arguments":{}}]}
    ```
    """
    if not text or not isinstance(text, str):
        return []
    t = text.strip()
    if "```" not in t:
        return []

    blocks = t.split("```")
    candidates: List[str] = []
    for i in range(1, len(blocks), 2):
        blk = blocks[i].strip()
        if not blk:
            continue
        lines = blk.splitlines()
        if len(lines) >= 2 and len(lines[0]) <= 24 and all(ch.isalnum() or ch in ("_", "-") for ch in lines[0].strip()):
            lang = lines[0].strip().lower()
            if lang in ("json", "tool_calls", "tool", "tools", "python"):
                blk = "\n".join(lines[1:]).strip()
        candidates.append(blk)

    for cand in candidates:
        cand_s = cand.strip()
        try:
            obj = json.loads(cand_s)
        except Exception:
            try:
                repaired = re.sub(r",\s*([\]}])", r"\1", cand_s)
                obj = json.loads(repaired)
            except Exception:
                continue

        tool_calls_raw = None
        if isinstance(obj, dict) and isinstance(obj.get("tool_calls"), list):
            tool_calls_raw = obj["tool_calls"]
        elif isinstance(obj, dict) and isinstance(obj.get("tool"), str):
            tool_calls_raw = [{"name": obj.get("tool"), "arguments": obj.get("arguments") or {}}]

        if not tool_calls_raw:
            continue

        out: List[Dict[str, Any]] = []
        for idx, tc in enumerate(tool_calls_raw):
            if not isinstance(tc, dict):
                continue
            name = str(tc.get("name") or tc.get("tool") or "").strip()
            args = tc.get("arguments", {})
            if not name:
                continue
            try:
                args_str = json.dumps(args if isinstance(args, dict) else {}, ensure_ascii=False)
            except Exception:
                args_str = "{}"
            out.append({
                "id": str(tc.get("id") or f"text_tc_{idx}"),
                "type": "function",
                "function": {"name": name, "arguments": args_str},
            })
        if out:
            return out

    return []


def looks_like_misformatted_tool_use(text: Optional[str]) -> bool:
    """Heuristic: model tried to run tools by printing code/thoughts instead of tool_calls JSON."""
    if not text or not isinstance(text, str):
        return False
    t = text.lower()
    if (
        "```python" not in t
        and "<think" not in t
        and "</think" not in t
        and ("```" not in t or "tool_calls" not in t)
    ):
        return False

    if "tool_calls" in t and "```" in t:
        return True
    needles = (
        "drive_read(", "drive_write(", "drive_list(",
        "repo_read(", "repo_list(", "repo_write(", "repo_edit(",
        "shell(", "web_search(", "browse_page(", "browser_action(",
        "switch_model(", "enable_tools(", "list_available_tools(",
    )
    return any(n in t for n in needles)
