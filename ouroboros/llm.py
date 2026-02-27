"""
Ouroboros — LLM client.

The only module that communicates with the LLM API (OpenRouter).
Contract: chat(), default_model(), available_models(), add_usage().
"""

from __future__ import annotations

import logging
import os
import pathlib
import time
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger(__name__)

DEFAULT_LIGHT_MODEL = "google/gemini-3-pro-preview"

# Optional Zhipu AI SDK (ChatGLM). Used when model id starts with "zhipu/".
try:
    from zhipuai import ZhipuAI  # type: ignore
except Exception:  # pragma: no cover
    ZhipuAI = None  # type: ignore


def _is_zhipu_model_id(model: str) -> bool:
    m = str(model or "").strip().lower()
    return m.startswith("zhipu/") or m.startswith("zhipuai/")


def _zhipu_api_key() -> str:
    return (
        str(os.environ.get("OUROBOROS_ZHIPU_API_KEY") or "").strip()
        or str(os.environ.get("ZHIPUAI_API_KEY") or "").strip()
        or str(os.environ.get("ZHIPU_API_KEY") or "").strip()
    )


def _zhipu_base_url() -> str:
    # Zhipu v4 API base. Allow override for enterprise/mirrors.
    return str(os.environ.get("OUROBOROS_ZHIPU_BASE_URL") or "https://open.bigmodel.cn/api/paas/v4/").strip()


def _strip_zhipu_prefix(model: str) -> str:
    m = str(model or "").strip()
    if m.lower().startswith("zhipu/"):
        return m.split("/", 1)[1].strip()
    if m.lower().startswith("zhipuai/"):
        return m.split("/", 1)[1].strip()
    return m


def _env_llm_base_url() -> str:
    return str(os.environ.get("OUROBOROS_LLM_BASE_URL") or "").strip()


def _is_openrouter_base_url(base_url: str) -> bool:
    b = str(base_url or "").lower()
    return "openrouter.ai" in b


def _is_anthropic_base_url(base_url: str) -> bool:
    b = str(base_url or "").lower()
    return "api.anthropic.com" in b


def normalize_reasoning_effort(value: str, default: str = "medium") -> str:
    allowed = {"none", "minimal", "low", "medium", "high", "xhigh"}
    v = str(value or "").strip().lower()
    return v if v in allowed else default


def reasoning_rank(value: str) -> int:
    order = {"none": 0, "minimal": 1, "low": 2, "medium": 3, "high": 4, "xhigh": 5}
    return int(order.get(str(value or "").strip().lower(), 3))


def add_usage(total: Dict[str, Any], usage: Dict[str, Any]) -> None:
    """Accumulate usage from one LLM call into a running total."""
    for k in ("prompt_tokens", "completion_tokens", "total_tokens", "cached_tokens", "cache_write_tokens"):
        total[k] = int(total.get(k) or 0) + int(usage.get(k) or 0)
    if usage.get("cost"):
        total["cost"] = float(total.get("cost") or 0) + float(usage["cost"])


def fetch_openrouter_pricing() -> Dict[str, Tuple[float, float, float]]:
    """
    Fetch current pricing from OpenRouter API.

    Returns dict of {model_id: (input_per_1m, cached_per_1m, output_per_1m)}.
    Returns empty dict on failure.
    """
    import logging
    log = logging.getLogger("ouroboros.llm")

    try:
        import requests
    except ImportError:
        log.warning("requests not installed, cannot fetch pricing")
        return {}

    # If the runtime is pointed at a local LLM (Ollama/LM Studio/etc),
    # don't spam OpenRouter pricing fetches.
    if _env_llm_base_url() and not _is_openrouter_base_url(_env_llm_base_url()):
        return {}

    try:
        url = "https://openrouter.ai/api/v1/models"
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()

        data = resp.json()
        models = data.get("data", [])

        # Prefixes we care about
        prefixes = ("anthropic/", "openai/", "google/", "meta-llama/", "x-ai/", "qwen/")

        pricing_dict = {}
        for model in models:
            model_id = model.get("id", "")
            if not model_id.startswith(prefixes):
                continue

            pricing = model.get("pricing", {})
            if not pricing or not pricing.get("prompt"):
                continue

            # OpenRouter pricing is in dollars per token (raw values)
            raw_prompt = float(pricing.get("prompt", 0))
            raw_completion = float(pricing.get("completion", 0))
            raw_cached_str = pricing.get("input_cache_read")
            raw_cached = float(raw_cached_str) if raw_cached_str else None

            # Convert to per-million tokens
            prompt_price = round(raw_prompt * 1_000_000, 4)
            completion_price = round(raw_completion * 1_000_000, 4)
            if raw_cached is not None:
                cached_price = round(raw_cached * 1_000_000, 4)
            else:
                cached_price = round(prompt_price * 0.1, 4)  # fallback: 10% of prompt

            # Sanity check: skip obviously wrong prices
            if prompt_price > 1000 or completion_price > 1000:
                log.warning(f"Skipping {model_id}: prices seem wrong (prompt={prompt_price}, completion={completion_price})")
                continue

            pricing_dict[model_id] = (prompt_price, cached_price, completion_price)

        log.info(f"Fetched pricing for {len(pricing_dict)} models from OpenRouter")
        return pricing_dict

    except (requests.RequestException, ValueError, KeyError) as e:
        log.warning(f"Failed to fetch OpenRouter pricing: {e}")
        return {}


class LLMClient:
    """LLM API wrapper. Defaults to OpenRouter; can be overridden to local OpenAI-compatible endpoints (e.g. Ollama)."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://openrouter.ai/api/v1",
        base_url_override: Optional[str] = None,
    ):
        env_base = _env_llm_base_url()
        # Allow callers (e.g. model monitor) to force a base_url without being overridden by OUROBOROS_LLM_BASE_URL.
        self._base_url = (str(base_url_override).strip() if base_url_override else "") or env_base or base_url
        # Prefer explicit override for local endpoints; fall back to OpenRouter key.
        self._api_key = (
            api_key
            or os.environ.get("OUROBOROS_LLM_API_KEY", "")
            or os.environ.get("OPENROUTER_API_KEY", "")
            or "ollama"
        )
        self._client = None
        self._zhipu_client = None

    def _read_model_override_file(self) -> str:
        path_raw = str(os.environ.get("OUROBOROS_MODEL_OVERRIDE_FILE", "") or "").strip()
        if not path_raw:
            return ""
        try:
            p = pathlib.Path(path_raw).expanduser()
            if not p.exists() or not p.is_file():
                return ""
            raw = p.read_text(encoding="utf-8").strip()
            # Single-line model id only
            return raw.splitlines()[0].strip() if raw else ""
        except Exception:
            return ""

    def _get_client(self):
        if self._client is None:
            if _is_anthropic_base_url(self._base_url):
                from anthropic import Anthropic
                self._client = Anthropic(api_key=self._api_key)
            else:
                from openai import OpenAI
                headers = {"X-Title": "Ouroboros"}
                if _is_openrouter_base_url(self._base_url):
                    headers["HTTP-Referer"] = "https://colab.research.google.com/"
                self._client = OpenAI(base_url=self._base_url, api_key=self._api_key, default_headers=headers)
        return self._client

    def _get_zhipu_client(self):
        if self._zhipu_client is None:
            key = _zhipu_api_key()
            if not key:
                raise RuntimeError("Missing Zhipu API key (set OUROBOROS_ZHIPU_API_KEY or ZHIPUAI_API_KEY)")
            if ZhipuAI is None:
                raise RuntimeError("Zhipu SDK not installed (pip install zhipuai)")
            # Force v4 base_url; older endpoints can yield "model not found" (code 1211).
            self._zhipu_client = ZhipuAI(api_key=key, base_url=_zhipu_base_url())
        return self._zhipu_client

    def _chat_zhipu(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        tools: Optional[List[Dict[str, Any]]],
        tool_choice: str,
        max_tokens: int,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Zhipu AI chat call (text-only fallback).

        Note: Zhipu tool-calling is not integrated into Ouroboros' OpenAI tool loop;
        this is intended as a resilience fallback when OpenRouter models are unavailable.
        """
        client = self._get_zhipu_client()
        z_model = _strip_zhipu_prefix(model)

        # Zhipu expects OpenAI-like messages but may reject extra fields.
        cleaned: List[Dict[str, Any]] = []
        for m in messages:
            role = str(m.get("role") or "user")
            content = m.get("content")

            # Convert non-string content (blocks) to a plain string best-effort.
            if isinstance(content, list):
                parts: List[str] = []
                for b in content:
                    if isinstance(b, str):
                        parts.append(b)
                    elif isinstance(b, dict):
                        if b.get("type") == "text":
                            parts.append(str(b.get("text", "")))
                        else:
                            parts.append(str(b))
                    else:
                        parts.append(str(b))
                content_str = "\n".join(p for p in parts if p).strip()
            else:
                content_str = str(content or "").strip()

            # OpenAI "tool" role is not universal — map to user message.
            if role == "tool":
                role = "user"
                content_str = f"[tool_result]\n{content_str}".strip()

            cleaned.append({"role": role, "content": content_str})

        kwargs: Dict[str, Any] = {
            "model": z_model,
            "messages": cleaned,
            "max_tokens": max_tokens,
            "temperature": 0.7,
            # Disable "thinking"/reasoning channel to ensure `message.content` is populated
            # and to avoid leaking internal reasoning to the user.
            "thinking": {"type": "disabled"},
        }

        # Best-effort tool calling support (OpenAI-like schema).
        # If the selected Zhipu model doesn't support tools, the API may ignore or reject these fields.
        if tools:
            kwargs["tools"] = tools
            # Zhipu appears to support OpenAI-like tool_choice ("auto"/"none"/{"type":"function"...}).
            kwargs["tool_choice"] = tool_choice or "auto"

        resp = client.chat.completions.create(
            model=z_model,
            messages=cleaned,
            max_tokens=max_tokens,
            temperature=0.7,
            thinking={"type": "disabled"},
        )

        # Extract content + tool_calls (best-effort; SDK shapes can vary).
        tool_calls_out: List[Dict[str, Any]] = []
        try:
            msg_obj = resp.choices[0].message
            content_out = msg_obj.content
            raw_tool_calls = getattr(msg_obj, "tool_calls", None)
            if raw_tool_calls:
                import json as _json
                for i, tc in enumerate(raw_tool_calls):
                    # tc may be dict-like or typed object; normalize via model_dump if present
                    if hasattr(tc, "model_dump"):
                        tc_d = tc.model_dump()
                    elif isinstance(tc, dict):
                        tc_d = tc
                    else:
                        tc_d = dict(tc) if hasattr(tc, "items") else {}

                    fn = tc_d.get("function") or {}
                    name = fn.get("name") or tc_d.get("name") or ""
                    args = fn.get("arguments") if isinstance(fn, dict) else None
                    if args is None:
                        args = tc_d.get("arguments")
                    if isinstance(args, dict):
                        args_str = _json.dumps(args, ensure_ascii=False)
                    else:
                        args_str = str(args or "{}")
                    tool_calls_out.append({
                        "id": tc_d.get("id") or f"zhipu_tc_{i}",
                        "type": "function",
                        "function": {"name": str(name), "arguments": args_str},
                    })
        except Exception:
            # Fallback to dict dump if shape differs
            dumped = getattr(resp, "model_dump", None)
            if callable(dumped):
                data = dumped()
                content_out = (((data.get("choices") or [{}])[0]).get("message") or {}).get("content") or ""
                raw_tool_calls2 = (((data.get("choices") or [{}])[0]).get("message") or {}).get("tool_calls") or []
                if raw_tool_calls2:
                    import json as _json
                    for i, tc_d in enumerate(raw_tool_calls2):
                        fn = (tc_d or {}).get("function") or {}
                        name = fn.get("name") or (tc_d or {}).get("name") or ""
                        args = fn.get("arguments") if isinstance(fn, dict) else None
                        if args is None:
                            args = (tc_d or {}).get("arguments")
                        if isinstance(args, dict):
                            args_str = _json.dumps(args, ensure_ascii=False)
                        else:
                            args_str = str(args or "{}")
                        tool_calls_out.append({
                            "id": (tc_d or {}).get("id") or f"zhipu_tc_{i}",
                            "type": "function",
                            "function": {"name": str(name), "arguments": args_str},
                        })
            else:
                content_out = ""

        msg: Dict[str, Any] = {
            "role": "assistant",
            "content": str(content_out or ""),
            "tool_calls": tool_calls_out,
        }

        # Usage (best-effort; may be missing)
        usage: Dict[str, Any] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "cost": 0.0}
        raw_usage = getattr(resp, "usage", None)
        if raw_usage is not None:
            usage["prompt_tokens"] = int(getattr(raw_usage, "prompt_tokens", 0) or 0)
            usage["completion_tokens"] = int(getattr(raw_usage, "completion_tokens", 0) or 0)
            usage["total_tokens"] = int(getattr(raw_usage, "total_tokens", 0) or (usage["prompt_tokens"] + usage["completion_tokens"]))

        return msg, usage

    def _fetch_generation_cost(self, generation_id: str) -> Optional[float]:
        """Fetch cost from OpenRouter Generation API as fallback."""
        try:
            import requests
            url = f"{self._base_url.rstrip('/')}/generation?id={generation_id}"
            resp = requests.get(url, headers={"Authorization": f"Bearer {self._api_key}"}, timeout=5)
            if resp.status_code == 200:
                data = resp.json().get("data") or {}
                cost = data.get("total_cost") or data.get("usage", {}).get("cost")
                if cost is not None:
                    return float(cost)
            # Generation might not be ready yet — retry once after short delay
            time.sleep(0.5)
            resp = requests.get(url, headers={"Authorization": f"Bearer {self._api_key}"}, timeout=5)
            if resp.status_code == 200:
                data = resp.json().get("data") or {}
                cost = data.get("total_cost") or data.get("usage", {}).get("cost")
                if cost is not None:
                    return float(cost)
        except Exception:
            log.debug("Failed to fetch generation cost from OpenRouter", exc_info=True)
            pass
        return None

    def _chat_anthropic(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        tools: Optional[List[Dict[str, Any]]],
        max_tokens: int,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Anthropic SDK call — translates to/from OpenAI message format."""
        client = self._get_client()

        def _content_to_str(content: Any) -> str:
            """Convert content (str or list of blocks) to plain string."""
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                parts = []
                for block in content:
                    if isinstance(block, str):
                        parts.append(block)
                    elif isinstance(block, dict):
                        if block.get("type") == "text":
                            parts.append(str(block.get("text", "")))
                        elif block.get("type") == "tool_result":
                            parts.append(str(block.get("content", "")))
                return "\n".join(parts)
            return str(content)

        # Extract system prompt (Anthropic has separate `system` param)
        system_parts = [_content_to_str(m["content"]) for m in messages if m.get("role") == "system"]
        system = "\n\n".join(system_parts) if system_parts else None
        non_system = [m for m in messages if m.get("role") != "system"]

        # Normalize message content for Anthropic (must be str or list of content blocks)
        ant_messages = []
        for m in non_system:
            msg_copy = dict(m)
            content = msg_copy.get("content")
            # tool_calls in assistant messages → Anthropic tool_use blocks
            if m.get("role") == "assistant" and m.get("tool_calls"):
                blocks = []
                if content:
                    blocks.append({"type": "text", "text": _content_to_str(content)})
                for tc in m["tool_calls"]:
                    import json as _json
                    fn = tc.get("function", {})
                    try:
                        inp = _json.loads(fn.get("arguments", "{}"))
                    except Exception:
                        inp = {}
                    blocks.append({
                        "type": "tool_use",
                        "id": tc.get("id", ""),
                        "name": fn.get("name", ""),
                        "input": inp,
                    })
                msg_copy = {"role": "assistant", "content": blocks}
            elif m.get("role") == "tool":
                # OpenAI tool result → Anthropic user message with tool_result block
                msg_copy = {
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": m.get("tool_call_id", ""),
                        "content": _content_to_str(content),
                    }]
                }
            elif isinstance(content, list):
                # Already a list of blocks — pass as-is
                pass
            else:
                msg_copy["content"] = _content_to_str(content)
            ant_messages.append(msg_copy)

        # Build Anthropic tool schema
        ant_tools = None
        if tools:
            ant_tools = []
            for t in tools:
                fn = t.get("function", t)
                ant_tools.append({
                    "name": fn.get("name", ""),
                    "description": fn.get("description", ""),
                    "input_schema": fn.get("parameters", {"type": "object", "properties": {}}),
                })

        kwargs: Dict[str, Any] = {
            "model": model,
            "messages": ant_messages,
            "max_tokens": max_tokens,
        }
        if system:
            kwargs["system"] = system
        if ant_tools:
            kwargs["tools"] = ant_tools

        resp = client.messages.create(**kwargs)

        # Convert Anthropic response → OpenAI-style message dict
        tool_calls = []
        text_content = ""
        for block in resp.content:
            if block.type == "text":
                text_content = block.text
            elif block.type == "tool_use":
                import json as _json
                tool_calls.append({
                    "id": block.id,
                    "type": "function",
                    "function": {
                        "name": block.name,
                        "arguments": _json.dumps(block.input),
                    },
                })

        msg: Dict[str, Any] = {"role": "assistant", "content": text_content}
        if tool_calls:
            msg["tool_calls"] = tool_calls

        usage: Dict[str, Any] = {
            "prompt_tokens": resp.usage.input_tokens,
            "completion_tokens": resp.usage.output_tokens,
            "total_tokens": resp.usage.input_tokens + resp.usage.output_tokens,
            "cached_tokens": getattr(resp.usage, "cache_read_input_tokens", 0) or 0,
            "cache_write_tokens": getattr(resp.usage, "cache_creation_input_tokens", 0) or 0,
            "cost": 0.0,
        }
        return msg, usage

    def chat(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        tools: Optional[List[Dict[str, Any]]] = None,
        reasoning_effort: str = "medium",
        max_tokens: int = 16384,
        tool_choice: str = "auto",
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Single LLM call. Returns: (response_message_dict, usage_dict with cost)."""
        # Special-case Zhipu model IDs: "zhipu/<glm-model>".
        if _is_zhipu_model_id(model):
            return self._chat_zhipu(messages=messages, model=model, tools=tools, tool_choice=tool_choice, max_tokens=max_tokens)

        client = self._get_client()
        effort = normalize_reasoning_effort(reasoning_effort)

        extra_body: Dict[str, Any] = {}
        if _is_openrouter_base_url(self._base_url):
            # OpenRouter supports this knob; some local servers may reject unknown fields.
            extra_body = {"reasoning": {"effort": effort, "exclude": True}}

        # Pin Anthropic models to Anthropic provider for prompt caching
        if _is_openrouter_base_url(self._base_url) and model.startswith("anthropic/"):
            extra_body["provider"] = {
                "order": ["Anthropic"],
                "allow_fallbacks": False,
                "require_parameters": True,
            }

        # Groq free tier rejects requests with max_tokens > 8192
        if "groq.com" in str(self._base_url).lower() and max_tokens > 8192:
            max_tokens = 8192

        kwargs: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
        }
        if extra_body:
            kwargs["extra_body"] = extra_body
        if tools:
            # OpenRouter/Anthropic supports tool schema caching. Local endpoints may reject unknown fields.
            if _is_openrouter_base_url(self._base_url):
                tools_with_cache = [t for t in tools]  # shallow copy
                if tools_with_cache:
                    last_tool = {**tools_with_cache[-1]}  # copy last tool
                    last_tool["cache_control"] = {"type": "ephemeral", "ttl": "1h"}
                    tools_with_cache[-1] = last_tool
                kwargs["tools"] = tools_with_cache
            else:
                kwargs["tools"] = tools
            kwargs["tool_choice"] = tool_choice

        if _is_anthropic_base_url(self._base_url):
            return self._chat_anthropic(messages, model, tools, max_tokens)

        resp = client.chat.completions.create(**kwargs)
        resp_dict = resp.model_dump()
        usage = resp_dict.get("usage") or {}
        choices = resp_dict.get("choices") or [{}]
        msg = (choices[0] if choices else {}).get("message") or {}

        # Extract cached_tokens from prompt_tokens_details if available
        if not usage.get("cached_tokens"):
            prompt_details = usage.get("prompt_tokens_details") or {}
            if isinstance(prompt_details, dict) and prompt_details.get("cached_tokens"):
                usage["cached_tokens"] = int(prompt_details["cached_tokens"])

        # Extract cache_write_tokens from prompt_tokens_details if available
        # OpenRouter: "cache_write_tokens"
        # Native Anthropic: "cache_creation_tokens" or "cache_creation_input_tokens"
        if not usage.get("cache_write_tokens"):
            prompt_details_for_write = usage.get("prompt_tokens_details") or {}
            if isinstance(prompt_details_for_write, dict):
                cache_write = (prompt_details_for_write.get("cache_write_tokens")
                              or prompt_details_for_write.get("cache_creation_tokens")
                              or prompt_details_for_write.get("cache_creation_input_tokens"))
                if cache_write:
                    usage["cache_write_tokens"] = int(cache_write)

        # Ensure cost is present in usage (OpenRouter includes it, but fallback if missing)
        if not usage.get("cost"):
            gen_id = resp_dict.get("id") or ""
            if gen_id:
                cost = self._fetch_generation_cost(gen_id)
                if cost is not None:
                    usage["cost"] = cost

        return msg, usage

    def vision_query(
        self,
        prompt: str,
        images: List[Dict[str, Any]],
        model: str = "anthropic/claude-sonnet-4.6",
        max_tokens: int = 1024,
        reasoning_effort: str = "low",
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Send a vision query to an LLM. Lightweight — no tools, no loop.

        Args:
            prompt: Text instruction for the model
            images: List of image dicts. Each dict must have either:
                - {"url": "https://..."} — for URL images
                - {"base64": "<b64>", "mime": "image/png"} — for base64 images
            model: VLM-capable model ID
            max_tokens: Max response tokens
            reasoning_effort: Effort level

        Returns:
            (text_response, usage_dict)
        """
        # Build multipart content
        content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
        for img in images:
            if "url" in img:
                content.append({
                    "type": "image_url",
                    "image_url": {"url": img["url"]},
                })
            elif "base64" in img:
                mime = img.get("mime", "image/png")
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime};base64,{img['base64']}"},
                })
            else:
                log.warning("vision_query: skipping image with unknown format: %s", list(img.keys()))

        messages = [{"role": "user", "content": content}]
        response_msg, usage = self.chat(
            messages=messages,
            model=model,
            tools=None,
            reasoning_effort=reasoning_effort,
            max_tokens=max_tokens,
        )
        text = response_msg.get("content") or ""
        return text, usage

    def default_model(self) -> str:
        """Return the single default model from env. LLM switches via tool if needed."""
        override = self._read_model_override_file()
        if override:
            return override
        return os.environ.get("OUROBOROS_MODEL", "anthropic/claude-sonnet-4.6")

    def available_models(self) -> List[str]:
        """Return list of available models from env (for switch_model tool schema)."""
        override = self._read_model_override_file()
        main = os.environ.get("OUROBOROS_MODEL", "anthropic/claude-sonnet-4.6")
        code = os.environ.get("OUROBOROS_MODEL_CODE", "")
        light = os.environ.get("OUROBOROS_MODEL_LIGHT", "")
        models = [m for m in [override, main] if m]
        if code and code != main:
            models.append(code)
        if light and light != main and light != code:
            models.append(light)
        return models
