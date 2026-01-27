# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

# utils_LLM.py
from __future__ import annotations
import os, time, json, re
from typing import Any, Callable, Dict, Optional
from dotenv import load_dotenv

try:
    import tiktoken  # type: ignore
except Exception:
    tiktoken = None  # optional

try:
    from saver import saver
    log = saver.log_info
except Exception:
    def log(msg: str):  # fallback logger
        print(msg)


# =========================
# Public API
# =========================

def get_llm(model_name: str, llm_engine_type: Optional[str] = None, **llm_args) -> Any:
    """
    Returns an llm_agent handle that llm_inference(...) can use.

    Accepts:
      - provider names: "openai", "anthropic", "http"
      - model ids (esp. OpenRouter style like "x-ai/grok-4-fast:free")
        -> treated as OpenAI-compatible via OpenRouter unless overridden.

    Common llm_args:
      - client          : prebuilt client (OpenAI or Anthropic). If omitted, we create one.
      - base_url        : override base URL (defaults to OpenRouter if model id includes "/" or ":")
      - api_key_env     : env var name to fetch API key from (default OPENAI_API_KEY or OPENROUTER_API_KEY heuristic)
      - model           : optional explicit model override (defaults to model_name)
      - headers         : dict of extra headers (for OpenRouter Referer/Title)
      - system, temperature, max_tokens, stop : standard gen args (optional)
    """
    name = (model_name or "").strip().lower()
    model_id = llm_args.get("model") or model_name

    # If caller passed a ready-made callable, return it
    fn = llm_args.get("fn")
    if llm_engine_type == "callable" or callable(fn):
        if not callable(fn):
            raise TypeError("get_llm('callable') requires fn=<callable>.")
        return fn

    # Provider explicitly requested
    provider = (llm_engine_type or "").strip().lower()
    if provider in {"openai", "anthropic", "http"}:
        return _make_agent(provider=provider, model=model_id, **llm_args)

    # Otherwise infer:
    # - If name matches a known provider, use it
    if name in {"openai", "anthropic", "http", "rest", "vllm", "tgi"}:
        mapped = "http" if name in {"http", "rest", "vllm", "tgi"} else name
        return _make_agent(provider=mapped, model=model_id, **llm_args)

    # - If it *looks* like an OpenRouter model id ("vendor/model" or ":tag"), route to OpenAI-compatible client @ OpenRouter
    if "/" in model_name or ":" in model_name:
        return _make_agent(provider="openai", model=model_id, default_to_openrouter=True, **llm_args)

    # - Fallback: assume OpenAI-compatible local/OpenAI
    return _make_agent(provider="openai", model=model_id, **llm_args)


def llm_inference(llm_agent: Any, prompt: str, tag: str) -> str:
    """
    Execute a single-turn chat-style request and return the text.
    llm_agent can be:
      - a callable: fn(prompt, tag) -> str
      - a dict from get_llm()
    """
    if llm_agent is None:
        raise ValueError("llm_inference: llm_agent is None")

    if callable(llm_agent):
        return _ensure_text(llm_agent(prompt, tag))

    if isinstance(llm_agent, dict) and "provider" in llm_agent:
        prov = llm_agent["provider"]
        if prov == "openai":
            return _call_openai(llm_agent, prompt, tag)
        if prov == "anthropic":
            return _call_anthropic(llm_agent, prompt, tag)
        if prov == "http":
            return _call_http(llm_agent, prompt, tag)

    raise TypeError("llm_inference: Unsupported llm_agent. Use get_llm(...) or a callable.")


def count_prompt_tokens(text: str, model_name: Optional[str] = None) -> int:
    """
    Token count with best-effort accuracy.
    Uses tiktoken if present (encoding_for_model or cl100k_base fallback), else heuristics.
    """
    s = (text or "").strip()
    if not s:
        return 0

    if tiktoken is not None:
        enc = _get_tiktoken_encoding(model_name)
        if enc is not None:
            try:
                return len(enc.encode(s))
            except Exception:
                pass

    # Heuristics: conservative (take max)
    approx = max(1, int(round(len(s) / 4.0)))           # ~4 chars per token
    ws_tokens = max(1, len(re.findall(r"\S+", s)))      # whitespace split
    return max(approx, ws_tokens)


# =========================
# Provider builders
# =========================

def _make_agent(*, provider: str, model: str, **kw) -> Dict[str, Any]:
    agent: Dict[str, Any] = {
        "provider": provider,
        "model": model
    }

    if provider == "openai":
        client = kw.get("client")
        base_url = kw.get("base_url")
        headers = kw.get("headers") or kw.get("extra_headers") or {}
        api_key = kw.get("api_key")

        if not api_key:
            # 1. Try to get the OpenRouter key first, then the OpenAI key as a fallback
            # getenv returns None if neither exists
            load_dotenv()
            api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
            print(f'api_key:{api_key}')
            # 2. If both returned None, raise your error
            if not api_key:
                raise RuntimeError("No API key found: set OPENROUTER_API_KEY or OPENAI_API_KEY.")
        
        if client is None:
            try:
                from openai import OpenAI  
                client = OpenAI(api_key=api_key, base_url=base_url, default_headers=headers or None)
            except Exception as e:
                raise RuntimeError("OpenAI client not provided and auto-import failed.") from e

        agent.update({"client": client})
        return agent

    if provider == "anthropic":
        client = kw.get("client")
        if client is None:
            try:
                import anthropic  # type: ignore
                key = os.getenv(kw.get("api_key_env") or "ANTHROPIC_API_KEY")
                if not key:
                    raise RuntimeError("No API key found: set ANTHROPIC_API_KEY.")
                client = anthropic.Anthropic(api_key=key)
            except Exception as e:
                raise RuntimeError("Anthropic client not provided and auto-import failed.") from e
        agent.update({"client": client})
        return agent

    if provider == "http":
        base_url = kw.get("base_url")
        if not base_url:
            raise ValueError("get_llm('http') requires base_url=...")
        headers = kw.get("headers", {"Content-Type": "application/json"})
        agent.update({
            "base_url": base_url,
            "headers": headers
        })
        return agent

    raise ValueError(f"Unsupported provider: {provider!r}")


# =========================
# Provider calls
# =========================

def _call_openai(agent: Dict[str, Any], prompt: str, tag: str) -> str:
    client = agent["client"]
    model = agent["model"]

    def do_chat():
        return client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": prompt},
            ],
        )

    # Try chat.completions (OpenAI v1)
    if hasattr(client, "chat") and hasattr(client.chat, "completions"):
        return _retry(lambda: _extract_text_openai(do_chat()), f"openai.chat:{model}")

    # Try responses API
    if hasattr(client, "responses"):
        def do_resp():
            return client.responses.create(
                model=model,
                input=prompt,
            )
        return _retry(lambda: _extract_text_openai(do_resp()), f"openai.responses:{model}")

    raise RuntimeError("OpenAI client has no known methods (chat.completions/responses).")


def _call_anthropic(agent: Dict[str, Any], prompt: str, tag: str) -> str:
    client = agent["client"]
    model = agent["model"]
    system = agent.get("system", "You are a helpful assistant.")
    temperature = agent.get("temperature", 0.0)
    max_tokens = agent.get("max_tokens", 1024)
    stop = agent.get("stop")

    if not (hasattr(client, "messages") and hasattr(client.messages, "create")):
        raise RuntimeError("Anthropic client lacks messages.create().")

    def do():
        resp = client.messages.create(
            model=model,
            system=f"{system} (tag={tag})",
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
            stop_sequences=stop,
        )
        return _extract_text_anthropic(resp)

    return _retry(do, f"anthropic:{model}")


def _call_http(agent: Dict[str, Any], prompt: str, tag: str) -> str:
    import urllib.request
    base_url = agent["base_url"]
    headers = agent.get("headers", {"Content-Type": "application/json"})
    model = agent.get("model", "")
    system = agent.get("system", "You are a helpful assistant.")
    temperature = agent.get("temperature", 0.0)
    max_tokens = agent.get("max_tokens", 1024)
    stop = agent.get("stop")

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": f"{system} (tag={tag})"},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if stop:
        payload["stop"] = stop

    data = json.dumps(payload).encode("utf-8")

    def do():
        req = urllib.request.Request(base_url, data=data, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=60) as resp:
            obj = json.loads(resp.read().decode("utf-8"))
        try:
            return obj["choices"][0]["message"]["content"]
        except Exception:
            pass
        try:
            return obj["choices"][0]["text"]
        except Exception:
            pass
        return str(obj)

    return _retry(do, f"http:{base_url}")


# =========================
# Helpers
# =========================

def _retry(fn: Callable[[], str], tag: str, tries: int = 3, backoff: float = 1.5) -> str:
    last = None
    for i in range(1, tries + 1):
        try:
            out = fn()
            return _ensure_text(out)
        except Exception as e:
            last = e
            log(f"[llm_inference] ({tag}) attempt {i}/{tries} failed: {e}")
            if i < tries:
                time.sleep(backoff ** i)
    raise last  # type: ignore[misc]

def _ensure_text(x: Any) -> str:
    if isinstance(x, str):
        return x
    # OpenAI chat.completions
    try: return x.choices[0].message.content  # type: ignore[attr-defined]
    except Exception: pass
    # OpenAI responses
    try: return x.output_text  # type: ignore[attr-defined]
    except Exception: pass
    # Anthropic messages format
    try:
        parts = []
        for block in getattr(x, "content", []) or []:
            if getattr(block, "type", None) == "text":
                parts.append(getattr(block, "text", ""))
        if parts: return "".join(parts)
    except Exception:
        pass
    return str(x)

def _extract_text_openai(resp: Any) -> str:
    try: return resp.choices[0].message.content
    except Exception: pass
    try: return resp.output_text
    except Exception: pass
    return _ensure_text(resp)

def _extract_text_anthropic(resp: Any) -> str:
    parts = []
    for block in getattr(resp, "content", []) or []:
        if getattr(block, "type", None) == "text":
            parts.append(getattr(block, "text", ""))
    return "".join(parts) if parts else _ensure_text(resp)

def _get_tiktoken_encoding(model_name: Optional[str]):
    if tiktoken is None:
        return None
    try:
        if model_name:
            return tiktoken.encoding_for_model(model_name)
        return tiktoken.get_encoding("cl100k_base")
    except Exception:
        try:
            return tiktoken.get_encoding("cl100k_base")
        except Exception:
            return None
