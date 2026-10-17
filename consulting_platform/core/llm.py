"""Provider-agnostic LLM client (REST only, no SDKs).

Supported: Gemini (GEMINI_API_KEY / GOOGLE_API_KEY), OpenAI (OPENAI_API_KEY),
Anthropic (ANTHROPIC_API_KEY), Ollama (local), or 'none' (extractive fallback, no LLM).
"""
from __future__ import annotations

import json
import os
import re
import time
from typing import Optional

import requests

import config


class LLMError(RuntimeError):
    pass


import threading

# at most N concurrent requests and a minimum gap between calls (free-tier friendly; tune via env)
_SEM = threading.Semaphore(int(os.getenv("LLM_MAX_CONCURRENCY", "3")))
_LOCK = threading.Lock()
_LAST = [0.0]


def _throttle():
    gap = float(os.getenv("LLM_MIN_INTERVAL", "0.5"))
    with _LOCK:
        wait = _LAST[0] + gap - time.time()
        if wait > 0:
            time.sleep(wait)
        _LAST[0] = time.time()


class LLM:
    def __init__(self, provider: Optional[str] = None):
        self.provider = (provider or config.LLM_PROVIDER or "auto").lower()
        if self.provider == "auto":
            self.provider = self._detect()
        self.model = {
            "gemini": config.GEMINI_MODEL, "openai": config.OPENAI_MODEL,
            "anthropic": config.ANTHROPIC_MODEL, "ollama": config.OLLAMA_MODEL,
        }.get(self.provider, "extractive")
        self.calls, self.last_error = 0, None

    @staticmethod
    def _detect() -> str:
        if os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"):
            return "gemini"
        if os.getenv("OPENAI_API_KEY"):
            return "openai"
        if os.getenv("ANTHROPIC_API_KEY"):
            return "anthropic"
        try:
            requests.get(f"{config.OLLAMA_URL}/api/tags", timeout=0.5)
            return "ollama"
        except Exception:
            return "none"

    @property
    def available(self) -> bool:
        return self.provider != "none"

    @property
    def label(self) -> str:
        return "No LLM (extractive mode)" if not self.available else f"{self.provider} / {self.model}"

    # ------------------------------------------------------------------ public
    def generate(self, prompt: str, system: str = "", temperature: float = 0.2, max_tokens: int = 1800,
                 json_mode: bool = False) -> str:
        if not self.available:
            raise LLMError("No LLM provider configured")
        if time.time() < getattr(self, "_down_until", 0):
            # circuit breaker: after quota exhaustion, fail fast so agents fall back instead of hanging
            raise LLMError(f"{self.provider} temporarily unavailable (rate limit/quota): {self.last_error}")
        fn = getattr(self, f"_{self.provider}")
        last = None
        for attempt in range(3):
            try:
                with _SEM:
                    _throttle()
                    self.calls += 1
                    return fn(prompt, system, temperature, max_tokens, json_mode)
            except requests.HTTPError as e:
                last = e
                code = e.response.status_code if e.response is not None else 0
                body = e.response.text if e.response is not None else ""
                if code == 429 and re.search(r"PerDay|per day|daily", body, re.I):
                    self.last_error = "daily quota exhausted"
                    self._down_until = time.time() + 3600
                    raise LLMError(f"{self.provider}: daily quota exhausted") from e
                if code in (429, 500, 502, 503, 529):
                    # free tiers rate-limit per minute: honour the server's retry hint when present
                    m = re.search(r'"retryDelay":\s*"(\d+)', body)
                    time.sleep(min(int(m.group(1)) + 1, 30) if m else 4 * (attempt + 1))
                    continue
                raise LLMError(f"{self.provider} HTTP {code}: {body[:300]}") from e
            except requests.RequestException as e:
                last = e
                time.sleep(1.5)
        self.last_error = str(last)[:200]
        self._down_until = time.time() + 45
        raise LLMError(f"{self.provider} failed: {last}")

    def generate_json(self, prompt: str, system: str = "") -> dict:
        txt = self.generate(prompt, system, temperature=0.0, max_tokens=800, json_mode=True)
        m = re.search(r"\{.*\}", txt, re.S)
        if not m:
            raise LLMError(f"No JSON in response: {txt[:200]}")
        return json.loads(m.group(0))

    # ------------------------------------------------------------------ providers
    def _gemini(self, prompt, system, temperature, max_tokens, json_mode):
        key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
        # newer Gemini models "think" and those tokens count toward maxOutputTokens -> leave generous headroom
        gen = {"temperature": temperature, "maxOutputTokens": max_tokens + 6000}
        if json_mode:
            gen["responseMimeType"] = "application/json"
        body = {"contents": [{"role": "user", "parts": [{"text": prompt}]}], "generationConfig": gen}
        if system:
            body["systemInstruction"] = {"parts": [{"text": system}]}
        r = requests.post(url, headers={"x-goog-api-key": key}, json=body, timeout=120)
        if r.status_code in (401, 403) and not key.startswith("AIza"):
            # some Google credentials are bearer tokens rather than API keys
            r = requests.post(url, headers={"Authorization": f"Bearer {key}"}, json=body, timeout=120)
        if r.status_code == 404 and not getattr(self, "_discovered", False):
            # configured model id retired/unknown -> discover the newest Flash model available to this key
            self._discovered = True
            self.model = self._discover_gemini(key) or self.model
            return self._gemini(prompt, system, temperature, max_tokens, json_mode)
        r.raise_for_status()
        data = r.json()
        try:
            return "".join(p.get("text", "") for p in data["candidates"][0]["content"]["parts"])
        except (KeyError, IndexError):
            raise LLMError(f"Gemini returned no text: {str(data)[:300]}")

    @staticmethod
    def _discover_gemini(key: str) -> Optional[str]:
        try:
            r = requests.get("https://generativelanguage.googleapis.com/v1beta/models",
                             params={"key": key, "pageSize": 200}, timeout=20)
            r.raise_for_status()
            names = [m["name"].split("/")[-1] for m in r.json().get("models", [])
                     if "generateContent" in m.get("supportedGenerationMethods", [])]
            flash = [n for n in names if "flash" in n and not any(x in n for x in
                     ("lite", "image", "tts", "audio", "live", "exp", "preview", "thinking"))]
            def ver(n):
                m = re.search(r"(\d+(?:\.\d+)?)", n)
                return float(m.group(1)) if m else 0.0
            pool = flash or names
            return sorted(pool, key=ver)[-1] if pool else None
        except Exception:
            return None

    def _openai(self, prompt, system, temperature, max_tokens, json_mode):
        msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
        body = {"model": self.model, "messages": msgs, "temperature": temperature, "max_tokens": max_tokens}
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        r = requests.post(os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1") + "/chat/completions",
                          headers={"Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}"}, json=body, timeout=120)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]

    def _anthropic(self, prompt, system, temperature, max_tokens, json_mode):
        body = {"model": self.model, "max_tokens": max_tokens, "temperature": temperature,
                "messages": [{"role": "user", "content": prompt}]}
        if system:
            body["system"] = system
        r = requests.post("https://api.anthropic.com/v1/messages", json=body, timeout=120,
                          headers={"x-api-key": os.getenv("ANTHROPIC_API_KEY", ""),
                                   "anthropic-version": "2023-06-01"})
        r.raise_for_status()
        return "".join(b.get("text", "") for b in r.json()["content"])

    def _ollama(self, prompt, system, temperature, max_tokens, json_mode):
        msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
        body = {"model": self.model, "messages": msgs, "stream": False,
                "options": {"temperature": temperature, "num_predict": max_tokens,
                            "num_ctx": int(os.getenv("OLLAMA_NUM_CTX", "8192"))}}  # default ctx would truncate evidence
        if json_mode:
            body["format"] = "json"
        r = requests.post(f"{config.OLLAMA_URL}/api/chat", json=body, timeout=300)
        r.raise_for_status()
        return r.json()["message"]["content"]


_default: Optional[LLM] = None


def get_llm(provider: Optional[str] = None) -> LLM:
    global _default
    if provider:
        return LLM(provider)
    if _default is None:
        _default = LLM()
    return _default
