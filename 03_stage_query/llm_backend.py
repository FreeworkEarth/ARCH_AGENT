"""
LLM backend abstraction for ARCH_AGENT.

Supports Ollama (local), vLLM (OpenAI-compatible, cluster), and any OpenAI-compatible
HTTP API (GLM-5 via ZhipuAI, OpenAI, etc.).

Configuration via environment variables:
  ARCH_AGENT_LLM_BACKEND   = ollama | vllm | api   (default: ollama)
  ARCH_AGENT_LLM_BASE_URL  = http://host:port       (default: http://localhost:11434 for ollama,
                                                       http://localhost:8000 for vllm)
  ARCH_AGENT_LLM_API_KEY   = <key>                  (required for backend=api; optional for vllm)

Usage:
  from llm_backend import LLMBackend
  llm = LLMBackend(model="deepseek-r1:32b")
  answer = llm.generate(prompt="Explain M-score")

  # Override backend at runtime:
  llm = LLMBackend(model="deepseek-r1:32b", backend="vllm", base_url="http://gpu-node:8000")
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import urllib.error
import urllib.request
from typing import Optional


_OLLAMA_DEFAULT = "http://localhost:11434"
_VLLM_DEFAULT = "http://localhost:8000"

# Claude model mapping: short names → CLI model names
_CLAUDE_MODELS = {
    "claude-opus": "opus",
    "claude-sonnet": "sonnet",
    "claude-haiku": "haiku",
    "opus": "opus",
    "sonnet": "sonnet",
    "haiku": "haiku",
}


def _strip_thinking(text: str) -> str:
    """Strip leading Thinking.../Done thinking block emitted by deepseek-r1."""
    if not text:
        return text
    lines = text.splitlines()
    if lines and lines[0].strip().lower().startswith("thinking"):
        for i, ln in enumerate(lines[:300]):
            if "done thinking" in (ln or "").lower():
                return "\n".join(lines[i + 1:]).lstrip()
    return text


class LLMBackend:
    """Swappable LLM backend: Ollama / vLLM / OpenAI-compatible HTTP API."""

    def __init__(
        self,
        model: str = "deepseek-r1:32b",
        backend: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        num_ctx: int = 4096,
    ):
        self.model = model
        self.num_ctx = num_ctx

        # Auto-detect claude backend from model name
        if backend:
            self.backend = backend.lower()
        elif model.lower().replace("-", "").replace(" ", "") in ("claudeopus", "claudesonnet", "claudehaiku") or model.lower() in _CLAUDE_MODELS:
            self.backend = "claude"
        else:
            self.backend = os.environ.get("ARCH_AGENT_LLM_BACKEND", "ollama").lower()

        self.api_key = api_key or os.environ.get("ARCH_AGENT_LLM_API_KEY", "")

        if base_url:
            self.base_url = base_url.rstrip("/")
        elif os.environ.get("ARCH_AGENT_LLM_BASE_URL"):
            self.base_url = os.environ["ARCH_AGENT_LLM_BASE_URL"].rstrip("/")
        elif self.backend == "vllm":
            self.base_url = _VLLM_DEFAULT
        elif self.backend == "claude":
            self.base_url = ""  # not used for Claude CLI
        else:
            self.base_url = _OLLAMA_DEFAULT

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self, prompt: str, system: Optional[str] = None, timeout_s: int = 900) -> str:
        """Generate a response. Returns plain text (thinking block stripped)."""
        if self.backend == "claude":
            raw = self._claude_cli(prompt, system, timeout_s)
        elif self.backend == "ollama":
            raw = self._ollama_api(prompt, timeout_s)
        elif self.backend in ("vllm", "api"):
            raw = self._openai_compat(prompt, system, timeout_s)
        else:
            raise ValueError(f"Unknown LLM backend: {self.backend!r}. Use ollama, vllm, api, or claude.")
        return _strip_thinking(raw)

    # ------------------------------------------------------------------
    # Backend implementations
    # ------------------------------------------------------------------

    def _claude_cli(self, prompt: str, system: Optional[str], timeout_s: int) -> str:
        """
        Call Claude via the Claude CLI (uses subscription, no API costs).
        Runs: claude -p --model <model> with prompt on stdin.
        """
        claude_bin = shutil.which("claude")
        if not claude_bin:
            return "[LLM ERROR: Claude CLI not found. Install: https://docs.anthropic.com/en/docs/claude-cli]"

        # Map model name to CLI model variant
        cli_model = _CLAUDE_MODELS.get(self.model.lower(), "opus")

        # Build full prompt with system message if provided
        full_prompt = prompt
        if system:
            full_prompt = f"{system}\n\n{prompt}"

        cmd = [claude_bin, "-p", "--model", cli_model]

        try:
            result = subprocess.run(
                cmd,
                input=full_prompt,
                capture_output=True,
                text=True,
                timeout=timeout_s,
            )
            if result.returncode != 0:
                err = (result.stderr or "").strip()[:300]
                return f"[LLM ERROR: Claude CLI returned code {result.returncode}: {err}]"
            return (result.stdout or "").strip()
        except subprocess.TimeoutExpired:
            return f"[LLM ERROR: Claude CLI timed out after {timeout_s}s]"
        except Exception as e:
            return f"[LLM ERROR: Claude CLI failed: {e}]"

    def _ollama_api(self, prompt: str, timeout_s: int) -> str:
        """
        Call Ollama via HTTP /api/generate — supports num_ctx option to control
        context window size and avoid loading the model with 131k default context.
        """
        payload = json.dumps({
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0,
                "num_ctx": self.num_ctx,
            },
        }).encode("utf-8")

        url = f"{self.base_url}/api/generate"
        req = urllib.request.Request(
            url, data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            return (data.get("response") or "").strip()
        except TimeoutError:
            return "[LLM ERROR: timed out]"
        except urllib.error.URLError as e:
            if "refused" in str(e).lower():
                return "[LLM ERROR: Ollama not running — start with: ollama serve]"
            return f"[LLM ERROR: {e}]"
        except Exception as e:
            return f"[LLM ERROR: {e}]"

    def _openai_compat(self, prompt: str, system: Optional[str], timeout_s: int) -> str:
        """
        Call any OpenAI-compatible /v1/chat/completions endpoint.
        Works for: vLLM, ZhipuAI (GLM-5), OpenAI, local proxies.
        """
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload = json.dumps({
            "model": self.model,
            "messages": messages,
            "temperature": 0,
            "max_tokens": 4096,
        }).encode("utf-8")

        url = f"{self.base_url}/v1/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            return (
                data.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
                .strip()
            )
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            return f"[LLM ERROR: HTTP {e.code} — {body[:300]}]"
        except Exception as e:
            return f"[LLM ERROR: {e}]"

    # ------------------------------------------------------------------
    # Factory helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_env(cls, model: str = "deepseek-r1:32b") -> "LLMBackend":
        """Create backend from environment variables (convenience method)."""
        return cls(model=model)

    def __repr__(self) -> str:
        return f"LLMBackend(backend={self.backend!r}, model={self.model!r}, base_url={self.base_url!r})"
