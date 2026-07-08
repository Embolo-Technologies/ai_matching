"""
HTTP client for the vLLM OpenAI-compatible server.
Used only by the GPU batch-matching pipeline (fast_server.py). The
interactive chat/CLI tools (chat.py, server.py, matcher.py) still use
Qwen3Engine + llama-cpp-python directly and are untouched.
"""
import requests
from requests.adapters import HTTPAdapter

from qwen3_engine.config import VLLM_BASE_URL, VLLM_MODEL_NAME, VLLM_MAX_WORKERS

# One shared session with a large connection pool — all worker threads
# reuse it concurrently instead of opening a new TCP connection per request.
_session = requests.Session()
_adapter = HTTPAdapter(pool_connections=VLLM_MAX_WORKERS, pool_maxsize=VLLM_MAX_WORKERS)
_session.mount("http://", _adapter)
_session.mount("https://", _adapter)


class VLLMEngine:
    """Drop-in replacement for the subset of Qwen3Engine's interface that
    FastMatcher needs (system_prompt, clear_history, generate, is_loaded).
    No local model weights — every call is an HTTP request to the vLLM
    server, which does its own continuous batching on the GPU."""

    def __init__(self, base_url: str = None, model_name: str = None, system_prompt: str = ""):
        self.base_url = (base_url or VLLM_BASE_URL).rstrip("/")
        self.model_name = model_name or VLLM_MODEL_NAME
        self.system_prompt = system_prompt
        self._ready = False

    def is_loaded(self) -> bool:
        return self._ready

    def check_ready(self, timeout: float = 3.0) -> bool:
        try:
            resp = _session.get(f"{self.base_url}/models", timeout=timeout)
            self._ready = resp.status_code == 200
        except Exception:
            self._ready = False
        return self._ready

    def clear_history(self) -> None:
        # Matching flow never uses multi-turn history — every call to
        # generate() is an independent request.
        pass

    def generate(self, prompt: str, max_tokens: int = 100, temperature: float = 0.0, **kwargs) -> str:
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        resp = _session.post(f"{self.base_url}/chat/completions", json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()
