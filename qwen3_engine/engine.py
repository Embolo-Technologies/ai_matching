import os
import re
import sys
import time
import threading
from typing import Generator, List, Dict, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qwen3_engine.config import (
    MODEL_REGISTRY, MODELS_DIR, TEMPLATES, DEFAULT_MODEL,
    N_CTX, N_CTX_MATCHER, N_GPU_LAYERS, N_THREADS, N_BATCH,
    MAX_TOKENS, TEMPERATURE, TOP_P, TOP_K, REPEAT_PENALTY,
    THINKING_MODE_DEFAULT, VERBOSE_LLAMA,
)

class Qwen3Engine:
    # One lock per GPU — workers on different GPUs run concurrently.
    # Populated on first use via _get_gpu_lock(); keyed by gpu_id (int).
    _gpu_locks: dict = {}
    _gpu_locks_mutex = threading.Lock()

    @classmethod
    def _get_gpu_lock(cls, gpu_id: int) -> threading.Lock:
        with cls._gpu_locks_mutex:
            if gpu_id not in cls._gpu_locks:
                cls._gpu_locks[gpu_id] = threading.Lock()
            return cls._gpu_locks[gpu_id]

    def __init__(
        self,
        model_key: Optional[str] = None,
        system_prompt: Optional[str] = None,
        thinking_mode: Optional[bool] = None,
        gpu_id: int = 0,
    ):
        self.model_key = model_key or DEFAULT_MODEL
        if self.model_key not in MODEL_REGISTRY:
            raise ValueError(f"Unknown model key: '{self.model_key}'")

        self._cfg          = MODEL_REGISTRY[self.model_key]
        self._template     = TEMPLATES[self._cfg["template"]]
        shm_path = os.path.join("/dev/shm", self._cfg["filename"])
        if os.path.exists(shm_path):
            self._model_path = shm_path
        else:
            self._model_path = os.path.join(MODELS_DIR, self._cfg["filename"])
        self.system_prompt = system_prompt or self._cfg["system_prompt"]
        self.thinking_mode = (
            thinking_mode
            if thinking_mode is not None
            else (THINKING_MODE_DEFAULT and self._cfg["supports_thinking"])
        )
        self._llm        = None
        self._gpu_id     = gpu_id
        self.history: List[Dict[str, str]] = []
        self._load_time  = 0.0
        self._loaded_ctx = 0

    @property
    def model_name(self) -> str:
        return self._cfg["name"]

    @property
    def supports_thinking(self) -> bool:
        return self._cfg["supports_thinking"]

    def load(self, n_ctx: int = None, num_gpus: int = 1) -> None:
        if not os.path.exists(self._model_path):
            print(f"[Engine] Model file not found at {self._model_path}. Downloading automatically...")
            try:
                from qwen3_engine.downloader import download_model
                success = download_model(self.model_key)
                if not success or not os.path.exists(self._model_path):
                    raise FileNotFoundError(f"Auto-download failed for model '{self.model_key}'")
            except Exception as e:
                raise FileNotFoundError(
                    f"Model file not found: {self._model_path} and auto-download failed: {e}\n"
                    f"Download it with:  bash run.sh download {self.model_key}"
                )
        try:
            from llama_cpp import Llama
        except ImportError:
            raise ImportError("llama-cpp-python not installed. Run: bash setup.sh")

        t0 = time.time()
        n_gpu_layers = self._cfg.get("n_gpu_layers", N_GPU_LAYERS)
        effective_ctx = n_ctx if n_ctx is not None else N_CTX
        self._loaded_ctx = effective_ctx

        # Build tensor_split to pin this instance to self._gpu_id.
        # tensor_split[i] = fraction of model layers placed on GPU i.
        # Setting slot gpu_id=1.0 and all others=0.0 routes the entire model
        # to that GPU, regardless of how many GPUs the system exposes.
        if num_gpus > 1:
            tensor_split = [0.0] * num_gpus
            tensor_split[self._gpu_id] = 1.0
        else:
            tensor_split = None

        llm_kwargs = dict(
            model_path=self._model_path,
            n_ctx=effective_ctx,
            n_gpu_layers=n_gpu_layers,
            main_gpu=self._gpu_id,
            n_threads=N_THREADS,
            n_batch=N_BATCH,
            verbose=VERBOSE_LLAMA,
            flash_attn=True,
            use_mmap=True,
        )
        if tensor_split is not None:
            llm_kwargs["tensor_split"] = tensor_split

        self._llm = Llama(**llm_kwargs)
        self._load_time = time.time() - t0

    def is_loaded(self) -> bool:
        return self._llm is not None

    def get_load_time(self) -> float:
        return self._load_time

    def _build_prompt(self, user_message: str) -> str:
        t    = self._template
        tmpl = self._cfg["template"]
        parts = []

        if tmpl == "chatml":
            parts.append(t["system_start"] + self.system_prompt + t["system_end"])
            for turn in self.history:
                if turn["role"] == "user":
                    parts.append(t["user_start"] + turn["content"] + t["user_end"])
                elif turn["role"] == "assistant":
                    parts.append(t["assistant_start"] + turn["content"] + t["assistant_end"])
            parts.append(t["user_start"] + user_message.rstrip() + t["user_end"])
            if not self.thinking_mode:
                parts.append(t["assistant_start"] + "<think>\n\n</think>\n\n")
            else:
                parts.append(t["assistant_start"])

        elif tmpl == "gemma":
            for i, turn in enumerate(self.history):
                if turn["role"] == "user":
                    content = turn["content"]
                    if i == 0 and self.system_prompt:
                        content = self.system_prompt + "\n\n" + content
                    parts.append(t["user_start"] + content + t["user_end"])
                elif turn["role"] == "assistant":
                    parts.append(t["assistant_start"] + turn["content"] + t["assistant_end"])
            user_content = user_message.rstrip()
            if not self.history and self.system_prompt:
                user_content = self.system_prompt + "\n\n" + user_content
            parts.append(t["user_start"] + user_content + t["user_end"])
            parts.append(t["assistant_start"])

        elif tmpl == "llama3":
            parts.append(t["system_start"] + self.system_prompt + t["system_end"])
            for turn in self.history:
                if turn["role"] == "user":
                    parts.append(t["user_start"] + turn["content"] + t["user_end"])
                elif turn["role"] == "assistant":
                    parts.append(t["assistant_start"] + turn["content"] + t["assistant_end"])
            parts.append(t["user_start"] + user_message.rstrip() + t["user_end"])
            parts.append(t["assistant_start"])

        return "".join(parts)

    def clear_history(self) -> None:
        self.history = []

    def _trim_history(self) -> None:
        max_hist_tokens = N_CTX - MAX_TOKENS - 300
        while self.history:
            approx = len(self._build_prompt("[check]")) // 4
            if approx < max_hist_tokens:
                break
            self.history = self.history[2:] if len(self.history) >= 2 else []

    def stream(
        self,
        user_message: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        repeat_penalty: Optional[float] = None,
    ) -> Generator[str, None, None]:
        if not self.is_loaded():
            raise RuntimeError("Model not loaded. Call engine.load() first.")

        self._trim_history()
        prompt = self._build_prompt(user_message)

        effective_max = max_tokens or MAX_TOKENS
        if self.thinking_mode and self.supports_thinking:
            effective_max = effective_max * 4

        raw_tokens = []
        for chunk in self._llm(
            prompt,
            max_tokens=effective_max,
            temperature=temperature if temperature is not None else TEMPERATURE,
            top_p=top_p or TOP_P,
            top_k=top_k or TOP_K,
            repeat_penalty=repeat_penalty or REPEAT_PENALTY,
            stream=True,
            stop=self._template["stop_tokens"],
        ):
            raw_tokens.append(chunk["choices"][0]["text"])

        raw_text   = "".join(raw_tokens)
        clean_text = re.sub(r"<think>.*?</think>\s*", "", raw_text, flags=re.DOTALL).lstrip("\n")

        if clean_text:
            for i in range(0, len(clean_text), 4):
                yield clean_text[i:i + 4]

        assistant_text = clean_text.strip()
        if assistant_text:
            self.history.append({"role": "user",      "content": user_message})
            self.history.append({"role": "assistant",  "content": assistant_text})

    def generate(self, user_message: str, **kwargs) -> str:
        return "".join(self.stream(user_message, **kwargs))

    def info(self) -> dict:
        return {
            "model_key":       self.model_key,
            "model_name":      self.model_name,
            "model_path":      self._model_path,
            "model_loaded":    self.is_loaded(),
            "load_time_s":     round(self._load_time, 2),
            "n_ctx":           self._loaded_ctx or N_CTX,
            "n_gpu_layers":    self._cfg.get("n_gpu_layers", N_GPU_LAYERS),
            "n_threads":       N_THREADS,
            "n_batch":         N_BATCH,
            "thinking_mode":   self.thinking_mode,
            "supports_thinking": self.supports_thinking,
            "template":        self._cfg["template"],
            "history_turns":   len(self.history),
            "system_prompt":   self.system_prompt[:80] + "…" if len(self.system_prompt) > 80 else self.system_prompt,
        }
