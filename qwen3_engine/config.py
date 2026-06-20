import os
import multiprocessing

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")

# ─── Model Registry ───────────────────────────────────────────────────────────
MODEL_REGISTRY = {
    "qwen3": {
        "name":              "Qwen3 0.6B",
        "filename":          "Qwen3-0.6B-Q4_K_M.gguf",
        "hf_repo":           "unsloth/Qwen3-0.6B-GGUF",
        "hf_file":           "Qwen3-0.6B-Q4_K_M.gguf",
        "size_mb":           378,
        "template":          "chatml",
        "supports_thinking": True,
        "system_prompt": (
            "You are a helpful, concise, and knowledgeable assistant. "
            "Provide accurate and clear answers. "
            "When you don't know something, say so honestly."
        ),
        "description": "Fast · 0.6B · Thinking mode · ChatML",
    },
    "qwen3_1.7b": {
        "name":              "Qwen3 1.7B",
        "filename":          "Qwen3-1.7B-Q4_K_M.gguf",
        "hf_repo":           "unsloth/Qwen3-1.7B-GGUF",
        "hf_file":           "Qwen3-1.7B-Q4_K_M.gguf",
        "size_mb":           1100,
        "template":          "chatml",
        "supports_thinking": True,
        "n_gpu_layers":      32,
        "system_prompt": (
            "You are a strict, deterministic medical data-matching engine. "
            "Your ONLY job is to select the single correct candidate from a list."
        ),
        "description": "Smart · 1.7B · Thinking mode · ChatML",
    },
    "gemma3": {
        "name":              "Gemma 3 270M",
        "filename":          "gemma-3-270m-it-Q4_K_M.gguf",
        "hf_repo":           "unsloth/gemma-3-270m-it-GGUF",
        "hf_file":           "gemma-3-270m-it-Q4_K_M.gguf",
        "size_mb":           170,
        "template":          "gemma",
        "supports_thinking": False,
        "system_prompt": (
            "You are a helpful and concise assistant. "
            "Answer clearly and accurately."
        ),
        "description": "Tiny · 270M · Ultra-fast · Gemma template",
    },
    "gemma3_1b": {
        "name":              "Gemma 3 1B",
        "filename":          "gemma-3-1b-it-Q4_K_M.gguf",
        "hf_repo":           "unsloth/gemma-3-1b-it-GGUF",
        "hf_file":           "gemma-3-1b-it-Q4_K_M.gguf",
        "size_mb":           900,
        "template":          "gemma",
        "supports_thinking": False,
        "system_prompt": (
            "You are a helpful and concise assistant. "
            "Answer clearly and accurately."
        ),
        "description": "Small · 1B · Fast & capable · Gemma template",
    },
    "llama3_2_3b": {
        "name":              "Llama 3.2 3B Instruct",
        "filename":          "Llama-3.2-3B-Instruct-Q4_K_M.gguf",
        "hf_repo":           "unsloth/Llama-3.2-3B-Instruct-GGUF",
        "hf_file":           "Llama-3.2-3B-Instruct-Q4_K_M.gguf",
        "size_mb":           2020,
        "template":          "llama3",
        "supports_thinking": False,
        "system_prompt": (
            "You are a deterministic medical data-matching engine. "
            "Your ONLY job is to select the single correct candidate from a list."
        ),
        "description": "Fast · 3B · Excellent reasoning · Llama 3 template",
    },
    "gemma4_2b": {
        "name":              "Gemma 4 E2B Instruct",
        "filename":          "gemma-4-E2B-it-Q4_K_M.gguf",
        "hf_repo":           "unsloth/gemma-4-E2B-it-GGUF",
        "hf_file":           "gemma-4-E2B-it-Q4_K_M.gguf",
        "size_mb":           1400,
        "template":          "gemma",
        "supports_thinking": False,
        "n_gpu_layers":      32, # Safe VRAM boundary for Apple M-series Metal watchdog stability
        "system_prompt": (
            "You are a helpful, concise, and knowledgeable assistant."
        ),
        "description": "Tiny · 2B · Gemma 4 Effective · GPU offloaded · Gemma template",
    },
}

DEFAULT_MODEL = "gemma4_2b"

# ─── Chat Templates ────────────────────────────────────────────────────────────
TEMPLATES = {
    "chatml": {
        "system_start":    "<|im_start|>system\n",
        "system_end":      "<|im_end|>\n",
        "user_start":      "<|im_start|>user\n",
        "user_end":        "<|im_end|>\n",
        "assistant_start": "<|im_start|>assistant\n",
        "assistant_end":   "<|im_end|>\n",
        "think_start":     "<think>",
        "think_end":       "</think>",
        "stop_tokens":     ["<|im_end|>", "<|endoftext|>"],
    },
    "gemma": {
        "system_start":    "<start_of_turn>user\n",
        "system_end":      "",
        "user_start":      "<start_of_turn>user\n",
        "user_end":        "<end_of_turn>\n",
        "assistant_start": "<start_of_turn>model\n",
        "assistant_end":   "<end_of_turn>\n",
        "think_start":     "",
        "think_end":       "",
        "stop_tokens":     ["<end_of_turn>", "<eos>"],
    },
    "llama3": {
        "system_start":    "<|start_header_id|>system<|end_header_id|>\n\n",
        "system_end":      "<|eot_id|>",
        "user_start":      "<|start_header_id|>user<|end_header_id|>\n\n",
        "user_end":        "<|eot_id|>",
        "assistant_start": "<|start_header_id|>assistant<|end_header_id|>\n\n",
        "assistant_end":   "<|eot_id|>",
        "think_start":     "",
        "think_end":       "",
        "stop_tokens":     ["<|eot_id|>", "<|end_of_text|>"],
    },
}

CHAT_TEMPLATE = TEMPLATES["chatml"]
MODEL_PATH    = os.path.join(MODELS_DIR, MODEL_REGISTRY[DEFAULT_MODEL]["filename"])

# ─── Inference Settings ───────────────────────────────────────────────────────
N_CTX         = 2048
N_CTX_MATCHER = 2048
N_GPU_LAYERS  = -1    # -1 = offload all layers to GPU
N_THREADS     = min(8, max(6, multiprocessing.cpu_count() - 2))
N_BATCH       = 1024

# ─── Generation Defaults ──────────────────────────────────────────────────────
MAX_TOKENS     = 512
TEMPERATURE    = 0.7
TOP_P          = 0.9
TOP_K          = 40
REPEAT_PENALTY = 1.1

DEFAULT_SYSTEM_PROMPT  = MODEL_REGISTRY[DEFAULT_MODEL]["system_prompt"]
THINKING_MODE_DEFAULT  = False
VERBOSE_LLAMA          = False

SERVER_HOST  = "0.0.0.0"
SERVER_PORT  = 8080
SERVER_DEBUG = False
