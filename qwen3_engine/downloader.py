import os
import sys
import requests
from tqdm import tqdm
from qwen3_engine.config import MODEL_REGISTRY, MODELS_DIR

def is_downloaded(model_key: str) -> bool:
    if model_key not in MODEL_REGISTRY:
        return False
    cfg = MODEL_REGISTRY[model_key]
    path = os.path.join(MODELS_DIR, cfg["filename"])
    return os.path.exists(path)

def model_path(model_key: str) -> str:
    cfg = MODEL_REGISTRY[model_key]
    return os.path.join(MODELS_DIR, cfg["filename"])

def download_model(model_key: str, force: bool = False) -> bool:
    if model_key not in MODEL_REGISTRY:
        print(f"Error: Model key '{model_key}' not found in configuration.")
        return False

    cfg = MODEL_REGISTRY[model_key]
    dest_path = os.path.join(MODELS_DIR, cfg["filename"])

    if os.path.exists(dest_path) and not force:
        print(f"Model '{cfg['name']}' already downloaded at: {dest_path}")
        return True

    os.makedirs(MODELS_DIR, exist_ok=True)

    url = f"https://huggingface.co/{cfg['hf_repo']}/resolve/main/{cfg['hf_file']}"
    print(f"Downloading {cfg['name']}...")
    print(f"URL: {url}")
    print(f"Destination: {dest_path}")

    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024 * 1024 # 1 MB chunks

        with open(dest_path, "wb") as f, tqdm(
            desc=cfg["name"],
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(block_size):
                f.write(data)
                bar.update(len(data))

        print(f"\n✓ Successfully downloaded {cfg['name']}!")
        return True
    except Exception as e:
        print(f"\nError downloading model: {e}")
        if os.path.exists(dest_path):
            os.remove(dest_path)
        return False
