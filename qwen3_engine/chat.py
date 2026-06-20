import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qwen3_engine.engine import Qwen3Engine
from qwen3_engine.config import MODEL_REGISTRY, DEFAULT_MODEL

def main(model_key: str = None, thinking_mode: bool = None):
    model_key = model_key or DEFAULT_MODEL
    print(f"\nLoading model '{model_key}'...")
    
    try:
        engine = Qwen3Engine(model_key=model_key, thinking_mode=thinking_mode)
        engine.load()
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)
        
    info = engine.info()
    print(f"Model Name : {info['model_name']}")
    print(f"Load Time  : {info['load_time_s']}s")
    print(f"Context    : {info['n_ctx']} tokens")
    print(f"GPU Layers : {info['n_gpu_layers']}")
    print(f"Thinking   : {info['thinking_mode']}")
    print("\nChat session started. Type '/exit' or '/quit' to leave.")
    print("--------------------------------------------------")

    while True:
        try:
            user_input = input("\nUser > ").strip()
            if not user_input:
                continue
            if user_input.lower() in ('/exit', '/quit'):
                break
                
            print("\nAssistant > ", end="", flush=True)
            for token in engine.stream(user_input):
                print(token, end="", flush=True)
            print()
        except KeyboardInterrupt:
            print("\nSession interrupted.")
            break
            
    print("\nChat session closed.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Interactive console chat")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--think", action="store_true")
    args = parser.parse_args()
    main(model_key=args.model, thinking_mode=args.think)
