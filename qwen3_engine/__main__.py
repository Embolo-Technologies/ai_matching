import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qwen3_engine.config import MODEL_REGISTRY, DEFAULT_MODEL
from qwen3_engine.downloader import download_model, is_downloaded

def list_models():
    print()
    print("  Available Models:")
    print("  " + "─" * 60)
    for key, cfg in MODEL_REGISTRY.items():
        status = "Downloaded" if is_downloaded(key) else "Not Downloaded"
        print(f"  * {cfg['name']} ({key})")
        print(f"    Status: {status}")
        print(f"    Size  : {cfg['size_mb']} MB")
        print(f"    Desc  : {cfg['description']}")
        print()

def main():
    if len(sys.argv) < 2:
        print("\n  Local AI Engine Command Interface")
        print("  Usage: python -m qwen3_engine <command> [args]")
        print("\n  Commands:")
        print("    chat       Start interactive terminal chat session")
        print("    server     Start REST API server")
        print("    download   Download models from HuggingFace")
        print("    models     List registered models and download status")
        print()
        sys.exit(0)

    cmd = sys.argv[1]
    sys.argv.pop(1) # Remove command so argparsers in submodules work correctly

    if cmd == "chat":
        from qwen3_engine import chat
        chat.main()
    elif cmd == "server":
        from qwen3_engine import server
        server.main()
    elif cmd == "download":
        parser = argparse.ArgumentParser(description="Download model")
        parser.add_argument("model", type=str, default=DEFAULT_MODEL, nargs="?")
        parser.add_argument("--force", action="store_true")
        args = parser.parse_args()
        
        if args.model == "all":
            for m in MODEL_REGISTRY:
                download_model(m, force=args.force)
        else:
            download_model(args.model, force=args.force)
    elif cmd == "models":
        list_models()
    else:
        print(f"Unknown command: '{cmd}'")
        sys.exit(1)

if __name__ == "__main__":
    main()
