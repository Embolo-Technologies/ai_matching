import os
import sys
import time
import json
import uuid

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qwen3_engine.engine import Qwen3Engine
from qwen3_engine.downloader import download_model
from qwen3_engine.config import (
    MODEL_PATH, SERVER_HOST, SERVER_PORT, SERVER_DEBUG,
    MAX_TOKENS, TEMPERATURE, TOP_P, TOP_K, DEFAULT_MODEL,
    N_CTX_MATCHER, N_THREADS, N_BATCH,
)
from qwen3_engine.matcher import HybridMatcher
from qwen3_engine.searcher import FuzzySearcher

try:
    from flask import Flask, request, jsonify, Response, stream_with_context
    from flask_cors import CORS
except ImportError:
    print("\n  Flask or flask-cors not installed. Run: bash setup.sh\n")
    sys.exit(1)

app = Flask(__name__)
CORS(app)

_matcher: HybridMatcher = None
_searcher: FuzzySearcher = None

def get_engine() -> Qwen3Engine:
    return _matcher.engine if _matcher else None

@app.route("/match", methods=["POST"])
def match_item_route():
    if not _matcher:
        return jsonify({"error": "Matcher not initialized."}), 503

    data = request.get_json(silent=True) or {}
    name = data.get("name", "").strip()
    pack = data.get("pack", "").strip()
    compname = data.get("company", "").strip() or data.get("compname", "").strip()
    query = data.get("query", "").strip()
    if not query:
        query = f"{name} {pack}".strip()
    if not query:
        return jsonify({"error": "Field 'query' or 'name' is required."}), 400

    try:
        t0 = time.time()
        result = _matcher.match(query, top_k=5, name=name, pack=pack, compname=compname)
        elapsed = time.time() - t0
        return jsonify({
            "query": query,
            "match": result,
            "latency_ms": round(elapsed * 1000, 2)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/search", methods=["POST"])
def search_route():
    if not _searcher or not _searcher.is_loaded():
        return jsonify({"error": "Searcher not initialized."}), 503

    data = request.get_json(silent=True) or {}
    name = data.get("name", "").strip()
    if not name:
        return jsonify({"error": "Field 'name' is required."}), 400

    pack   = data.get("pack", "").strip()
    top_k  = int(data.get("top_k", 5))
    top_k  = max(1, min(top_k, 20))

    try:
        results, elapsed_ms = _searcher.search(name, pack=pack, top_k=top_k)
        return jsonify({
            "query":     {"name": name, "pack": pack},
            "results":   results,
            "count":     len(results),
            "search_ms": elapsed_ms,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/health", methods=["GET"])
def health():
    engine = get_engine()
    return jsonify({
        "status": "ok",
        "model_loaded": engine.is_loaded() if engine else False,
        "model": os.path.basename(MODEL_PATH),
        "timestamp": time.time(),
    })

@app.route("/v1/models", methods=["GET"])
def list_models():
    return jsonify({
        "object": "list",
        "data": [
            {
                "id": "qwen3-1.7b",
                "object": "model",
                "created": 1748000000,
                "owned_by": "local",
                "permission": [],
            }
        ]
    })

@app.route("/v1/chat/completions", methods=["POST"])
def chat_completions():
    engine = get_engine()
    if not engine or not engine.is_loaded():
        return jsonify({"error": "Model not loaded"}), 503

    data = request.get_json(silent=True) or {}
    messages = data.get("messages", [])
    if not messages:
        return jsonify({"error": "messages field is required"}), 400

    stream_mode = data.get("stream", False)
    max_tokens = int(data.get("max_tokens", MAX_TOKENS))
    temperature = float(data.get("temperature", TEMPERATURE))
    top_p = float(data.get("top_p", TOP_P))
    top_k = int(data.get("top_k", TOP_K))

    user_message = None
    new_system_prompt = None
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role == "system":
            new_system_prompt = content
        elif role == "user":
            user_message = content

    if not user_message:
        return jsonify({"error": "No user message found"}), 400

    original_system = engine.system_prompt
    if new_system_prompt:
        engine.system_prompt = new_system_prompt

    engine.history = []
    for msg in messages[:-1]:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role in ("user", "assistant"):
            engine.history.append({"role": role, "content": content})

    completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())

    if stream_mode:
        def generate_stream():
            try:
                for token in engine.stream(
                    user_message,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                ):
                    chunk = {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": "qwen3-1.7b",
                        "choices": [{
                            "index": 0,
                            "delta": {"content": token},
                            "finish_reason": None,
                        }],
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"
                
                done_chunk = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": "qwen3-1.7b",
                    "choices": [{
                        "index": 0,
                        "delta": {},
                        "finish_reason": "stop",
                    }],
                }
                yield f"data: {json.dumps(done_chunk)}\n\n"
                yield "data: [DONE]\n\n"
            finally:
                engine.system_prompt = original_system

        return Response(stream_with_context(generate_stream()), content_type="text/event-stream")

    try:
        t0 = time.time()
        response_text = engine.generate(
            user_message,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )
        elapsed = time.time() - t0
    finally:
        engine.system_prompt = original_system

    return jsonify({
        "id": completion_id,
        "object": "chat.completion",
        "created": created,
        "model": "qwen3-1.7b",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": response_text,
            },
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens": -1,
            "completion_tokens": len(response_text.split()),
            "total_tokens": -1,
        },
        "_meta": {
            "latency_s": round(elapsed, 2),
        }
    })

def main():
    global _matcher, _searcher
    import argparse

    parser = argparse.ArgumentParser(description="REST API Server for local AI engine & Matcher")
    parser.add_argument("--csv", type=str, default="", help="Path to catalog CSV")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Model key from config.py")
    parser.add_argument("--host", type=str, default=SERVER_HOST, help="Host to bind to")
    parser.add_argument("--port", type=int, default=SERVER_PORT, help="Port to bind to")
    args, unknown = parser.parse_known_args()

    csv_path = args.csv
    if not csv_path:
        candidate_paths = [
            "/Users/admin/Downloads/item_export_2026-05-30_09-08-22.csv",
            "/Users/admin/Downloads/Item_export_2026-05-29_17-33-30.csv"
        ]
        for p in candidate_paths:
            if os.path.exists(p):
                csv_path = p
                break

    if not csv_path or not os.path.exists(csv_path):
        print(f"Error: Unified CSV database not found.")
        sys.exit(1)

    # Ensure model is downloaded
    from qwen3_engine.downloader import is_downloaded
    if not is_downloaded(args.model):
        print(f"Model '{args.model}' not found. Download it first using: bash run.sh download {args.model}")
        sys.exit(1)

    print("Initializing matcher & catalog...")
    _matcher = HybridMatcher(master_csv_path=csv_path, model_key=args.model)
    _matcher.load_catalog()

    print("Initializing fuzzy searcher...")
    _searcher = FuzzySearcher(csv_path=csv_path)
    _searcher.load()

    print("Loading LLM model on GPU...")
    _matcher.engine.load(n_ctx=N_CTX_MATCHER)

    print(f"✓ Matcher API Server ready on http://{args.host}:{args.port}")
    app.run(
        host=args.host,
        port=args.port,
        debug=SERVER_DEBUG,
        threaded=False,
        use_reloader=False,
    )

if __name__ == "__main__":
    main()
