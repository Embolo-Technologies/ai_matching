"""
Gunicorn hooks for fast_server.

With --preload, gunicorn imports the app once in the master process before
forking workers, so the master's catalog fetch + search-index build
(initialize_on_import in fast_server.py) happens once and all workers
inherit it via copy-on-write, instead of each of the N workers redundantly
re-fetching and re-indexing the full catalog on their own.

The engine pool is different: it holds live HTTP connections to
llama-server, which do not survive being copied into a forked child. So it
must be created fresh in each worker AFTER fork — that's what post_fork
does here, calling init_engine_pool_for_worker() once per worker process.
"""
import threading


def post_fork(server, worker):
    from qwen3_engine.fast_server import init_engine_pool_for_worker
    threading.Thread(target=init_engine_pool_for_worker, daemon=True).start()
