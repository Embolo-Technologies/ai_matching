#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# run.sh — Launcher for local AI engine
# ─────────────────────────────────────────────────────────────────────────────

RESET="\033[0m"; BOLD="\033[1m"; DIM="\033[2m"
GREEN="\033[92m"; YELLOW="\033[93m"; CYAN="\033[96m"; RED="\033[91m"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PYTHON="${SCRIPT_DIR}/qwen3_engine/venv/bin/python3"

if [ ! -f "${VENV_PYTHON}" ]; then
    echo ""
    echo -e "  ${RED}✗${RESET} Virtual environment not found."
    echo -e "  ${DIM}Run setup first:${RESET} ${CYAN}bash setup.sh${RESET}"
    echo ""
    exit 1
fi

CMD="${1:-}"

if [ -z "${CMD}" ]; then
    echo ""
    echo -e "${BOLD}${CYAN}  Local AI Engine${RESET}  ${DIM}(Qwen3, Gemma3, Gemma4, Llama3)${RESET}"
    echo ""
    echo -e "  ${CYAN}chat${RESET}                   Interactive terminal chat session"
    echo -e "  ${CYAN}server${RESET}                 REST API server on :8080"
    echo -e "  ${CYAN}download <model>${RESET}       Download model (e.g. qwen3, qwen3_1.7b, all)"
    echo -e "  ${CYAN}models${RESET}                 List all registered models"
    echo ""
    exit 0
fi

export PYTHONPATH="${SCRIPT_DIR}"
export PYTHONIOENCODING=utf-8

shift

case "${CMD}" in
    chat)
        exec "${VENV_PYTHON}" -m qwen3_engine chat "$@"
        ;;
    server)
        exec "${VENV_PYTHON}" -m qwen3_engine server "$@"
        ;;
    download)
        exec "${VENV_PYTHON}" -m qwen3_engine download "$@"
        ;;
    models)
        exec "${VENV_PYTHON}" -m qwen3_engine models "$@"
        ;;
    *)
        echo ""
        echo -e "  ${RED}✗${RESET} Unknown command: '${CMD}'"
        echo -e "  Valid: ${CYAN}chat${RESET}, ${CYAN}server${RESET}, ${CYAN}download${RESET}, ${CYAN}models${RESET}"
        echo ""
        exit 1
        ;;
esac
