#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# setup.sh — Environment bootstrap script for AI Matcher Engine
# ─────────────────────────────────────────────────────────────────────────────
set -e

RESET="\033[0m"; BOLD="\033[1m"; DIM="\033[2m"
GREEN="\033[92m"; YELLOW="\033[93m"; CYAN="\033[96m"; RED="\033[91m"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${SCRIPT_DIR}/qwen3_engine/venv"
MODELS_DIR="${SCRIPT_DIR}/models"

echo ""
echo -e "${BOLD}${CYAN}  AI Matcher Engine — Setup${RESET}"
echo -e "  ${DIM}Project : ${SCRIPT_DIR}${RESET}"
echo ""

# Find best Python
if command -v python3.11 &>/dev/null; then
    PYTHON_BIN="python3.11"
elif command -v python3.12 &>/dev/null; then
    PYTHON_BIN="python3.12"
elif command -v python3 &>/dev/null; then
    PYTHON_BIN="python3"
else
    echo "Error: Python 3.9+ is required."
    exit 1
fi

echo -e "  Using Python: $(${PYTHON_BIN} --version 2>&1)"

if [ -d "${VENV_DIR}" ]; then
    echo "  Virtual environment already exists. Skipping creation."
else
    echo "  Creating virtual environment inside SSD..."
    "${PYTHON_BIN}" -m venv "${VENV_DIR}"
fi

source "${VENV_DIR}/bin/activate"
export PYTHONIOENCODING=utf-8

echo "  Upgrading pip..."
pip install --quiet --upgrade pip

echo "  Installing Flask, RapidFuzz, Flask-CORS, requests, tqdm..."
pip install --quiet requests Flask flask-cors rapidfuzz tqdm rich prompt_toolkit pygments

echo "  Compiling llama-cpp-python with Metal GPU support..."
CMAKE_ARGS="-DGGML_METAL=on" pip install llama-cpp-python --no-cache-dir --force-reinstall 2>&1 | tail -n 5 || true

mkdir -p "${MODELS_DIR}"

echo ""
echo -e "${BOLD}${GREEN}  ✓ Setup complete!${RESET}"
echo -e "  To start the server tomorrow after downloading models, run:"
echo -e "  ${CYAN}bash run.sh download qwen3_1.7b${RESET}"
echo -e "  ${CYAN}bash run.sh server${RESET}"
echo ""
