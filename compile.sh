#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# compile.sh — Standalone Executable Compiler for low-spec PCs
# ─────────────────────────────────────────────────────────────────────────────
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PYTHON="${SCRIPT_DIR}/qwen3_engine/venv/bin/python3"
VENV_PIP="${SCRIPT_DIR}/qwen3_engine/venv/bin/pip"

# Ensure venv exists
if [ ! -f "${VENV_PYTHON}" ]; then
    echo "Creating virtual environment inside SSD..."
    if command -v python3.11 &>/dev/null; then
        PYTHON_BIN="python3.11"
    elif command -v python3.12 &>/dev/null; then
        PYTHON_BIN="python3.12"
    else
        PYTHON_BIN="python3"
    fi
    "${PYTHON_BIN}" -m venv "${SCRIPT_DIR}/qwen3_engine/venv"
fi

echo "Activating virtual environment and installing Flask, RapidFuzz, Flask-CORS, PyInstaller..."
"${VENV_PIP}" install --quiet Flask flask-cors rapidfuzz pyinstaller

# Resolve CSV file
CSV_FILE=""
CANDIDATE_CSVS=(
    "/Users/admin/Downloads/item_export_2026-05-30_09-08-22.csv"
    "/Users/admin/Downloads/Item_export_2026-05-29_17-33-30.csv"
)
for p in "${CANDIDATE_CSVS[@]}"; do
    if [ -f "$p" ]; then
        CSV_FILE="$p"
        break
    fi
done

if [ -z "${CSV_FILE}" ]; then
    echo "Error: Unified catalog CSV not found in Downloads folder."
    exit 1
fi

DEST_CSV="${SCRIPT_DIR}/Item_export_2026-05-29_17-33-30.csv"
echo "Copying catalog database: ${CSV_FILE} -> ${DEST_CSV}"
cp "${CSV_FILE}" "${DEST_CSV}"

echo "Compiling standalone executable..."
cd "${SCRIPT_DIR}"
"${SCRIPT_DIR}/qwen3_engine/venv/bin/pyinstaller" \
    --onefile \
    --clean \
    --name fast_search_server \
    --add-data "Item_export_2026-05-29_17-33-30.csv:." \
    qwen3_engine/fast_server.py

echo "Cleaning up build assets..."
mv dist/fast_search_server "${SCRIPT_DIR}/"
rm -rf build dist fast_search_server.spec

echo ""
echo "  ✓ stand-alone executable built successfully!"
echo "  Filename: ${SCRIPT_DIR}/fast_search_server"
echo "  Size    : $(du -sh "${SCRIPT_DIR}/fast_search_server" | cut -f1)"
echo ""
echo "  You can now copy 'fast_search_server' to any PC/SSD."
echo "  When double-clicked/executed, it starts the matching server on http://localhost:8080"
echo "  without needing Python or Node.js."
echo ""
