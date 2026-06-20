#!/bin/bash
# ==============================================================================
# Automated GCP GPU Matching Deployment Script
# ==============================================================================
set -e

VM_NAME="instance-20260612-075658"
ZONE="us-central1-a"
LOCAL_CODE_DIR="/Volumes/ssd embolo/embolo/ai"
LOCAL_GAMES_DIR="/Volumes/ssd embolo/Games"

echo "===================================================================="
echo "          STARTING GOOGLE CLOUD GPU DEPLOYMENT"
echo "===================================================================="

# 1. Check if gcloud CLI is installed
if ! command -v gcloud &> /dev/null; then
    echo "⚠️ Google Cloud SDK (gcloud) is not installed on your Mac."
    echo "Installing it now using Homebrew..."
    if ! command -v brew &> /dev/null; then
        echo "❌ Homebrew is not installed. Please install Homebrew or Google Cloud SDK first."
        exit 1
    fi
    brew install --cask google-cloud-sdk
fi

# 2. Login to gcloud
echo -e "\n[1/6] Verifying Google Cloud login..."
gcloud auth list --filter=status=ACTIVE --format="value(account)" | grep -q "@" || gcloud auth login

# 3. Ask for Project ID
echo -e "\n[2/6] Configuring project..."
CURRENT_PROJECT=$(gcloud config get-value project 2>/dev/null || echo "")
PROJECT_ID=${PROJECT_ID:-$CURRENT_PROJECT}
echo "Using Project ID: $PROJECT_ID"

if [ -z "$PROJECT_ID" ]; then
    echo "❌ Project ID is required."
    exit 1
fi
gcloud config set project "$PROJECT_ID"

# 4. Copy code and spreadsheet files to the VM
echo -e "\n[3/6] Uploading code and data to VM..."
# Temporarily move models and heavy CSV out to speed up transfer
mv "$LOCAL_CODE_DIR/models" "$LOCAL_CODE_DIR/../models_temp" || true
mv "$LOCAL_CODE_DIR/Item_export_2026-05-29_17-33-30.csv" "$LOCAL_CODE_DIR/../Item_export_2026-05-29_17-33-30.csv_temp" || true

# Perform scp of code directory (now lightweight)
gcloud compute scp --recurse "$LOCAL_CODE_DIR" "$VM_NAME:~/" --zone="$ZONE"

# Restore files
mv "$LOCAL_CODE_DIR/../models_temp" "$LOCAL_CODE_DIR/models" || true
mv "$LOCAL_CODE_DIR/../Item_export_2026-05-29_17-33-30.csv_temp" "$LOCAL_CODE_DIR/Item_export_2026-05-29_17-33-30.csv" || true

gcloud compute scp "$LOCAL_GAMES_DIR/masterdata.xlsx" "$VM_NAME:~/ai/" --zone="$ZONE"
gcloud compute scp "$LOCAL_GAMES_DIR/input.xlsx" "$VM_NAME:~/ai/" --zone="$ZONE"

# 5. Install GPU Drivers and dependencies on the VM if not already present
echo -e "\n[4/6] Checking GPU drivers on the VM..."
if gcloud compute ssh "$VM_NAME" --zone="$ZONE" --command="nvidia-smi" &>/dev/null; then
    echo "✓ GPU drivers already installed. Skipping installation and reboot."
else
    echo "Installing GPU drivers and CUDA on the VM (This takes 2-3 mins)..."
    gcloud compute ssh "$VM_NAME" --zone="$ZONE" --command="
      sudo apt update &&
      sudo apt install -y software-properties-common &&
      sudo apt-add-repository contrib non-free -y &&
      sudo apt update &&
      sudo apt install -y nvidia-driver nvidia-cuda-toolkit python3-pip python3-venv git &&
      echo '✓ Drivers installed! Rebooting VM...' &&
      sudo reboot
    " || true
    echo "⏳ Waiting 60 seconds for the VM to reboot and activate GPU..."
    sleep 60
fi

# 6. Run the matching pipeline on the GPU VM
echo -e "\n[5/6] Starting AI batch matching on the GPU VM..."
gcloud compute ssh "$VM_NAME" --zone="$ZONE" --command="
  cd ~/ai &&
  python3 -m venv venv &&
  source venv/bin/activate &&
  pip install pandas openpyxl rapidfuzz flask flask-cors requests tqdm &&
  if python3 -c "import llama_cpp; assert llama_cpp.llama_supports_gpu_offload()" &>/dev/null; then
    echo "✓ llama-cpp-python with CUDA is already installed."
  else
    echo "Compiling llama-cpp with CUDA support..." &&
    CMAKE_ARGS='-GGML_CUDA=on' pip install --force-reinstall --no-cache-dir llama-cpp-python
  fi &&
  echo '✓ Ensuring Gemma 4 2B model is downloaded on the GPU server...' &&
  export PYTHONPATH=. &&
  python3 -c \"from qwen3_engine.downloader import download_model; download_model('gemma4_2b')\" &&
  echo '✓ Running matching run...' &&
  python run_batch_match.py --master-xlsx masterdata.xlsx --input-xlsx input.xlsx --output-xlsx full_matching_results.xlsx
"

# 7. Download results back to Mac
echo -e "\n[6/6] Matching complete! Downloading results back to Mac..."
gcloud compute scp "$VM_NAME:~/ai/full_matching_results.xlsx" "$LOCAL_GAMES_DIR/full_matching_results.xlsx" --zone="$ZONE"

echo "===================================================================="
echo "🎉 SUCCESS! Results saved to: $LOCAL_GAMES_DIR/full_matching_results.xlsx"
echo "===================================================================="
