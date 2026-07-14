#!/usr/bin/env bash
# Local setup, CPU + 6GB laptop GPU. Runs Phases 0-2 (data recon + baseline chain).
set -euo pipefail
cd "$(dirname "$0")/.."

python3 -m venv venv
# shellcheck disable=SC1091
source venv/bin/activate
pip install --upgrade pip

# torch: CUDA 12.1 wheel (RTX 3060 laptop). Swap to the cpu index-url if no GPU.
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

pip install -r requirements.txt

echo "OK. Activate with: source venv/bin/activate"
echo "Next: cp .env.example .env  &&  fill HF_TOKEN, then: hf auth login"
