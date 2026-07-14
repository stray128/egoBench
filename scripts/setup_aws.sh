#!/usr/bin/env bash
# AWS setup, 24GB GPU instance (e.g. g5.xlarge / A10). Runs Phase 3 heavy models.
# Same requirements.txt as local; only the torch CUDA build + research repos differ.
set -euo pipefail
cd "$(dirname "$0")/.."

python3 -m venv venv
# shellcheck disable=SC1091
source venv/bin/activate
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

# Heavy research repos, installed here only, NOT pip deps (conflicting pins,
# non-commercial weights). Clone + follow each repo's install into ./third_party.
mkdir -p third_party
echo ">> Clone WiLoR, HaMeR, MapAnything, HaWoR into third_party/ per Phase 3."
echo ">> DATA_ROOT should point at the EBS/instance mount, e.g.:"
echo "     export DATA_ROOT=/mnt/data/egobench"
echo "OK."
