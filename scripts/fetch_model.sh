#!/usr/bin/env bash
# Fetches all-MiniLM-L6-v2 in ONNX form + its WordPiece vocab into models/.
# Uses the Xenova ONNX export (public, no auth). ~90 MB.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p models
BASE="https://huggingface.co/Xenova/all-MiniLM-L6-v2/resolve/main"

echo "Downloading model.onnx (~90 MB)..."
curl -fL "$BASE/onnx/model.onnx" -o models/model.onnx
echo "Downloading vocab.txt..."
curl -fL "$BASE/vocab.txt" -o models/vocab.txt

echo "Done:"
ls -lh models/
