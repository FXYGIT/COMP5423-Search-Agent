#!/bin/bash
# Move indexes/models into the project so paths are self-contained.
# Run from anywhere on the cloud instance:  bash scripts/migrate_layout.sh
set -e
cd "$(dirname "$0")/.."

mkdir -p indexes/dense models

if [ -d /root/rivermind-data/dense-index ]; then
  find /root/rivermind-data/dense-index -mindepth 1 -maxdepth 1 -exec mv {} indexes/dense/ \;
  rmdir /root/rivermind-data/dense-index 2>/dev/null || true
fi

if [ -d /root/rivermind-data/models ]; then
  find /root/rivermind-data/models -mindepth 1 -maxdepth 1 -exec mv {} models/ \;
  rmdir /root/rivermind-data/models 2>/dev/null || true
fi

python3 - <<'PY'
from pathlib import Path

path = Path(".env")
if path.exists():
    text = path.read_text(encoding="utf-8")
    text = text.replace("/root/rivermind-data/dense-index", "./indexes/dense")
    text = text.replace("/root/rivermind-data/models", "./models")
    path.write_text(text, encoding="utf-8")
    print("env updated to relative paths")
PY

echo "--- layout ---"
du -sh indexes/* models/* 2>/dev/null || true
echo "MIGRATE_DONE"
