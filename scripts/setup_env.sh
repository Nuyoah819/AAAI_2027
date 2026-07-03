#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="$ROOT_DIR/.venv"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required to create this environment." >&2
  exit 1
fi

uv venv "$VENV_DIR" --python python3 --prompt AAAI0622
uv pip install --python "$VENV_DIR/bin/python" --link-mode=copy -r "$ROOT_DIR/requirements.txt"
uv pip install --python "$VENV_DIR/bin/python" --link-mode=copy pip

echo "Environment ready: $VENV_DIR"
echo "Run commands with: $ROOT_DIR/scripts/run_in_venv.sh <python-script> [args...]"
