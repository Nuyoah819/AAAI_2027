#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="$ROOT_DIR/.venv/bin/python"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Virtual environment not found at $ROOT_DIR/.venv" >&2
  echo "Create it first with: $ROOT_DIR/scripts/setup_env.sh" >&2
  exit 1
fi

cd "$ROOT_DIR"
exec "$PYTHON_BIN" "$@"
