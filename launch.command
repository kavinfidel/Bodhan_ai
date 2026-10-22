#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"

if [[ "$(uname -s)" != "Darwin" || "$(uname -m)" != "arm64" ]]; then
  echo "This demo needs an Apple silicon Mac (M1/M2/M3/M4)."
  read -r -p "Press Return to close..."
  exit 1
fi

PYTHON_BIN="$(command -v python3.11 || command -v python3.13 || command -v python3 || true)"
if [[ -z "$PYTHON_BIN" ]]; then
  echo "Python 3.11 or 3.13 is required. Install one with Homebrew: brew install python@3.11"
  read -r -p "Press Return to close..."
  exit 1
fi

PYTHON_VERSION="$($PYTHON_BIN -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
case "$PYTHON_VERSION" in
  3.11|3.13) ;;
  *)
    echo "Found Python $PYTHON_VERSION at $PYTHON_BIN. Use Python 3.11 or 3.13 for MLX."
    echo "Install Python 3.11 with: brew install python@3.11"
    read -r -p "Press Return to close..."
    exit 1
    ;;
esac

if [[ ! -d .venv ]]; then
  "$PYTHON_BIN" -m venv .venv
fi
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m streamlit run app.py
