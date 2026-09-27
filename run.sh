#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

python_bin="${PYTHON:-python3}"
if ! command -v "$python_bin" >/dev/null 2>&1; then
    echo "Error: $python_bin was not found. Install Python 3.12 or newer." >&2
    exit 1
fi

if ! "$python_bin" -c 'import sys; raise SystemExit(sys.version_info < (3, 12))'; then
    echo "Error: GammaForge requires Python 3.12 or newer." >&2
    exit 1
fi

if [[ ! -x .venv/bin/python ]]; then
    echo "Creating the GammaForge virtual environment..."
    "$python_bin" -m venv .venv
fi

if ! .venv/bin/python -c 'import gammaforge, nicegui, plotly' >/dev/null 2>&1; then
    echo "Installing GammaForge and its GUI dependencies..."
    .venv/bin/python -m pip install --upgrade pip
    .venv/bin/python -m pip install -e '.[gui]'
fi

echo "Starting GammaForge..."
exec .venv/bin/python -m gammaforge.gui "$@"
