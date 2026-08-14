#!/usr/bin/env bash
# ── TIP — start the interactive web UI (macOS / Linux) ────────────────────
# The Windows equivalent is run_gui.bat. Sim4Life is not needed to run this;
# only montage re-solving requires it, and that is Windows-only.
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONIOENCODING=utf-8

echo "[tip-gui] starting the server... a browser will open at http://127.0.0.1:8765"
echo "[tip-gui] press Ctrl+C in this window to stop"

# Prefer an explicit interpreter, then a local venv, then whatever python3 is on PATH.
# numpy and scipy must be importable; matplotlib is only needed for slice images.
if [ -n "${TIP_PYTHON:-}" ]; then
    PY="$TIP_PYTHON"
elif [ -x ".venv/bin/python" ]; then
    PY=".venv/bin/python"
elif [ -x "../.venv/bin/python" ]; then
    PY="../.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
    PY="python3"
else
    echo
    echo "[tip-gui] error: no Python found."
    echo "          Needs Python >= 3.10 with numpy and scipy, e.g."
    echo "            python3 -m venv .venv && .venv/bin/pip install -e \".[plot]\""
    echo "          Or set TIP_PYTHON to an interpreter that already has them."
    exit 1
fi

echo "[tip-gui] interpreter: $PY"
exec "$PY" src/tip/gui/app.py
