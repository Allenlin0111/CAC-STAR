#!/usr/bin/env bash
set -euo pipefail
python -V
python scripts/smoke_forward.py
echo "[OK] smoke forward passed."