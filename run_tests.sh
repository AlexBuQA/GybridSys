#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
source .venv/bin/activate
pytest -q
python scripts/validate_artifacts.py
