#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
source .venv/bin/activate
python scripts/recommend.py --user-id 1 --n 10
