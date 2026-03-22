#!/usr/bin/env bash
# Run from project root so `python3 -m rag.app` finds the `rag` package.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
if [[ -f .venv/bin/activate ]]; then
  # shellcheck source=/dev/null
  source .venv/bin/activate
fi
exec python3 -m rag.app
