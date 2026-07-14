#!/usr/bin/env bash
# Run label audit + train baseline from Code/
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ -x "$HOME/.venvs/drowsiness-dds/bin/python" ]]; then
  PY="$HOME/.venvs/drowsiness-dds/bin/python"
elif [[ -x "$ROOT/.venv/bin/python" ]]; then
  PY="$ROOT/.venv/bin/python"
else
  PY="python3"
fi

echo "Using: $PY"
"$PY" -m src.audit_labels --out artifacts/label_audit.json
"$PY" -m src.train --config configs/default.yaml
echo "Done."
