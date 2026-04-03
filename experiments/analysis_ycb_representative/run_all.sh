#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
MANIFEST="$ROOT/experiments.json"

mapfile -t EXPERIMENTS < <(python3 - <<'PY' "$MANIFEST"
import json
import sys

with open(sys.argv[1], "r") as f:
    data = json.load(f)

for name in data["experiments"]:
    print(name)
PY
)

for exp_name in "${EXPERIMENTS[@]}"; do
  "$ROOT/run_experiment.sh" "$exp_name"
done
