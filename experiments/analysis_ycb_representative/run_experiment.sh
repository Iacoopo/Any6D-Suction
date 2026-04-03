#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
MANIFEST="$ROOT/experiments.json"

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <experiment_name>"
  exit 1
fi

EXPERIMENT_NAME="$1"

mapfile -t RUN_INFO < <(python3 - <<'PY' "$MANIFEST" "$EXPERIMENT_NAME"
import json
import shlex
import sys

manifest_path = sys.argv[1]
experiment_name = sys.argv[2]

with open(manifest_path, "r") as f:
    data = json.load(f)

common = data["common"]
exp = data["experiments"].get(experiment_name)
if exp is None:
    raise SystemExit(f"Unknown experiment: {experiment_name}")

# Use specific dataset_root if provided in experiment, else fallback to common
dataset_root = exp.get("dataset_root", common.get("dataset_root"))

args = [
    common["wrapper"],
    "python",
    common["runner"],
    "--name", experiment_name,
    "--dataset_root", dataset_root,
    "--ycb_model_path", common["ycb_model_path"],
    "--factor", exp["factor"],
    "--object_name", exp["object_name"],
    "--mask_type", common["mask_type"],
    "--gt_camera_convention", common["gt_camera_convention"],
    "--gt_pose_field", common["gt_pose_field"],
]

if common.get("object_frame_corrections_json"):
    args.extend([
        "--object_frame_corrections_json", common["object_frame_corrections_json"],
    ])

print(common["workdir"])
print(" ".join(shlex.quote(x) for x in args))
PY
)

WORKDIR="${RUN_INFO[0]}"
CMD="${RUN_INFO[1]}"

echo "[analysis] Running: $EXPERIMENT_NAME"
echo "[analysis] Workdir: $WORKDIR"
echo "[analysis] Command: $CMD"

cd "$WORKDIR"
eval "$CMD"
