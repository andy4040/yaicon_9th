#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source .runtime-env.sh
python -m glint.train_two_set \
  --config configs/exps/glint/3d-front-t/scene_3.yaml \
  --data-root /root/data/3d-front-t/scene_3 \
  --environment-ply /root/data/3d-front-t/scene_3/envs/points3D.ply \
  --output-dir results/two_set/scene_3_direct_beta \
  --transport-mode direct_beta --max-steps 60000 "$@"
