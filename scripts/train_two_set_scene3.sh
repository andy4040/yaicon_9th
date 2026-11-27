#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ -f .runtime-env.sh ]]; then
    source .runtime-env.sh
fi
PYTHON="${PYTHON:-python}"
DATA_ROOT="${DATA_ROOT:-/root/data/3d-front-t/scene_3}"
MODE="${1:-smoke}"
if [[ $# -gt 0 ]]; then shift; fi
COMMON=(--config configs/exps/glint/3d-front-t/scene_3.yaml --data-root "$DATA_ROOT")
if [[ -f "$DATA_ROOT/envs/points3D.ply" ]]; then
    COMMON+=(--environment-ply "$DATA_ROOT/envs/points3D.ply")
fi
case "$MODE" in
    smoke)
        "$PYTHON" -m glint.train_two_set "${COMMON[@]}" \
            --output-dir "${OUTPUT_DIR:-results/two_set/scene_3_smoke}" \
            --ratio 0.125 --max-points 20000 --max-steps 160 \
            --warmup-steps 50 --probation-steps 100 --selection-views 8 \
            --no-refinement --num-workers 0 --log-every 10 --save-every 50 "$@"
        ;;
    train)
        "$PYTHON" -m glint.train_two_set "${COMMON[@]}" \
            --output-dir "${OUTPUT_DIR:-results/two_set/scene_3}" \
            --max-steps 60000 "$@"
        ;;
    *) echo "Usage: $0 {smoke|train} [extra training arguments]" >&2; exit 2 ;;
esac
