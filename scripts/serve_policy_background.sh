#!/bin/bash
# Background policy serving script
# Usage: ./scripts/serve_policy_background.sh

CUDA_DEVICE=${CUDA_VISIBLE_DEVICES:-6}
LOG_FILE="serve_policy_$(date +%Y%m%d_%H%M%S).log"

echo "Starting policy server on GPU $CUDA_DEVICE"
echo "Log file: $LOG_FILE"

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE nohup uv run --active scripts/serve_policy.py \
    --env LIBERO \
    policy:checkpoint \
    --policy.config pi05_libero \
    --policy.dir /data/zijianzhang/train_ckpts/pi05_libero/gaussian_world_model_exp0309/15000/ \
    > "$LOG_FILE" 2>&1 &

PID=$!
echo "Policy server started with PID: $PID"
echo "To stop: kill $PID"
echo "To view logs: tail -f $LOG_FILE"
