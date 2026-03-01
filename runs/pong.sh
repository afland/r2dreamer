#!/bin/bash

# Usage: bash runs/pong.sh [OPTIONS]
#   --rep_loss    dreamer / r2dreamer          (default: dreamer)
#   --thick       enable THICK                 (flag, no argument)
#   --seed        int                          (default: 0)
#   --gpu         device ids, e.g. 0 or 0,1,2 (default: 0)

REP_LOSS="dreamer"
THICK="false"
SEED=0
GPU="0"

while [[ $# -gt 0 ]]; do
    case $1 in
        --rep_loss) REP_LOSS="$2"; shift 2 ;;
        --thick)    THICK="true";  shift ;;
        --seed)     SEED="$2";     shift 2 ;;
        --gpu)      GPU="$2";      shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

mkdir -p logs

# Build run name from config combo
if [ "$THICK" = "true" ]; then
    VARIANT="thick_${REP_LOSS}"
else
    VARIANT="${REP_LOSS}"
fi

TIMESTAMP=$(date +%m%d_%H%M%S)
RUN_NAME="${VARIANT}_pong_s${SEED}_${TIMESTAMP}"

THICK_FLAG=""
if [ "$THICK" = "true" ]; then
    THICK_FLAG="model.thick.enabled=True"
fi

CUDA_VISIBLE_DEVICES=$GPU nohup python -u train.py \
    env=atari100k \
    env.task=atari_pong \
    model=size50M \
    model.rep_loss=${REP_LOSS} \
    model.compile=True \
    buffer.storage_device=cpu \
    env.steps=4e6 \
    ${THICK_FLAG} \
    logdir=logdir/${RUN_NAME} \
    seed=${SEED} \
    > logs/${RUN_NAME}.log &

echo "Launched: ${RUN_NAME} (GPU=${GPU}, PID=$!)"
