#!/bin/bash

# Usage: bash runs/pinpad.sh [OPTIONS]
#   --rep_loss    dreamer / r2dreamer / infonce       (default: dreamer)
#   --thick       enable THICK                        (flag, no argument)
#   --gate_type   gatelord / gatelord_binary / timelord (default: gatelord)
#   --coarse_critic  enable coarse critic + mixed value target (flag)
#   --sparse      sparse loss scale                   (default: 10.0)
#   --sparse_free sparse free nats                    (default: 0.0)
#   --seed        int                                 (default: 0)
#   --gpu         device ids, e.g. 0 or 0,1,2        (default: 0)

REP_LOSS="dreamer"
THICK="false"
GATE_TYPE="gatelord"
COARSE_CRITIC="false"
SPARSE="1.0"
SPARSE_FREE="0.0"
SEED=0
GPU="0"

while [[ $# -gt 0 ]]; do
    case $1 in
        --rep_loss)   REP_LOSS="$2";   shift 2 ;;
        --thick)      THICK="true";    shift ;;
        --gate_type)  GATE_TYPE="$2";  shift 2 ;;
        --coarse_critic) COARSE_CRITIC="true"; shift ;;
        --sparse)     SPARSE="$2";     shift 2 ;;
        --sparse_free) SPARSE_FREE="$2"; shift 2 ;;
        --seed)       SEED="$2";       shift 2 ;;
        --gpu)        GPU="$2";        shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

mkdir -p logs

# Build run name from config combo
if [ "$THICK" = "true" ]; then
    if [ "$COARSE_CRITIC" = "true" ]; then
        VARIANT="thick_${GATE_TYPE}_cc_${REP_LOSS}"
    else
        VARIANT="thick_${GATE_TYPE}_${REP_LOSS}"
    fi
else
    VARIANT="${REP_LOSS}"
fi

TIMESTAMP=$(date +%m%d_%H%M%S)
RUN_NAME="pinpad_${VARIANT}_s${SEED}_${TIMESTAMP}"

THICK_FLAGS=""
if [ "$THICK" = "true" ]; then
    THICK_FLAGS="model.thick.enabled=True model.thick.gate_type=${GATE_TYPE} model.loss_scales.sparse=${SPARSE} model.thick.sparse_free=${SPARSE_FREE}"
    if [ "$COARSE_CRITIC" = "true" ]; then
        THICK_FLAGS="${THICK_FLAGS} model.thick.coarse_critic=True"
    fi
fi

CUDA_VISIBLE_DEVICES=$GPU nohup python -u train.py \
    env=pinpad \
    env.task=pinpad_three \
    model=size12M \
    model.rep_loss=${REP_LOSS} \
    model.compile=True \
    buffer.storage_device=cpu \
    ${THICK_FLAGS} \
    logdir=logdir/${RUN_NAME} \
    seed=${SEED} \
    > logs/${RUN_NAME}.log &

echo "Launched: ${RUN_NAME} (GPU=${GPU}, PID=$!)"
