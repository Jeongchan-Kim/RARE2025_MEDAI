#!/bin/bash

# GPU ID 설정
GPU_ID=0

# ============================
# Configuration
# ============================
DATA_DIR="/Path/to/data"
MODEL="ResNet50"
BATCH_SIZE=128
EPOCHS=100
LR=1e-4
ALPHA=0.2
DECAY=1e-4
NUM_WORKERS=8
SAMPLING='weighted'
PRETRAINED_PATH="/Path/to/gastronet/RN50_GastroNet-5M_DINOv1.pth"
SEED=42
MIXUP='balanced' 

# ============================
# 환경변수 설정
# ============================
export CUDA_VISIBLE_DEVICES=${GPU_ID}

# ============================
# Command
# ============================
CMD="/Path/to/train/code/train.py \
    --data_folder ${DATA_DIR} \
    --model ${MODEL} \
    --batch_size ${BATCH_SIZE} \
    --epoch ${EPOCHS} \
    --lr ${LR} \
    --alpha ${ALPHA} \
    --decay ${DECAY} \
    --num_workers ${NUM_WORKERS} \
    --seed ${SEED} \
    --mixup ${MIXUP} \
    --ema \
    --optimizer adamw \
    --sched cosine \
    --warmup_epochs 5 \
    --bn_recalc \
    --augment \
    --split_ratio 0.1 \
    --mode platt \
    "

if [[ -n "${PRETRAINED_PATH}" ]]; then
    CMD="${CMD} --pretrained_path ${PRETRAINED_PATH}"
fi

# ============================
# Execute
# ============================
echo "[INFO] Using GPU: ${GPU_ID}"
echo "[INFO] Running command: ${CMD}"

python ${CMD}
