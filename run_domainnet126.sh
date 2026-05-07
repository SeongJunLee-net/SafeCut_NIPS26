#!/bin/bash
set -e
GPU_ID=${1:-0}
DATASET="domainnet126"
METHOD="safecut"
PYTHON=${PYTHON:-python}
EPOCHS=50
BATCH_SIZE=192
WEIGHT_BATCH_SIZE=192
WORKERS=8
LR=1e-2
IIC_WEIGHT=1.3
DIV_WEIGHT_T=0.05
DIV_WEIGHT_P=0.05
K_NEIGHBORS=50
SAFECUT_EMA_M=0.95
PROMPT_LR=5e-4
TARGET_MODEL_LR_SCHEDULE="inv"
RELIABILITY_CMP_BETA=6.0
TEMPORAL_WEIGHT=1.0
PEER_MODEL_LR_FLOOR=0.1
mkdir -p "outer_log/${DATASET}"
SRC_DOMAINS=("c" "p" "r" "s")
TGT_DOMAINS=("c" "p" "r" "s")
for s in "${SRC_DOMAINS[@]}"; do
    for t in "${TGT_DOMAINS[@]}"; do
        if [[ "$s" == "$t" ]]; then
            continue
        fi
        DSET="${s}2${t}"
        LOG="outer_log/${DATASET}/${DSET}.log"
        echo ""
        echo "------------------------------------------------------------------"
        echo " Running: ${DATASET}  ${DSET}  (GPU=${GPU_ID})"
        echo " Log    : ${LOG}"
        echo "------------------------------------------------------------------"
        CUDA_VISIBLE_DEVICES=${GPU_ID} PYTHONPATH=. \
        ${PYTHON} -m src.methods.safecut.train \
            --dataset      "${DATASET}" \
            --dset         "${DSET}" \
            --gpu_id       0 \
            --epochs       "${EPOCHS}" \
            --batch_size   "${BATCH_SIZE}" \
            --weight_batch_size "${WEIGHT_BATCH_SIZE}" \
            --worker       "${WORKERS}" \
            --lr           "${LR}" \
            --target_model_lr_schedule "${TARGET_MODEL_LR_SCHEDULE}" \
            --reliability_cmp_beta "${RELIABILITY_CMP_BETA}" \
            --iic_weight   "${IIC_WEIGHT}" \
            --div_weight_t "${DIV_WEIGHT_T}" \
            --div_weight_p "${DIV_WEIGHT_P}" \
            --k_neighbors  "${K_NEIGHBORS}" \
            --safecut_ema_m    "${SAFECUT_EMA_M}" \
            --eval_interval 10 \
            --bottleneck   256 \
            --peer_model_lr_floor "${PEER_MODEL_LR_FLOOR}" \
            --prompt_lr     "${PROMPT_LR}" \
            --temporal_weight "${TEMPORAL_WEIGHT}" \
            --isolated_fallback_regular_knn \
            --diag_log     "outer_log/${DATASET}/${DSET}_diag.log" \
        2>&1 | tee "${LOG}"
        echo " Finished: ${DSET}"
    done
done
echo ""
echo "=================================================================="
echo " [Done] DomainNet-126 All Done"
echo "=================================================================="