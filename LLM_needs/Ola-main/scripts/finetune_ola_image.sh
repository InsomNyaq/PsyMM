#!/bin/bash
set -euo pipefail

export LOWRES_RESIZE=384x32
export VIDEO_RESIZE="0x32"
export HIGHRES_BASE="0x32"
export MAXRES=1536
export MINRES=0
export VIDEO_MAXRES=448
export VIDEO_MINRES=288
export PAD2STRIDE=1
export FORCE_NO_DOWNSAMPLE=1
export LOAD_VISION_EARLY=1

PROJECT_ROOT="${PSYMM_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)}"
export PSYMM_WEIGHTS="${PSYMM_WEIGHTS:-$PROJECT_ROOT/weights}"
export PYTHONPATH="$PROJECT_ROOT/LLM_needs/Ola-main:${PYTHONPATH:-}"

GPU_IDS="${GPU_IDS:?Set GPU_IDS to the allocated GPU list}"
MASTER_PORT="12324"
EXP_NAME="ola_7b_esconv_stage2"
DATA="${DATA:-$PROJECT_ROOT/infer_outputs_esconv/esconv_for_ola_train.json}"
OUTPUT_DIR="./checkpoints/${EXP_NAME}"
ALLOW_RESUME="${ALLOW_RESUME:-0}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-2}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-16}"
LOGGING_STEPS="${LOGGING_STEPS:-5}"


MODEL_CHECKPOINT="${MODEL_CHECKPOINT:-$PROJECT_ROOT/LLM_needs/Ola-main/checkpoints/ola_7b_esconv_finetune}"
BASE_WEIGHTS="$PSYMM_WEIGHTS/ola_weights"

SPEECH_ENCODER="$BASE_WEIGHTS/large-v3.pt"
MUSIC_ENCODER="$BASE_WEIGHTS/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt"

if [[ -f "$MODEL_CHECKPOINT/speech_projector.bin" ]]; then
    SPEECH_PROJECTOR="$MODEL_CHECKPOINT/speech_projector.bin"
else
    SPEECH_PROJECTOR="$BASE_WEIGHTS/speech_projector.bin"
fi


if [[ -d "$PSYMM_WEIGHTS/HG_data/clip-vit-large-patch14-336" ]]; then
    VISION_TOWER="$PSYMM_WEIGHTS/HG_data/clip-vit-large-patch14-336"
elif [[ -d "$PSYMM_WEIGHTS/clip-vit-large-patch14" ]]; then
    VISION_TOWER="$PSYMM_WEIGHTS/clip-vit-large-patch14"
else
    VISION_TOWER="openai/clip-vit-large-patch14"
fi

for required_path in "$MODEL_CHECKPOINT" "$BASE_WEIGHTS" "$SPEECH_PROJECTOR" "$SPEECH_ENCODER" "$MUSIC_ENCODER" "$DATA"; do
    if [[ ! -e "$required_path" ]]; then
        echo "Error: missing required path: $required_path"
        exit 1
    fi
done

if [[ ! -d "$VISION_TOWER" && "$VISION_TOWER" != openai/* ]]; then
    echo "Error: invalid local vision tower path: $VISION_TOWER"
    exit 1
fi

echo "Starting LoRA training on GPU ${GPU_IDS} at port ${MASTER_PORT}"
echo "Vision tower: ${VISION_TOWER}"
echo "Model checkpoint: ${MODEL_CHECKPOINT}"
echo "Training data: ${DATA}"
echo "Output dir: ${OUTPUT_DIR}"
echo "Train batch size per device: ${TRAIN_BATCH_SIZE}"
echo "Gradient accumulation steps: ${GRAD_ACCUM_STEPS}"
echo "Logging every ${LOGGING_STEPS} steps"

if compgen -G "${OUTPUT_DIR}/checkpoint-*" > /dev/null && [[ "${ALLOW_RESUME}" != "1" ]]; then
    echo "Error: found existing checkpoints in ${OUTPUT_DIR}."
    echo "Set ALLOW_RESUME=1 to continue from old checkpoints, or change EXP_NAME for a fresh run."
    exit 1
fi

if command -v deepspeed >/dev/null 2>&1; then
    DS_LAUNCHER=(deepspeed)
elif python -c "import deepspeed" >/dev/null 2>&1; then
    DS_LAUNCHER=(python -m deepspeed)
else
    echo "Error: DeepSpeed is not available in current env. Install with: pip install deepspeed"
    exit 1
fi

"${DS_LAUNCHER[@]}" --include "localhost:${GPU_IDS}" --master_port "${MASTER_PORT}" \
    "$PROJECT_ROOT/LLM_needs/Ola-main/ola/train/train.py" \
    --deepspeed "$PROJECT_ROOT/LLM_needs/Ola-main/scripts/zero2.json" \
    --lora_enable True \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.05 \
    --run_name "$EXP_NAME" \
    --model_name_or_path "$MODEL_CHECKPOINT" \
    --tokenizer_name_or_path "$BASE_WEIGHTS" \
    --pretrain_speech_projector "$SPEECH_PROJECTOR" \
    --vision_tower "$VISION_TOWER" \
    --mm_projector_type ola_mlp \
    --speech_projector_type linear \
    --mm_vision_select_layer -1 \
    --mm_use_im_patch_token False \
    --tune_speech_adapter False \
    --version qwen_1_5 \
    --data_path "$DATA" \
    --bf16 True \
    --output_dir "$OUTPUT_DIR" \
    --sample_independently False \
    --fix_speech_encoder True \
    --freeze_mm_vision_tower True \
    --speech_encoder "$SPEECH_ENCODER" \
    --music_encoder "$MUSIC_ENCODER" \
    --speech_encoder_type dual \
    --speech_encoder_hidden_size 2048 \
    --speech_encoder_ds_rate 10 \
    --num_train_epochs 5 \
    --per_device_train_batch_size "$TRAIN_BATCH_SIZE" \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps "$GRAD_ACCUM_STEPS" \
    --evaluation_strategy no \
    --save_strategy steps \
    --save_steps 500 \
    --save_total_limit 3 \
    --learning_rate 2e-5 \
    --weight_decay 0.01 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type cosine \
    --logging_first_step True \
    --logging_steps "$LOGGING_STEPS" \
    --tf32 True \
    --model_max_length 4096 \
    --gradient_checkpointing True \
    --dataloader_num_workers 2 \
    --frames_upbound 16 \
    --lazy_preprocess True \
    --report_to wandb
