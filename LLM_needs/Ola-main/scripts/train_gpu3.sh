#!/bin/bash

echo "======== Allocated GPU: ${CUDA_VISIBLE_DEVICES:?Set CUDA_VISIBLE_DEVICES} ========"


export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:?Set CUDA_VISIBLE_DEVICES to an allocated GPU}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID


export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128


export FLASH_ATTENTION_FORCE_DISABLE=1

echo "当前可见 GPU:"
python - <<EOF
import torch
print("PyTorch看到GPU数量:", torch.cuda.device_count())
print("当前GPU:", torch.cuda.get_device_name(0))
EOF

echo "================================"



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
export PYTHONPATH="$PROJECT_ROOT/LLM_needs/Ola-main:${PYTHONPATH:-}"



EXP_NAME="ola_7b_avamerg_gpu3_single"

DATA="${DATA:-${PSYMM_DATA:-$PROJECT_ROOT/data}/AvaMERG/train_ola_converted_full.json}"
CHECKPOINT="${CHECKPOINT:-${PSYMM_WEIGHTS:-$PROJECT_ROOT/weights}/ola_weights}"

mkdir -p ./checkpoints/$EXP_NAME

echo "实验名: $EXP_NAME"
echo "使用 GPU: CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "================================"


python ola/train/train.py \
    --run_name $EXP_NAME \
    --model_name_or_path "$CHECKPOINT" \
    --pretrain_speech_projector "$CHECKPOINT/speech_projector.bin" \
    --vision_tower "openai/clip-vit-large-patch14" \
    --mm_projector_type "ola_mlp" \
    --speech_projector_type "linear" \
    --mm_vision_select_layer -1 \
    --mm_use_im_patch_token "False" \
    --freeze_backbone "True"\
    --tune_mm_mlp_adapter "True"\
    --tune_speech_adapter "True"\
    --fix_speech_encoder "True"\
    --freeze_mm_vision_tower "True"\
    --version "qwen_1_5" \
    --data_path "$DATA" \
    --bf16 "True" \
    --output_dir "./checkpoints/$EXP_NAME" \
    --sample_independently "True" \
    --speech_encoder "$CHECKPOINT/large-v3.pt" \
    --music_encoder "$CHECKPOINT/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt" \
    --speech_encoder_type "dual" \
    --speech_encoder_hidden_size 2048 \
    --speech_encoder_ds_rate 10 \
    --learning_rate 2e-4 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --weight_decay 0.0 \
    --warmup_ratio 0.05 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --tf32 "True" \
    --model_max_length 1024 \
    --gradient_checkpointing "True" \
    --dataloader_num_workers 1 \
    --frames_upbound 8 \
    --lazy_preprocess "True" \
    --report_to "none"
