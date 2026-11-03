#!/usr/bin/env bash
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
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

EXP_NAME="ola_7b_esconv_stage3"
DATA="${DATA:-$PROJECT_ROOT/infer_outputs_esconv/esconv_stage3_multimodal.json}"
E0_MANIFEST="${E0_MANIFEST:?Set E0_MANIFEST to the verified experiment manifest}"


python3 "$PROJECT_ROOT/experiments/e0_preflight.py" \
    --manifest "$E0_MANIFEST" \
    --input "$DATA"


STAGE2_RESULT="${STAGE2_RESULT:-$PROJECT_ROOT/LLM_needs/Ola-main/checkpoints/ola_7b_esconv_stage2}"

MODEL_CHECKPOINT="${MODEL_CHECKPOINT:-$PROJECT_ROOT/LLM_needs/Ola-main/checkpoints/ola_7b_esconv_stage2_merged}"
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

GPU_IDS="${GPU_IDS:?Set GPU_IDS to the allocated GPU list}"
MASTER_PORT="12325"
MODEL_MAX_LENGTH="${MODEL_MAX_LENGTH:-2048}"

if [[ -f "$STAGE2_RESULT/adapter_config.json" && -f "$STAGE2_RESULT/adapter_model.safetensors" ]]; then
    if [[ ! -f "$MODEL_CHECKPOINT/config.json" ]]; then
        echo "Error: $STAGE2_RESULT is a LoRA adapter directory."
        echo "Please merge stage2 adapter first, then rerun this script:"
        echo "python $PROJECT_ROOT/merge_lora.py --base-path $BASE_WEIGHTS --lora-path $STAGE2_RESULT --output-path $MODEL_CHECKPOINT"
        exit 1
    fi
fi

for required_path in "$STAGE2_RESULT" "$MODEL_CHECKPOINT" "$BASE_WEIGHTS" "$SPEECH_PROJECTOR" "$SPEECH_ENCODER" "$MUSIC_ENCODER" "$DATA"; do
    if [[ ! -e "$required_path" ]]; then
        echo "Error: missing required path: $required_path"
        exit 1
    fi
done

echo "Starting stage-3 LoRA finetune on GPU ${GPU_IDS}"
echo "Model checkpoint: ${MODEL_CHECKPOINT}"
echo "Tokenizer source: ${BASE_WEIGHTS}"
echo "Vision tower: ${VISION_TOWER}"
echo "Model max length: ${MODEL_MAX_LENGTH}"

if ! python -c "import deepspeed" >/dev/null 2>&1; then
    echo "Error: DeepSpeed is not available in current env. Install with: python -m pip install 'deepspeed<0.16.0'"
    exit 1
fi


export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
python ola/train/train.py \
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
    --version qwen_1_5 \
    --data_path "$DATA" \
    --bf16 True \
    --output_dir "./checkpoints/$EXP_NAME" \
    --fix_speech_encoder True \
    --freeze_mm_vision_tower True \
    --speech_encoder "$SPEECH_ENCODER" \
    --music_encoder "$MUSIC_ENCODER" \
    --speech_encoder_type dual \
    --speech_encoder_hidden_size 2048 \
    --speech_encoder_ds_rate 10 \
    --num_train_epochs 5 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-5 \
    --model_max_length 4096 \
    --gradient_checkpointing True \
    --frames_upbound 32 \
    --logging_steps 1 \
    --warmup_ratio 0.03 \
    --resume_from_checkpoint ./checkpoints/ola_7b_esconv_stage3/checkpoint-500 \
    --lazy_preprocess True
