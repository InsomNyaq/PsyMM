#!/bin/bash
set -euo pipefail


export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
PROJECT_ROOT="${PSYMM_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)}"
export PSYMM_WEIGHTS="${PSYMM_WEIGHTS:-$PROJECT_ROOT/weights}"
export PYTHONPATH="$PROJECT_ROOT/LLM_needs/Ola-main:${PYTHONPATH:-}"


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


BASE_DIR="$PROJECT_ROOT"
ESCONV_ALIGNED="${ESCONV_ALIGNED:-$BASE_DIR/infer_outputs_esconv/esconv_aligned.json}"
DATA="${DATA:-$BASE_DIR/infer_outputs_esconv/esconv_for_ola_train.json}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"

EXP_NAME="${EXP_NAME:-ola_7b_esconv_finetune}"
CHECKPOINT="${CHECKPOINT:-$PSYMM_WEIGHTS/ola_weights}"
VISION_TOWER="${VISION_TOWER:-openai/clip-vit-large-patch14}"
GPU_IDS="${GPU_IDS:?Set GPU_IDS to the allocated GPU list}"
MASTER_PORT="${MASTER_PORT:-12324}"

SPEECH_ENCODER="$CHECKPOINT/large-v3.pt"
MUSIC_ENCODER="$CHECKPOINT/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt"

echo "=== ESConv -> Ola LoRA Finetune ==="
echo "Input aligned json: $ESCONV_ALIGNED"
echo "Output train json:  $DATA"
echo "Experiment name:    $EXP_NAME"
echo "Checkpoint:         $CHECKPOINT"
echo "GPUs:               $GPU_IDS"
echo "MAX_SAMPLES:        $MAX_SAMPLES"

if [[ ! -f "$ESCONV_ALIGNED" ]]; then
  echo "Error: missing input file: $ESCONV_ALIGNED"
  exit 1
fi



export ESCONV_ALIGNED DATA MAX_SAMPLES
python - <<'PY'
import json
import os
from pathlib import Path

src = Path(os.environ["ESCONV_ALIGNED"])
dst = Path(os.environ["DATA"])
max_samples = int(os.environ.get("MAX_SAMPLES", "0"))

with src.open("r", encoding="utf-8") as f:
    data = json.load(f)

samples = []
for conv in data:
    conv_id = str(conv.get("conversation_id", "esconv_unknown"))
    topic = str(conv.get("topic", "")).strip()

    for turn in conv.get("turns", []):
        response = str(turn.get("response", "")).strip()
        if not response:
            continue

        context = str(turn.get("context", "")).replace("\n", " ").strip()
        history_lines = []
        for utt in turn.get("dialogue_history", []):
            role = str(utt.get("role", "speaker")).lower()
            role_name = "Seeker" if role == "speaker" else "Supporter"
            text = str(utt.get("utterance", "")).replace("\n", " ").strip()
            if text:
                history_lines.append(f"{role_name}: {text}")

        prompt_parts = [
            "You are an empathetic supporter in a counseling conversation.",
        ]
        if topic:
            prompt_parts.append(f"Topic: {topic}")
        if context:
            prompt_parts.append(f"Situation: {context}")
        if history_lines:
            prompt_parts.append("Dialogue history:\n" + "\n".join(history_lines[-20:]))
        prompt_parts.append("Provide the next Supporter reply.")
        prompt = "\n".join(prompt_parts)

        sample_id = str(turn.get("file_id") or f"{conv_id}_T{turn.get('turn_id', '0')}")
        samples.append(
            {
                "id": sample_id,
                "conversations": [
                    {"from": "human", "value": prompt},
                    {"from": "gpt", "value": response},
                ],
            }
        )

        if max_samples > 0 and len(samples) >= max_samples:
            break

    if max_samples > 0 and len(samples) >= max_samples:
        break

dst.parent.mkdir(parents=True, exist_ok=True)
with dst.open("w", encoding="utf-8") as f:
    json.dump(samples, f, ensure_ascii=False, indent=2)

print(f"Built {len(samples)} training samples -> {dst}")
PY

if [[ ! -f "$DATA" ]]; then
  echo "Error: failed to generate train json: $DATA"
  exit 1
fi

TRAIN_COUNT=$(python - <<'PY'
import json, os
with open(os.environ["DATA"], "r", encoding="utf-8") as f:
    x = json.load(f)
print(len(x))
PY
)

if [[ "$TRAIN_COUNT" -eq 0 ]]; then
  echo "Error: generated dataset is empty: $DATA"
  exit 1
fi

echo "Train samples: $TRAIN_COUNT"

for req in "$CHECKPOINT" "$CHECKPOINT/speech_projector.bin" "$SPEECH_ENCODER" "$MUSIC_ENCODER"; do
  if [[ ! -e "$req" ]]; then
    echo "Error: missing required file: $req"
    exit 1
  fi
done

mkdir -p "./checkpoints_esconv/$EXP_NAME"

echo "Starting LoRA training..."
deepspeed --include "localhost:${GPU_IDS}" --master_port "$MASTER_PORT" \
  "$PROJECT_ROOT/LLM_needs/Ola-main/ola/train/train.py" \
  --deepspeed "$PROJECT_ROOT/LLM_needs/Ola-main/scripts/zero2.json" \
  --lora_enable True \
  --lora_r 64 \
  --lora_alpha 128 \
  --lora_dropout 0.05 \
  --run_name "$EXP_NAME" \
  --model_name_or_path "$CHECKPOINT" \
  --pretrain_speech_projector "$CHECKPOINT/speech_projector.bin" \
  --vision_tower "$VISION_TOWER" \
  --mm_projector_type ola_mlp \
  --speech_projector_type linear \
  --mm_vision_select_layer -1 \
  --mm_use_im_patch_token False \
  --tune_speech_adapter False \
  --version qwen_1_5 \
  --data_path "$DATA" \
  --bf16 True \
  --output_dir "./checkpoints/$EXP_NAME" \
  --sample_independently False \
  --fix_speech_encoder True \
  --freeze_mm_vision_tower True \
  --speech_encoder "$SPEECH_ENCODER" \
  --music_encoder "$MUSIC_ENCODER" \
  --speech_encoder_type dual \
  --speech_encoder_hidden_size 2048 \
  --speech_encoder_ds_rate 10 \
  --num_train_epochs 1 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --evaluation_strategy no \
  --save_strategy steps \
  --save_steps 500 \
  --save_total_limit 3 \
  --learning_rate 5e-5 \
  --weight_decay 0.0 \
  --warmup_ratio 0.1 \
  --lr_scheduler_type cosine \
  --logging_steps 1 \
  --tf32 True \
  --model_max_length 4096 \
  --gradient_checkpointing True \
  --dataloader_num_workers 2 \
  --frames_upbound 4 \
  --lazy_preprocess True \
  --report_to none

echo "Training finished. Output: ./checkpoints_esconv/$EXP_NAME"
