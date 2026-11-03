import torch
from safetensors import safe_open
import os
import sys

files = [
    "checkpoints/ola_7b_esconv_finetune/adapter_model.safetensors",
    "checkpoints/ola_7b_esconv_stage2_finetune/adapter_model.safetensors"
]

for file_path in files:
    print(f"\nAnalyzing: {file_path}")
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        continue

    try:
        with safe_open(file_path, framework="pt", device="cpu") as f:
            keys = sorted(f.keys())

            print("First 30 keys:")
            for k in keys[:30]:
                print(f"  {k}")


            patterns = ['.lora_A', '.lora_B', '.lora_A.default', '.lora_B.default', '.lora_A.lora_A']
            for pattern in patterns:
                matched = [k for k in keys if pattern in k]
                print(f"Pattern '{pattern}' found: {len(matched)} times")
                if matched:
                    print(f"  Example: {matched[0]}")
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
    sys.stdout.flush()
