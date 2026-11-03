#!/bin/bash

component_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
image_path="$component_root/examples/img/female.png"
audio_path="$component_root/examples/wav/female.wav"
emotion_path="$component_root/examples/emo/happy.npy"
output_path="$component_root/results/output.mp4"

python3 "$component_root/demo.py" --image_path "$image_path" --audio_path "$audio_path" --emotion_path "$emotion_path" --output_path "$output_path"
