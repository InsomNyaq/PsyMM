import gradio as gr
import os
import numpy as np
from pydub import AudioSegment
import hashlib
from dice_talk import DICE_Talk
import shutil
from pathlib import Path

pipe = DICE_Talk(0)


def get_md5(content):
    md5hash = hashlib.md5(content)
    md5 = md5hash.hexdigest()
    return md5

def get_video_res(img_path, audio_path, emotion_path, res_video_path, ref_scale=None, emo_scale=None, crop=False):

    expand_ratio = 0.5
    min_resolution = 512
    inference_steps = 25

    face_info = pipe.preprocess(img_path, expand_ratio=expand_ratio)
    print(face_info)
    if face_info['face_num'] > 0:
        if crop:
            crop_image_path = img_path + '.crop.png'
            pipe.crop_image(img_path, crop_image_path, face_info['crop_bbox'])
            img_path = crop_image_path
        os.makedirs(os.path.dirname(res_video_path), exist_ok=True)
        pipe.process(img_path, audio_path, emotion_path, res_video_path, min_resolution=min_resolution, inference_steps=inference_steps, ref_scale=ref_scale, emo_scale=emo_scale)
    else:
        return -1
component_root = Path(__file__).resolve().parent
tmp_path = str(component_root / 'tmp_path') + os.sep
res_path = str(component_root / 'res_path') + os.sep
os.makedirs(tmp_path,exist_ok=True)
os.makedirs(res_path,exist_ok=True)

def process_dice(image,audio, emotion, s0, s1, crop=False):
    img_md5= get_md5(np.array(image))
    audio_md5 = get_md5(audio[1])

    print(img_md5,audio_md5)
    sampling_rate, arr = audio[:2]
    if len(arr.shape)==1:
        arr = arr[:,None]
    audio = AudioSegment(
        arr.tobytes(),
        frame_rate=sampling_rate,
        sample_width=arr.dtype.itemsize,
        channels=arr.shape[1]
    )
    audio = audio.set_frame_rate(sampling_rate)
    image_path = os.path.abspath(tmp_path+'{0}.png'.format(img_md5))
    audio_path = os.path.abspath(tmp_path+'{0}.wav'.format(audio_md5))
    emotion_path = os.path.abspath(tmp_path+'{0}.npy'.format(emotion))
    if not os.path.exists(image_path):
        image.save(image_path)
    if not os.path.exists(audio_path):
        audio.export(audio_path, format="wav")
    if not os.path.exists(emotion_path):
        src_emo_path = component_root / 'examples/emo' / f'{emotion}.npy'
        shutil.copy(src_emo_path, emotion_path)
    res_video_path = os.path.abspath(res_path+f'{img_md5}_{audio_md5}_{emotion}_{s0}_{s1}_{int(crop)}.mp4')
    if os.path.exists(res_video_path):
        return res_video_path
    else:
        get_video_res(image_path, audio_path, emotion_path, res_video_path, s0, s1, crop=crop)
    return res_video_path

inputs = [
    gr.Image(type='pil',label="Upload Image"),
    gr.Audio(label="Upload Audio"),
    gr.Dropdown(
        choices=['contempt', 'sad', 'happy', 'surprised', 'angry', 'disgusted', 'fear', 'neutral'],
        label="Choose Emotion"
    ),
    gr.Slider(0.0, 10.0, value=3.0, step=0.1, label="Reference", info="Increase/decrease to obtain stronger/weaker identity preservation"),
    gr.Slider(0.0, 10.0, value=6.0, step=0.1, label="Emotion", info="Increase/decrease to obtain stronger/weaker emotions"),
    gr.Checkbox(label="Crop image", value=False)
]
outputs = gr.Video(label="output.mp4")


html_description = """
<div style="display: flex; justify-content: center; align-items: center;">
  <a href="https://github.com/toto222/DICE-Talk" style="margin: 0 2px;">
    <img src='https://img.shields.io/badge/GitHub-Repo-blue?style=flat&logo=GitHub' alt='GitHub'>
  </a>
  <a href="https://arxiv.org/abs/2504.18087" style="margin: 0 2px;">
    <img src='https://img.shields.io/badge/arXiv-2504.18087-red?style=flat&logo=arXiv&logoColor=red' alt='arxiv'>
  </a>
  <a href='https://toto222.github.io/DICE-Talk/' style="margin: 0 2px;">
    <img src='https://img.shields.io/badge/Webpage-Project-silver?style=flat&logo=&logoColor=orange' alt='webpage'>
  </a>
  <a href="https://github.com/toto222/DICE-Talk/blob/main/LICENSE" style="margin: 0 2px;">
    <img src='https://img.shields.io/badge/License-CC BY--NC--SA--4.0-lightgreen?style=flat&logo=Lisence' alt='License'>
  </a>
</div>

The demo can only be used for <b>Non-commercial Use</b>.
<br>If you like our work, please star <a href='https://github.com/toto222/DICE-Talk' style="margin: 0 2px;">DICE-Talk</a>.
"""

def get_example():
    img_dir = str(component_root / 'examples/img') + os.sep
    wav_dir = str(component_root / 'examples/wav') + os.sep
    return [
        [f"{img_dir}nazha.png", f"{wav_dir}female-zh.wav", "happy", 3.0, 6.0],
        [f"{img_dir}pyy.jpg", f"{wav_dir}male-zh.wav", "neutral", 4.0, 7.5],
        [f"{img_dir}female.png", f"{wav_dir}female.wav", "happy", 3.0, 6.0],
        [f"{img_dir}hg.jpeg", f"{wav_dir}male.wav", "surprised", 3.0, 6.0],
    ]

with gr.Blocks(title="DICE-Talk") as demo:
    gr.Interface(fn=process_dice, inputs=inputs, outputs=outputs, title="Disentangle Identity, Cooperate Emotion: Correlation-Aware\
        Emotional Talking Portrait Generation", description=html_description)
    gr.Examples(
        examples=get_example(),
        fn=process_dice,
        inputs=inputs,
        outputs=outputs,
        cache_examples=False,)

demo.queue()
demo.launch(server_name='0.0.0.0', server_port=8081, share=False)
