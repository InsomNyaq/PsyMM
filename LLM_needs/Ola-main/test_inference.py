import os
from pathlib import Path
import json
import torch
import librosa
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer
from peft import PeftModel
import whisper


from ola.model.language_model.ola_qwen import OlaQwenForCausalLM
from ola.constants import IMAGE_TOKEN_INDEX, DEFAULT_SPEECH_TOKEN


def manual_tokenizer_speech(prompt, tokenizer):
    """手动处理 <speech> 占位符，转换为模型识别的特征索引"""
    parts = prompt.split(DEFAULT_SPEECH_TOKEN)
    ids = []
    for i, part in enumerate(parts):
        if part:
            ids.extend(tokenizer.encode(part, add_special_tokens=False))
        if i < len(parts) - 1:
            ids.append(IMAGE_TOKEN_INDEX)
    return torch.tensor([ids], dtype=torch.long).cuda()


root = Path(__file__).resolve().parents[2]
weights = Path(os.environ.get('PSYMM_WEIGHTS', root / 'weights'))
data = Path(os.environ.get('PSYMM_DATA', root / 'data'))
BASE_MODEL_PATH = str(weights / 'ola_weights')
CHECKPOINT_PATH = str(Path(__file__).resolve().parent / 'checkpoints/ola_7b_avamerg_finetune/checkpoint-3548')
TEST_JSON_PATH = './test_standard.json'
AUDIO_DIR = str(data / 'AvaMERG/audio_v5_0/test_audio_v5_0')
OUTPUT_FILE = str(Path(__file__).resolve().parent / 'checkpoints/ola_7b_avamerg_finetune/avamerg_predictions.json')


def load_audio_chunks(audio_file):
    """Split/trim audio to 16000Hz mel chunks matching Ola inference path."""
    speech_wav, _ = librosa.load(audio_file, sr=16000)
    if speech_wav.ndim > 1:
        speech_wav = speech_wav[:, 0]
    speech_wav = speech_wav.astype(np.float32)
    CHUNK_LIM = 480000
    speechs = []
    speech_wavs = []
    if len(speech_wav) <= CHUNK_LIM:
        chunk = whisper.pad_or_trim(speech_wav)
        speechs.append(chunk)
        speech_wavs.append(torch.from_numpy(chunk).unsqueeze(0))
    else:
        for i in range(0, len(speech_wav), CHUNK_LIM):
            chunk = speech_wav[i : i + CHUNK_LIM]
            if len(chunk) < CHUNK_LIM:
                chunk = whisper.pad_or_trim(chunk)
            speechs.append(chunk)
            speech_wavs.append(torch.from_numpy(chunk).unsqueeze(0))
    mels = []
    for chunk in speechs:
        mel = whisper.log_mel_spectrogram(chunk, n_mels=128).permute(1, 0).unsqueeze(0)
        mels.append(mel)
    mels = torch.cat(mels, dim=0)
    speech_wavs = torch.cat(speech_wavs, dim=0)
    if mels.shape[0] > 25:
        mels = mels[:25]
        speech_wavs = speech_wavs[:25]
    speech_length = torch.LongTensor([mels.shape[1]] * mels.shape[0])
    speech_chunks = torch.LongTensor([mels.shape[0]])
    return mels, speech_length, speech_chunks, speech_wavs

def main():
    print("🚀 步骤 1: 加载 Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)

    print("🚀 步骤 2: 加载基础模型...")
    base_model = OlaQwenForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    ).cuda()

    print("🚀 步骤 3: 深度初始化多模态组件...")
    model_core = base_model.get_model()


    config_map = {
        'mm_vision_tower': 'vision_tower',
        'mm_projector_type': 'projector_type',
        'mm_hidden_size': 'mm_hidden_size',
        'pretrain_mm_mlp_adapter': 'pretrain_mm_mlp_adapter'
    }

    for src, dst in config_map.items():
        if hasattr(base_model.config, src):
            setattr(base_model.config, dst, getattr(base_model.config, src))


    if not hasattr(base_model.config, 'pretrain_mm_mlp_adapter'):
        base_model.config.pretrain_mm_mlp_adapter = None


    try:

        if hasattr(model_core, 'initialize_ola_modules'):
            model_core.initialize_ola_modules(base_model.config, for_inference=True)
        else:

            model_core.initialize_vision_modules(base_model.config)
        print(" 多模态模块初始化完成")
    except Exception as e:
        print(f" 初始化警告 (尝试继续): {e}")



    if not hasattr(model_core, 'mm_projector') or model_core.mm_projector is None:
        print(" 正在手动重建 mm_projector...")
        model_core.initialize_vision_modules(base_model.config)
    if not hasattr(model_core, 'mm_projector') or model_core.mm_projector is None:
        print(" mm_projector 仍为空，使用 Identity 兜底以避免 NoneType 错误")
        model_core.mm_projector = torch.nn.Identity().to(dtype=torch.bfloat16, device='cuda')


    for attr in ['speech_encoder', 'speech_tower', 'mm_speech_tower']:
        tower = getattr(model_core, attr, None)
        if tower is not None:
            tower.to(dtype=torch.bfloat16, device='cuda')
            print(f"✅ 已激活音频塔: {attr}")
            break


    if hasattr(model_core, 'mm_projector') and model_core.mm_projector is not None:
        model_core.mm_projector.to(dtype=torch.bfloat16, device='cuda')
        print("✅ 已激活多模态桥梁 (mm_projector)")
    print(f"🚀 步骤 4: 融合微调 LoRA 权重...")
    model = PeftModel.from_pretrained(base_model, CHECKPOINT_PATH)
    model.eval()

    print("🚀 步骤 5: 预加载并建立音频文件索引...")
    all_audio_files = os.listdir(AUDIO_DIR)

    audio_map = {f.split('utt')[0]: f for f in all_audio_files if 'utt' in f}


    with open(TEST_JSON_PATH, 'r', encoding='utf-8') as f:
        test_data = json.load(f)

    results = []



    for item in tqdm(test_data[:5], desc="推理进度"):
        conv_id = item['id']
        search_key = f"dia{conv_id}"


        matched_file = audio_map.get(search_key)
        if not matched_file:

            matched_file = next((f for f in all_audio_files if f.startswith(search_key)), None)

        if not matched_file:
            print(f"\n[跳过] 找不到 ID {conv_id} 对应的音频文件")
            continue

        audio_path = os.path.join(AUDIO_DIR, matched_file)


        prompt = DEFAULT_SPEECH_TOKEN + "\n" + item['text']
        input_ids = manual_tokenizer_speech(prompt, tokenizer)

        try:

            speech, speech_length, speech_chunk, speech_wav = load_audio_chunks(audio_path)
            speech = speech.bfloat16().cuda()
            speech_length = speech_length.to('cuda')
            speech_chunk = speech_chunk.to('cuda')
            speech_wav = speech_wav.to('cuda')


            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    speech=speech,
                    speech_lengths=speech_length,
                    speech_chunks=speech_chunk,
                    speech_wav=speech_wav,
                    do_sample=False,
                    max_new_tokens=64,
                    use_cache=True
                )


            response = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

            results.append({
                "id": conv_id,
                "audio_used": matched_file,
                "prediction": response,
                "ground_truth": item['label']
            })

        except Exception as e:
            print(f"\n[报错] 处理 ID {conv_id} 时发生异常: {e}")


    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    print(f"\n🎉 测试运行结束！已成功生成 {len(results)} 条结果。")
    print(f"请使用 `cat {OUTPUT_FILE}` 查看内容。如果结果正常，请去掉代码中的 `[:5]` 运行全量测试！")

if __name__ == "__main__":
    main()
