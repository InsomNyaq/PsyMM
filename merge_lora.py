"""Merge Ola adapters in training order, including non-LoRA multimodal weights."""
import argparse
import json
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'LLM_needs/Ola-main'))


def validate_full_model(path):
    path = Path(path)
    if not (path / 'config.json').is_file():
        raise FileNotFoundError(f'Missing model config: {path}')
    direct = [path / n for n in ('model.safetensors', 'pytorch_model.bin')]
    indices = [path / n for n in ('model.safetensors.index.json', 'pytorch_model.bin.index.json')]
    files = [p for p in direct if p.is_file()]
    for index in indices:
        if index.exists():
            files += [path / n for n in set(json.loads(index.read_text())['weight_map'].values())]
    if not files or any(not p.is_file() or p.stat().st_size < 1024 for p in files):
        raise ValueError(f'{path} is not a complete model; adapter-only directories cannot be bases')


def normalize_non_lora_keys(state):
    state = {(k[11:] if k.startswith('base_model.') else k): v for k,v in state.items()}
    if any(k.startswith('model.model.') for k in state):
        state = {(k[6:] if k.startswith('model.') else k): v for k,v in state.items()}
    return state


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base-path', required=True)
    p.add_argument('--parent-lora-path', default='')
    p.add_argument('--lora-path', required=True)
    p.add_argument('--output-path', required=True)
    p.add_argument('--tokenizer-path')
    args = p.parse_args()
    validate_full_model(args.base_path)
    output = Path(args.output_path).resolve()
    if output.exists():
        raise FileExistsError('Use a new output directory; never overwrite checkpoints')
    adapters = [Path(x).resolve() for x in (args.parent_lora_path,args.lora_path) if x]
    if output in [Path(args.base_path).resolve(), *adapters]:
        raise ValueError('Output must differ from all inputs')
    for adapter in adapters:
        if not (adapter / 'adapter_config.json').exists() or not any((adapter/n).exists() for n in ('adapter_model.safetensors','adapter_model.bin')):
            raise FileNotFoundError(f'Incomplete adapter: {adapter}')
    import torch
    from peft import PeftModel
    from transformers import AutoTokenizer
    from ola.model.language_model.ola_qwen import OlaQwenForCausalLM, OlaConfigQwen
    config = OlaConfigQwen.from_pretrained(str(adapters[0]))
    model = OlaQwenForCausalLM.from_pretrained(args.base_path, config=config,
                    torch_dtype=torch.bfloat16, low_cpu_mem_usage=False, device_map=None)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path or args.base_path, trust_remote_code=True)
    model.resize_token_embeddings(len(tokenizer))
    report = []
    for adapter in adapters:
        non_lora_path = adapter / 'non_lora_trainables.bin'
        loaded = []
        if non_lora_path.exists():
            state = normalize_non_lora_keys(torch.load(str(non_lora_path), map_location='cpu'))
            incompatible = model.load_state_dict(state, strict=False)
            if incompatible.unexpected_keys:
                raise ValueError(f'Unrestored multimodal tensors: {incompatible.unexpected_keys}')
            loaded = list(state)

        model = PeftModel.from_pretrained(model, str(adapter)).merge_and_unload()
        report.append(dict(adapter=str(adapter), non_lora_keys=loaded))
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=output.name+'.partial.',dir=output.parent))
    model.save_pretrained(str(temporary), safe_serialization=True, max_shard_size='4GB')
    tokenizer.save_pretrained(str(temporary))
    validate_full_model(temporary)
    (temporary/'merge_provenance.json').write_text(json.dumps(dict(base=args.base_path,adapters=report),indent=2))
    os.replace(temporary, output)
    print(f'Full model saved to {output}; run inference validation before experiments')


if __name__ == '__main__':
    main()
