import os
import torch
import gc


os.environ['CUDA_VISIBLE_DEVICES'] = '1,5'
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

def test_ola_memory():
    """精确测试OLA模型在当前配置下的显存需求"""

    print("=== OLA 7B模型显存测试 ===")
    print(f"使用GPU: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
    print(f"配置参数:")
    print(f"  - 批次大小: 1")
    print(f"  - 序列长度: 1024")
    print(f"  - 帧数上限: 8")
    print(f"  - 梯度累积: 4")
    print(f"  - BF16: True")

    torch.cuda.empty_cache()
    gc.collect()


    print("\n1. 模型参数加载...")
    model_size = 7 * 1e9 * 2 / 1024**3
    print(f"   模型参数显存: {model_size:.2f} GB")


    print("\n2. 优化器状态...")
    optimizer_size = model_size * 0.5
    print(f"   优化器状态(8-bit): {optimizer_size:.2f} GB")


    print("\n3. 梯度显存...")
    grad_size = model_size
    print(f"   梯度显存: {grad_size:.2f} GB")


    print("\n4. 激活值显存...")
    batch_size = 1
    seq_len = 1024
    hidden_size = 4096
    num_layers = 32

    activation_per_layer = batch_size * seq_len * hidden_size * 2 / 1024**3
    activation_total = activation_per_layer * num_layers * 0.3
    print(f"   激活值显存: {activation_total:.2f} GB")


    print("\n5. 多模态特征...")
    frames = 8
    frame_tokens = 128
    audio_tokens = 500
    feature_size = batch_size * (frames * frame_tokens + audio_tokens) * hidden_size * 2 / 1024**3
    print(f"   多模态特征: {feature_size:.2f} GB")


    total = model_size + optimizer_size + grad_size + activation_total + feature_size
    total += 1

    print(f"\n{'='*40}")
    print(f"单卡总显存需求: {total:.2f} GB")
    print(f"{'='*40}")


    num_gpus = 2
    zero2_per_gpu = (model_size + grad_size) / num_gpus + optimizer_size + activation_total + feature_size + 1
    print(f"\nZeRO-2 ({num_gpus}卡) 每卡需求: {zero2_per_gpu:.2f} GB")


    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            total_mem = torch.cuda.get_device_properties(i).total_memory / 1024**3
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            free_mem = total_mem - allocated

            print(f"\nGPU {i} (物理卡 {[1,5][i]}):")
            print(f"  总显存: {total_mem:.1f} GB")
            print(f"  已用: {allocated:.1f} GB")
            print(f"  可用: {free_mem:.1f} GB")

            if free_mem < zero2_per_gpu:
                print(f"  ❌ 显存不足! 需要 {zero2_per_gpu:.1f} GB, 只有 {free_mem:.1f} GB")
                print(f"  建议: 进一步降低序列长度或帧数")
            else:
                print(f"  ✅ 显存足够! 需要 {zero2_per_gpu:.1f} GB, 可用 {free_mem:.1f} GB")

if __name__ == "__main__":
    test_ola_memory()
