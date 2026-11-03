#!/usr/bin/env python3
"""
调试版train.py，添加详细日志
"""
import sys

print("=== 开始调试 train.py ===")
print(f"命令行参数: {sys.argv}")
print(f"参数数量: {len(sys.argv)}")


vision_tower_found = False
for i, arg in enumerate(sys.argv):
    if arg == '--vision_tower':
        vision_tower_found = True
        if i+1 < len(sys.argv):
            print(f"找到 --vision_tower 参数，值: {sys.argv[i+1]}")
        else:
            print("错误: --vision_tower 后面没有值!")
    elif '--vision_tower=' in arg:
        vision_tower_found = True
        print(f"找到 --vision_tower= 参数，值: {arg.split('=')[1]}")

if not vision_tower_found:
    print("警告: 没有找到 --vision_tower 参数")


import ola.train.train as original_train


original_train_original = original_train.train

def debug_train():
    print("\n=== 进入原始train函数 ===")


    from transformers import HfArgumentParser
    from dataclasses import dataclass, field
    from typing import Optional

    @dataclass
    class DebugModelArgs:
        vision_tower: Optional[str] = field(default=None)

    @dataclass
    class DebugDataArgs:
        data_path: Optional[str] = field(default=None)

    @dataclass
    class DebugTrainingArgs:
        output_dir: str = field(default="./output")

    debug_parser = HfArgumentParser((DebugModelArgs, DebugDataArgs, DebugTrainingArgs))
    try:
        debug_args = debug_parser.parse_args_into_dataclasses()
        print(f"调试解析器结果: vision_tower={debug_args[0].vision_tower}")
    except Exception as e:
        print(f"调试解析器错误: {e}")


    return original_train_original()

original_train.train = debug_train


if __name__ == "__main__":
    original_train.train()
