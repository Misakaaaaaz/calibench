#!/usr/bin/env python3
"""
批量训练 CIFAR-100 脚本

按顺序训练所有 loss function 和 seed：
1. 对每个 loss function
2.   对每个 seed (1-5)
3.     训练模型
4.     上传到 Hugging Face
5.     删除本地文件
6.     继续下一个 seed

目录结构: cifar100/{model}/{loss}/seed_{seed}/

支持的模型: resnet50, resnet152, densenet121, wide_resnet50, mlp_mixer_b_16, mobilenetv2
"""

import os
import sys
import subprocess
import argparse

# 配置
LOSS_FUNCTIONS = [
    'nll', 'brier', 'mmce', 'ls-0.05', 
    'flsd-53', 'flsd-3', 'dfl', 'softece', 'smoothsoftece'
]

SEEDS = [1, 2, 3, 4, 5]

# 支持的模型
MODELS = ['resnet50', 'resnet152', 'densenet121', 'wide_resnet50', 'mlp_mixer_b_16', 'mobilenetv2']

# 获取脚本所在目录
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_SCRIPT = os.path.join(SCRIPT_DIR, 'train_cifar100.py')


def run_training(loss_func, seed, model_name, data_root, save_dir, num_epochs, batch_size):
    """运行训练脚本"""
    print(f"\n{'='*80}")
    print(f"Training: {loss_func.upper()} | Seed: {seed}")
    print(f"{'='*80}")
    
    cmd = [
        sys.executable, TRAIN_SCRIPT,
        '--model', model_name,
        '--loss', loss_func,
        '--seed', str(seed),
        '--data_root', data_root,
        '--save_dir', save_dir,
        '--num_epochs', str(num_epochs),
        '--batch_size', str(batch_size),
    ]
    
    try:
        result = subprocess.run(cmd, check=True, cwd=SCRIPT_DIR)
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ 训练失败: {loss_func} seed {seed}")
        print(f"错误: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Batch training for CIFAR-100')
    parser.add_argument('--data_root', type=str, default='/share/datasets/',
                        help='Root directory for CIFAR-100 dataset')
    parser.add_argument('--save_dir', type=str, default='./checkpoints',
                        help='Base directory to save results')
    parser.add_argument('--num_epochs', type=int, default=350,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size')
    parser.add_argument('--model', type=str, default='resnet50', choices=MODELS,
                        help=f'Model architecture ({", ".join(MODELS)})')
    parser.add_argument('--loss', type=str, default=None, choices=LOSS_FUNCTIONS,
                        help='Train only specific loss function (default: all)')
    parser.add_argument('--seed', type=int, nargs='+', default=None,
                        help='Train only specific seed(s), e.g., --seed 2 3 4 5 (default: all)')
    
    args = parser.parse_args()
    
    # 验证 seed 值
    if args.seed:
        invalid_seeds = [s for s in args.seed if s not in SEEDS]
        if invalid_seeds:
            parser.error(f"Invalid seed values: {invalid_seeds}. Must be one of {SEEDS}")
    
    # 转换为绝对路径
    save_dir = os.path.abspath(os.path.expanduser(args.save_dir))
    
    print("="*80)
    print("CIFAR-100 Batch Training")
    print("="*80)
    print(f"Model: {args.model}")
    print(f"Loss functions: {LOSS_FUNCTIONS if args.loss is None else [args.loss]}")
    print(f"Seeds: {SEEDS if args.seed is None else args.seed}")
    print(f"Save directory: {save_dir}")
    print("="*80)
    
    # 确定要训练的组合
    loss_funcs = [args.loss] if args.loss else LOSS_FUNCTIONS
    seeds = args.seed if args.seed else SEEDS
    
    total_runs = len(loss_funcs) * len(seeds)
    current_run = 0
    
    success_count = 0
    failed_count = 0
    
    # 遍历所有组合
    for loss_func in loss_funcs:
        for seed in seeds:
            current_run += 1
            print(f"\n\n进度: {current_run}/{total_runs}")
            
            # 训练（包含上传和删除）
            if run_training(loss_func, seed, args.model, args.data_root, save_dir, 
                          args.num_epochs, args.batch_size):
                success_count += 1
                print(f"✓ 完成: {args.model.upper()} | {loss_func.upper()} | Seed {seed}")
            else:
                failed_count += 1
                print(f"✗ 失败: {args.model.upper()} | {loss_func.upper()} | Seed {seed}")
    
    # 总结
    print("\n" + "="*80)
    print("Training Summary")
    print("="*80)
    print(f"成功: {success_count}")
    print(f"失败: {failed_count}")
    print(f"总计: {total_runs}")
    print("="*80)


if __name__ == '__main__':
    main()


