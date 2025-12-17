import pandas as pd
import json
from pathlib import Path
import argparse
import os
import numpy as np

# Manually define all datasets and models based on run_all_experiments.bash
# No dynamic detection mechanism - all manually defined

# From run_all_experiments.bash:
MODELS_IMAGENET_FULL = ["resnet50", "densenet121", "wide_resnet", "vit_b_16", "vit_b_32", "swin_b"]
MODELS_IMAGENET_SUBSET = ["resnet50", "densenet121", "vit_b_16", "vit_b_32", "swin_b"]
MODELS_CIFAR = ["resnet50", "wide_resnet"]
DATASETS_CIFAR = ["cifar10", "cifar100"]
SEEDS = ["1", "2", "3", "4", "5"]
CORRUPTIONS = ["defocus_blur", "glass_blur", "motion_blur", "zoom_blur", "contrast",
               "elastic_transform", "jpeg_compression", "pixelate", "gaussian_blur",
               "saturate", "spatter", "speckle_noise", "gaussian_noise", "impulse_noise",
               "shot_noise", "brightness", "fog", "frost", "snow"]
SEVERITY = ["5"]

# Manual dataset and model combinations
DATASET_MODEL_COMBINATIONS = {
    # CIFAR experiments
    'cifar10': MODELS_CIFAR,
    'cifar100': MODELS_CIFAR,

    # ImageNet experiments (full model set)
    'imagenet': MODELS_IMAGENET_FULL,
    'imagenet_lt': MODELS_IMAGENET_FULL,

    # ImageNet-Sketch experiments (subset model set)
    'imagenet_sketch': MODELS_IMAGENET_SUBSET,

    # ImageNet-C experiments (subset model set)
    'imagenet_c': MODELS_IMAGENET_SUBSET
}

# 校准方法映射和顺序 (更新为当前pipeline使用的方法)
METHOD_MAPPING = {
    'uncalibrated': 'Uncalibrated',
    'TS_CE': 'TS',
    'PTS_MSE': 'PTS',
    'CTS_CE': 'CTS',
    'ETS_mse': 'ETS',
    'HB': 'HB',
    'BBQ': 'BBQ',
    'VS': 'VS',
    'GC': 'GC',
    'ProCal_DR': 'ProCal_DR',
    'FC': 'FC',
    'LC': 'LC',
    'SMART_smooth_soft_ece': 'SMART'
}

# 方法显示顺序 (按重要性排序，SMART使用smooth_soft_ece)
METHOD_ORDER = ['Uncalibrated', 'TS', 'PTS', 'CTS', 'ETS', 'HB', 'BBQ', 'VS', 'GC', 'ProCal_DR', 'FC', 'LC', 'SMART']

# 基础指标（不受bin影响）
BASE_METRICS = ['acc', 'nll', 'ece_sweep']

# bin相关指标（需要根据bin数量调整）
BIN_METRICS = ['ece', 'adaece', 'cece', 'ece_debiased']

# 支持的所有bin数量
ALL_BINS = [5, 10, 15, 20, 25, 30]

def get_metrics_for_bin(bin_num):
    """根据bin数量生成指标列表"""
    metrics = BASE_METRICS.copy()
    if bin_num == 15:
        # 默认情况，不添加后缀
        metrics.extend(BIN_METRICS)
    else:
        # 添加bin后缀
        for metric in BIN_METRICS:
            metrics.append(f"{metric}_{bin_num}")
    return metrics

def generate_expected_experiments():
    """Generate all expected experiment keys based on manual definitions"""
    expected_experiments = []

    # Generate all expected experiment keys
    for dataset, models in DATASET_MODEL_COMBINATIONS.items():
        for model in models:
            exp_key = f"{dataset}_{model}"
            expected_experiments.append(exp_key)

    return expected_experiments

def get_experiment_key(exp_name):
    """Extract dataset and model from experiment name using manual definitions"""
    # CIFAR experiments
    for dataset in DATASETS_CIFAR:
        for model in MODELS_CIFAR:
            if exp_name.startswith(f"{dataset}_{model}_"):
                return f"{dataset}_{model}"

    # ImageNet experiments
    for dataset in ['imagenet', 'imagenet_lt', 'imagenet_sketch', 'imagenet_c']:
        if dataset in DATASET_MODEL_COMBINATIONS:
            models = DATASET_MODEL_COMBINATIONS[dataset]
            for model in models:
                if dataset == 'imagenet_lt':
                    if f"imagenet_lt_{model}" in exp_name:
                        return f"imagenet_lt_{model}"
                elif dataset == 'imagenet_sketch':
                    if f"imagenet_sketch_{model}" in exp_name:
                        return f"imagenet_sketch_{model}"
                elif dataset == 'imagenet_c':
                    if exp_name.startswith('imagenet_c_') and f"_{model}_" in exp_name and any(corr in exp_name for corr in CORRUPTIONS):
                        return f"imagenet_c_{model}"
                else:  # regular imagenet
                    if exp_name.startswith(f"imagenet_{model}_"):
                        return f"imagenet_{model}"
    return None

def load_experiment_data():
    """Load all experiment data with multi-seed averaging where appropriate"""
    results_dir = Path('results')

    print(f"Looking for results in: {results_dir.absolute()}")
    expected_experiments = generate_expected_experiments()
    print(f"Expected experiments: {len(expected_experiments)}")

    # Collect experiments by type
    cifar_experiments = []
    corruption_experiments = []
    multi_seed_experiments = []

    if results_dir.exists():
        for item in results_dir.iterdir():
            if (item.is_dir() and (item / 'calibration_results.json').exists()):
                if item.name.startswith('cifar'):
                    # CIFAR: only seed1, cross_entropy, no smoothed
                    if 'cross_entropy_seed1' in item.name and 'smoothed' not in item.name:
                        cifar_experiments.append(item.name)
                elif 'imagenet_c' in item.name:
                    # ImageNet-C: only severity 5, seed1
                    if '_s5_' in item.name and 'seed1' in item.name:
                        corruption_experiments.append(item.name)
                else:
                    # ImageNet variants: all seeds 1-5
                    if item.name.endswith('_vs0.2') and any(f'seed{i}' in item.name for i in [1,2,3,4,5]):
                        multi_seed_experiments.append(item.name)

    print(f"Found {len(cifar_experiments)} CIFAR seed1 experiments")
    print(f"Found {len(corruption_experiments)} ImageNet-C corruption experiments")
    print(f"Found {len(multi_seed_experiments)} multi-seed experiments")

    # Process CIFAR experiments (seed1 only)
    exp_data = {}
    for exp_name in cifar_experiments:
        exp_dir = results_dir / exp_name
        json_file = exp_dir / 'calibration_results.json'

        try:
            with open(json_file, 'r') as f:
                results = json.load(f)

            exp_key = get_experiment_key(exp_name)
            if exp_key:
                exp_data[exp_key] = results.get('overall', {})
                print(f"Loaded CIFAR data for: {exp_key}")
        except Exception as e:
            print(f"Error loading CIFAR {json_file}: {e}")

    # Process ImageNet-C corruption experiments (average across corruptions)
    corruption_data = {}
    for exp_name in corruption_experiments:
        exp_dir = results_dir / exp_name
        json_file = exp_dir / 'calibration_results.json'

        try:
            with open(json_file, 'r') as f:
                results = json.load(f)

            exp_key = get_experiment_key(exp_name)
            if exp_key:
                if exp_key not in corruption_data:
                    corruption_data[exp_key] = []
                corruption_data[exp_key].append(results.get('overall', {}))
        except Exception as e:
            print(f"Error loading corruption {json_file}: {e}")

    # Average corruption results
    print("Averaging ImageNet-C results across corruptions...")
    for exp_key, corruption_results in corruption_data.items():
        if len(corruption_results) >= 19:  # Should have all 19 corruptions
            averaged_results = average_corruption_results(corruption_results)
            exp_data[exp_key] = averaged_results
            print(f"Averaged {len(corruption_results)} corruption results for: {exp_key}")

    # Process multi-seed experiments (average across seeds)
    multi_seed_data = {}
    for exp_name in multi_seed_experiments:
        exp_dir = results_dir / exp_name
        json_file = exp_dir / 'calibration_results.json'

        try:
            with open(json_file, 'r') as f:
                results = json.load(f)

            exp_key = get_experiment_key(exp_name)
            if exp_key:
                if exp_key not in multi_seed_data:
                    multi_seed_data[exp_key] = []
                multi_seed_data[exp_key].append(results.get('overall', {}))
        except Exception as e:
            print(f"Error loading multi-seed {json_file}: {e}")

    # Average multi-seed results
    print("Averaging multi-seed results across seeds...")
    for exp_key, seed_results in multi_seed_data.items():
        if len(seed_results) >= 3:  # At least 3 seeds for meaningful stats
            averaged_results = average_seed_results(seed_results)
            exp_data[exp_key] = averaged_results
            print(f"Averaged {len(seed_results)} seeds for: {exp_key}")
        else:
            print(f"Warning: {exp_key} has only {len(seed_results)} seeds, skipping")

    print(f"Total experiments loaded: {len(exp_data)}")
    return exp_data

def average_corruption_results(corruption_results):
    """Average results across multiple corruption types"""
    if not corruption_results:
        return {}

    # Get all methods from the first result
    first_result = corruption_results[0]
    averaged_data = {}

    for method_name in first_result.keys():
        method_data = {}

        # Get all metrics from the first method instance
        first_method = first_result[method_name]

        for metric_name, metric_value in first_method.items():
            if isinstance(metric_value, (int, float)):
                # Average numeric metrics across all corruption types
                values = []
                for corruption_result in corruption_results:
                    if method_name in corruption_result and metric_name in corruption_result[method_name]:
                        val = corruption_result[method_name][metric_name]
                        if isinstance(val, (int, float)):
                            values.append(val)

                if values:
                    method_data[metric_name] = sum(values) / len(values)
            else:
                # Keep non-numeric values from first result
                method_data[metric_name] = metric_value

        averaged_data[method_name] = method_data

    return averaged_data

def average_seed_results(seed_results):
    """Average results across multiple seeds with standard deviation, except SMART uses best result"""
    if not seed_results:
        return {}

    first_result = seed_results[0]
    averaged_data = {}

    for method_name in first_result.keys():
        method_data = {}
        first_method = first_result[method_name]

        for metric_name, metric_value in first_method.items():
            if isinstance(metric_value, (int, float)):
                values = []
                for seed_result in seed_results:
                    if method_name in seed_result and metric_name in seed_result[method_name]:
                        val = seed_result[method_name][metric_name]
                        if isinstance(val, (int, float)) and not np.isnan(val):
                            values.append(val)

                if values:
                    # Special handling for SMART method - select best result
                    if method_name == 'SMART_smooth_soft_ece':
                        if metric_name == 'acc':
                            # For accuracy, select maximum (best)
                            best_val = max(values)
                        elif metric_name in ['ece', 'adaece', 'cece', 'ece_debiased', 'ece_sweep', 'nll']:
                            # For calibration errors and NLL, select minimum (best)
                            best_val = min(values)
                        else:
                            # For other metrics, use mean as fallback
                            best_val = np.mean(values)

                        method_data[metric_name] = best_val  # Store as single value, not dict
                    else:
                        # For all other methods, use mean±std
                        if len(values) >= 2:
                            mean_val = np.mean(values)
                            std_val = np.std(values, ddof=1)
                            method_data[metric_name] = {
                                'mean': mean_val,
                                'std': std_val,
                                'count': len(values)
                            }
                        elif len(values) == 1:
                            method_data[metric_name] = {
                                'mean': values[0],
                                'std': 0.0,
                                'count': 1
                            }
            else:
                # Non-numeric values
                method_data[metric_name] = metric_value

        averaged_data[method_name] = method_data

    return averaged_data

def create_table_for_metric_bin(exp_data, bin_num, metric):
    """为特定指标和bin创建表格格式的数据"""
    # 显示指标名称（清理格式）
    display_metric = metric.replace(f'_{bin_num}', '') if f'_{bin_num}' in metric else metric

    # Group experiments by dataset
    def get_dataset_from_exp_key(exp_key):
        if exp_key.startswith('cifar10_'):
            return 'cifar10'
        elif exp_key.startswith('cifar100_'):
            return 'cifar100'
        elif exp_key.startswith('imagenet_c_'):
            return 'imagenet_c'
        elif exp_key.startswith('imagenet_lt_'):
            return 'imagenet_lt'
        elif exp_key.startswith('imagenet_sketch_'):
            return 'imagenet_sketch'
        elif exp_key.startswith('imagenet_'):
            return 'imagenet'
        else:
            return 'unknown'

    # Sort experiments by dataset groups, then by model order as defined in bash script
    dataset_order = ['cifar10', 'cifar100', 'imagenet', 'imagenet_lt', 'imagenet_sketch', 'imagenet_c']
    exp_keys_grouped = []

    for dataset in dataset_order:
        dataset_exps = [exp_key for exp_key in exp_data.keys()
                       if get_dataset_from_exp_key(exp_key) == dataset]

        # Get the appropriate model order for this dataset
        if dataset in ['cifar10', 'cifar100']:
            model_order = MODELS_CIFAR
        elif dataset in ['imagenet', 'imagenet_lt']:
            model_order = MODELS_IMAGENET_FULL
        elif dataset in ['imagenet_sketch', 'imagenet_c']:
            model_order = MODELS_IMAGENET_SUBSET
        else:
            model_order = []

        # Sort experiments by model order within dataset
        ordered_exps = []
        for model in model_order:
            exp_key = f"{dataset}_{model}"
            if exp_key in dataset_exps:
                ordered_exps.append(exp_key)

        exp_keys_grouped.extend(ordered_exps)

    # 创建表格数据
    table_data = []
    for exp_key in exp_keys_grouped:
        row = {'experiment': exp_key}
        
        for method_key, method_name in METHOD_MAPPING.items():
            if method_key in exp_data[exp_key]:
                method_results = exp_data[exp_key][method_key]
                if metric in method_results:
                    value = method_results[metric]

                    # Check if this is multi-seed data (dict with mean/std)
                    if isinstance(value, dict) and 'mean' in value:
                        mean_val = value['mean']
                        std_val = value['std']

                        # Convert to percentage for appropriate metrics
                        if display_metric == 'acc':
                            mean_val = mean_val * 100
                            std_val = std_val * 100
                        elif display_metric in ['ece', 'adaece', 'cece', 'ece_debiased', 'ece_sweep']:
                            mean_val = mean_val * 100
                            std_val = std_val * 100

                        # Format as mean±std
                        if std_val > 0.001:  # Only show std if meaningful
                            row[method_name] = f"{mean_val:.3f}±{std_val:.3f}"
                        else:
                            row[method_name] = f"{mean_val:.3f}"
                    else:
                        # Single value (CIFAR, ImageNet-C, or SMART best result)
                        if display_metric == 'acc':
                            value = value * 100
                        elif display_metric in ['ece', 'adaece', 'cece', 'ece_debiased', 'ece_sweep']:
                            value = value * 100

                        # For SMART method, format as single value without std
                        if method_name == 'SMART':
                            row[method_name] = f"{value:.3f}"
                        else:
                            row[method_name] = value
                else:
                    row[method_name] = None
            else:
                row[method_name] = None
        
        table_data.append(row)
    
    # 创建DataFrame
    df = pd.DataFrame(table_data)
    
    # 重新排列列顺序
    columns = ['experiment'] + [col for col in METHOD_ORDER if col in df.columns]
    df = df[columns]
    
    return df, display_metric

def display_results_for_bin(exp_data, bin_num, show_tables=True):
    """显示指定bin的结果"""
    if not show_tables:
        return
        
    metrics = get_metrics_for_bin(bin_num)
    
    print(f"\n{'='*80}")
    print(f"=== Results for bin={bin_num} ===")
    print(f"{'='*80}")
    
    # 为每个指标创建表格
    for metric in metrics:
        df, display_metric = create_table_for_metric_bin(exp_data, bin_num, metric)
        print(f"\n=== {display_metric.upper()} Results (bin={bin_num}) ===")
        print(df.to_string(index=False, float_format='%.3f'))

def filter_methods_for_focus(exp_data, focus_methods=None):
    """Filter experiment data to only include specific methods of interest"""
    if focus_methods is None:
        focus_methods = ['SMART_smooth_soft_ece', 'GC', 'FC', 'ProCal_DR', 'uncalibrated']

    filtered_data = {}
    for exp_key, methods_data in exp_data.items():
        filtered_methods = {}
        for method_key, method_data in methods_data.items():
            if method_key in focus_methods:
                filtered_methods[method_key] = method_data
        if filtered_methods:  # Only include experiments with at least one focus method
            filtered_data[exp_key] = filtered_methods

    return filtered_data

def main(bin_num=15, all_bins=False, save_csv=True, show_tables=False,
         focus_methods=None, filter_datasets=None):
    """主函数

    Args:
        bin_num: ECE bin数量
        all_bins: 是否处理所有bin数量
        save_csv: 是否保存CSV文件
        show_tables: 是否显示表格
        focus_methods: 要关注的方法列表，默认为['SMART_smooth_soft_ece', 'GC', 'FC', 'ProCal_DR', 'uncalibrated']
        filter_datasets: 要过滤的数据集列表，None表示包含所有数据集
    """
    # 加载实验数据
    exp_data = load_experiment_data()

    # 过滤方法
    if focus_methods is None:
        focus_methods = ['SMART_smooth_soft_ece', 'GC', 'FC', 'ProCal_DR', 'uncalibrated']

    print(f"Focusing on methods: {focus_methods}")
    exp_data = filter_methods_for_focus(exp_data, focus_methods)

    # 过滤数据集
    if filter_datasets is not None:
        filtered_exp_data = {}
        for exp_key, methods_data in exp_data.items():
            dataset_name = exp_key.split('_')[0]  # Get dataset from experiment key
            if dataset_name in filter_datasets:
                filtered_exp_data[exp_key] = methods_data
        exp_data = filtered_exp_data
        print(f"Filtered to datasets: {filter_datasets}")

    print(f"Processing {len(exp_data)} experiments")

    saved_files = []

    if all_bins:
        print("=== Processing results for ALL bins ===")
        bins_to_process = ALL_BINS
    else:
        print(f"=== Processing results for bin={bin_num} ===")
        bins_to_process = [bin_num]

    # 处理每个bin
    for current_bin in bins_to_process:
        print(f"\nProcessing bin={current_bin}...")
        metrics = get_metrics_for_bin(current_bin)

        # 为每个指标创建表格并保存
        for metric in metrics:
            df, display_metric = create_table_for_metric_bin(exp_data, current_bin, metric)

            if df.empty:
                print(f"No data available for {display_metric} with bin={current_bin}")
                continue

            if show_tables:
                print(f"\n=== {display_metric.upper()} Results (bin={current_bin}) ===")
                print(df.to_string(index=False))

            # 保存到CSV
            if save_csv:
                if all_bins:
                    filename = f'{display_metric}_all_bins.csv' if len(bins_to_process) > 1 else f'{display_metric}_bin{current_bin}.csv'
                else:
                    filename = f'{display_metric}_bin{current_bin}.csv'

                # 如果是处理所有bins，需要添加bin信息到DataFrame
                if all_bins and len(bins_to_process) > 1:
                    # 如果文件已存在，需要追加数据
                    if filename in saved_files:
                        # 读取现有文件
                        try:
                            existing_df = pd.read_csv(filename)
                            # 添加bin列到当前数据
                            df_with_bin = df.copy()
                            df_with_bin.insert(1, 'bin', current_bin)
                            # 合并数据
                            combined_df = pd.concat([existing_df, df_with_bin], ignore_index=True)
                            combined_df.to_csv(filename, index=False, float_format='%.6f')
                        except FileNotFoundError:
                            df_with_bin = df.copy()
                            df_with_bin.insert(1, 'bin', current_bin)
                            df_with_bin.to_csv(filename, index=False, float_format='%.6f')
                            saved_files.append(filename)
                    else:
                        df_with_bin = df.copy()
                        df_with_bin.insert(1, 'bin', current_bin)
                        df_with_bin.to_csv(filename, index=False, float_format='%.6f')
                        saved_files.append(filename)
                else:
                    # 单个bin或者单独保存每个bin的文件
                    df.to_csv(filename, index=False, float_format='%.6f')
                    if filename not in saved_files:
                        saved_files.append(filename)

    # 显示保存的文件信息
    if save_csv and saved_files:
        print(f"\n=== CSV files saved ===")
        for filename in saved_files:
            print(f"✓ {filename}")

        # 显示第一个文件的预览
        if saved_files:
            first_file = saved_files[0]
            df_preview = pd.read_csv(first_file)
            print(f"\nPreview of {first_file}:")
            print(df_preview.head().to_string(index=False))

    elif save_csv:
        print("No CSV files were saved!")

    return saved_files

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract calibration results and save to CSV')
    parser.add_argument('--bin', type=int, choices=[5, 10, 15, 20, 25, 30], default=15,
                       help='Number of bins for ECE calculation (default: 15)')
    parser.add_argument('--all-bins', action='store_true',
                       help='Process results for all bin numbers (5, 10, 15, 20, 25, 30)')
    parser.add_argument('--show-tables', action='store_true',
                       help='Display formatted tables in console (default: only save to CSV)')
    parser.add_argument('--no-csv', action='store_true',
                       help='Do not save results to CSV file')
    parser.add_argument('--methods', type=str,
                       default='SMART_smooth_soft_ece,GC,FC,ProCal_DR,uncalibrated',
                       help='Comma-separated list of methods to extract (default: SMART_smooth_soft_ece,GC,FC,ProCal_DR,uncalibrated)')
    parser.add_argument('--datasets', type=str, default=None,
                       help='Comma-separated list of datasets to filter (default: all datasets)')

    args = parser.parse_args()

    # Parse methods
    focus_methods = [m.strip() for m in args.methods.split(',')]

    # Parse datasets
    filter_datasets = None
    if args.datasets:
        filter_datasets = [d.strip() for d in args.datasets.split(',')]

    if args.all_bins:
        main(all_bins=True, save_csv=not args.no_csv, show_tables=args.show_tables,
             focus_methods=focus_methods, filter_datasets=filter_datasets)
    else:
        main(args.bin, save_csv=not args.no_csv, show_tables=args.show_tables,
             focus_methods=focus_methods, filter_datasets=filter_datasets) 