from pathlib import Path
import json
from typing import Dict, Any

# ANSI 颜色代码
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def format_metric_value(value: float, precision: int = 1) -> str:
    """格式化指标值，保留指定小数位数"""
    if isinstance(value, (int, float)):
        return f"{value:.{precision}f}"
    return str(value)

def print_metrics_oneline(data: Dict[str, Any], model_name: str, dataset_name: str):
    """以一行格式打印指标"""
    # 提取关键指标
    pass_at_1_avg = data.get('accuracy_avg', data.get('mean_accuracy', 'N/A'))*100
    pass_at_1_std = data.get('accuracy_std', data.get('std_accuracy', 'N/A'))*100
    pass_at_k = data.get('pass@k', data.get('pass@4', 'N/A'))*100
    maj_at_k = data.get('maj@k', data.get('maj@4', 'N/A'))*100
    
    # 格式化 Pass@1 (合并平均值和标准差)
    if pass_at_1_avg != 'N/A' and pass_at_1_std != 'N/A':
        pass_at_1_display = f"{format_metric_value(pass_at_1_avg)}±{format_metric_value(pass_at_1_std)}"
    elif pass_at_1_avg != 'N/A':
        pass_at_1_display = format_metric_value(pass_at_1_avg)
    else:
        pass_at_1_display = 'N/A'
    
    # 格式化其他指标
    pass_at_k_display = format_metric_value(pass_at_k) if pass_at_k != 'N/A' else 'N/A'
    maj_at_k_display = format_metric_value(maj_at_k) if maj_at_k != 'N/A' else 'N/A'
    
    # 一行输出格式
    print(f"{Colors.BOLD}{model_name:<30}{Colors.ENDC} | "
          f"{Colors.OKCYAN}{dataset_name:<20}{Colors.ENDC} | "
          f"{Colors.OKGREEN}Pass@1: {pass_at_1_display:<12}{Colors.ENDC} | "
          f"{Colors.OKGREEN}Pass@k: {pass_at_k_display:<8}{Colors.ENDC} | "
          f"{Colors.OKGREEN}Maj@k: {maj_at_k_display:<8}{Colors.ENDC}")

def print_metrics_ckpt_oneline(data: Dict[str, Any], model_name: str, ckpt_name: str, dataset_name: str):
    """以一行格式打印指标"""
    # 提取关键指标
    pass_at_1_avg = data.get('accuracy_avg', data.get('mean_accuracy', 'N/A'))*100
    pass_at_1_std = data.get('accuracy_std', data.get('std_accuracy', 'N/A'))*100
    pass_at_k = data.get('pass@k', data.get('pass@4', 'N/A'))*100
    maj_at_k = data.get('maj@k', data.get('maj@4', 'N/A'))*100
    
    # 格式化 Pass@1 (合并平均值和标准差)
    if pass_at_1_avg != 'N/A' and pass_at_1_std != 'N/A':
        pass_at_1_display = f"{format_metric_value(pass_at_1_avg)}±{format_metric_value(pass_at_1_std)}"
    elif pass_at_1_avg != 'N/A':
        pass_at_1_display = format_metric_value(pass_at_1_avg)
    else:
        pass_at_1_display = 'N/A'
    
    # 格式化其他指标
    pass_at_k_display = format_metric_value(pass_at_k) if pass_at_k != 'N/A' else 'N/A'
    maj_at_k_display = format_metric_value(maj_at_k) if maj_at_k != 'N/A' else 'N/A'
    
    # 一行输出格式
    print(f"{Colors.BOLD}{model_name:<30}{Colors.ENDC} | "
        f"{Colors.BOLD}{ckpt_name:<30}{Colors.ENDC} | "
          f"{Colors.OKCYAN}{dataset_name:<20}{Colors.ENDC} | "
          f"{Colors.OKGREEN}Pass@1: {pass_at_1_display:<12}{Colors.ENDC} | "
          f"{Colors.OKGREEN}Pass@k: {pass_at_k_display:<8}{Colors.ENDC} | "
          f"{Colors.OKGREEN}Maj@k: {maj_at_k_display:<8}{Colors.ENDC}")

def print_math_results(dir_path: Path):
    """打印数学评估结果"""
    processed_count = 0
    
    # 打印表头
    print(f"\n{Colors.BOLD}{Colors.HEADER}{'模型名称':<30} | {'数据集':<20} | {'Pass@1':<18} | {'Pass@k':<14} | {'Maj@k':<14}{Colors.ENDC}")
    print(f"{Colors.BOLD}{'─' * 30} | {'─' * 20} | {'─' * 18} | {'─' * 14} | {'─' * 14}{Colors.ENDC}")
    
    # 遍历模型目录
    for model_dir in sorted(dir_path.iterdir()):
        if model_dir.is_dir():
            # 遍历数据集目录
            for dataset_dir in sorted(model_dir.iterdir()):
                if dataset_dir.is_dir():
                    # 查找 metrics.json 文件
                    metrics_files = list(dataset_dir.glob('*metrics.json'))
                    if metrics_files:
                        metrics_file = metrics_files[0]  # 取第一个找到的文件
                        try:
                            with open(metrics_file, 'r', encoding='utf-8') as f:
                                data = json.load(f)
                            
                            # 从路径中提取模型和数据集信息
                            model_name = model_dir.name
                            dataset_name = dataset_dir.name
                            
                            print_metrics_oneline(data, model_name, dataset_name)
                            processed_count += 1
                            
                        except (json.JSONDecodeError, FileNotFoundError) as e:
                            print(f"{Colors.FAIL}❌ 读取文件失败 {metrics_file}: {e}{Colors.ENDC}")
    
    # 打印处理总结
    print(f"\n{Colors.BOLD}{Colors.OKCYAN}📈 处理完成！共处理了 {processed_count} 个模型/数据集组合{Colors.ENDC}")

def print_math_ckpt_results(dir_path: Path):
    """打印数学评估结果"""
    processed_count = 0
    
    # 打印表头
    print(f"\n{Colors.BOLD}{Colors.HEADER}{'模型名称':<30} | {'ckpt': <30} | {'数据集':<20} | {'Pass@1':<18} | {'Pass@k':<14} | {'Maj@k':<14}{Colors.ENDC}")
    print(f"{Colors.BOLD}{'─' * 30} | {'─' * 30} | {'─' * 20} | {'─' * 18} | {'─' * 14} | {'─' * 14}{Colors.ENDC}")
    
    
    # 遍历模型目录
    for model_dir in sorted(dir_path.iterdir()):
        if model_dir.is_dir():
            # 遍历数据集目录
            for ckpt_dir in sorted(model_dir.iterdir()):
                if ckpt_dir.is_dir():
                    for dataset_dir in sorted(ckpt_dir.iterdir()):
                        if dataset_dir.is_dir():
                            # 查找 metrics.json 文件
                            metrics_files = list(dataset_dir.glob('*metrics.json'))
                            if metrics_files:
                                metrics_file = metrics_files[0]  # 取第一个找到的文件
                                try:
                                    with open(metrics_file, 'r', encoding='utf-8') as f:
                                        data = json.load(f)
                                    
                                    # 从路径中提取模型和数据集信息
                                    model_name = model_dir.name
                                    ckpt_name = ckpt_dir.name
                                    dataset_name = dataset_dir.name
                                    
                                    print_metrics_ckpt_oneline(data, model_name, ckpt_name, dataset_name)
                                    processed_count += 1
                                    
                                except (json.JSONDecodeError, FileNotFoundError) as e:
                                    print(f"{Colors.FAIL}❌ 读取文件失败 {metrics_file}: {e}{Colors.ENDC}")
    
    # 打印处理总结
    print(f"\n{Colors.BOLD}{Colors.OKCYAN}📈 处理完成！共处理了 {processed_count} 个模型/数据集组合{Colors.ENDC}")

def print_gpqa_results(dir_path: Path):
    """打印数学评估结果"""
    processed_count = 0
    
    # 打印表头
    print(f"\n{Colors.BOLD}{Colors.HEADER}{'模型名称':<30} | {'数据集':<20} | {'Pass@1':<18} | {'Pass@k':<14} | {'Maj@k':<14}{Colors.ENDC}")
    print(f"{Colors.BOLD}{'─' * 30} | {'─' * 20} | {'─' * 18} | {'─' * 14} | {'─' * 14}{Colors.ENDC}")
    
    # 遍历模型目录
    for model_dir in sorted(dir_path.iterdir()):
        if model_dir.is_dir():
            # 遍历数据集目录
            for dataset_dir in sorted(model_dir.iterdir()):
                if dataset_dir.is_dir():
                    # 查找 metrics.json 文件
                    metrics_files = list(dataset_dir.glob('*metric.json'))
                    if metrics_files:
                        metrics_file = metrics_files[0]  # 取第一个找到的文件
                        try:
                            with open(metrics_file, 'r', encoding='utf-8') as f:
                                data = json.load(f)
                            
                            # 从路径中提取模型和数据集信息
                            model_name = model_dir.name
                            dataset_name = dataset_dir.name
                            
                            print_metrics_oneline(data, model_name, dataset_name)
                            processed_count += 1
                            
                        except (json.JSONDecodeError, FileNotFoundError) as e:
                            print(f"{Colors.FAIL}❌ 读取文件失败 {metrics_file}: {e}{Colors.ENDC}")
    
    # 打印处理总结
    print(f"\n{Colors.BOLD}{Colors.OKCYAN}📈 处理完成！共处理了 {processed_count} 个模型/数据集组合{Colors.ENDC}")

def print_gpqa_ckpt_results(dir_path: Path):
    """打印数学评估结果"""
    processed_count = 0
    
    # 打印表头
    print(f"\n{Colors.BOLD}{Colors.HEADER}{'模型名称':<30} | {'ckpt': <30} | {'数据集':<20} | {'Pass@1':<18} | {'Pass@k':<14} | {'Maj@k':<14}{Colors.ENDC}")
    print(f"{Colors.BOLD}{'─' * 30} | {'─' * 30} | {'─' * 20} | {'─' * 18} | {'─' * 14} | {'─' * 14}{Colors.ENDC}")
    
    # 遍历模型目录
    for model_dir in sorted(dir_path.iterdir()):
        if model_dir.is_dir():
            # 遍历数据集目录
            for ckpt_dir in sorted(model_dir.iterdir()):
                if ckpt_dir.is_dir():
                    for dataset_dir in sorted(ckpt_dir.iterdir()):
                        if dataset_dir.is_dir():
                            # 查找 metrics.json 文件
                            metrics_files = list(dataset_dir.glob('*metric.json'))
                            if metrics_files:
                                metrics_file = metrics_files[0]  # 取第一个找到的文件
                                try:
                                    with open(metrics_file, 'r', encoding='utf-8') as f:
                                        data = json.load(f)
                                    
                                    # 从路径中提取模型和数据集信息
                                    model_name = model_dir.name
                                    ckpt_name = ckpt_dir.name
                                    dataset_name = dataset_dir.name
                                    
                                    print_metrics_ckpt_oneline(data, model_name, ckpt_name, dataset_name)
                                    processed_count += 1
                                    
                                except (json.JSONDecodeError, FileNotFoundError) as e:
                                    print(f"{Colors.FAIL}❌ 读取文件失败 {metrics_file}: {e}{Colors.ENDC}")
    
    # 打印处理总结
    print(f"\n{Colors.BOLD}{Colors.OKCYAN}📈 处理完成！共处理了 {processed_count} 个模型/数据集组合{Colors.ENDC}")

if __name__ == '__main__':
    # dir_path = Path("/pubshare/fwk/code/sae/SAE-Reasoning2/vllm_sae_evaluation/outputs/deepseek-ai/DeepSeek-R1-Distill-Llama-8B/math_eval_sampling_4")
    # print_math_results(dir_path)
    
    dir_path = Path("/pubshare/fwk/code/sae/SAE-Reasoning2/vllm_sae_evaluation/outputs/deepseek-ai/DeepSeek-R1-Distill-Llama-8B/math_eval_sampling_4")
    print_math_ckpt_results(dir_path)

    # dir_path = Path("/pubshare/fwk/code/sae/SAE-Reasoning2/vllm_sae_evaluation/eval_results/pubshare/LLM/deepseek-ai/DeepSeek-R1-Distill-Llama-8B")
    # print_gpqa_results(dir_path)

    # dir_path = Path("/pubshare/fwk/code/sae/SAE-Reasoning2/vllm_sae_evaluation/outputs/pubshare/LLM/deepseek-ai/DeepSeek-R1-Distill-Llama-8B/gpqa_sampling_4")
    # print_gpqa_results(dir_path)