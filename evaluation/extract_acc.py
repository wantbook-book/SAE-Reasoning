#!/usr/bin/env python3
"""
从评估结果 JSON 文件中提取准确率信息

用法:
python extract_accuracy.py
"""

import json
import re
import os
import glob

def extract_accuracy_info(json_file_path):
    """
    从指定的 JSON 文件中提取 exact_match,none、task 和 sae_feature_idx 信息
    
    Args:
        json_file_path: JSON 结果文件路径
    
    Returns:
        dict: 包含提取信息的字典
    """
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 提取 exact_match,none
        exact_match = None
        task_name = None
        
        # 从 results 中提取 exact_match 和 task 名称
        if 'results' in data:
            for task, metrics in data['results'].items():
                if 'exact_match,none' in metrics:
                    exact_match = metrics['exact_match,none']
                    task_name = task
                    break
        
        # 从 model_args 中提取 sae_feature_idx
        sae_feature_idx = None
        if 'config' in data and 'model_args' in data['config']:
            model_args = data['config']['model_args']
            # 使用正则表达式提取 sae_feature_idx
            match = re.search(r'sae_feature_idx=([^,]+)', model_args)
            if match:
                sae_feature_idx = match.group(1)
        
        return {
            'exact_match,none': exact_match,
            'task': task_name,
            'sae_feature_idx': sae_feature_idx
        }
        
    except FileNotFoundError:
        print(f"错误: 找不到文件 {json_file_path}")
        return None
    except json.JSONDecodeError:
        print(f"错误: 无法解析 JSON 文件 {json_file_path}")
        return None
    except Exception as e:
        print(f"错误: {e}")
        return None

def main():
    # 评估目录路径
    evaluation_dir = "/angel/fwk/code/SAE-Reasoning/evaluation/results"
    
    # 查找所有 deepseek-llama-8b-xxxx 格式的目录
    pattern = os.path.join(evaluation_dir, "deepseek-llama-8b-*")
    directories = glob.glob(pattern)
    
    if not directories:
        print("未找到任何 deepseek-llama-8b-xxxx 格式的目录")
        return
    
    print(f"找到 {len(directories)} 个目录，开始处理...\n")
    
    # 存储所有结果
    all_results = []
    
    for dir_path in sorted(directories):
        dir_name = os.path.basename(dir_path)
        print(f"处理目录: {dir_name}")
        
        # 查找该目录下的 results 文件
        model_dir = os.path.join(dir_path, "deepseek-ai__DeepSeek-R1-Distill-Llama-8B")
        if not os.path.exists(model_dir):
            print(f"  警告: 未找到模型目录 {model_dir}")
            continue
            
        # 查找 results_*.json 文件
        results_pattern = os.path.join(model_dir, "results_*.json")
        results_files = glob.glob(results_pattern)
        
        if not results_files:
            print(f"  警告: 未找到 results 文件")
            continue
            
        # 处理找到的第一个 results 文件
        results_file = results_files[0]
        print(f"  处理文件: {os.path.basename(results_file)}")
        
        # 提取信息
        result = extract_accuracy_info(results_file)
        
        if result:
            result['directory'] = dir_name
            result['file_path'] = results_file
            all_results.append(result)
            print(f"  exact_match,none: {result['exact_match,none']}")
            print(f"  task: {result['task']}")
            print(f"  sae_feature_idx: {result['sae_feature_idx']}")
        else:
            print(f"  错误: 提取失败")
        
        print()  # 空行分隔
    
    # 输出汇总结果
    if all_results:
        # 按SAE特征索引排序
        def sort_key(result):
            sae_idx = result['sae_feature_idx']
            if sae_idx is None or sae_idx == 'N/A':
                return float('inf')  # 将None或N/A排到最后
            try:
                return int(sae_idx)
            except ValueError:
                return float('inf')  # 无法转换为整数的排到最后
        
        all_results.sort(key=sort_key)
        
        print("=" * 80)
        print("汇总结果:")
        print("=" * 80)
        print(f"{'目录名':<40} {'任务':<25} {'准确率':<10} {'SAE特征索引':<15}")
        print("-" * 80)
        
        for result in all_results:
            dir_name = result['directory']
            task = result['task'] or 'N/A'
            accuracy = f"{result['exact_match,none']:.4f}" if result['exact_match,none'] is not None else 'N/A'
            sae_idx = result['sae_feature_idx'] or 'N/A'
            print(f"{dir_name:<40} {task:<25} {accuracy:<10} {sae_idx:<15}")
    else:
        print("未成功提取任何结果")

if __name__ == "__main__":
    main()