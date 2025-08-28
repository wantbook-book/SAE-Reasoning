"""
数据标注模块：从指定数据集抽取reasoning和non-reasoning数据
"""
import os
import json
import fire
import torch
import numpy as np
from typing import List, Dict, Tuple, Optional
from datasets import Dataset, load_dataset, concatenate_datasets
from dataclasses import dataclass
from tqdm import tqdm
from transformers import AutoTokenizer
import debugpy
# debugpy.listen(5678)
# debugpy.wait_for_client()

@dataclass
class DataLabelingConfig:
    """数据标注配置"""
    reasoning_dataset: str = "open-thoughts/OpenThoughts-114k"  # reasoning数据集
    non_reasoning_dataset: str = "lmsys/lmsys-chat-1m"  # non-reasoning数据集
    reasoning_text_column: str = "text"  # reasoning数据集的文本列
    non_reasoning_text_column: str = "text"  # non-reasoning数据集的文本列
    min_text_length: int = 20  # 最小文本长度
    max_text_length: int = 2000  # 最大文本长度
    reasoning_ratio: float = 0.5  # reasoning数据的比例
    tokenizer_path: Optional[str] = None  # tokenizer模型路径
    
class DatasetBasedLabeler:
    """基于数据集的标注器"""
    
    def __init__(self, config: DataLabelingConfig):
        self.config = config
        self.tokenizer = None
        
    def _get_tokenizer(self):
        """获取tokenizer用于apply_chat_template"""
        if self.tokenizer is None:
            try:
                # 确定tokenizer路径
                tokenizer_path = self.config.tokenizer_path
                if tokenizer_path is None:
                    # 如果没有指定，则报错
                    raise ValueError("请指定tokenizer路径")
                else:
                    print(f">>> 使用指定的tokenizer路径: {tokenizer_path}")
                
                # 加载tokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(
                    tokenizer_path, 
                    trust_remote_code=True
                )
                
                # 确保有chat_template
                if not hasattr(self.tokenizer, 'chat_template') or self.tokenizer.chat_template is None:
                    # 则报错
                    raise ValueError("tokenizer没有chat_template")
                else:
                    print(">>> 使用tokenizer自带的chat template") 
                
                print(">>> 成功加载tokenizer用于chat template处理")
            except Exception as e:
                print(f">>> 警告: 无法加载tokenizer: {e}")
                print(">>> 将使用手动格式化方法")
                self.tokenizer = None
        return self.tokenizer
        
    def load_reasoning_data(self, max_samples: int) -> Dataset:
        """加载reasoning数据"""
        print(f">>> 加载reasoning数据集: {self.config.reasoning_dataset}")
        
        try:
            # 加载OpenThoughts数据集
            dataset = load_dataset(
                self.config.reasoning_dataset, 
                "metadata",
                split="train", 
                streaming=False
            )
            # 只取math类别的数据
            dataset = dataset.filter(lambda x: x["domain"] == "math")
            
            # 随机采样
            if len(dataset) > max_samples:
                dataset = dataset.shuffle(seed=42).select(range(max_samples))
                print(f">>> 采样后reasoning数据量: {len(dataset)}")


            print(f">>> 原始reasoning数据量: {len(dataset)}")
            
            # 预处理OpenThoughts数据格式
            def preprocess_openthoughts(example):
                """将OpenThoughts数据转换为统一格式"""
                problem = example.get("problem", "")
                deepseek_reasoning = example.get("deepseek_reasoning", "")
                deepseek_solution = example.get("deepseek_solution", "")
                
                # 构建统一的文本格式
                text = f"<｜begin▁of▁sentence｜><｜User｜>{problem}\nPlease reason step by step, and put your final answer within \\boxed{{}}.\n<｜Assistant｜><think>{deepseek_reasoning}</think><answer>{deepseek_solution}</answer><｜end▁of▁sentence｜>"
                
                return {
                    "text": text,
                }
            
            print(">>> 预处理OpenThoughts数据格式...")
            dataset = dataset.map(preprocess_openthoughts)
            
            # 过滤文本长度
            # def filter_text_length(example):
            #     text = example.get("text", "")
            #     if isinstance(text, str):
            #         return (self.config.min_text_length <= len(text) <= self.config.max_text_length)
            #     return False
            
            # dataset = dataset.filter(filter_text_length)
            # print(f">>> 过滤后reasoning数据量: {len(dataset)}")
            
            # 添加标签
            labels = [1] * len(dataset)  # reasoning标签为1
            dataset = dataset.add_column("reasoning_label", labels)
            
            return dataset
            
        except Exception as e:
            print(f"❌ 加载reasoning数据集失败: {e}")
            raise
    
    def load_non_reasoning_data(self, max_samples: int) -> Dataset:
        """加载non-reasoning数据"""
        print(f">>> 加载non-reasoning数据集: {self.config.non_reasoning_dataset}")
        
        try:
            # 加载LMSYS Chat数据集
            dataset = load_dataset(
                self.config.non_reasoning_dataset, 
                split="train", 
                streaming=False
            )
            
            print(f">>> 原始non-reasoning数据量: {len(dataset)}")

            # 随机采样
            if len(dataset) > max_samples:
                dataset = dataset.shuffle(seed=42).select(range(max_samples))
                print(f">>> 采样后non-reasoning数据量: {len(dataset)}")

            # 预处理LMSYS数据格式
            def preprocess_lmsys(example):
                """将LMSYS数据转换为统一格式"""
                conversation = example.get("conversation", [])
                if isinstance(conversation, list) and len(conversation) > 0:
                    try:
                        # 首先尝试使用tokenizer的apply_chat_template
                        tokenizer = self._get_tokenizer()
                        
                        # 使用apply_chat_template处理对话
                        text = tokenizer.apply_chat_template(
                            conversation, 
                            tokenize=False,
                            add_generation_prompt=False
                        )

                        return {
                            "text": text,
                        }
                        
                    except Exception as e:
                        print(f"警告: 处理对话失败: {e}")
                        raise ValueError("处理对话失败")
                
            
            print(">>> 预处理LMSYS数据格式...")
            dataset = dataset.map(preprocess_lmsys)
            
            # 过滤空文本和长度
            # def filter_text_length(example):
            #     text = example.get("text", "")
            #     if isinstance(text, str) and text.strip():
            #         return (self.config.min_text_length <= len(text) <= self.config.max_text_length)
            #     return False
            
            # dataset = dataset.filter(filter_text_length)
            # print(f">>> 过滤后non-reasoning数据量: {len(dataset)}")
            
            # 添加标签
            labels = [0] * len(dataset)  # non-reasoning标签为0
            dataset = dataset.add_column("reasoning_label", labels)
            
            return dataset
            
        except Exception as e:
            print(f"❌ 加载non-reasoning数据集失败: {e}")
            raise
    
    def create_balanced_dataset(self, total_samples: int) -> Dataset:
        """创建平衡的数据集"""
        # 计算各类数据的样本数
        reasoning_samples = int(total_samples * self.config.reasoning_ratio)
        non_reasoning_samples = total_samples - reasoning_samples
        
        print(f">>> 目标样本分布:")
        print(f"    Reasoning: {reasoning_samples}")
        print(f"    Non-reasoning: {non_reasoning_samples}")
        
        # 加载数据
        reasoning_data = self.load_reasoning_data(reasoning_samples)
        non_reasoning_data = self.load_non_reasoning_data(non_reasoning_samples)
        
        # 确保实际获得的样本数
        actual_reasoning = len(reasoning_data)
        actual_non_reasoning = len(non_reasoning_data)
        
        print(f">>> 实际获得样本分布:")
        print(f"    Reasoning: {actual_reasoning}")
        print(f"    Non-reasoning: {actual_non_reasoning}")
        
        # 合并数据集
        print(">>> 合并数据集...")
        combined_dataset = concatenate_datasets([reasoning_data, non_reasoning_data])
        
        # 打乱数据
        combined_dataset = combined_dataset.shuffle(seed=42)
        
        # 只保留必要的字段: text 和 reasoning_label
        print(">>> 清理数据集字段...")
        def clean_dataset(example):
            return {
                "text": example["text"],
                "reasoning_label": example["reasoning_label"]
            }
        
        combined_dataset = combined_dataset.map(clean_dataset, remove_columns=[
            col for col in combined_dataset.column_names 
            if col not in ["text", "reasoning_label"]
        ])
        
        print(f">>> 最终数据集大小: {len(combined_dataset)}")
        print(f">>> 最终数据集字段: {combined_dataset.column_names}")
        print(f">>> 实际reasoning比例: {actual_reasoning / len(combined_dataset):.2%}")
        
        return combined_dataset

def create_reasoning_dataset(
    output_path: str,
    total_samples: int = 10000,
    reasoning_ratio: float = 0.5,
    reasoning_dataset: str = "open-thoughts/OpenThoughts-114k",
    non_reasoning_dataset: str = "lmsys/lmsys-chat-1m",
    reasoning_text_column: str = "text",
    non_reasoning_text_column: str = "text",
    min_text_length: int = 20,
    max_text_length: int = 2000,
    tokenizer_path: Optional[str] = None
):
    """创建reasoning和non-reasoning混合数据集"""
    
    # 创建配置
    config = DataLabelingConfig(
        reasoning_dataset=reasoning_dataset,
        non_reasoning_dataset=non_reasoning_dataset,
        reasoning_text_column=reasoning_text_column,
        non_reasoning_text_column=non_reasoning_text_column,
        min_text_length=min_text_length,
        max_text_length=max_text_length,
        reasoning_ratio=reasoning_ratio,
        tokenizer_path=tokenizer_path
    )
    
    # 创建标注器
    labeler = DatasetBasedLabeler(config)
    
    # 创建平衡数据集
    print(">>> 开始创建混合数据集...")
    balanced_dataset = labeler.create_balanced_dataset(total_samples)
    
    # 保存结果
    print(f">>> 保存数据集到 {output_path}")
    os.makedirs(output_path, exist_ok=True)
    balanced_dataset.save_to_disk(output_path)
    
    # 计算统计信息
    reasoning_data = balanced_dataset.filter(lambda x: x["reasoning_label"] == 1)
    non_reasoning_data = balanced_dataset.filter(lambda x: x["reasoning_label"] == 0)
    
    stats = {
        "total_samples": len(balanced_dataset),
        "reasoning_samples": len(reasoning_data),
        "non_reasoning_samples": len(non_reasoning_data),
        "reasoning_ratio": len(reasoning_data) / len(balanced_dataset),
        "actual_reasoning_ratio": len(reasoning_data) / len(balanced_dataset),
        "config": {
            "reasoning_dataset": config.reasoning_dataset,
            "non_reasoning_dataset": config.non_reasoning_dataset,
            "target_total_samples": total_samples,
            "target_reasoning_ratio": reasoning_ratio,
            "min_text_length": min_text_length,
            "max_text_length": max_text_length
        }
    }
    
    # 保存统计信息
    stats_path = os.path.join(output_path, "dataset_stats.json")
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    
    print(">>> 数据集创建完成!")
    print(f">>> 统计信息:")
    print(f"    总样本数: {stats['total_samples']}")
    print(f"    Reasoning样本: {stats['reasoning_samples']}")
    print(f"    Non-reasoning样本: {stats['non_reasoning_samples']}")
    print(f"    实际reasoning比例: {stats['actual_reasoning_ratio']:.2%}")
    print(f">>> 数据集保存到: {output_path}")
    print(f">>> 统计信息保存到: {stats_path}")
    
    return output_path

def create_custom_reasoning_dataset(
    reasoning_texts: List[str],
    non_reasoning_texts: List[str],
    output_path: str,
    total_samples: Optional[int] = None,
    reasoning_ratio: float = 0.5
) -> str:
    """从自定义文本列表创建reasoning数据集"""
    
    print(">>> 创建自定义reasoning数据集...")
    
    # 如果没有指定总样本数，使用所有可用数据
    if total_samples is None:
        total_samples = len(reasoning_texts) + len(non_reasoning_texts)
    
    # 计算各类数据的目标样本数
    target_reasoning = int(total_samples * reasoning_ratio)
    target_non_reasoning = total_samples - target_reasoning
    
    # 限制样本数量
    target_reasoning = min(target_reasoning, len(reasoning_texts))
    target_non_reasoning = min(target_non_reasoning, len(non_reasoning_texts))
    
    print(f">>> 目标样本分布:")
    print(f"    Reasoning: {target_reasoning} (可用: {len(reasoning_texts)})")
    print(f"    Non-reasoning: {target_non_reasoning} (可用: {len(non_reasoning_texts)})")
    
    # 随机采样
    np.random.seed(42)
    
    reasoning_indices = np.random.choice(len(reasoning_texts), target_reasoning, replace=False)
    non_reasoning_indices = np.random.choice(len(non_reasoning_texts), target_non_reasoning, replace=False)
    
    selected_reasoning = [reasoning_texts[i] for i in reasoning_indices]
    selected_non_reasoning = [non_reasoning_texts[i] for i in non_reasoning_indices]
    
    # 创建数据集
    all_texts = selected_reasoning + selected_non_reasoning
    all_labels = [1] * len(selected_reasoning) + [0] * len(selected_non_reasoning)
    
    # 打乱数据
    combined_data = list(zip(all_texts, all_labels))
    np.random.shuffle(combined_data)
    all_texts, all_labels = zip(*combined_data)
    
    # 创建Dataset对象
    from datasets import Dataset
    dataset = Dataset.from_dict({
        "text": list(all_texts),
        "reasoning_label": list(all_labels)
    })
    
    # 保存数据集
    os.makedirs(output_path, exist_ok=True)
    dataset.save_to_disk(output_path)
    
    # 统计信息
    stats = {
        "total_samples": len(dataset),
        "reasoning_samples": len(selected_reasoning),
        "non_reasoning_samples": len(selected_non_reasoning),
        "reasoning_ratio": len(selected_reasoning) / len(dataset)
    }
    
    stats_path = os.path.join(output_path, "dataset_stats.json")
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    
    print(f">>> 自定义数据集创建完成!")
    print(f">>> 最终样本分布: Reasoning={len(selected_reasoning)}, Non-reasoning={len(selected_non_reasoning)}")
    print(f">>> 数据集保存到: {output_path}")
    
    return output_path

if __name__ == "__main__":
    fire.Fire(create_reasoning_dataset)
