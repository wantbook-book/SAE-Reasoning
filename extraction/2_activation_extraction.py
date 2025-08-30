"""
激活提取模块：从LLM指定层提取activations并使用SAE进行编码
"""
import os
import json
import fire
import torch
import numpy as np
import re
from typing import List, Dict, Tuple, Optional, Union
from datasets import Dataset, load_from_disk
from dataclasses import dataclass
from tqdm import tqdm
import pickle

from transformer_lens import HookedTransformer
from transformer_lens.hook_points import HookPoint
from sae_lens import SAE
import debugpy
debugpy.listen(5678)
debugpy.wait_for_client()

@dataclass
class ActivationExtractionConfig:
    """激活提取配置"""
    model_path: str
    sae_path: str
    hook_point: str  # 例如 "blocks.12.hook_resid_post"
    sae_id: Optional[str] = None
    batch_size: int = 32
    max_length: int = 512
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: str = "bfloat16"
    # 分块参数，避免OOM
    minibatch_size_tokens: int = 64  # token级分块大小
    minibatch_size_features: int = 256  # feature级分块大小（如果需要）


class ActivationExtractor:
    """LLM激活提取器"""
    
    def __init__(self, config: ActivationExtractionConfig):
        self.config = config
        self.model = None
        self.sae = None
        
    def load_model_and_sae(self):
        """加载模型和SAE"""
        print(f">>> 加载模型: {self.config.model_path}")
        self.model = HookedTransformer.from_pretrained_no_processing(
            self.config.model_path,
            dtype=torch.bfloat16 if self.config.dtype == "bfloat16" else torch.float32,
            device=self.config.device,
        )
        
        # 设置pad token
        if self.model.tokenizer.pad_token_id == self.model.tokenizer.eos_token_id:
            self.model.tokenizer.add_special_tokens({"pad_token": "<PAD>"})
        
        print(f">>> 加载SAE: {self.config.sae_path}")
        if self.config.sae_id is None:
            self.sae = SAE.load_from_pretrained(self.config.sae_path, device=self.config.device)
        else:
            self.sae, _, _ = SAE.from_pretrained(self.config.sae_path, self.config.sae_id, device=self.config.device)
        
        print(f">>> SAE维度: {self.sae.cfg.d_sae}, Hook点: {self.config.hook_point}")
    
    def tokenize_texts(self, texts: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """对文本进行tokenization"""
        tokens = self.model.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.max_length
        )
        input_ids = tokens["input_ids"].to(self.config.device)
        attention_mask = tokens["attention_mask"].to(self.config.device)
        return input_ids, attention_mask
    
    def extract_activations(self, tokens: torch.Tensor) -> torch.Tensor:
        """从指定层提取activations，返回完整的序列激活，支持分块处理避免OOM"""
        batch_size, seq_len = tokens.shape
        print(f">>> 处理 {batch_size} 个样本，每个长度 {seq_len}")
        
        # 如果批次大小超过minibatch_size_tokens，进行分块处理
        if batch_size > self.config.minibatch_size_tokens:
            print(f">>> 批次过大，分块处理: {batch_size} -> {self.config.minibatch_size_tokens}")
            return self._extract_activations_chunked(tokens)
        else:
            return self._extract_activations_single(tokens)
    
    def _extract_activations_single(self, tokens: torch.Tensor) -> torch.Tensor:
        """处理单个批次的激活提取"""
        def hook_fn_store_act(activation: torch.Tensor, hook: HookPoint):
            """Hook函数，用于捕获激活"""
            hook.ctx["activation"] = activation.detach()
        
        hooks = [(self.config.hook_point, hook_fn_store_act)]
        
        # 获取hook对应的层数，用于早停优化
        layer_match = re.match(r"blocks\.(\d+)\.", self.config.hook_point)
        if layer_match:
            hook_layer = int(layer_match.group(1))
        else:
            hook_layer = None
        
        with torch.no_grad():
            # 使用run_with_hooks进行前向传播
            if hook_layer is not None:
                self.model.run_with_hooks(
                    tokens, 
                    stop_at_layer=hook_layer + 1, 
                    fwd_hooks=hooks, 
                    return_type=None
                )
            else:
                self.model.run_with_hooks(
                    tokens, 
                    fwd_hooks=hooks, 
                    return_type=None
                )
            
            # 从hook context中获取激活
            activation = self.model.hook_dict[self.config.hook_point].ctx.pop("activation")
        
        return activation  # [batch_size, seq_len, hidden_dim]
    
    def _extract_activations_chunked(self, tokens: torch.Tensor) -> torch.Tensor:
        """分块处理大批次的激活提取，避免OOM"""
        batch_size = tokens.shape[0]
        all_activations = []
        
        # 将tokens分成小批次
        token_minibatches = tokens.split(self.config.minibatch_size_tokens)
        
        print(f">>> 分块处理: {len(token_minibatches)} 个小批次")
        
        for i, minibatch in enumerate(token_minibatches):
            print(f">>> 处理小批次 {i+1}/{len(token_minibatches)}, 大小: {minibatch.shape[0]}")
            
            # 确保minibatch在正确的设备上
            minibatch = minibatch.to(self.config.device)
            
            # 提取当前小批次的激活
            batch_activations = self._extract_activations_single(minibatch)
            
            # 移动到CPU以节省显存
            all_activations.append(batch_activations.cpu())
            
            # 清理显存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # 合并所有激活并移回目标设备
        final_activations = torch.cat(all_activations, dim=0)
        return final_activations.to(self.config.device)
    
    def _compute_masked_average(self, activations: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """计算masked平均激活，排除填充token"""
        # activations: [batch, seq_len, hidden_dim]
        # attention_mask: [batch, seq_len] (1表示真实token，0表示填充token)
        
        # 扩展attention_mask到hidden_dim维度
        mask_expanded = attention_mask.unsqueeze(-1)  # [batch, seq_len, 1]
        
        # 将填充位置的激活置零
        masked_activations = activations * mask_expanded  # [batch, seq_len, hidden_dim]
        
        # 计算每个样本的有效token数量
        valid_lengths = attention_mask.sum(dim=1, keepdim=True).float()  # [batch, 1]
        
        # 计算平均激活，避免除零
        sum_activations = masked_activations.sum(dim=1)  # [batch, hidden_dim]
        averaged_activations = sum_activations / torch.clamp(valid_lengths, min=1.0)  # [batch, hidden_dim]
        
        return averaged_activations
    
    def encode_with_sae(self, activations: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """使用SAE编码激活，然后对每个句子的SAE特征做平均，支持分块处理避免OOM"""
        # activations: [batch, seq_len, hidden_dim]
        # attention_mask: [batch, seq_len]
        
        batch_size, seq_len, hidden_dim = activations.shape
        total_tokens = batch_size * seq_len
        
        # 将激活reshape为 [batch * seq_len, hidden_dim] 以便批量SAE编码
        activations_flat = activations.reshape(-1, hidden_dim)
        
        # 如果token数量过多，进行分块处理
        if total_tokens > self.config.minibatch_size_features * 4:  # 使用4倍的feature batch size作为阈值
            print(f">>> SAE编码分块处理: {total_tokens} tokens")
            sae_features_flat = self._encode_sae_chunked(activations_flat)
        else:
            # 直接编码
            activations_flat = activations_flat.to(self.sae.device)
        with torch.no_grad():
                sae_features_flat = self.sae.encode(activations_flat)
        
        # 将SAE特征reshape回 [batch, seq_len, sae_dim]
        sae_dim = sae_features_flat.shape[-1]
        sae_features = sae_features_flat.reshape(batch_size, seq_len, sae_dim).cpu()
        
        # 对每个句子的SAE特征做平均（排除填充token）
        averaged_sae_features = self._compute_masked_average(sae_features, attention_mask.cpu())
        
        return averaged_sae_features
    
    def _encode_sae_chunked(self, activations_flat: torch.Tensor) -> torch.Tensor:
        """分块进行SAE编码，避免OOM"""
        chunk_size = self.config.minibatch_size_features * 4  # 每次处理的token数量
        all_sae_features = []
        
        num_chunks = (activations_flat.shape[0] + chunk_size - 1) // chunk_size
        print(f">>> SAE编码分块: {num_chunks} 个块")
        
        for i in range(0, activations_flat.shape[0], chunk_size):
            chunk = activations_flat[i:i+chunk_size].to(self.sae.device)
            
            with torch.no_grad():
                chunk_features = self.sae.encode(chunk)
            
            # 移动到CPU节省显存
            all_sae_features.append(chunk_features.cpu())
            
            # 清理显存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        return torch.cat(all_sae_features, dim=0)
    
    def extract_and_encode_batch(self, texts: List[str]) -> torch.Tensor:
        """批量提取并编码文本"""
        # Tokenize
        tokens, attention_mask = self.tokenize_texts(texts)
        
        # 提取激活（完整序列激活）
        activations = self.extract_activations(tokens)
        
        # SAE编码并对每个句子的SAE特征做平均
        sae_features = self.encode_with_sae(activations, attention_mask)
        
        return sae_features
    
    def process_dataset(self, dataset: Dataset, text_column: str = "text") -> Tuple[torch.Tensor, torch.Tensor]:
        """处理整个数据集"""
        all_features = []
        all_labels = []
        
        print(">>> 开始提取激活和SAE特征...")
        
        # 按批处理数据
        batch_size = self.config.batch_size
        for i in tqdm(range(0, len(dataset), batch_size), desc="处理批次"):
            batch_data = dataset[i:i+batch_size]
            texts = batch_data[text_column] if isinstance(batch_data[text_column], list) else [batch_data[text_column]]
            labels = batch_data["reasoning_label"] if isinstance(batch_data["reasoning_label"], list) else [batch_data["reasoning_label"]]
            
            # 提取特征
            features = self.extract_and_encode_batch(texts)
            
            all_features.append(features)
            all_labels.extend(labels)
        
        # 合并所有特征
        all_features = torch.cat(all_features, dim=0)
        all_labels = torch.tensor(all_labels)
        
        print(f">>> 提取完成! 特征形状: {all_features.shape}, 标签形状: {all_labels.shape}")
        
        return all_features, all_labels


def extract_activations_from_dataset(
    dataset_path: str,
    model_path: str,
    sae_path: str,
    hook_point: str,
    output_path: str,
    sae_id: Optional[str] = None,
    text_column: str = "text",
    batch_size: int = 32,
    max_length: int = 512,
    device: str = "auto",
    minibatch_size_tokens: int = 64,
    minibatch_size_features: int = 256
):
    """从数据集中提取激活和SAE特征"""
    
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 配置
    config = ActivationExtractionConfig(
        model_path=model_path,
        sae_path=sae_path,
        hook_point=hook_point,
        sae_id=sae_id,
        batch_size=batch_size,
        max_length=max_length,
        device=device,
        minibatch_size_tokens=minibatch_size_tokens,
        minibatch_size_features=minibatch_size_features
    )
    
    # 初始化提取器
    extractor = ActivationExtractor(config)
    extractor.load_model_and_sae()
    
    # 加载数据集
    print(f">>> 加载数据集: {dataset_path}")
    dataset = load_from_disk(dataset_path)
    
    # 提取特征
    features, labels = extractor.process_dataset(dataset, text_column)
    
    # 保存结果
    os.makedirs(output_path, exist_ok=True)
    
    features_path = os.path.join(output_path, "sae_features.pt")
    labels_path = os.path.join(output_path, "labels.pt")
    config_path = os.path.join(output_path, "extraction_config.json")
    
    torch.save(features, features_path)
    torch.save(labels, labels_path)
    
    # 保存配置信息
    config_dict = {
        "model_path": config.model_path,
        "sae_path": config.sae_path,
        "hook_point": config.hook_point,
        "sae_id": config.sae_id,
        "batch_size": config.batch_size,
        "max_length": config.max_length,
        "device": config.device,
        "feature_dim": features.shape[1],
        "num_samples": features.shape[0],
        "num_reasoning": labels.sum().item(),
        "num_non_reasoning": (labels == 0).sum().item()
    }
    
    with open(config_path, 'w') as f:
        json.dump(config_dict, f, indent=2)
    
    print(f">>> 特征保存到: {features_path}")
    print(f">>> 标签保存到: {labels_path}")
    print(f">>> 配置保存到: {config_path}")
    print(f">>> 特征维度: {features.shape}")
    print(f">>> Reasoning样本: {labels.sum().item()}/{len(labels)}")


def extract_activations_from_texts(
    texts: List[str],
    labels: List[int],
    model_path: str,
    sae_path: str,
    hook_point: str,
    output_path: str,
    sae_id: Optional[str] = None,
    batch_size: int = 32,
    max_length: int = 512,
    device: str = "auto",
    minibatch_size_tokens: int = 64,
    minibatch_size_features: int = 256
):
    """从文本列表中提取激活和SAE特征"""
    
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 配置
    config = ActivationExtractionConfig(
        model_path=model_path,
        sae_path=sae_path,
        hook_point=hook_point,
        sae_id=sae_id,
        batch_size=batch_size,
        max_length=max_length,
        device=device,
        minibatch_size_tokens=minibatch_size_tokens,
        minibatch_size_features=minibatch_size_features
    )
    
    # 初始化提取器
    extractor = ActivationExtractor(config)
    extractor.load_model_and_sae()
    
    # 处理文本
    all_features = []
    print(">>> 开始提取激活和SAE特征...")
    
    for i in tqdm(range(0, len(texts), batch_size), desc="处理批次"):
        batch_texts = texts[i:i+batch_size]
        features = extractor.extract_and_encode_batch(batch_texts)
        all_features.append(features)
    
    # 合并特征
    all_features = torch.cat(all_features, dim=0)
    all_labels = torch.tensor(labels)
    
    # 保存结果
    os.makedirs(output_path, exist_ok=True)
    
    features_path = os.path.join(output_path, "sae_features.pt")
    labels_path = os.path.join(output_path, "labels.pt")
    
    torch.save(all_features, features_path)
    torch.save(all_labels, labels_path)
    
    print(f">>> 特征保存到: {features_path}")
    print(f">>> 标签保存到: {labels_path}")
    print(f">>> 特征维度: {all_features.shape}")


if __name__ == "__main__":
    fire.Fire(extract_activations_from_dataset)