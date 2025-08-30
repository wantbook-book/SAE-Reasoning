"""
特征二值化模块：对SAE特征激活值进行阈值二值化
"""
import os
import json
import fire
import torch
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns


@dataclass
class BinarizationConfig:
    """二值化配置"""
    threshold_method: str = "percentile"  # "percentile", "mean", "median", "fixed", "adaptive"
    threshold_value: float = 0.95  # 对于percentile是百分位数，对于fixed是固定阈值
    adaptive_k: float = 2.0  # 对于adaptive方法，threshold = mean + k * std
    min_activation_count: int = 10  # 最少激活次数，低于此数的特征将被忽略


class FeatureBinarizer:
    """特征二值化器"""
    
    def __init__(self, config: BinarizationConfig):
        self.config = config
        self.thresholds = None
        self.feature_stats = None
        
    def compute_feature_statistics(self, features: torch.Tensor) -> Dict:
        """计算特征统计信息"""
        print(">>> 计算特征统计信息...")
        
        stats = {
            "mean": features.mean(dim=0),
            "std": features.std(dim=0),
            "median": features.median(dim=0).values,
            "min": features.min(dim=0).values,
            "max": features.max(dim=0).values,
            "percentiles": {}
        }
        
        # 计算不同百分位数
        percentiles = [50, 75, 90, 95, 99]
        for p in percentiles:
            stats["percentiles"][p] = torch.quantile(features, p/100, dim=0)
        
        # 计算非零激活的比例
        stats["activation_rate"] = (features > 0).float().mean(dim=0)
        stats["activation_count"] = (features > 0).sum(dim=0)
        
        self.feature_stats = stats
        return stats
    
    def compute_thresholds(self, features: torch.Tensor) -> torch.Tensor:
        """计算二值化阈值"""
        print(f">>> 使用方法 '{self.config.threshold_method}' 计算阈值...")
        
        if self.feature_stats is None:
            self.compute_feature_statistics(features)
        
        n_features = features.shape[1]
        thresholds = torch.zeros(n_features)
        
        if self.config.threshold_method == "percentile":
            thresholds = self.feature_stats["percentiles"][int(self.config.threshold_value * 100)]
            
        elif self.config.threshold_method == "mean":
            thresholds = self.feature_stats["mean"]
            
        elif self.config.threshold_method == "median":
            thresholds = self.feature_stats["median"]
            
        elif self.config.threshold_method == "fixed":
            thresholds = torch.full((n_features,), self.config.threshold_value)
            
        elif self.config.threshold_method == "adaptive":
            thresholds = (self.feature_stats["mean"] + 
                         self.config.adaptive_k * self.feature_stats["std"])
        
        else:
            raise ValueError(f"未知的阈值方法: {self.config.threshold_method}")
        
        # 过滤激活次数过少的特征
        low_activation_mask = self.feature_stats["activation_count"] < self.config.min_activation_count
        thresholds[low_activation_mask] = float('inf')  # 设置为无穷大，这样永远不会被激活
        
        print(f">>> 阈值范围: {thresholds[thresholds != float('inf')].min().item():.4f} - {thresholds[thresholds != float('inf')].max().item():.4f}")
        print(f">>> 过滤了 {low_activation_mask.sum().item()} 个低激活特征")
        
        self.thresholds = thresholds
        return thresholds
    
    def binarize_features(self, features: torch.Tensor, thresholds: torch.Tensor = None) -> torch.Tensor:
        """对特征进行二值化"""
        if thresholds is None:
            if self.thresholds is None:
                thresholds = self.compute_thresholds(features)
            else:
                thresholds = self.thresholds
        
        # 二值化
        binary_features = (features > thresholds.unsqueeze(0)).float()
        
        return binary_features
    
    def analyze_binarization_quality(self, features: torch.Tensor, binary_features: torch.Tensor) -> Dict:
        """分析二值化质量"""
        print(">>> 分析二值化质量...")
        
        # 计算每个特征的激活率
        activation_rates = binary_features.mean(dim=0)
        
        # 计算信息保留率（简单的相关性度量）
        correlations = []
        for i in range(features.shape[1]):
            if activation_rates[i] > 0 and activation_rates[i] < 1:  # 避免全0或全1的特征
                corr = torch.corrcoef(torch.stack([features[:, i], binary_features[:, i]]))[0, 1]
                if not torch.isnan(corr):
                    correlations.append(corr.item())
        
        analysis = {
            "mean_activation_rate": activation_rates.mean().item(),
            "std_activation_rate": activation_rates.std().item(),
            "min_activation_rate": activation_rates.min().item(),
            "max_activation_rate": activation_rates.max().item(),
            "mean_correlation": np.mean(correlations) if correlations else 0.0,
            "num_active_features": (activation_rates > 0).sum().item(),
            "num_always_active": (activation_rates == 1).sum().item(),
            "num_never_active": (activation_rates == 0).sum().item()
        }
        
        return analysis
    
    def plot_threshold_analysis(self, features: torch.Tensor, output_dir: str):
        """绘制阈值分析图"""
        if self.feature_stats is None:
            self.compute_feature_statistics(features)
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. 激活率分布
        plt.figure(figsize=(12, 8))
        plt.subplot(2, 3, 1)
        plt.hist(self.feature_stats["activation_rate"].numpy(), bins=50, alpha=0.7)
        plt.xlabel("激活率")
        plt.ylabel("特征数量")
        plt.title("特征激活率分布")
        
        # 2. 激活值分布（log scale）
        plt.subplot(2, 3, 2)
        non_zero_features = features[features > 0]
        plt.hist(torch.log10(non_zero_features + 1e-8).numpy(), bins=50, alpha=0.7)
        plt.xlabel("log10(激活值)")
        plt.ylabel("频次")
        plt.title("非零激活值分布")
        
        # 3. 阈值分布
        if self.thresholds is not None:
            plt.subplot(2, 3, 3)
            valid_thresholds = self.thresholds[self.thresholds != float('inf')]
            plt.hist(valid_thresholds.numpy(), bins=50, alpha=0.7)
            plt.xlabel("阈值")
            plt.ylabel("特征数量")
            plt.title("阈值分布")
        
        # 4. 特征均值 vs 标准差
        plt.subplot(2, 3, 4)
        plt.scatter(self.feature_stats["mean"].numpy(), self.feature_stats["std"].numpy(), alpha=0.5)
        plt.xlabel("均值")
        plt.ylabel("标准差")
        plt.title("特征均值 vs 标准差")
        
        # 5. 激活次数分布
        plt.subplot(2, 3, 5)
        plt.hist(self.feature_stats["activation_count"].numpy(), bins=50, alpha=0.7)
        plt.xlabel("激活次数")
        plt.ylabel("特征数量")
        plt.title("特征激活次数分布")
        plt.yscale('log')
        
        # 6. 百分位数比较
        plt.subplot(2, 3, 6)
        percentiles = [50, 75, 90, 95, 99]
        percentile_means = [self.feature_stats["percentiles"][p].mean().item() for p in percentiles]
        plt.plot(percentiles, percentile_means, 'o-')
        plt.xlabel("百分位数")
        plt.ylabel("平均阈值")
        plt.title("不同百分位数的平均阈值")
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "threshold_analysis.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f">>> 阈值分析图保存到: {os.path.join(output_dir, 'threshold_analysis.png')}")


def binarize_sae_features(
    features_path: str,
    output_path: str,
    threshold_method: str = "percentile",
    threshold_value: float = 0.95,
    adaptive_k: float = 2.0,
    min_activation_count: int = 10,
    save_analysis: bool = True
):
    """对SAE特征进行二值化"""
    
    # 配置
    config = BinarizationConfig(
        threshold_method=threshold_method,
        threshold_value=threshold_value,
        adaptive_k=adaptive_k,
        min_activation_count=min_activation_count
    )
    
    # 初始化二值化器
    binarizer = FeatureBinarizer(config)
    
    # 加载特征
    print(f">>> 加载特征: {features_path}")
    features = torch.load(features_path, map_location="cpu")
    print(f">>> 特征形状: {features.shape}")
    
    # 计算阈值
    thresholds = binarizer.compute_thresholds(features)
    
    # 二值化
    print(">>> 进行二值化...")
    binary_features = binarizer.binarize_features(features, thresholds)
    
    # 分析质量
    analysis = binarizer.analyze_binarization_quality(features, binary_features)
    print(">>> 二值化质量分析:")
    for key, value in analysis.items():
        print(f"    {key}: {value:.4f}")
    
    # 保存结果
    os.makedirs(output_path, exist_ok=True)
    
    binary_features_path = os.path.join(output_path, "binary_features.pt")
    thresholds_path = os.path.join(output_path, "thresholds.pt")
    config_path = os.path.join(output_path, "binarization_config.json")
    analysis_path = os.path.join(output_path, "binarization_analysis.json")
    
    torch.save(binary_features, binary_features_path)
    torch.save(thresholds, thresholds_path)
    
    # 保存配置
    config_dict = {
        "threshold_method": config.threshold_method,
        "threshold_value": config.threshold_value,
        "adaptive_k": config.adaptive_k,
        "min_activation_count": config.min_activation_count,
        "input_shape": list(features.shape),
        "output_shape": list(binary_features.shape)
    }
    
    with open(config_path, 'w') as f:
        json.dump(config_dict, f, indent=2)
    
    with open(analysis_path, 'w') as f:
        json.dump(analysis, f, indent=2)
    
    # 保存分析图
    if save_analysis:
        binarizer.plot_threshold_analysis(features, output_path)
    
    print(f">>> 二值化特征保存到: {binary_features_path}")
    print(f">>> 阈值保存到: {thresholds_path}")
    print(f">>> 配置保存到: {config_path}")
    print(f">>> 分析保存到: {analysis_path}")


def compare_threshold_methods(
    features_path: str,
    output_path: str,
    methods: List[str] = None,
    threshold_values: List[float] = None
):
    """比较不同阈值方法的效果"""
    
    if methods is None:
        methods = ["percentile", "mean", "median", "adaptive"]
    
    if threshold_values is None:
        threshold_values = [0.9, 0.95, 0.99] if "percentile" in methods else [0.95]
    
    # 加载特征
    features = torch.load(features_path, map_location="cpu")
    
    results = {}
    
    for method in methods:
        for value in threshold_values:
            if method == "percentile":
                key = f"{method}_{int(value*100)}"
            else:
                key = f"{method}_{value}"
            
            config = BinarizationConfig(
                threshold_method=method,
                threshold_value=value,
                min_activation_count=10
            )
            
            binarizer = FeatureBinarizer(config)
            thresholds = binarizer.compute_thresholds(features)
            binary_features = binarizer.binarize_features(features, thresholds)
            analysis = binarizer.analyze_binarization_quality(features, binary_features)
            
            results[key] = analysis
    
    # 保存比较结果
    os.makedirs(output_path, exist_ok=True)
    comparison_path = os.path.join(output_path, "threshold_comparison.json")
    
    with open(comparison_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # 绘制比较图
    plt.figure(figsize=(15, 10))
    
    metrics = ["mean_activation_rate", "mean_correlation", "num_active_features"]
    
    for i, metric in enumerate(metrics):
        plt.subplot(2, 2, i+1)
        
        methods_list = list(results.keys())
        values = [results[method][metric] for method in methods_list]
        
        plt.bar(range(len(methods_list)), values)
        plt.xticks(range(len(methods_list)), methods_list, rotation=45)
        plt.ylabel(metric)
        plt.title(f"{metric} 比较")
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "threshold_methods_comparison.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f">>> 阈值方法比较结果保存到: {comparison_path}")
    print(">>> 最佳方法推荐:")
    
    # 简单的评分系统来推荐最佳方法
    scores = {}
    for method, analysis in results.items():
        score = (
            analysis["mean_correlation"] * 0.4 +  # 相关性权重40%
            (1 - abs(analysis["mean_activation_rate"] - 0.1)) * 0.3 +  # 激活率接近10%权重30%
            (analysis["num_active_features"] / features.shape[1]) * 0.3  # 活跃特征比例权重30%
        )
        scores[method] = score
    
    best_method = max(scores, key=scores.get)
    print(f"    推荐方法: {best_method} (评分: {scores[best_method]:.4f})")


if __name__ == "__main__":
    fire.Fire(binarize_sae_features)
