"""
互信息计算模块：计算特征与数据标签的互信息并进行可视化
"""
import os
import json
import fire
import torch
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mutual_info_score
from sklearn.feature_selection import mutual_info_classif
import pandas as pd


@dataclass
class MutualInformationConfig:
    """互信息计算配置"""
    method: str = "sklearn"  # "sklearn", "manual", "kl_divergence"
    bins: int = 10  # 用于离散化连续特征的bins数量
    random_state: int = 42
    n_neighbors: int = 3  # KNN方法的邻居数
    discrete_features: bool = True  # 特征是否已经是离散的


class MutualInformationCalculator:
    """互信息计算器"""
    
    def __init__(self, config: MutualInformationConfig):
        self.config = config
        self.mi_scores = None
        self.feature_stats = None
        
    def calculate_mi_sklearn(self, features: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """使用sklearn计算互信息"""
        print(">>> 使用sklearn方法计算互信息...")
        
        if self.config.discrete_features:
            # 对于离散特征（二值化后的特征）
            mi_scores = mutual_info_classif(
                features, labels,
                discrete_features=True,
                random_state=self.config.random_state
            )
        else:
            # 对于连续特征
            mi_scores = mutual_info_classif(
                features, labels,
                discrete_features=False,
                n_neighbors=self.config.n_neighbors,
                random_state=self.config.random_state
            )
        
        return mi_scores
    
    def calculate_mi_manual(self, features: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """手动计算互信息（适用于二值特征）"""
        print(">>> 使用手动方法计算互信息...")
        
        n_features = features.shape[1]
        mi_scores = np.zeros(n_features)
        
        for i in tqdm(range(n_features), desc="计算特征互信息"):
            feature = features[:, i]
            mi_scores[i] = self._mutual_info_binary(feature, labels)
        
        return mi_scores
    
    def _mutual_info_binary(self, feature: np.ndarray, labels: np.ndarray) -> float:
        """计算二值特征与标签的互信息"""
        # 计算联合分布和边缘分布
        unique_features = np.unique(feature)
        unique_labels = np.unique(labels)
        
        if len(unique_features) == 1:  # 特征没有变化
            return 0.0
        
        # 计算概率
        n = len(feature)
        mi = 0.0
        
        for f_val in unique_features:
            for l_val in unique_labels:
                # P(X=f, Y=l)
                p_joint = np.sum((feature == f_val) & (labels == l_val)) / n
                
                if p_joint > 0:
                    # P(X=f)
                    p_feature = np.sum(feature == f_val) / n
                    # P(Y=l)
                    p_label = np.sum(labels == l_val) / n
                    
                    # MI += P(X,Y) * log(P(X,Y) / (P(X) * P(Y)))
                    mi += p_joint * np.log2(p_joint / (p_feature * p_label))
        
        return mi
    
    def calculate_mi_kl_divergence(self, features: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """使用KL散度方法计算互信息"""
        print(">>> 使用KL散度方法计算互信息...")
        
        n_features = features.shape[1]
        mi_scores = np.zeros(n_features)
        
        for i in tqdm(range(n_features), desc="计算特征互信息"):
            feature = features[:, i]
            mi_scores[i] = self._kl_divergence_mi(feature, labels)
        
        return mi_scores
    
    def _kl_divergence_mi(self, feature: np.ndarray, labels: np.ndarray) -> float:
        """使用KL散度计算互信息"""
        # 对于每个标签类别，计算特征的条件分布
        unique_labels = np.unique(labels)
        
        if len(unique_labels) == 1:
            return 0.0
        
        # 计算整体特征分布 P(X)
        if self.config.discrete_features:
            unique_features = np.unique(feature)
            p_x = np.array([np.mean(feature == f) for f in unique_features])
        else:
            # 对连续特征进行分箱
            feature_binned = np.digitize(feature, np.linspace(feature.min(), feature.max(), self.config.bins))
            unique_features = np.unique(feature_binned)
            p_x = np.array([np.mean(feature_binned == f) for f in unique_features])
            feature = feature_binned
        
        # 计算条件分布 P(X|Y)
        mi = 0.0
        for label in unique_labels:
            p_y = np.mean(labels == label)
            if p_y == 0:
                continue
            
            mask = labels == label
            feature_given_label = feature[mask]
            
            # P(X|Y=label)
            p_x_given_y = np.array([np.mean(feature_given_label == f) for f in unique_features])
            
            # KL(P(X|Y=label) || P(X))
            for i, (p_cond, p_marg) in enumerate(zip(p_x_given_y, p_x)):
                if p_cond > 0 and p_marg > 0:
                    mi += p_y * p_cond * np.log2(p_cond / p_marg)
        
        return max(0, mi)  # 互信息不能为负
    
    def compute_mutual_information(self, features: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """计算互信息"""
        print(f">>> 计算 {features.shape[1]} 个特征的互信息...")
        
        if self.config.method == "sklearn":
            mi_scores = self.calculate_mi_sklearn(features, labels)
        elif self.config.method == "manual":
            mi_scores = self.calculate_mi_manual(features, labels)
        elif self.config.method == "kl_divergence":
            mi_scores = self.calculate_mi_kl_divergence(features, labels)
        else:
            raise ValueError(f"未知的计算方法: {self.config.method}")
        
        self.mi_scores = mi_scores
        return mi_scores
    
    def analyze_mi_distribution(self, mi_scores: np.ndarray) -> Dict:
        """分析互信息分布"""
        analysis = {
            "mean": float(np.mean(mi_scores)),
            "std": float(np.std(mi_scores)),
            "min": float(np.min(mi_scores)),
            "max": float(np.max(mi_scores)),
            "median": float(np.median(mi_scores)),
            "q25": float(np.percentile(mi_scores, 25)),
            "q75": float(np.percentile(mi_scores, 75)),
            "q90": float(np.percentile(mi_scores, 90)),
            "q95": float(np.percentile(mi_scores, 95)),
            "q99": float(np.percentile(mi_scores, 99)),
            "non_zero_count": int(np.sum(mi_scores > 0)),
            "high_mi_count_90": int(np.sum(mi_scores > np.percentile(mi_scores, 90))),
            "high_mi_count_95": int(np.sum(mi_scores > np.percentile(mi_scores, 95))),
            "high_mi_count_99": int(np.sum(mi_scores > np.percentile(mi_scores, 99)))
        }
        
        return analysis
    
    def plot_mi_distribution(self, mi_scores: np.ndarray, output_dir: str, labels: np.ndarray = None):
        """绘制互信息分布图"""
        os.makedirs(output_dir, exist_ok=True)
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. 互信息直方图
        axes[0, 0].hist(mi_scores, bins=50, alpha=0.7, edgecolor='black')
        axes[0, 0].set_xlabel("互信息值")
        axes[0, 0].set_ylabel("特征数量")
        axes[0, 0].set_title("互信息分布直方图")
        axes[0, 0].axvline(np.mean(mi_scores), color='red', linestyle='--', label=f'均值: {np.mean(mi_scores):.4f}')
        axes[0, 0].axvline(np.median(mi_scores), color='green', linestyle='--', label=f'中位数: {np.median(mi_scores):.4f}')
        axes[0, 0].legend()
        
        # 2. 对数尺度直方图
        axes[0, 1].hist(mi_scores[mi_scores > 0], bins=50, alpha=0.7, edgecolor='black')
        axes[0, 1].set_xlabel("互信息值")
        axes[0, 1].set_ylabel("特征数量")
        axes[0, 1].set_title("非零互信息分布（对数尺度）")
        axes[0, 1].set_yscale('log')
        
        # 3. 累积分布函数
        sorted_mi = np.sort(mi_scores)
        axes[0, 2].plot(sorted_mi, np.arange(1, len(sorted_mi) + 1) / len(sorted_mi))
        axes[0, 2].set_xlabel("互信息值")
        axes[0, 2].set_ylabel("累积概率")
        axes[0, 2].set_title("互信息累积分布函数")
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Top-K特征的互信息值
        top_k = min(100, len(mi_scores))
        top_indices = np.argsort(mi_scores)[-top_k:]
        axes[1, 0].bar(range(top_k), mi_scores[top_indices])
        axes[1, 0].set_xlabel("特征排名（倒序）")
        axes[1, 0].set_ylabel("互信息值")
        axes[1, 0].set_title(f"Top-{top_k} 特征的互信息值")
        
        # 5. 互信息 vs 特征索引（散点图）
        sample_indices = np.random.choice(len(mi_scores), min(1000, len(mi_scores)), replace=False)
        axes[1, 1].scatter(sample_indices, mi_scores[sample_indices], alpha=0.5, s=1)
        axes[1, 1].set_xlabel("特征索引")
        axes[1, 1].set_ylabel("互信息值")
        axes[1, 1].set_title("互信息 vs 特征索引")
        
        # 6. 箱线图
        percentiles = [50, 75, 90, 95, 99]
        thresholds = [np.percentile(mi_scores, p) for p in percentiles]
        high_mi_counts = [np.sum(mi_scores > t) for t in thresholds]
        
        axes[1, 2].bar(range(len(percentiles)), high_mi_counts)
        axes[1, 2].set_xticks(range(len(percentiles)))
        axes[1, 2].set_xticklabels([f'>{p}%' for p in percentiles])
        axes[1, 2].set_ylabel("高互信息特征数量")
        axes[1, 2].set_title("不同阈值下的高互信息特征数量")
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "mutual_information_analysis.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 绘制详细的Top特征分析
        self.plot_top_features_analysis(mi_scores, output_dir)
        
        print(f">>> 互信息分析图保存到: {os.path.join(output_dir, 'mutual_information_analysis.png')}")
    
    def plot_top_features_analysis(self, mi_scores: np.ndarray, output_dir: str, top_k: int = 50):
        """绘制Top特征的详细分析"""
        # 获取Top-K特征
        top_indices = np.argsort(mi_scores)[-top_k:][::-1]  # 降序排列
        top_scores = mi_scores[top_indices]
        
        # 创建Top特征图
        plt.figure(figsize=(15, 8))
        
        plt.subplot(2, 1, 1)
        bars = plt.bar(range(top_k), top_scores)
        plt.xlabel("特征排名")
        plt.ylabel("互信息值")
        plt.title(f"Top-{top_k} 特征的互信息值")
        plt.xticks(range(0, top_k, max(1, top_k//10)))
        
        # 为最高的几个特征添加数值标签
        for i in range(min(10, top_k)):
            plt.text(i, top_scores[i] + max(top_scores) * 0.01, 
                    f'{top_scores[i]:.4f}', ha='center', va='bottom', fontsize=8)
        
        plt.subplot(2, 1, 2)
        # 互信息值的变化率
        if top_k > 1:
            score_diffs = np.diff(top_scores)
            plt.plot(range(1, top_k), -score_diffs, 'o-')  # 负号因为是降序
            plt.xlabel("特征排名")
            plt.ylabel("互信息差值")
            plt.title("相邻排名特征间的互信息差值")
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"top_{top_k}_features_analysis.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 保存Top特征信息
        top_features_info = {
            "top_features": {
                int(idx): {
                    "feature_index": int(idx),
                    "mutual_information": float(score),
                    "rank": int(rank + 1)
                }
                for rank, (idx, score) in enumerate(zip(top_indices, top_scores))
            },
            "summary": {
                "top_k": top_k,
                "highest_mi": float(top_scores[0]),
                "lowest_top_mi": float(top_scores[-1]),
                "mean_top_mi": float(np.mean(top_scores)),
                "std_top_mi": float(np.std(top_scores))
            }
        }
        
        with open(os.path.join(output_dir, f"top_{top_k}_features.json"), 'w') as f:
            json.dump(top_features_info, f, indent=2)


def compute_mutual_information(
    features_path: str,
    labels_path: str,
    output_path: str,
    method: str = "sklearn",
    discrete_features: bool = True,
    bins: int = 10,
    n_neighbors: int = 3,
    top_k: int = 100
):
    """计算特征与标签的互信息"""
    
    # 配置
    config = MutualInformationConfig(
        method=method,
        bins=bins,
        discrete_features=discrete_features,
        n_neighbors=n_neighbors
    )
    
    # 初始化计算器
    calculator = MutualInformationCalculator(config)
    
    # 加载数据
    print(f">>> 加载特征: {features_path}")
    features = torch.load(features_path, map_location="cpu")
    
    print(f">>> 加载标签: {labels_path}")
    labels = torch.load(labels_path, map_location="cpu")
    
    print(f">>> 数据形状 - 特征: {features.shape}, 标签: {labels.shape}")
    
    # 转换为numpy
    features_np = features.numpy()
    labels_np = labels.numpy()
    
    # 检查数据
    print(f">>> 数据统计:")
    print(f"    特征范围: {features_np.min():.4f} - {features_np.max():.4f}")
    print(f"    标签分布: {np.bincount(labels_np)}")
    
    # 计算互信息
    mi_scores = calculator.compute_mutual_information(features_np, labels_np)
    
    # 分析结果
    analysis = calculator.analyze_mi_distribution(mi_scores)
    print(">>> 互信息分析结果:")
    for key, value in analysis.items():
        print(f"    {key}: {value}")
    
    # 保存结果
    os.makedirs(output_path, exist_ok=True)
    
    mi_scores_path = os.path.join(output_path, "mutual_information_scores.pt")
    config_path = os.path.join(output_path, "mi_config.json")
    analysis_path = os.path.join(output_path, "mi_analysis.json")
    
    torch.save(torch.tensor(mi_scores), mi_scores_path)
    
    # 保存配置
    config_dict = {
        "method": config.method,
        "bins": config.bins,
        "discrete_features": config.discrete_features,
        "n_neighbors": config.n_neighbors,
        "features_shape": list(features.shape),
        "labels_shape": list(labels.shape)
    }
    
    with open(config_path, 'w') as f:
        json.dump(config_dict, f, indent=2)
    
    with open(analysis_path, 'w') as f:
        json.dump(analysis, f, indent=2)
    
    # 绘制分析图
    calculator.plot_mi_distribution(mi_scores, output_path, labels_np)
    
    # 获取Top-K特征
    top_indices = np.argsort(mi_scores)[-top_k:][::-1]
    top_mi_features = {
        "indices": top_indices.tolist(),
        "scores": mi_scores[top_indices].tolist(),
        "threshold_95": float(np.percentile(mi_scores, 95)),
        "threshold_99": float(np.percentile(mi_scores, 99))
    }
    
    top_features_path = os.path.join(output_path, f"top_{top_k}_reasoning_features.json")
    with open(top_features_path, 'w') as f:
        json.dump(top_mi_features, f, indent=2)
    
    print(f">>> 互信息得分保存到: {mi_scores_path}")
    print(f">>> 分析结果保存到: {analysis_path}")
    print(f">>> Top-{top_k} reasoning特征保存到: {top_features_path}")
    print(f">>> 推荐的reasoning特征数量: {len(top_indices)} (Top-{top_k})")


def compare_mi_methods(
    features_path: str,
    labels_path: str,
    output_path: str,
    methods: List[str] = None
):
    """比较不同互信息计算方法"""
    
    if methods is None:
        methods = ["sklearn", "manual"]
    
    # 加载数据
    features = torch.load(features_path, map_location="cpu").numpy()
    labels = torch.load(labels_path, map_location="cpu").numpy()
    
    results = {}
    
    for method in methods:
        print(f">>> 使用方法: {method}")
        
        config = MutualInformationConfig(method=method, discrete_features=True)
        calculator = MutualInformationCalculator(config)
        
        mi_scores = calculator.compute_mutual_information(features, labels)
        analysis = calculator.analyze_mi_distribution(mi_scores)
        
        results[method] = {
            "scores": mi_scores.tolist(),
            "analysis": analysis
        }
    
    # 保存比较结果
    os.makedirs(output_path, exist_ok=True)
    comparison_path = os.path.join(output_path, "mi_methods_comparison.json")
    
    with open(comparison_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # 绘制比较图
    plt.figure(figsize=(15, 10))
    
    for i, method in enumerate(methods):
        plt.subplot(2, len(methods), i+1)
        plt.hist(results[method]["scores"], bins=50, alpha=0.7, label=method)
        plt.xlabel("互信息值")
        plt.ylabel("特征数量")
        plt.title(f"{method} 方法")
        plt.legend()
        
        plt.subplot(2, len(methods), i+1+len(methods))
        scores = np.array(results[method]["scores"])
        top_100 = np.sort(scores)[-100:]
        plt.plot(range(100), top_100)
        plt.xlabel("Top特征排名")
        plt.ylabel("互信息值")
        plt.title(f"{method} Top-100特征")
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "mi_methods_comparison.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f">>> 方法比较结果保存到: {comparison_path}")


if __name__ == "__main__":
    fire.Fire(compute_mutual_information)
