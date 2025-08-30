# SAE Reasoning特征提取器

这个模块实现了从LLM中提取reasoning相关特征的完整流程，使用SAE（Sparse Autoencoder）对模型激活进行分析。

## 📋 功能概述

该系统按照以下步骤工作：

1. **数据标注**: 区分reasoning和non-reasoning数据，为数据添加二分类标签
2. **激活提取**: 从LLM指定层提取activations，对每个token用SAE编码，然后对每个句子的SAE特征做平均
3. **特征二值化**: 对SAE特征激活值进行阈值二值化处理
4. **互信息计算**: 计算每个特征与reasoning标签的互信息
5. **特征选择**: 筛选出与reasoning高度相关的特征

## 🚀 快速开始

### 基本使用

```python
from reasoning_feature_extractor import extract_reasoning_features

# 运行完整流程 (自动从OpenThoughts和LMSYS数据集获取数据)
reasoning_features_path = extract_reasoning_features(
    model_path="microsoft/DialoGPT-medium",
    sae_path="your/sae/path", 
    hook_point="blocks.12.hook_resid_post",
    output_dir="./results",
    max_samples=5000,
    reasoning_ratio=0.5,  # 50% reasoning数据，50% non-reasoning数据
    top_k=100
)
```

### 分步执行

```python
from reasoning_feature_extractor import ReasoningFeatureExtractor, ReasoningFeatureExtractionConfig

# 创建配置
config = ReasoningFeatureExtractionConfig(
    dataset_path="your/dataset/path",
    model_path="microsoft/DialoGPT-medium", 
    sae_path="your/sae/path",
    hook_point="blocks.12.hook_resid_post",
    output_dir="./results"
)

# 初始化并分步执行
extractor = ReasoningFeatureExtractor(config)
labeled_data = extractor.step1_label_data()
features, labels = extractor.step2_extract_activations(labeled_data)
binary_features = extractor.step3_binarize_features(features)
mi_scores = extractor.step4_compute_mutual_information(binary_features, labels)
reasoning_features = extractor.step5_select_reasoning_features(mi_scores)
```

## 📁 模块说明

### 1. data_labeling.py
负责数据标注，区分reasoning和non-reasoning文本。

**主要功能**:
- 基于关键词检测reasoning文本
- 支持自定义reasoning指示词
- 数据集平衡和采样

**关键参数**:
- `reasoning_indicators`: reasoning指示词列表
- `min_reasoning_length`: reasoning文本最小长度
- `reasoning_ratio`: reasoning数据比例

### 2. activation_extraction.py
从LLM中提取指定层的激活并使用SAE编码。

**主要功能**:
- LLM前向传播和激活提取
- SAE特征编码
- 批处理和内存管理

**关键参数**:
- `hook_point`: LLM层的hook点（如"blocks.12.hook_resid_post"）
- `batch_size`: 批处理大小
- `max_length`: 序列最大长度

### 3. feature_binarization.py
对连续的SAE特征进行二值化处理。

**主要功能**:
- 多种阈值计算方法（百分位数、均值、自适应等）
- 特征质量分析
- 可视化分析图表

**关键参数**:
- `threshold_method`: 阈值方法（"percentile", "mean", "adaptive"等）
- `threshold_value`: 阈值参数
- `min_activation_count`: 最小激活次数

### 4. mutual_information.py
计算特征与reasoning标签的互信息。

**主要功能**:
- 多种互信息计算方法
- 分布分析和可视化
- Top特征识别

**关键参数**:
- `method`: 计算方法（"sklearn", "manual", "kl_divergence"）
- `discrete_features`: 特征是否为离散型
- `bins`: 连续特征分箱数

### 5. reasoning_feature_extractor.py
主要的流程管理器，整合所有功能。

**主要功能**:
- 完整流程管理
- 结果保存和报告生成
- 可视化和分析

## ⚙️ 配置参数

### 数据配置
- `dataset_path`: 输入数据集路径
- `text_column`: 文本列名（默认"text"）
- `max_samples`: 最大样本数
- `reasoning_ratio`: reasoning数据比例

### 模型配置  
- `model_path`: LLM模型路径
- `sae_path`: SAE模型路径
- `hook_point`: 提取激活的层
- `sae_id`: SAE模型ID（可选）

### 处理配置
- `batch_size`: 批处理大小
- `max_length`: 序列最大长度
- `device`: 计算设备（"cuda"/"cpu"/"auto"）

### 二值化配置
- `threshold_method`: 阈值方法
- `threshold_value`: 阈值参数
- `min_activation_count`: 最小激活次数

### 特征选择配置
- `selection_method`: 选择方法（"top_k", "threshold", "percentile"）
- `top_k`: Top-K特征数量
- `mi_threshold`: 互信息阈值
- `percentile_threshold`: 百分位数阈值

## 📊 输出文件

运行完成后，输出目录将包含：

```
output_dir/
├── extraction_config.json          # 提取配置
├── labeled_data/                   # 标注数据
├── labeling_stats.json            # 标注统计
├── sae_features.pt                # SAE特征
├── labels.pt                      # 数据标签  
├── binary_features.pt             # 二值化特征
├── thresholds.pt                  # 二值化阈值
├── mutual_information_scores.pt   # 互信息得分
├── reasoning_features.json        # 最终reasoning特征
├── final_report.json             # 完整报告
└── *.png                         # 分析图表
```

## 🎯 使用示例

### 示例1: 使用OpenThoughts数据集

```python
reasoning_features = extract_reasoning_features(
    dataset_path="open-thoughts/OpenThoughts-114k",
    model_path="microsoft/DialoGPT-medium",
    sae_path="your_sae_path",
    hook_point="blocks.12.hook_resid_post", 
    output_dir="./openthoughts_results",
    max_samples=10000,
    reasoning_ratio=0.5,
    selection_method="top_k",
    top_k=100
)
```

### 示例2: 自定义reasoning指示词

```python
import json

# 定义自定义指示词
custom_indicators = [
    "analyze", "reasoning", "logic", "step by step",
    "问题", "分析", "推理", "逻辑"  # 支持中文
]

# 保存指示词文件
with open("custom_indicators.json", "w") as f:
    json.dump(custom_indicators, f)

# 使用自定义指示词
config = ReasoningFeatureExtractionConfig(
    dataset_path="your_dataset",
    reasoning_indicators_path="custom_indicators.json",
    # ... 其他配置
)
```

### 示例3: 不同的特征选择策略

```python
# Top-K选择
extract_reasoning_features(
    # ... 基础配置
    selection_method="top_k",
    top_k=50
)

# 阈值选择  
extract_reasoning_features(
    # ... 基础配置
    selection_method="threshold", 
    mi_threshold=0.01
)

# 百分位数选择
extract_reasoning_features(
    # ... 基础配置
    selection_method="percentile",
    percentile_threshold=95
)
```

## 🔧 高级用法

### 批量实验

```python
# 不同阈值方法的比较
from feature_binarization import compare_threshold_methods

compare_threshold_methods(
    features_path="./results/sae_features.pt",
    output_path="./threshold_comparison",
    methods=["percentile", "mean", "adaptive"],
    threshold_values=[0.9, 0.95, 0.99]
)

# 不同互信息方法的比较  
from mutual_information import compare_mi_methods

compare_mi_methods(
    features_path="./results/binary_features.pt",
    labels_path="./results/labels.pt", 
    output_path="./mi_comparison",
    methods=["sklearn", "manual"]
)
```

### 自定义数据处理

```python
from data_labeling import label_custom_data

# 处理自定义文本
texts = ["This is reasoning text...", "This is not reasoning..."]
labeled_texts, labels = label_custom_data(
    texts, 
    reasoning_indicators_path="custom_indicators.json"
)
```

## 📈 结果分析

系统会自动生成多种分析图表：

1. **数据分布图**: reasoning vs non-reasoning数据分布
2. **激活分析图**: SAE特征激活模式
3. **阈值分析图**: 二值化阈值效果
4. **互信息分布图**: 特征互信息分布和Top特征
5. **特征选择图**: 最终选择的reasoning特征

## ⚠️ 注意事项

1. **内存需求**: 大型模型和数据集需要充足的GPU内存
2. **计算时间**: 完整流程可能需要较长时间，建议先用小数据集测试
3. **模型兼容性**: 确保SAE与LLM模型兼容
4. **数据质量**: reasoning标注的质量直接影响最终结果

## 🤝 贡献

欢迎提交issue和pull request来改进这个工具！

## 📄 许可证

MIT License
