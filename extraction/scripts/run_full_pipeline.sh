#!/bin/bash

# SAE Reasoning特征提取完整流水线
# 按顺序运行四个主要模块：数据标注 → 激活提取 → 特征二值化 → 互信息计算

set -e  # 遇到错误时退出

echo "🚀 开始运行SAE Reasoning特征提取完整流水线..."
echo "=================================================="

# 创建输出目录
mkdir -p data

# 步骤1: 数据标注
echo ""
echo "📍 步骤 1/4: 数据标注"
echo "=================================================="
bash scripts/run_data_labeling.sh

# 检查步骤1是否成功
if [ ! -f "data/labeled_dataset/dataset_stats.json" ]; then
    echo "❌ 步骤1失败：数据标注未完成"
    exit 1
fi

# 步骤2: 激活提取
echo ""
echo "📍 步骤 2/4: 激活提取"
echo "=================================================="
bash scripts/run_activation_extraction.sh

# 检查步骤2是否成功
if [ ! -f "data/sae_features/sae_features.pt" ]; then
    echo "❌ 步骤2失败：激活提取未完成"
    exit 1
fi

# 步骤3: 特征二值化
echo ""
echo "📍 步骤 3/4: 特征二值化"
echo "=================================================="
bash scripts/run_feature_binarization.sh

# 检查步骤3是否成功
if [ ! -f "data/binary_features/binary_features.pt" ]; then
    echo "❌ 步骤3失败：特征二值化未完成"
    exit 1
fi

# 步骤4: 互信息计算
echo ""
echo "📍 步骤 4/4: 互信息计算"
echo "=================================================="
bash scripts/run_mutual_information.sh

# 检查步骤4是否成功
if [ ! -f "data/mutual_information/top_100_reasoning_features.json" ]; then
    echo "❌ 步骤4失败：互信息计算未完成"
    exit 1
fi

echo ""
echo "🎉 完整流水线执行成功！"
echo "=================================================="
echo "📂 输出文件结构："
echo "data/"
echo "├── labeled_dataset/          # 标注后的数据集"
echo "│   ├── dataset_stats.json    # 数据统计"
echo "│   └── ..."
echo "├── sae_features/             # SAE特征"
echo "│   ├── sae_features.pt       # SAE特征文件"
echo "│   ├── labels.pt             # 标签文件"
echo "│   └── extraction_config.json # 提取配置"
echo "├── binary_features/          # 二值化特征"
echo "│   ├── binary_features.pt    # 二值化特征文件"
echo "│   ├── thresholds.pt         # 阈值文件"
echo "│   ├── binarization_analysis.json # 二值化分析"
echo "│   └── threshold_analysis.png # 阈值分析图"
echo "└── mutual_information/       # 互信息分析"
echo "    ├── mutual_information_scores.pt # 互信息得分"
echo "    ├── top_100_reasoning_features.json # Top-100特征"
echo "    ├── mi_analysis.json      # 互信息分析"
echo "    └── mutual_information_analysis.png # 分析图"
echo ""
echo "🏆 推荐查看的关键文件："
echo "1. data/mutual_information/top_100_reasoning_features.json - Top reasoning特征"
echo "2. data/mutual_information/mi_analysis.json - 互信息统计分析"
echo "3. data/binary_features/binarization_analysis.json - 特征二值化质量"
echo "4. data/labeled_dataset/dataset_stats.json - 数据集统计信息"
