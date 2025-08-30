#!/bin/bash

# 运行互信息计算模块
# 计算每个特征与reasoning标签的互信息，筛选出与reasoning高度相关的特征

echo "📊 开始运行互信息计算模块..."

python 4_mutual_information.py \
    --features_path data/binary_features/binary_features.pt \
    --labels_path data/sae_features/labels.pt \
    --output_path data/mutual_information \
    --method "sklearn" \
    --discrete_features true \
    --bins 10 \
    --n_neighbors 3 \
    --top_k 100

echo "✅ 互信息计算完成！结果保存在: data/mutual_information"
echo "📊 查看分析结果: cat data/mutual_information/mi_analysis.json"
echo "🏆 查看Top-100 reasoning特征: cat data/mutual_information/top_100_reasoning_features.json"
echo "📈 查看互信息分析图: data/mutual_information/mutual_information_analysis.png"
