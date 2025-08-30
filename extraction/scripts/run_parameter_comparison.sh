#!/bin/bash

# 参数对比实验脚本
# 比较不同阈值方法和互信息计算方法的效果

echo "🔬 开始运行参数对比实验..."
echo "=================================================="

# 确保已有SAE特征
if [ ! -f "data/sae_features/sae_features.pt" ]; then
    echo "❌ 请先运行激活提取步骤生成SAE特征"
    echo "运行: bash scripts/run_activation_extraction.sh"
    exit 1
fi

# 创建对比实验目录
mkdir -p data/comparisons

echo ""
echo "📍 实验1: 比较不同阈值方法"
echo "=================================================="

# 比较不同阈值方法
python 3_feature_binarization.py compare_threshold_methods \
    --features_path data/sae_features/sae_features.pt \
    --output_path data/comparisons/threshold_methods \
    --methods "percentile,mean,median,adaptive" \
    --threshold_values "0.9,0.95,0.99"

echo ""
echo "📍 实验2: 使用最佳阈值方法进行二值化"
echo "=================================================="

# 使用percentile 95%阈值（通常效果较好）
python 3_feature_binarization.py \
    --features_path data/sae_features/sae_features.pt \
    --output_path data/comparisons/binary_features_95 \
    --threshold_method "percentile" \
    --threshold_value 0.95 \
    --min_activation_count 10

# 使用percentile 99%阈值（更稀疏）
python 3_feature_binarization.py \
    --features_path data/sae_features/sae_features.pt \
    --output_path data/comparisons/binary_features_99 \
    --threshold_method "percentile" \
    --threshold_value 0.99 \
    --min_activation_count 5

echo ""
echo "📍 实验3: 比较不同互信息计算方法"
echo "=================================================="

# 使用95%阈值的二值化特征
python 4_mutual_information.py compare_mi_methods \
    --features_path data/comparisons/binary_features_95/binary_features.pt \
    --labels_path data/sae_features/labels.pt \
    --output_path data/comparisons/mi_methods_95 \
    --methods "sklearn,manual"

# 使用99%阈值的二值化特征
python 4_mutual_information.py compare_mi_methods \
    --features_path data/comparisons/binary_features_99/binary_features.pt \
    --labels_path data/sae_features/labels.pt \
    --output_path data/comparisons/mi_methods_99 \
    --methods "sklearn,manual"

echo ""
echo "📍 实验4: 生成最终推荐的reasoning特征"
echo "=================================================="

# 使用推荐的参数组合
python 4_mutual_information.py \
    --features_path data/comparisons/binary_features_95/binary_features.pt \
    --labels_path data/sae_features/labels.pt \
    --output_path data/comparisons/final_reasoning_features \
    --method "sklearn" \
    --discrete_features true \
    --top_k 200

echo ""
echo "🎉 参数对比实验完成！"
echo "=================================================="
echo "📂 对比实验结果："
echo "data/comparisons/"
echo "├── threshold_methods/        # 阈值方法比较"
echo "│   ├── threshold_comparison.json"
echo "│   └── threshold_methods_comparison.png"
echo "├── binary_features_95/       # 95%阈值二值化"
echo "├── binary_features_99/       # 99%阈值二值化"
echo "├── mi_methods_95/           # 95%阈值的MI方法比较"
echo "├── mi_methods_99/           # 99%阈值的MI方法比较"
echo "└── final_reasoning_features/ # 最终推荐特征"
echo "    ├── top_200_reasoning_features.json"
echo "    └── mutual_information_analysis.png"
echo ""
echo "🏆 推荐查看："
echo "1. data/comparisons/threshold_methods/threshold_comparison.json - 阈值方法比较"
echo "2. data/comparisons/final_reasoning_features/top_200_reasoning_features.json - 最终特征"
echo "3. data/comparisons/mi_methods_95/mi_methods_comparison.json - MI方法比较"
