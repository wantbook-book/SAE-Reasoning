#!/bin/bash

# 运行特征二值化模块
# 对SAE特征激活值进行阈值二值化处理

echo "🔢 开始运行特征二值化模块..."

python 3_feature_binarization.py \
    --features_path data/sae_features/sae_features.pt \
    --output_path data/binary_features \
    --threshold_method "percentile" \
    --threshold_value 0.95 \
    --adaptive_k 2.0 \
    --min_activation_count 10 \
    --save_analysis true

echo "✅ 特征二值化完成！二值化特征保存在: data/binary_features"
echo "📊 查看分析结果: cat data/binary_features/binarization_analysis.json"
echo "📈 查看阈值分析图: data/binary_features/threshold_analysis.png"
