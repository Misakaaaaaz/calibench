#!/bin/bash

# Gradient Analysis for SMART Loss Functions
# This script analyzes how different loss functions (SmoothSoftECE, SoftECE, NLL)
# produce different gradients and convergence behavior

# Activate conda environment if available
if command -v conda &> /dev/null; then
    echo "Activating conda base environment..."
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate base
fi

echo "=================================="
echo "SMART Gradient Analysis"
echo "=================================="
echo ""
echo "Python: $(which python)"
echo "Python version: $(python --version 2>&1)"
echo ""

# Configuration
DATASET="imagenet_sketch"
MODEL="resnet50"
SEED=1
VALID_SIZE=0.2
NUM_SAMPLES=2000
EPOCHS=500

# Create output directory with timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="gradient_analysis_${DATASET}_${MODEL}_${TIMESTAMP}"

echo "Configuration:"
echo "  Dataset: ${DATASET}"
echo "  Model: ${MODEL}"
echo "  Seed: ${SEED}"
echo "  Valid Size: ${VALID_SIZE}"
echo "  Samples: ${NUM_SAMPLES}"
echo "  Training Epochs: ${EPOCHS}"
echo "  Output: ${OUTPUT_DIR}"
echo ""

# Run gradient analysis
python analyze_loss_gradients.py \
    --dataset ${DATASET} \
    --model ${MODEL} \
    --seed ${SEED} \
    --valid_size ${VALID_SIZE} \
    --num_samples ${NUM_SAMPLES} \
    --epochs ${EPOCHS} \
    --output_dir ${OUTPUT_DIR}

echo ""
echo "=================================="
echo "Analysis Complete!"
echo "=================================="
echo ""
echo "Results saved to: ${OUTPUT_DIR}/"
echo ""
echo "Generated files:"
echo "  1. gradient_field_analysis.png      - Gradient magnitude vs temperature"
echo "  2. sample_wise_gradient_analysis.png - Per-sample gradient distributions"
echo "  3. convergence_analysis.png          - Training convergence comparison"
echo "  4. gradient_summary.json             - Numerical summary statistics"
echo ""
echo "View the plots to see:"
echo "  • How SmoothSoftECE produces smoother gradients than SoftECE"
echo "  • How NLL gradients differ in sensitivity"
echo "  • Convergence speed and stability differences"
echo ""

