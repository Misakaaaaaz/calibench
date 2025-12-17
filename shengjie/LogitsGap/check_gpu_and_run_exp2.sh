#!/bin/bash
# Script to check GPU availability and run Experiment 2

echo "================================================================================"
echo "GPU Availability Check for Experiment 2"
echo "================================================================================"

# Check current GPU status
echo -e "\nCurrent GPU Status:"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,memory.free --format=csv

# Find GPU with most free memory
echo -e "\nFinding GPU with most free memory..."
GPU_INFO=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -k2 -rn | head -1)
GPU_ID=$(echo $GPU_INFO | cut -d',' -f1)
FREE_MEM=$(echo $GPU_INFO | cut -d',' -f2)

echo "Best GPU: GPU $GPU_ID with ${FREE_MEM}MB free"

# Check if enough memory is available (need at least 10GB)
if [ "$FREE_MEM" -gt 10000 ]; then
    echo -e "\n✓ Sufficient GPU memory available!"
    echo "Starting Experiment 2 on GPU $GPU_ID..."
    
    cd /hdd/haolan/SMART/LogitsGap
    source ~/anaconda3/etc/profile.d/conda.sh
    conda activate rapids-24.12
    
    CUDA_VISIBLE_DEVICES=$GPU_ID nohup python experiment2_smart_combination/run_combination.py \
        --dataset imagenet \
        --model vit_b_16 \
        --seed 1 \
        --valid_size 0.2 \
        --smart_epochs 2000 \
        --patience 200 \
        --n_class 1000 \
        > experiment2_full.log 2>&1 &
    
    PID=$!
    echo "Experiment 2 started with PID: $PID"
    echo "Monitor progress with: tail -f experiment2_full.log"
else
    echo -e "\n✗ Insufficient GPU memory (${FREE_MEM}MB < 10000MB required)"
    echo "All GPUs are currently occupied. Please try again later."
    echo ""
    echo "You can:"
    echo "1. Wait and run this script again: ./check_gpu_and_run_exp2.sh"
    echo "2. Check GPU users: nvidia-smi"
    echo "3. Set up automatic monitoring: see EXPERIMENT2_RESTART_GUIDE.md"
fi

echo "================================================================================"

