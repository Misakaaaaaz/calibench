#!/bin/bash
# export CUDA_VISIBLE_DEVICES=1

datasets=("cifar10" "cifar100")
models=("resnet50" "wide_resnet")
seeds=("1")

for dataset in "${datasets[@]}"; do
    for model in "${models[@]}"; do
        for seed in "${seeds[@]}"; do
            python smart_logitgap.py --model "$model" --dataset "$dataset" --random_seed "$seed" --run_methods "uncalibrated,TS,PTS,CTS,ETS,SMART" --train_loss "cross_entropy" --use_underfitted --underfitted_epochs 5
        done
    done
done
