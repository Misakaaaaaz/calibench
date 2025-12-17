#!/bin/bash

models=(beit_base convnext_base eva02_base)
run_methods="uncalibrated,TS,PTS,SMART"
# dac_base_methods=(TS PTS SMART)
datasets=(imagenet)

for model in "${models[@]}"; do
    for dataset in "${datasets[@]}"; do
        # Skip imagenet_c with resnet50
        if [[ "$model" == "resnet50" && "$dataset" == "imagenet_c" ]]; then
            continue
        fi
        # Run the regular methods once per model-dataset combination
        python smart_logitgap.py --run_methods $run_methods --model $model --dataset $dataset
        
        # Run DAC with each base method separately
        for dac_base_method in "${dac_base_methods[@]}"; do
            python smart_logitgap.py --run_methods DAC --dac_base_method $dac_base_method --model $model --dataset $dataset
        done
    done
done