#!/bin/bash


python convert_checkpoint.py \
    --model_dir ./lmsys--vicuna-7b-v1.3/ \
    --hydra_model_dir ./ankner--hydra-vicuna-7b-v1.3 \
    --output_dir ./tllm_checkpoint_1gpu_hydra \
    --dtype float16 \
    --num_hydra_heads 4 \
    --num_hydra_layers 4
