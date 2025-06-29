#!/bin/bash


trtllm-build \
    --checkpoint_dir tllm_checkpoint_1gpu_hydra \
    --output_dir ./tmp/hydra/7B/trt_engine/fp16/1-gpu/ \
    --gemm_plugin float16 \
    --speculative_decoding_mode hydra \
    --max_batch_size 4
