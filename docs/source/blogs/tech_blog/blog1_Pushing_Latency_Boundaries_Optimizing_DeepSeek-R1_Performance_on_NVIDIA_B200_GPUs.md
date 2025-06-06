# Pushing Latency Boundaries: Optimizing DeepSeek-R1 Performance on NVIDIA B200 GPUs
by NVIDIA TensorRT-LLM team
## Table of Contents

- [Pushing Latency Boundaries: Optimizing DeepSeek-R1 Performance on NVIDIA B200 GPUs](#pushing-latency-boundaries-optimizing-deepseek-r1-performance-on-nvidia-b200-gpus)
  - [Table of Contents](#table-of-contents)
  - [Background](#background)
  - [Implementation Configuration](#implementation-configuration)
    - [Workload Profile](#workload-profile)
    - [Model Architecture](#model-architecture)
    - [Precision Strategy](#precision-strategy)
    - [Parallelism Strategy](#parallelism-strategy)
    - [Everything in One Diagram](#everything-in-one-diagram)
  - [Key Optimizations](#key-optimizations)
    - [System Level optimizations](#system-level-optimizations)
      - [CUDA Graph \& Programmatic Dependent Launch](#cuda-graph--programmatic-dependent-launch)
      - [MTP](#mtp)
        - [Autoregressive MTP Layers](#autoregressive-mtp-layers)
        - [Relax Acceptance Verification](#relax-acceptance-verification)
      - [Multi-streams](#multi-streams)
      - [Sparse Experts as GEMMs (only works when moe\_backend=CUTLASS)](#sparse-experts-as-gemms-only-works-when-moe_backendcutlass)
      - [Re-balanced the sparse experts](#re-balanced-the-sparse-experts)
        - [Mixed ETP](#mixed-etp)
        - [Smart Router](#smart-router)
    - [Kernel Level optimizations](#kernel-level-optimizations)
      - [Attention Kernel](#attention-kernel)
      - [Grouped GEMM](#grouped-gemm)
        - [CUTLASS Backend (default backend)](#cutlass-backend-default-backend)
        - [TRTLLM Backend](#trtllm-backend)
      - [Communication Kernel](#communication-kernel)
      - [Dense GEMM optimization](#dense-gemm-optimization)
        - [Fuse\_A\_GEMM](#fuse_a_gemm)
        - [RouterGEMM](#routergemm)
      - [Kernel fusion](#kernel-fusion)
  - [How to reproduce](#how-to-reproduce)
  - [Future Works](#future-works)
  - [Acknowledgment](#acknowledgment)

## Background
Recent advancements in Large Language Reasoning Models have demonstrated remarkable success, while creating new deployment challenges. A critical challenge emerges from extended Output Sequence Lengths (OSL) due to complex "thinking and reasoning" processes. Longer OSL demands stricter Token-to-Token Latency (TTL) requirements, often forcing concurrency limitations. The most extreme case, single concurrency (min-latency scenario) , becomes particularly challenging for real-time applications.

This article explores how TensorRT-LLM achieves record-breaking performance for [DeepSeek-R1](https://huggingface.co/deepseek-ai/DeepSeek-R1) in min-latency scenarios on NVIDIA's 8×B200 GPU configuration progressing from 67 tokens per second (TPS) to 253 before GTC 2025(**3.7x** speed-up), and to our current number is 368 TPS (**5.5x** speed-up).


## Implementation Configuration

### Workload Profile
Input Sequence Length (ISL): 1k tokens

Output Sequence Length (OSL): 2k tokens

### Model Architecture
The base DeepSeek-R1 main model contains: 3x dense layers (initial) and 58x MoE layers, there is also 1x Multi-Tokens Prediction (MTP) layer (MoE-architecture equivalent) for speculative decoding.  Our optimized configuration extends the MTP layer to 3x layers using autoregressive styling for peak performance exploration.

<img src="https://github.com/NVIDIA/TensorRT-LLM/raw/main/docs/source/blogs/media/tech_blog1_model_overview.png?raw=true" alt="tech_blog1_model_overview" width="500" height="auto">

### Precision Strategy
We have explored a mixed precision recipe, which provides a better tradeoff between accuracy and performance.

|               Component               | Precision |
|:-------------------------------------:|:---------:|
|  64x Attention Modules                |   bf16*   |
|  3x Dense FFN Layers                  |  nvfp4**  |
|  58x MoE FFN Layers                   |   nvfp4   |
|  3x MTP Layers                        |   bf16    |
|  RouterGEMM***                        |   bf16    |

*TensorRT-LLM already supports [FP8 Attention](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/models/core/deepseek_v3#fp8-kv-cache-and-mla) while for this latency scenario low-precision attention computation doesn't help with performance so we choose to use bf16 precision for the Attention Modules.

** nvfp4 model checkpoint is generated by the [NVIDIA TensorRT Model Optimizer toolkit](https://github.com/NVIDIA/TensorRT-Model-Optimizer).

*** RouterGEMM uses bf16 inputs/weights with fp32 outputs for numerical stability


### Parallelism Strategy
We have also explored and introduced mixed parallel strategy on 8xB200 GPUs. Specifically, the best strategy for this latency scenario is 'TP8EP2', the definition represents

|       Component       |                   Parallelism Patterns                   |
|:---------------------:|:--------------------------------------------------------:|
| Attention Modules     | Tensor Parallelism 8 (TP8)                               |
| MoE Sparse Experts    | Mixed TP4 with Expert Parallelism 2 (EP2)               |
| MoE Shared Experts    | TP8                                                     |
| Fuse_A GEMM          | Data Parallelism 8 (DP8)                                 |
| RouterGEMM           | DP8                                                     |

### Everything in One Diagram
Now let's put everything into one diagram, which represents a MoE layer from a decoding iteration.

<img src="https://github.com/NVIDIA/TensorRT-LLM/raw/main/docs/source/blogs/media/tech_blog1_model_details.png?raw=true" alt="tech_blog1_model_details" width="1600" height="auto">


The modules in the diagram are:

- Input Module: A BF16 tensor with shape [m, 7168], where m is the number of tokens (for instance, m = 4 when using three MTP layers), and 7168 is the model's hidden size.

- Module1: Fuse_A_GEMM Concatenates the weights for [WDQ, WDKV, and WKR](https://arxiv.org/pdf/2412.19437) to reduce kernel launch overhead.

- Module2: 2× RMSNorm Performs normalization for Q/K tensors. These can be either overlapped on multiple streams or fused into a single grouped RMSNorm.

- Module3: UQ_QR_GEMM Concatenates WUQ and WQR weights to reduce kernel launch overhead.

- Module4: UK_BGEMM Uses WUK in a batched GEMM. We avoid absorbing Modules 3 and 4 to prevent weight-size inflation and extra loading costs.

- Module5: Concat KVCache & applyRope Merges K/V cache and applies ROPE (Rotary Positional Encoding).

- Module6: genAttention Performs MLA during generation, acting like an MQA with num_q_heads = 128 / TP8 = 16.

- Module7: UV_GEMM Executes a batched GEMM with WUV weights.

- Module8: WO_GEMM Runs a dense GEMM using WO weights. We do not absorb Modules 7 and 8 to avoid increased weight loading overhead.

- Module9: Fused Kernels Incorporates oneshotAllReduce, Add_RMSNorm, and DynamicQuant (BF16->NVFP4) in a single kernel.

- Module10: routerGEMM & topK Handles the router GEMM and topK selection.

- Module11: Shared Expert Overlaps partially with Module10 and Module 12.

- Module12: Sparse Experts Implements expert layers via grouped GEMM.

- Module13: Final Fused Kernels Performs localReduction, oneshotAllReduce, and Add_RMSNorm operations together.

## Key Optimizations
| Feature                                                   | TPS/User | Code Links / Notes                                                                                                                                          |
|:----------------------------------------------------------|:--------:|:------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Baseline: CUDA Graph + EP8TP8                             |   67     | [modeling_deepseekv3.py](https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/_torch/models/modeling_deepseekv3.py)                                |
| Multi Stream to overlap shared expert with sparse experts |   73     | [modeling_deepseekv3.py#L506](https://github.com/NVIDIA/TensorRT-LLM/blob/14bfb5e0d6e81aec3306a1324cf074566646f886/tensorrt_llm/_torch/models/modeling_deepseekv3.py#L506) |
| Optimize MLA Kernel                                       |   80     | [PR #3763](https://github.com/NVIDIA/TensorRT-LLM/pull/3763)                                                                                                |
| Optimize TopK Kernels                                     |   84     | • [RoutingKernel.cu](https://github.com/NVIDIA/TensorRT-LLM/blob/main/cpp/tensorrt_llm/kernels/trtllmGenKernels/blockScaleMoe/trtllmGenSrc/RoutingKernel.cu)<br/>• [noAuxTcKernels.cu](https://github.com/NVIDIA/TensorRT-LLM/blob/main/cpp/tensorrt_llm/kernels/noAuxTcKernels.cu) |
| Optimize Fuse_A_GEMM                                      |   89     | [attention.py#L345](https://github.com/NVIDIA/TensorRT-LLM/blob/d6b741ddfe7f8a80718c10d49773c42abc0a254f/tensorrt_llm/_torch/modules/attention.py#L345)     |
| MTP3_Vanilla                                              |   154    | evolve to MTP3_Autoregressive                                                                                                                                                           |
| Evolve to MTP3_Autoregressive + Optimize Router GEMM      |   164    | [modeling_deepseekv3.py#L304](https://github.com/NVIDIA/TensorRT-LLM/blob/d6b741ddfe7f8a80718c10d49773c42abc0a254f/tensorrt_llm/_torch/models/modeling_deepseekv3.py#L304) |
| Fuse oneshotAR + RMSNorm                                  |   168    | [allReduceFusionKernels.cu#L440](https://github.com/NVIDIA/TensorRT-LLM/blob/d6b741ddfe7f8a80718c10d49773c42abc0a254f/cpp/tensorrt_llm/kernels/communicationKernels/allReduceFusionKernels.cu#L440) |
| Enable PDL                                                |   173    | Set environment variable: `export TRTLLM_ENABLE_PDL=1`                                                                                                      |
| Multi-stream to overlap two RMS_norms                     |   180    | [attention.py#L546](https://github.com/NVIDIA/TensorRT-LLM/blob/d6b741ddfe7f8a80718c10d49773c42abc0a254f/tensorrt_llm/_torch/modules/attention.py#L546)     |
| MTP3_Autoregressive                                       |   204    | [modeling_deepseekv3.py#L823](https://github.com/NVIDIA/TensorRT-LLM/blob/d6b741ddfe7f8a80718c10d49773c42abc0a254f/tensorrt_llm/_torch/models/modeling_deepseekv3.py#L823) |
| Finetune clock/power                                      |   211    | `sudo nvidia-smi -pm 0; sudo nvidia-smi -pm 1; sudo nvidia-smi boost-slider --vboost 4`                                                                     |
| Optimize CUTLASS Grouped GEMM Kernels                     |   236    | The code is not open-source yet due to the dependency with internal base environment and we are planning to make it decoupled from internal base environment thus to be able to open-source in the future.|
| Optimize CUTLASS Flow: Sparse Experts as GEMMs            |   249    | The code is not open-source yet due to the dependency with internal base environment and we are planning to make it decoupled from internal base environment thus to be able to open-source in the future.|
| Introduce EP4TP2 for better workload balance              |   253    | Use `--tp 8 --ep 4` when benchmarking                                                                                                                       |
| Introduce moe_backend=TRTLLM, EP2TP4 for better balance   |   299    | [PR #4280](https://github.com/NVIDIA/TensorRT-LLM/pull/4280)                                                                                          |
| Optimize Fuse_A_GEMM and Router_GEMM                      |   340    | WIP                                                                                          |
| Relax Acceptance                                          |   **368**    | [deepseek_v3#multi-token-prediction-mtp](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/models/core/deepseek_v3#multi-token-prediction-mtp)     |

### System Level optimizations
#### CUDA Graph & Programmatic Dependent Launch
[CUDA Graph](https://developer.nvidia.com/blog/cuda-graphs/) is necessary to overcome the CPU-overhead for small workloads, while [Programmatic Dependent Launch](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html?highlight=Programmatic%2520Dependent%2520Launch#programmatic-dependent-launch-and-synchronization) can be used to reduce the kernel launch latency furthermore.
#### MTP
There are two optimizations based on MTP
##### Autoregressive MTP Layers

| Version     | Acceptance Rate | TPS/User | TPS/User Speedup |
|:-----------:|:---------------:|:--------:|:----------------:|
| Without MTP |       1.00      |   111    |       1.00       |
| MTP 1       |       1.92      |   198    |       1.78       |
| MTP 2       |       2.58      |   250    |       2.25       |
| MTP 3       |       2.82      |   253    |       2.28       |
| MTP 4       |       2.99      |   245    |       2.21       |
| MTP 5       |       3.01      |   239    |       2.15       |

Based on our exploration, 3x MTP layers configuration demonstrates optimal performance.

##### Relax Acceptance Verification
For the reasoning model (such as DeepSeek R1), the generation may consist of two phases: thinking phase and actual output. During the thinking phase, when relaxed acceptance is enabled, the draft token can be accepted when it is in a candidate set. This candidate is generated based on the logits topN and probability threshold.
- topN: The topN tokens are sampled from logits.
- Probability threshold. Based on topN candidates, only those tokens with a probability greater than the Top1's probability - delta can remain in the candidate set.

During the non-thinking phase, we still use strict acceptance.

| Version            | Acceptance Rate | TPS/User Speedup |
|:------------------:|:--------------:|:----------------:|
| MTP3_top1, d0.0    |      2.82      |       1.00       |
| MTP3_top10, d0.5   |      3.06      |       1.08       |
| MTP3_top10, d0.6   |      3.10      |       1.09       |
| MTP3_top15, d0.5   |      3.07      |       1.08       |

This is a relaxed way of verification and comparison, which can improve the acceptance rate and bring positive speedup with limited influence on accuracy.

|          Dataset          | Test Size | w/o Relaxed accuracy | w/ Relaxed accuracy |
|:-------------------------:|:---------:|:----------:|:----------:|
| MMLU-Pro                  | 12,032    | 84.0%      | 81.2%      |
| Humanity's Last Exam      | 2,684     | 9.0%       | 9.0%       |
| GPQA Diamond              | 198       | 71.0%      | 69.2%      |
| MATH-500                  | 500       | 96.0%      | 96.2%      |
| AIME 2024                 | 30        | 68.0%      | 74.0%      |
| SciCode                   | 338       | 36.0%      | 39.0%      |
| LiveCodeBench             | 315       | 62.0%      | 66.0%      |

For more information, please visit [multi-token-prediction-mtp](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/models/core/deepseek_v3#multi-token-prediction-mtp)



#### Multi-streams
We have introduced multi-streams based optimizations to hide some kernels' overhead, such as:
- Overlap shared experts with sparse experts
- Overlap Concat_KVCache kernel with GEMM


#### Sparse Experts as GEMMs (only works when moe_backend=CUTLASS)

<img src="https://github.com/NVIDIA/TensorRT-LLM/raw/main/docs/source/blogs/media/tech_blog1_sparse_exp_as_a_gemm.png?raw=true" alt="tech_blog1_sparse_exp_as_a_gemm" width="800" height="auto">

The existing CUTLASS-based Sparse Experts flow (illustrated in the figure) dispatches input tokens to their designated experts, then applies indexed local reduction on each expert's outputs before a global allreduce. Both dispatching and indexed local reduction incur high overhead in low-latency scenarios. To address this, we propose treating "Sparse Experts as GEMMs" by sending all tokens to each activated expert and masking out unneeded outputs before local reduction. Because grouped GEMMs are memory-bound, the extra computations from redundant tokens have minimal impact, effectively eliminating the costly dispatch and reduction overhead.

#### Re-balanced the sparse experts
For sparse experts, two parallelization strategies are commonly used: Expert Parallel (EP) and Tensor Parallel (TP). Expert Parallel (EP) maps each expert to a distinct GPU, achieving high memory and computational efficiency. However, token placement is data-dependent, distributing workloads unevenly across GPUs and revealing overhead in the AllReduce step after the MoE module. Tensor Parallel (TP) shards each expert evenly across GPUs, creating a balanced workload but sacrificing math/memory efficiency.


##### Mixed ETP
A combined EP/TP approach can mitigate both challenges. In practice, our experiments show that a configuration of TP4EP2 offers the best performance.

##### Smart Router
Alternatively, by storing all expert weights on a cluster of four GPUs and replicating them to another four-GPU cluster, a smart router can dynamically dispatch tokens across each cluster. This design keeps balanced workload distribution even without significantly impacting local memory and computation efficiency.


### Kernel Level optimizations
#### Attention Kernel
We have developed a customized MLA attention kernel to better utilize GPU resources for latency scenarios.
#### Grouped GEMM
##### CUTLASS Backend (default backend)
Our default MoE backend is based on CUTLASS, which is flexible/robust but may not be the best performance case.

##### TRTLLM Backend
The other MoE backend is TRTLLM, which provides better performance, and we are working to make it more flexible and robust, and in the future it will be switched as the default backend for Grouped GEMM computation for latency scenarios.

#### Communication Kernel
For small message sizes, regular NCCL latency-bound AllReduce kernels are inefficient, so we've developed a customized oneshot AllReduce kernel. It leverages the powerful NVSwitch HW capability by acting like an initial broadcast followed by local reduction, delivering better performance in min-latency scenarios.

#### Dense GEMM optimization
We focus on optimizing two kinds of dense GEMMs: Fuse_A_GEMM and RouterGEMM, because they dominate the execution time, suffer from low memory efficiency, and cannot be easily sharded (they are DP-based).

##### Fuse_A_GEMM
We developed a custom Fuse_A_GEMM that prefetches the majority of its weights into shared memory (enabled by PDL and overlapped with oneshot-AllReduce), significantly enhancing performance. The kernel shows substantial improvements over default GEMM implementation when num_tokens < 16.

<img src="https://github.com/NVIDIA/TensorRT-LLM/raw/main/docs/source/blogs/media/tech_blog1_fuse_a_gemm.png?raw=true" alt="tech_blog1_fuse_a_gemm" width="500" height="auto">

##### RouterGEMM
By leveraging our internal AI code generator, we automatically generate an optimized RouterGEMM kernel, which delivers substantial improvements over the default GEMM implementation when num_tokens <=30.

<img src="https://github.com/NVIDIA/TensorRT-LLM/raw/main/docs/source/blogs/media/tech_blog1_router_gemm.png?raw=true" alt="tech_blog1_router_gemm" width="500" height="auto">

#### Kernel fusion
Kernel fusion is necessary for min-latency scenario to reduce extra global memory write/read cost, and we support following fusion patterns now
- Fuse two overlapped RMS_Norms into one GroupedRMSNorm
- Fuse (LocalReduction) + AR+ RMS_Norm+ (Dynamic_Quant_bf16tonvfp4) into one kernel
- Fuse Grouped GEMM_FC1 + dot activation (when moe_backend=TRTLLM) into one kernel



## How to reproduce
https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/blogs/Best_perf_practice_on_DeepSeek-R1_in_TensorRT-LLM.md#b200-min-latency

Of note, the Relaxed Acceptance is specific for Deepseek-R1 model, if you want to enable it, you need to set `add_generation_prompt = True` when preparing the benchmark dataset, the code demo likes
```python
input_ids = tokenizer.encode(tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True), add_special_tokens=False)
```
It's also needed to set `use_relaxed_acceptance_for_thinking: true`, `relaxed_topk: 10` and `relaxed_delta: 0.6` in speculative_config.


## Future Works
- More Fusions
- More Overlap
- More optimization of Attention Kernel
- More Exploration of MTP

## Acknowledgment
Pushing the performance boundaries of DeepSeek R1 for latency-sensitive applications has been a remarkable engineering journey. The optimizations detailed in this post represent an exceptional cross-functional collaboration across the entire AI technology stack - spanning kernel-level optimizations, runtime enhancements, model quantization techniques, algorithmic improvements, and systematic performance analysis and tuning. While we can't individually acknowledge every contributor, we're proud to recognize the dedicated team of engineers whose collective expertise has helped advance the state-of-the-art in TensorRT-LLM performance engineering.

Through this collaborative endeavor, we've developed valuable insights into maximizing GPU utilization for large language model inference. We hope that the techniques and best practices shared in this blog will empower the developer community to better leverage NVIDIA GPU capabilities in their mission-critical LLM inference applications.
