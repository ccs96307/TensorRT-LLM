# SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import math
from typing import Optional, Union

import numpy as np
import torch
from transformers import AutoModelForCausalLM

from tensorrt_llm._utils import numpy_to_torch
from tensorrt_llm.models.hydra.weight import load_hydra_hf
from tensorrt_llm.models.llama.model import LLaMAForCausalLM, RmsNorm
from tensorrt_llm.models.qwen.model import QWenForCausalLM

from ..._common import default_net
from ..._utils import pad_vocab_size
from ...functional import (ACT2FN, add, cast, concat, constant, cos, div,
                           expand, matmul, mul, shape, sin, slice, softmax,
                           squeeze, stack, topk, transpose, unsqueeze, view)
from ...layers import ColumnLinear
from ...mapping import Mapping
from ...module import Module, ModuleList
from ..modeling_utils import PretrainedModel, QuantConfig
from .config import HydraConfig
from .weight import convert_hf_llama


# refer: https://github.com/zankner/Hydra/blob/main/hydra/model/hydra_heads/prefix_mlp_head.py#L44
class HydraResBlock(Module):

    def __init__(
            self,
            hidden_size,
            hidden_act="silu",
            num_condition=0,
            dtype=None,
            mapping=Mapping(),
    ):
        super().__init__()

        input_size = hidden_size * (num_condition + 1)
        self.linear = ColumnLinear(input_size,
                                   hidden_size,
                                   dtype=dtype,
                                   tp_group=mapping.tp_group,
                                   tp_size=mapping.tp_size,
                                   gather_output=True)
        self.res_connection = ColumnLinear(
            input_size,
            hidden_size,
            dtype=dtype,
            tp_group=mapping.tp_group,
            tp_size=mapping.tp_size,
            gather_output=True) if num_condition > 0 else torch.nn.Identity()

        self.hidden_act = hidden_act

    def forward(self, x):
        return self.res_connection(x) + ACT2FN[self.hidden_act](self.linear(x))


class HydraPrefixMLP(Module):

    def __init__(
            self,
            num_layers,
            hidden_size,
            vocab_size,
            hydra_head_idx,
            hidden_act="silu",
            dtype=None,
            mapping=Mapping(),
            lm_head_init_weight=None,
    ):
        super().__init__()
        self.hydra_mlp = HydraResBlock(hidden_size=hidden_size,
                                       num_condition=hydra_head_idx + 1,
                                       hidden_act=hidden_act,
                                       dtype=dtype,
                                       mapping=mapping)

        self.hydra_mlps = ModuleList([
            HydraResBlock(hidden_size=hidden_size,
                          hidden_act=hidden_act,
                          dtype=dtype,
                          mapping=mapping) for _ in range(num_layers)
        ])
        self.hydra_lm_head = ColumnLinear(hidden_size,
                                          vocab_size,
                                          bias=True,
                                          dtype=dtype,
                                          tp_group=mapping.tp_group,
                                          tp_size=mapping.tp_size,
                                          gather_output=True)

    def forward(self, x):
        hidden_states = self.hydra_mlp(x)

        for layer in self.hydra_mlps:
            hidden_states = layer(hidden_states)

        return self.hydra_lm_head(hidden_states)


def _compute_default_rope_parameters(
    config: Optional[HydraConfig] = None,
    **rope_kwargs,
):
    # if len(rope_kwargs) > 0:
    #     base = rope_kwargs["base"]
    #     dim = rope_kwargs["dim"]
    # elif config is not None:
    #     base = config.rope_theta
    #     partial_rotary_factor = config.partial_rotary_factor if hasattr(config, "partial_rotary_factor") else 1.0
    #     head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
    #     dim = int(head_dim * partial_rotary_factor)

    base = getattr(config, "rope_theta", 10000.0)
    head_dim = config.hidden_size // config.num_attention_heads
    partial_rotary_factor = getattr(config, "partial_rotary_factor", 1.0)
    dim = int(head_dim * partial_rotary_factor)

    attention_factor = 1.0  # Unused in this type of RoPE

    # Compute the inverse frequencies
    idx = np.arange(0, dim, 2, dtype=np.float32)
    inv_freq = 1.0 / (base**(idx / dim))

    return inv_freq, attention_factor


def _compute_llama3_parameters(
    config: HydraConfig,
    **rope_kwargs,
):
    # Gets the default RoPE parameters
    inv_freq, attention_factor = _compute_default_rope_parameters(
        config, **rope_kwargs)

    factor = 8.0
    low_freq_factor = 1.0
    high_freq_factor = 4.0
    old_context_len = 8192.0

    low_freq_wavelen = old_context_len / low_freq_factor
    high_freq_wavelen = old_context_len / high_freq_factor

    wavelen = 2 * math.pi / inv_freq

    # Use numpy
    inv_freq_llama = np.where(wavelen > low_freq_wavelen, inv_freq / factor,
                              inv_freq)

    smooth_factor = (old_context_len / wavelen -
                     low_freq_factor) / (high_freq_factor - low_freq_factor)
    smoothed_inv_freq = (
        1 - smooth_factor
    ) * inv_freq_llama / factor + smooth_factor * inv_freq_llama

    is_medium_freq = (wavelen <= low_freq_wavelen) & (wavelen
                                                      >= high_freq_wavelen)

    inv_freq_llama = np.where(is_medium_freq, smoothed_inv_freq, inv_freq_llama)

    return inv_freq_llama, attention_factor


def rotate_half(x):
    """
    TensorRT-LLM functional version of rotate_half.
    Assumes x is a 4D tensor: [batch, num_heads, seq_len, head_dim]
    Splits the last dimension in half, rotates the halves, and concatenates them.
    """
    # Get dimensions as scalar tensors
    dim0 = squeeze(shape(x, 0), dim=0)
    dim1 = squeeze(shape(x, 1), dim=0)
    dim2 = squeeze(shape(x, 2), dim=0)
    last_dim = squeeze(shape(x, 3), dim=0)

    # Compute half of last_dim
    two = constant(np.array([2], dtype="int64"))
    half_dim = squeeze(div(last_dim, two), dim=0)

    # Create scalar zero for use in starts
    zero = constant(np.array(0, dtype="int64"))

    # Define starts and sizes for slicing
    starts1 = stack([zero, zero, zero, zero], dim=0)
    sizes1 = stack([dim0, dim1, dim2, half_dim], dim=0)
    starts2 = stack([zero, zero, zero, half_dim], dim=0)

    # Slice tensors into two halves along the last dimension
    x1 = slice(x, starts=starts1, sizes=sizes1)
    x2 = slice(x, starts=starts2, sizes=sizes1)

    # Negate the second half and concatenate
    neg_x2 = mul(x2, -1.0)
    return concat([neg_x2, x1], dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    """
    TensorRT-LLM functional version of apply_rotary_pos_emb.
    """
    # PyTorch: cos.unsqueeze(unsqueeze_dim)
    cos_expanded = unsqueeze(cos, unsqueeze_dim)
    sin_expanded = unsqueeze(sin, unsqueeze_dim)

    # PyTorch: (q * cos) + (rotate_half(q) * sin)
    rotated_q = rotate_half(q)
    q_embed = add(mul(q, cos_expanded), mul(rotated_q, sin_expanded))

    # PyTorch: (k * cos) + (rotate_half(k) * sin)
    rotated_k = rotate_half(k)
    k_embed = add(mul(k, cos_expanded), mul(rotated_k, sin_expanded))

    return q_embed, k_embed


def repeat_kv(hidden_states, n_rep: int):
    """
    TensorRT-LLM functional version of repeat_kv
    """
    if n_rep == 1:
        return hidden_states

    batch, num_key_value_heads, slen, head_dim = (shape(hidden_states, 0),
                                                  shape(hidden_states, 1),
                                                  shape(hidden_states, 2),
                                                  shape(hidden_states, 3))

    hidden_states_unsqueezed = unsqueeze(hidden_states, 2)
    hidden_states_expanded = expand(
        hidden_states_unsqueezed,
        [batch, num_key_value_heads, n_rep, slen, head_dim])

    final_shape = [batch, num_key_value_heads * n_rep, slen, head_dim]
    return view(hidden_states_expanded, final_shape)


def eager_attention_forward(
    query,
    key,
    value,
    num_key_value_groups: int,
    scaling: float,
    dropout: float = 0.0,
    attention_mask=None,
    **kwargs,
):
    key_states = repeat_kv(key, num_key_value_groups)
    value_states = repeat_kv(value, num_key_value_groups)

    # Attetion Scores: (Q @ K.T) * scaling
    key_states_T = transpose(key_states, 2, 3)
    attn_scores = matmul(query, key_states_T)
    attn_scores_scaled = mul(attn_scores, scaling)

    if attention_mask is not None:
        key_len = shape(key_states, 2)
        mask_shape = shape(attention_mask)

        causal_mask = slice(
            attention_mask,
            starts=[0, 0, 0, 0],
            sizes=[mask_shape[0], mask_shape[1], mask_shape[2], key_len])
        attn_scores_masked = add(attn_scores_scaled, causal_mask)
    else:
        attn_scores_masked = attn_scores_scaled

    # Softmax
    query_dtype = query.dtype
    attn_weights_fp32 = softmax(cast(attn_scores_masked, "float32"), dim=-1)
    attn_weights = cast(attn_weights_fp32, query_dtype)

    # Ignore dropout
    # Compute Attention Output: attn_weights @ V
    attn_output = matmul(attn_weights, value_states)

    # Transpose
    attn_output = transpose(attn_output, 1, 2)

    return attn_output


class LlamaRotaryEmbedding(Module):

    def __init__(self, config: HydraConfig, mapping=Mapping()):
        super().__init__()
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings
        self.config = config
        self.rope_init_fn = _compute_llama3_parameters

        self.original_inv_freq, self.attention_scaling = self.rope_init_fn(
            self.config)

    def forward(self, x, position_ids):
        x_dtype = x.dtype

        inv_freq = constant(self.original_inv_freq)

        inv_freq_unsqueezed = unsqueeze(unsqueeze(inv_freq, 0), 2)
        b = shape(position_ids, 0)
        inv_freq_dim0 = shape(inv_freq, 0)
        one = constant(np.array([1], dtype=np.int64))

        expand_shape = concat(
            [unsqueeze(b, 0), unsqueeze(inv_freq_dim0, 0), one], dim=0)

        inv_freq_expanded = expand(inv_freq_unsqueezed, expand_shape)
        position_ids_expanded = unsqueeze(position_ids, 1)

        inv_freq_float32 = cast(inv_freq_expanded, "float32")
        position_ids_float32 = cast(position_ids_expanded, "float32")

        freqs_t = matmul(inv_freq_float32, position_ids_float32)
        freqs = transpose(freqs_t, 1, 2)

        emb = concat([freqs, freqs], dim=-1)

        # 6. Apply cos, sin, and scaling
        # PyTorch: emb.cos() * self.attention_scaling
        cos_emb = cos(emb)
        cos_scaled = mul(cos_emb, self.attention_scaling)

        # PyTorch: emb.sin() * self.attention_scaling
        sin_emb = sin(emb)
        sin_scaled = mul(sin_emb, self.attention_scaling)

        # 7. Cast back to original dtype
        # PyTorch: .to(dtype=x.dtype)
        final_cos = cast(cos_scaled, x_dtype)
        final_sin = cast(sin_scaled, x_dtype)

        return final_cos, final_sin


class LlamaMLP(Module):

    def __init__(self, config, dtype=None, mapping=Mapping()):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        self.gate_proj = ColumnLinear(
            in_features=self.hidden_size,
            out_features=self.intermediate_size,
            bias=False,
            dtype=dtype,
            tp_group=mapping.tp_group,
            tp_size=mapping.tp_size,
            gather_output=True,
        )

        self.up_proj = ColumnLinear(
            in_features=self.hidden_size,
            out_features=self.intermediate_size,
            bias=False,
            dtype=dtype,
            tp_group=mapping.tp_group,
            tp_size=mapping.tp_size,
            gather_output=True,
        )

        self.down_proj = ColumnLinear(
            in_features=self.intermediate_size,
            out_features=self.hidden_size,
            bias=False,
            dtype=dtype,
            tp_group=mapping.tp_group,
            tp_size=mapping.tp_size,
            gather_output=True,
        )

        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        # PyTorch: self.act_fn(self.gate_proj(x)) * self.up_proj(x)
        gated_x = self.act_fn(self.gate_proj(x))
        up_x = self.up_proj(x)
        fused_x = mul(gated_x, up_x)

        down_proj = self.down_proj(fused_x)
        return down_proj


class LlamaAttention(Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self,
                 config: HydraConfig,
                 layer_idx: int,
                 mapping=Mapping(),
                 dtype=None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(
            config, "head_dim",
            config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.is_causal = True

        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.tp_size = mapping.tp_size
        self.hidden_size = config.hidden_size

        self.q_proj = ColumnLinear(
            in_features=config.hidden_size,
            out_features=config.num_attention_heads * self.head_dim,
            bias=False,
            dtype=dtype,
            tp_group=mapping.tp_group,
            tp_size=mapping.tp_size,
            gather_output=True,
        )
        self.k_proj = ColumnLinear(
            in_features=config.hidden_size,
            out_features=config.num_key_value_heads * self.head_dim,
            bias=False,
            dtype=dtype,
            tp_group=mapping.tp_group,
            tp_size=mapping.tp_size,
            gather_output=True,
        )
        self.v_proj = ColumnLinear(
            in_features=config.hidden_size,
            out_features=config.num_key_value_heads * self.head_dim,
            bias=False,
            dtype=dtype,
            tp_group=mapping.tp_group,
            tp_size=mapping.tp_size,
            gather_output=True,
        )
        self.o_proj = ColumnLinear(
            in_features=config.num_attention_heads * self.head_dim,
            out_features=config.hidden_size,
            bias=False,
            dtype=dtype,
            tp_group=mapping.tp_group,
            tp_size=mapping.tp_size,
            gather_output=True,
        )

    def forward(
        self,
        hidden_states,
        position_embeddings,
        attention_mask=None,
        past_key_value=None,
        cache_position=None,
    ):
        b, s = shape(hidden_states, 0), shape(hidden_states, 1)

        # 1. Q, K, V Projections
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        print(f"q: {q.shape}")
        print(f"k: {q.shape}")
        print(f"v: {q.shape}")

        # 2. Reshape and Transpose to [batch, num_heads, seq_len, head_dim]
        # PyTorch: .view(hidden_shape).transpose(1, 2)
        query_states = transpose(
            view(
                q,
                [0, 0, self.num_attention_heads // self.tp_size, self.head_dim
                 ]), 1, 2)
        key_states = transpose(
            view(
                k,
                [0, 0, self.num_key_value_heads // self.tp_size, self.head_dim
                 ]), 1, 2)
        value_states = transpose(
            view(
                v,
                [0, 0, self.num_key_value_heads // self.tp_size, self.head_dim
                 ]), 1, 2)

        print(f"query_states: {query_states.shape}")
        print(f"key_states: {key_states.shape}")
        print(f"value_states: {value_states.shape}")

        # 3. Apply Rotary Position Embedding
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states,
                                                        key_states, cos, sin)

        # 4. Maybe pass it...
        # if past_key_value is not None:
        #     # sin and cos are specific to RoPE models; cache_position needed for the static cache
        #     cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        #     key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # 5. Attention Computation
        attn_output = eager_attention_forward(
            query=query_states,
            key=key_states,
            value=value_states,
            num_key_value_groups=self.num_key_value_groups,
            scaling=self.scaling,
            attention_mask=attention_mask,
        )

        # 6. Final Reshape and Projection
        # PyTorch: attn_output.reshape(*input_shape, -1).contiguous()
        print(f"attn_output (eager_attention_forward): {attn_output.shape}")

        attn_output = view(attn_output, [0, 0, -1])
        attn_output = self.o_proj(attn_output)

        return attn_output


class LlamaDecoderLayer(Module):

    def __init__(self, config: HydraConfig, layer_idx: int, mapping=Mapping()):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = LlamaAttention(config=config,
                                        layer_idx=layer_idx,
                                        mapping=mapping)

        self.mlp = LlamaMLP(config, mapping=mapping)
        self.input_layernorm = RmsNorm(
            normalized_shape=config.hidden_size,
            dtype=config.dtype,
        )
        self.post_attention_layernorm = RmsNorm(
            normalized_shape=config.hidden_size,
            dtype=config.dtype,
        )

    def forward(
            self,
            hidden_states,
            attention_mask=None,
            position_ids=None,
            past_key_value=None,
            output_attentions: Optional[bool] = False,
            use_cache: Optional[bool] = False,
            cache_position=None,
            position_embeddings=None,  # necessary, but kept here for BC
    ):
        residual = hidden_states
        normed_hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        attn_output = self.self_attn(
            hidden_states=normed_hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
        )
        hidden_states = add(residual, attn_output)

        print(f"hidden_states (self_attn): {hidden_states.shape}")

        # Fully Connected
        residual = hidden_states
        normed_hidden_states = self.post_attention_layernorm(hidden_states)

        print(
            f"hidden_states (self.post_attention_layernorm): {normed_hidden_states.shape}"
        )

        mlp_output = self.mlp(normed_hidden_states)

        print(f"hidden_states (self.mlp): {mlp_output.shape}")

        hidden_states = add(residual, mlp_output)

        return hidden_states


# refer: https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py
class PrefixEmbeddingLayer(Module):

    def __init__(self, config: HydraConfig, mapping=Mapping()):
        super().__init__()
        self.vocab_size = config.vocab_size

        self.layer = LlamaDecoderLayer(config=config,
                                       layer_idx=0,
                                       mapping=mapping)

        self.norm = RmsNorm(
            normalized_shape=config.hidden_size,
            dtype=config.dtype,
        )
        self.rotary_emb = LlamaRotaryEmbedding(config=config, mapping=mapping)

    def forward(
        self,
        inputs_embeds,
        position_ids,
        attention_mask=None,
    ):
        # output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # output_hidden_states = (
        #     output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        # )
        # use_cache = use_cache if use_cache is not None else self.config.use_cache

        # if cache_position is None:
        #     past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        #     cache_position = torch.arange(
        #         past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
        #     )

        # if position_ids is None:
        #     position_ids = cache_position.unsqueeze(0)

        # causal_mask = self._update_causal_mask(
        #     attention_mask,
        #     inputs_embeds,
        #     cache_position,
        #     past_key_values,
        #     output_attentions
        # )

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        hidden_states = self.layer(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
            position_embeddings=position_embeddings,
        )

        print(f"hidden_states (after self.layer): {hidden_states.shape}")

        hidden_states = self.norm(hidden_states)
        print(f"hidden_states (after self.norm): {hidden_states.shape}")

        return hidden_states


# HydraForCausalLM is a thin wrapper that picks parent class for GenericHydraForCausalLM.
# All hydra functionality is defined in GenericHydraForCausalLM.
class HydraForCausalLM(PretrainedModel):
    config_class = HydraConfig

    def __init__(self, config: HydraConfig):
        super().__init__(config)

        BaseLM = QWenForCausalLM if hasattr(
            config,
            "model_type") and "qwen" in config.model_type else LLaMAForCausalLM

        class GenericHydraForCausalLM(BaseLM):

            def __init__(self, config: HydraConfig):
                super().__init__(config)
                self.num_hydra_heads = config.num_hydra_heads
                self.num_hydra_layers = config.num_hydra_layers
                self.hidden_size = config.hidden_size
                self.vocab_size = config.vocab_size
                vocab_size_padded = pad_vocab_size(self.vocab_size,
                                                   config.mapping.tp_size)

                base_kwargs = config.to_dict()
                prefix_config = BaseLM.config_class(**base_kwargs)
                self.prefix_embedding_layer = PrefixEmbeddingLayer(
                    prefix_config)

                self.hydra_heads = ModuleList([
                    HydraPrefixMLP(num_layers=self.num_hydra_layers - 1,
                                   hidden_size=config.hidden_size,
                                   vocab_size=vocab_size_padded,
                                   hydra_head_idx=i,
                                   hidden_act=config.hidden_act,
                                   dtype=config.dtype,
                                   mapping=config.mapping)
                    for i in range(self.num_hydra_heads)
                ])

                self.input_embed_fn = self.transformer.vocab_embedding
                self.max_hydra_token_len = config.max_draft_len

            def forward(self, *args, **kwargs):
                output_original = True
                hidden_states = super().forward(*args, **kwargs)

                if kwargs['use_cache']:
                    if default_net().plugin_config.paged_kv_cache:
                        lm_logits, hidden_states, _ = hidden_states
                    else:
                        lm_logits, presents, hidden_states = hidden_states

                if self.mapping.is_last_pp_rank():
                    # position_ids = arange(0, s, dtype="int32")
                    # position_ids = view(position_ids, [1, s])

                    position_ids = kwargs["position_ids"]

                    hidden_states_3d = unsqueeze(
                        hidden_states, 1)  # Shape: [B, H] -> [B, 1, H]
                    print(f"hidden_states: {hidden_states.shape}")
                    print(f"hidden_states_3d: {hidden_states_3d.shape}")

                    prefix_embedding = self.prefix_embedding_layer(
                        inputs_embeds=hidden_states_3d,
                        position_ids=position_ids,
                        attention_mask=None,
                    )

                    # prefix_embedding = prefix_embedding_output[0] if isinstance(prefix_embedding_output, tuple) else prefix_embedding_output

                    _, topk_ids = topk(lm_logits, k=1, dim=-1)
                    next_embedding = self.input_embed_fn(squeeze(topk_ids, -1))

                    # TODO: Need to convert back into for-loop
                    # prefix_embedding and next_embedding are 2D: [batch, hidden_size]
                    # prefix_embedding_3d = unsqueeze(prefix_embedding, 1) # -> [batch, 1, hidden_size]
                    next_embedding_3d = unsqueeze(
                        next_embedding, 1)  # -> [batch, 1, hidden_size]

                    print(f"prefix_embedding_3d: {prefix_embedding.shape}")
                    print(f"next_embedding_3d: {next_embedding_3d.shape}")

                    all_head_logits = []
                    # --- Head 0 ---
                    head_0_input = concat([prefix_embedding, next_embedding_3d],
                                          dim=2)
                    # head_0_input = concat([next_embedding_3d, next_embedding_3d], dim=2)
                    head_0_logits = self.hydra_heads[0](head_0_input)
                    all_head_logits.append(squeeze(head_0_logits, dim=1))

                    # --- Head 1 ---
                    _, next_token_ids_1 = topk(head_0_logits, k=1, dim=-1)
                    next_embedding_1 = self.input_embed_fn(
                        squeeze(next_token_ids_1, -1))

                    head_1_input = concat([head_0_input, next_embedding_1],
                                          dim=2)
                    head_1_logits = self.hydra_heads[1](head_1_input)
                    all_head_logits.append(squeeze(head_1_logits, dim=1))

                    # --- Head 2 ---
                    _, next_token_ids_2 = topk(head_1_logits, k=1, dim=-1)
                    next_embedding_2 = self.input_embed_fn(
                        squeeze(next_token_ids_2, -1))

                    head_2_input = concat([head_1_input, next_embedding_2],
                                          dim=2)
                    head_2_logits = self.hydra_heads[2](head_2_input)
                    all_head_logits.append(squeeze(head_2_logits, dim=1))

                    # --- Head 3 ---
                    _, next_token_ids_3 = topk(head_2_logits, k=1, dim=-1)
                    next_embedding_3 = self.input_embed_fn(
                        squeeze(next_token_ids_3, -1))

                    head_3_input = concat([head_2_input, next_embedding_3],
                                          dim=2)
                    head_3_logits = self.hydra_heads[3](head_3_input)
                    all_head_logits.append(squeeze(head_3_logits, dim=1))

                    medusa_logits = stack(all_head_logits, dim=0)

                    # medusa_logits = unsqueeze(lm_logits, 0)
                    # medusa_logits = concat([medusa_logits, medusa_logits, medusa_logits, medusa_logits], dim=0)

                    print()
                    print(f"medusa_logits.shape: {medusa_logits.shape}")
                    print(f"lm_logits.shape: {lm_logits.shape}")
                    print(f"type(medusa_logits): {type(medusa_logits)}")
                    print()

                    medusa_logits.mark_output('medusa_logits',
                                              self.config.logits_dtype)

                else:
                    hidden_states.mark_output('hidden_states_output',
                                              self.config.dtype)

                if kwargs['use_cache'] and default_net(
                ).plugin_config.paged_kv_cache == False:
                    if self.mapping.is_last_pp_rank():
                        if output_original:
                            return (medusa_logits, lm_logits, presents)
                        return (medusa_logits, presents)
                    return (hidden_states, presents)
                else:
                    if self.mapping.is_last_pp_rank():
                        if output_original:
                            return medusa_logits, lm_logits
                        return medusa_logits
                    return hidden_states

            def prepare_inputs(self, *args, **kwargs):
                kwargs['speculative_decoding_draft_tokens_external'] = False
                kwargs['max_draft_len'] = self.max_hydra_token_len
                return super().prepare_inputs(*args, **kwargs)

        self.model = GenericHydraForCausalLM(config)

    # Specialization to redirect accesses to self.model
    def __getattribute__(self, name):
        if name == 'model' or '__' in name:
            return object.__getattribute__(self, name)
        else:
            model = object.__getattribute__(self, 'model')
            return model.__getattribute__(name)

    # Override specialized __setattr__ defined in Module
    def __setattr__(self, name, value) -> None:
        object.__setattr__(self, name, value)

    @classmethod
    def from_hugging_face(
            cls,
            hf_model_or_dir: Union[str, 'transformers.PreTrainedModel'],
            dtype: str = 'auto',
            mapping: Optional[Mapping] = None,
            quant_config: Optional[QuantConfig] = None,
            **kwargs):
        import transformers

        assert hf_model_or_dir is not None
        speculative_model_dir = kwargs.get('speculative_model', None)

        use_preloading = isinstance(hf_model_or_dir,
                                    transformers.PreTrainedModel)
        if use_preloading:
            hf_model = hf_model_or_dir
            hf_config_or_dir = hf_model.config
        else:
            hf_model_dir = hf_model_or_dir
            hf_config_or_dir = hf_model_or_dir

        config = HydraConfig.from_hugging_face(hf_config_or_dir,
                                               dtype=dtype,
                                               mapping=mapping,
                                               quant_config=quant_config,
                                               **kwargs)

        # ModelOpt ckpt has combined base model and Hydra-head
        is_modelopt_ckpt = True if not speculative_model_dir else False

        if not use_preloading:
            trust_remote_code = kwargs.pop('trust_remote_code', True)

            if is_modelopt_ckpt:
                hf_model = LLaMAForCausalLM.from_hugging_face(
                    hf_model_dir,
                    dtype,
                    mapping=mapping,
                    quant_config=quant_config,
                    **kwargs)
            else:
                hf_model = AutoModelForCausalLM.from_pretrained(
                    hf_model_dir,
                    torch_dtype="auto",
                    trust_remote_code=trust_remote_code)

                assert isinstance(hf_model, transformers.PreTrainedModel)

        if is_modelopt_ckpt:
            weights = {
                name: numpy_to_torch(param.raw_value)
                for name, param in hf_model.named_parameters()
            }
        else:
            weights = convert_hf_llama(
                hf_model,
                config.mapping,
                dtype='float16',
                use_parallel_embedding=config.use_parallel_embedding)

        model = cls(config)

        if is_modelopt_ckpt:
            num_hydra_heads = config.config.num_hydra_heads
            num_hydra_layers = config.config.num_hydra_layers
            speculative_model_dir = hf_model_or_dir
        else:
            config_file = speculative_model_dir / "config.json"
            with open(config_file) as fp:
                model_config = json.load(fp)

            num_hydra_heads = kwargs[
                'speculative_config'].num_hydra_heads if 'speculative_config' in kwargs else model_config.get(
                    'hydra_num_heads', None)
            num_hydra_layers = model_config.get('hydra_num_layers', None)
        hydra_weights = load_hydra_hf(hydra_path=speculative_model_dir,
                                      num_hydra_heads=num_hydra_heads,
                                      num_hydra_layers=num_hydra_layers,
                                      mapping=mapping,
                                      dtype="float16",
                                      base_config=hf_model.config,
                                      is_modelopt_ckpt=is_modelopt_ckpt)
        weights.update(hydra_weights)
        model.load(weights)
        return model
