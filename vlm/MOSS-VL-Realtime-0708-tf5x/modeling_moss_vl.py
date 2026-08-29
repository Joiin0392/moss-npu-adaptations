# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch MossVL model - Qwen3VL Vision + Text with Cross Attention"""

import gc
import os
import time
import queue
import inspect
import warnings
from collections import deque
from dataclasses import dataclass
from typing import Dict, Any, Callable, Iterator, Optional, Union, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.generation import GenerationMixin, GenerateDecoderOnlyOutput
from transformers.generation.configuration_utils import GenerationConfig, GenerationMode
from transformers.generation.logits_process import LogitsProcessorList
from transformers.generation.stopping_criteria import StoppingCriteria, StoppingCriteriaList
from transformers.integrations import use_kernel_forward_from_hub
from transformers.masking_utils import create_causal_mask
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import BaseModelOutputWithPast, ModelOutput, CausalLMOutputWithPast
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs, auto_docstring, is_torchdynamo_compiling, logging
from transformers.utils.deprecation import deprecate_kwarg
try:
    from transformers.utils.generic import OutputRecorder
except ImportError:
    from transformers.utils.output_capturing import OutputRecorder

from .configuration_moss_vl import MossVLConfig, MossVLTextConfig, MossVLVisionConfig
import copy
import threading
from transformers.generation.stopping_criteria import StoppingCriteria, StoppingCriteriaList
from transformers.generation.streamers import TextIteratorStreamer


_OFFLINE_SYSTEM_PROMPTS = {
    "no_thinking": {
        "text_image": "You are a helpful AI assistant. Respond to the user's request based on the provided text and/or images.",
        "video": "You are a helpful AI assistant specializing in video analysis. Respond to the user's request based on the provided video content.",
    },
    "deep_thinking": {
        "text_image": "A conversation between User and Assistant. The user makes a request, and the assistant responds to it based on the provided text and/or images. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <thinking></thinking> and <answer></answer> tags, respectively, i.e., <thinking>reasoning process here</thinking><answer>answer here</answer>.",
        "video": "A conversation between User and Assistant specializing in video analysis. The user makes a request, and the assistant responds to it based on the provided video content. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <thinking></thinking> and <answer></answer> tags, respectively, i.e., <thinking>reasoning process here</thinking><answer>answer here</answer>.",
    },
}

class _OfflineCancelStoppingCriteria(StoppingCriteria):
    def __init__(self, cancel_event: threading.Event):
        self.cancel_event = cancel_event

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        return self.cancel_event.is_set()

class _OfflineQueueStreamer(TextIteratorStreamer):
    def __init__(self, tokenizer, output_text_queue: "queue.Queue[str]"):
        super().__init__(tokenizer, skip_prompt=True, skip_special_tokens=True)
        self.output_text_queue = output_text_queue
        self.collected_chunks: List[str] = []

    def on_finalized_text(self, text: str, stream_end: bool = False):
        if text:
            self.collected_chunks.append(text)
            self.output_text_queue.put(text)
        super().on_finalized_text(text, stream_end=stream_end)

_OFFLINE_THINKING_MODE_ALIASES = {
    "no_thinking": "no_thinking",
    "default": "no_thinking",
    "standard": "no_thinking",
    "deep_thinking": "deep_thinking",
    "thinking": "deep_thinking",
    "reasoning": "deep_thinking",
}

_OFFLINE_SYSTEM_PROMPT_TYPE_ALIASES = {
    "text_image": "text_image",
    "text-image": "text_image",
    "image_text": "text_image",
    "image-text": "text_image",
    "text": "text_image",
    "image": "text_image",
    "video": "video",
}



logger = logging.get_logger(__name__)


@dataclass
class MossVLModelOutputWithPast(ModelOutput):
    """
    Output class for MossVL model with additional vision_token_info and rope_deltas fields.
    
    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        past_key_values (`Cache`, *optional*):
            Contains pre-computed hidden-states (key and values in the self-attention blocks and 
            cross-attention blocks) that can be used to speed up sequential decoding.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for each layer).
        attentions (`tuple(torch.FloatTensor)`, *optional*):
            Tuple of `torch.FloatTensor` (one for each layer) of attention weights.
        vision_token_info (`List[dict]`, *optional*):
            Information about vision tokens for each sample, used to correctly expand cross-attention masks.
            This is cached during prefill and reused during decode to handle ViT padding correctly.
        rope_deltas (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Position offset due to vision tokens. Used for fast position computation in decode stage.
            rope_deltas = max_position - sequence_length
    """
    
    last_hidden_state: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Cache] = None
    hidden_states: Optional[tuple[torch.FloatTensor]] = None
    attentions: Optional[tuple[torch.FloatTensor]] = None
    vision_token_info: Optional[List[dict]] = None
    rope_deltas: Optional[torch.LongTensor] = None


@dataclass
class MossVLCausalLMOutputWithPast(ModelOutput):
    """
    Output class for MossVL causal language model with additional vision_token_info and rope_deltas fields.
    
    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*):
            Language modeling loss (for next-token prediction).
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head.
        past_key_values (`Cache`, *optional*):
            Contains pre-computed hidden-states for speed up sequential decoding.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*):
            Tuple of hidden-states at each layer.
        attentions (`tuple(torch.FloatTensor)`, *optional*):
            Tuple of attention weights.
        vision_token_info (`List[dict]`, *optional*):
            Information about vision tokens for each sample, cached for decode stage.
        rope_deltas (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Position offset due to vision tokens. Used for fast position computation in decode stage.
    """
    
    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Cache] = None
    hidden_states: Optional[tuple[torch.FloatTensor]] = None
    attentions: Optional[tuple[torch.FloatTensor]] = None
    vision_token_info: Optional[List[dict]] = None
    rope_deltas: Optional[torch.LongTensor] = None


# ==================== Vision Components (from Qwen3VL) ====================

class MossVLVisionMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.linear_fc1 = nn.Linear(self.hidden_size, self.intermediate_size, bias=True)
        self.linear_fc2 = nn.Linear(self.intermediate_size, self.hidden_size, bias=True)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_state):
        return self.linear_fc2(self.act_fn(self.linear_fc1(hidden_state)))


class MossVLVisionPatchEmbed(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.patch_size = config.patch_size
        self.temporal_patch_size = config.temporal_patch_size
        self.in_channels = config.in_channels
        self.embed_dim = config.hidden_size

        kernel_size = [self.temporal_patch_size, self.patch_size, self.patch_size]
        self.proj = nn.Conv3d(self.in_channels, self.embed_dim, kernel_size=kernel_size, stride=kernel_size, bias=True)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        target_dtype = self.proj.weight.dtype
        hidden_states = hidden_states.view(
            -1, self.in_channels, self.temporal_patch_size, self.patch_size, self.patch_size
        )
        hidden_states = self.proj(hidden_states.to(dtype=target_dtype)).view(-1, self.embed_dim)
        return hidden_states


class MossVLVisionRotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor

    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float, device="cpu") / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._inv_freq_clean = inv_freq.clone()

    def forward(self, seqlen: int) -> torch.Tensor:
        if torch.isnan(self.inv_freq).any() or torch.isinf(self.inv_freq).any():
            self.inv_freq = self._inv_freq_clean.to(self.inv_freq.device)
        seq = torch.arange(seqlen, dtype=self.inv_freq.dtype, device=self.inv_freq.device)
        freqs = torch.outer(seq, self.inv_freq)
        return freqs


class MossVLVisionPatchMerger(nn.Module):
    def __init__(self, config: MossVLVisionConfig, num_deepstack_features=0) -> None:
        super().__init__()
        # spatial_merge，维度变为原始的config.spatial_merge_size**2倍
        base_hidden_size = config.hidden_size * (config.spatial_merge_size**2)
        # 计算输入维度：spatial_merge 后的维度 * (1 + deepstack特征数)
        self.input_hidden_size = base_hidden_size * (1 + num_deepstack_features)
        
        # Use independent LayerNorms for each feature level
        # Total features = 1 (last layer) + num_deepstack_features
        num_features = 1 + num_deepstack_features
        self.norms = nn.ModuleList([
            nn.LayerNorm(config.hidden_size, eps=1e-6)
            for _ in range(num_features)
        ])
        
        self.hidden_size = config.hidden_size
        
        self.linear_fc1 = nn.Linear(self.input_hidden_size, self.input_hidden_size)
        self.act_fn = nn.GELU()
        self.linear_fc2 = nn.Linear(self.input_hidden_size, config.out_hidden_size)

    def forward(self, last_hidden_state: torch.Tensor, deepstack_features: List[torch.Tensor] = []) -> torch.Tensor:
        # 1. Collect all features: [last_hidden_state, deepstack_1, deepstack_2, ...]
        # self.norms[0] corresponds to last_hidden_state
        # self.norms[1:] corresponds to deepstack_features
        
        all_inputs = [last_hidden_state] + deepstack_features
        
        # 2. Apply Norm independently
        outs = []
        for i, feat in enumerate(all_inputs):
            outs.append(self.norms[i](feat))
        
        # 3. Concat once
        x = torch.cat(outs, dim=-1)

        # 做merge，维度变为原始的config.spatial_merge_size**2倍，len对应缩小为原来的1/config.spatial_merge_size**2
        x = x.view(-1, self.input_hidden_size)
        x = self.linear_fc2(self.act_fn(self.linear_fc1(x)))
        return x


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb_vision(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    orig_q_dtype = q.dtype
    orig_k_dtype = k.dtype
    q, k = q.float(), k.float()
    cos, sin = cos.unsqueeze(-2).float(), sin.unsqueeze(-2).float()
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    q_embed = q_embed.to(orig_q_dtype)
    k_embed = k_embed.to(orig_k_dtype)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs: Unpack[TransformersKwargs],
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


class MossVLVisionAttention(nn.Module):
    def __init__(self, config: MossVLVisionConfig) -> None:
        super().__init__()
        self.dim = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = self.dim // self.num_heads
        self.num_key_value_groups = 1
        self.qkv = nn.Linear(self.dim, self.dim * 3, bias=True)
        self.proj = nn.Linear(self.dim, self.dim)
        self.scaling = self.head_dim**-0.5
        self.config = config
        self.attention_dropout = 0.0
        self.is_causal = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        query_states, key_states, value_states = (
            self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
        )
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb_vision(query_states, key_states, cos, sin)

        query_states = query_states.transpose(0, 1).unsqueeze(0)
        key_states = key_states.transpose(0, 1).unsqueeze(0)
        value_states = value_states.transpose(0, 1).unsqueeze(0)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        if self.config._attn_implementation == "flash_attention_2":
            max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max()
            attn_output, _ = attention_interface(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask=None,
                scaling=self.scaling,
                dropout=0.0 if not self.training else self.attention_dropout,
                cu_seq_lens_q=cu_seqlens,
                cu_seq_lens_k=cu_seqlens,
                max_length_q=max_seqlen,
                max_length_k=max_seqlen,
                is_causal=False,
                **kwargs,
            )
        else:
            lengths = cu_seqlens[1:] - cu_seqlens[:-1]
            splits = [
                torch.split(tensor, lengths.tolist(), dim=2) for tensor in (query_states, key_states, value_states)
            ]

            attn_outputs = [
                attention_interface(
                    self,
                    q,
                    k,
                    v,
                    attention_mask=None,
                    scaling=self.scaling,
                    dropout=0.0 if not self.training else self.attention_dropout,
                    is_causal=False,
                    **kwargs,
                )[0]
                for q, k, v in zip(*splits)
            ]
            attn_output = torch.cat(attn_outputs, dim=1)

        attn_output = attn_output.reshape(seq_length, -1).contiguous()
        attn_output = self.proj(attn_output)
        return attn_output


class MossVLVisionBlock(GradientCheckpointingLayer):
    def __init__(self, config, attn_implementation: str = "sdpa") -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.attn = MossVLVisionAttention(config=config)
        self.mlp = MossVLVisionMLP(config=config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states),
            cu_seqlens=cu_seqlens,
            rotary_pos_emb=rotary_pos_emb,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states



# ==================== Text Components (from Qwen3 + Cross Attention) ====================

class MossVLTextRotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor

    def __init__(self, config: MossVLTextConfig, device=None):
        super().__init__()
        # BC: "rope_type" was originally "type"
        if hasattr(config, "rope_scaling") and config.rope_scaling is not None:
            self.rope_type = config.rope_scaling.get("rope_type", "default")
        else:
            self.rope_type = "default"
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        if self.rope_type == "default" or self.rope_type not in ROPE_INIT_FUNCTIONS:
            import torch as _torch
            base = config.rope_theta if hasattr(config, "rope_theta") else 10000.0
            head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
            inv_freq = 1.0 / (base ** (_torch.arange(0, head_dim, 2, device="cpu").float() / head_dim))
            self.attention_scaling = 1.0
        else:
            self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]
            inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

        # Guard: from_pretrained may corrupt inv_freq with garbage from the
        # state dict (persistent=False buffers can still be overwritten during
        # weight loading on some transformers versions). Re-validate on forward.
        self._inv_freq_computed = inv_freq.clone()


        if hasattr(config, "rope_scaling") and config.rope_scaling is not None:
             self.mrope_section = config.rope_scaling.get("mrope_section", [24, 20, 20])
        else:
             self.mrope_section = [24, 20, 20]

    def apply_interleaved_mrope(self, freqs, mrope_section):
        """Apply interleaved MRoPE to 3D rotary embeddings.
        Reorganizes frequency layout from chunked [TTT...HHH...WWW] to
        interleaved [THTHWHTHW...TT], preserving frequency continuity.
        args:
            x: (3, bs, seq_len, head_dim // 2)
            mrope_section: (3,)
        returns:
            x_t: (bs, seq_len, head_dim // 2)
        """
        freqs_t = freqs[0]  # just overwrite the first dimension T
        for dim, offset in enumerate((1, 2), start=1):  # H, W
            length = mrope_section[dim] * 3
            idx = slice(offset, length, 3)
            freqs_t[..., idx] = freqs[dim, ..., idx]
        return freqs_t

    @torch.no_grad()
    @dynamic_rope_update
    def forward(self, x, position_ids):
        # Guard: from_pretrained can corrupt inv_freq — restore from clean copy
        if torch.isnan(self.inv_freq).any() or torch.isinf(self.inv_freq).any():
            self.inv_freq = self._inv_freq_computed.to(self.inv_freq.device)
            self.original_inv_freq = self._inv_freq_computed.to(self.inv_freq.device)

        if position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)
        
        inv_freq_expanded = self.inv_freq[None, None, :, None].float().expand(3, position_ids.shape[1], -1, 1)
        position_ids_expanded = position_ids[:, :, None, :].float()  # shape (3, bs, 1, positions)

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (inv_freq_expanded.float().to(x.device) @ position_ids_expanded.float()).transpose(2, 3)
            freqs = self.apply_interleaved_mrope(freqs, self.mrope_section)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


@use_kernel_forward_from_hub("RMSNorm")
class MossVLTextRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"

# self attention rotary position embedding
def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

# cross attention rotary position embedding
def apply_rotary_pos_emb_cross_attention(states, cos, sin, position_ids=None, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    states_embed = (states * cos) + (rotate_half(states) * sin)
    return states_embed


class MossVLTextSelfAttention(nn.Module):
    """Self attention for text decoder"""

    def __init__(self, config: MossVLTextConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )
        self.q_norm = MossVLTextRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = MossVLTextRMSNorm(self.head_dim, eps=config.rms_norm_eps)

    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = False,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class MossVLTextCrossAttention(nn.Module):
    """Cross attention - for vision-text interaction"""

    def __init__(self, config: MossVLTextConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = False

        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=config.attention_bias)

        self.q_norm = MossVLTextRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = MossVLTextRMSNorm(self.head_dim, eps=config.rms_norm_eps)

    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        cross_attention_states: torch.Tensor,       
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: bool = None,
        cache_position: Optional[torch.LongTensor] = None,  # vision_cache_position
        query_position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        vision_position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> torch.Tensor:
        batch_size, seq_length, _ = hidden_states.size()
        
        # Query from text hidden states
        query_states = self.q_proj(hidden_states)
        query_states = query_states.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        query_states = self.q_norm(query_states)

        if cross_attention_states is not None:
            # Key and Value from vision cross_attention_states
            key_states = self.k_proj(cross_attention_states)
            value_states = self.v_proj(cross_attention_states)
            
            key_states = key_states.view(batch_size, -1, self.num_key_value_heads, self.head_dim).transpose(1, 2)
            key_states = self.k_norm(key_states)
            value_states = value_states.view(batch_size, -1, self.num_key_value_heads, self.head_dim).transpose(1, 2)

            # Apply different RoPE for query (text position) and key (vision position)
            if query_position_embeddings is not None:
                cos, sin = query_position_embeddings
                query_states = apply_rotary_pos_emb_cross_attention(query_states, cos, sin)
            
            if vision_position_embeddings is not None:
                vision_cos, vision_sin = vision_position_embeddings
                key_states = apply_rotary_pos_emb_cross_attention(key_states, vision_cos, vision_sin)


            if past_key_values is not None:
                # if we have a new image + new tokens, we only computed key_states on that new image
                # we still update the cross key states, past_image, new_image. And use it!
                key_states, value_states = past_key_values.update(
                    key_states, value_states, self.layer_idx, {"cache_position": cache_position}
                )     
            
        elif cache_position[0] != 0:
            key_states, value_states = (
                past_key_values.layers[self.layer_idx].keys,
                past_key_values.layers[self.layer_idx].values,
            ) 
        else:
            raise ValueError(
                "Cross attention layer can't find neither `cross_attn_states` nor cached values for key/values!"
            )

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            # 如果是flash attention，走sdpa_attention_forward
            if self.config._attn_implementation == "flash_attention_3" or self.config._attn_implementation == "flash_attention_2":
                attention_interface = ALL_ATTENTION_FUNCTIONS["sdpa"]
            else:
                attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(batch_size, seq_length, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class MossVLTextMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


class MossVLSelfAttentionDecoderLayer(GradientCheckpointingLayer):
    """Self-attention decoder layer"""

    def __init__(self, config: MossVLTextConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx

        self.self_attn = MossVLTextSelfAttention(config=config, layer_idx=layer_idx)
        self.mlp = MossVLTextMLP(config)
        self.input_layernorm = MossVLTextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = MossVLTextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        cross_attention_states: Optional[torch.Tensor] = None,
        cross_attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        full_text_row_masked_out_mask: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        vision_position_ids: Optional[torch.LongTensor] = None,
        vision_cache_position: Optional[torch.LongTensor] = None,
        vision_position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        # Self Attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )
        hidden_states = residual + hidden_states

        # MLP
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states


class MossVLCrossAttentionDecoderLayer(GradientCheckpointingLayer):
    """Cross-attention decoder layer with tanh-gated attention and MLP"""

    def __init__(self, config: MossVLTextConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx

        self.cross_attn = MossVLTextCrossAttention(config=config, layer_idx=layer_idx)
        self.mlp = MossVLTextMLP(config)
        
        self.input_layernorm = MossVLTextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = MossVLTextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        # Gates for cross attention (single scalar value).
        # Gate scalar = tanh(gate[0]), initialized to zero so tanh(0)=0 at start.
        self.cross_attn_attn_gate = nn.Parameter(torch.zeros(1))
        self.cross_attn_mlp_gate = nn.Parameter(torch.zeros(1))

    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        cross_attention_states: Optional[torch.Tensor] = None,
        cross_attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        full_text_row_masked_out_mask: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        vision_position_ids: Optional[torch.LongTensor] = None,
        vision_cache_position: Optional[torch.LongTensor] = None,
        vision_position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        # Cross Attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        
        hidden_states, _ = self.cross_attn(
            hidden_states=hidden_states,
            cross_attention_states=cross_attention_states,
            attention_mask=cross_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=vision_cache_position,
            query_position_embeddings=position_embeddings,
            vision_position_embeddings=vision_position_embeddings,
        )
        if full_text_row_masked_out_mask is not None:
            hidden_states = full_text_row_masked_out_mask[:, 0] * hidden_states

        hidden_states = residual + self.cross_attn_attn_gate.tanh() * hidden_states

        # MLP
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        if full_text_row_masked_out_mask is not None:
            hidden_states = full_text_row_masked_out_mask[:, 0] * hidden_states

        hidden_states = residual + self.cross_attn_mlp_gate.tanh() * hidden_states
        
        return hidden_states




@auto_docstring
class MossVLPreTrainedModel(PreTrainedModel):
    config: MossVLConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["MossVLSelfAttentionDecoderLayer", "MossVLCrossAttentionDecoderLayer", "MossVLVisionBlock"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn = True
    _supports_sdpa = True
    _can_compile_fullgraph = True
    _supports_attention_backend = True
    _can_record_outputs = {
        "hidden_states": [MossVLSelfAttentionDecoderLayer, MossVLCrossAttentionDecoderLayer],
        "attentions": [
            OutputRecorder(MossVLTextSelfAttention, index=1, layer_name="self_attn"),  # self-attention layers
            OutputRecorder(MossVLTextCrossAttention, index=1, layer_name="cross_attn"),  # cross-attention layers
        ],
    }

    def _init_weights(self, module):
        """Initialize the weights.
        
        Note: For loading pretrained weights:
        - Cross attention: can be initialized from the previous layer's self attention weights
        """
        std = getattr(self.config, "initializer_range", 0.02)
        if hasattr(self.config, "text_config") and hasattr(self.config.text_config, "initializer_range"):
            std = self.config.text_config.initializer_range

        if isinstance(module, MossVLVisionPatchMerger):
            # Initialize merger weights
            # Input: hidden_size * (1 + num_deepstack_features) -> Output: out_hidden_size
            # This projection handles concatenated features, so we might want specific initialization
            module.linear_fc1.weight.data.normal_(mean=0.0, std=std)
            module.linear_fc2.weight.data.normal_(mean=0.0, std=std)
            if module.linear_fc1.bias is not None:
                module.linear_fc1.bias.data.zero_()
            if module.linear_fc2.bias is not None:
                module.linear_fc2.bias.data.zero_()
            
            # Initialize separate LayerNorms
            if hasattr(module, "norms"):
                for norm in module.norms:
                    if hasattr(norm, "weight") and norm.weight is not None:
                        norm.weight.data.fill_(1.0)
                    if hasattr(norm, "bias") and norm.bias is not None:
                        norm.bias.data.zero_()





class MossVLVisionModel(MossVLPreTrainedModel):
    config: MossVLVisionConfig
    _no_split_modules = ["MossVLVisionBlock"]

    def __init__(self, config, *inputs, **kwargs) -> None:
        super().__init__(config, *inputs, **kwargs)
        self.spatial_merge_size = config.spatial_merge_size
        self.patch_size = config.patch_size
        self.spatial_merge_unit = self.spatial_merge_size * self.spatial_merge_size

        self.patch_embed = MossVLVisionPatchEmbed(config=config)
        self.pos_embed = nn.Embedding(config.num_position_embeddings, config.hidden_size)
        self.num_grid_per_side = int(config.num_position_embeddings**0.5)

        head_dim = config.hidden_size // config.num_heads
        self.rotary_pos_emb = MossVLVisionRotaryEmbedding(head_dim // 2)

        self.blocks = nn.ModuleList([MossVLVisionBlock(config) for _ in range(config.depth)])
        
        # DeepStack: 记录需要提取特征的层索引
        self.deepstack_visual_indexes = config.deepstack_visual_indexes
        num_deepstack_features = len(self.deepstack_visual_indexes)
        
        # Merger: 输入维度 = hidden_size * (1 + num_deepstack_features)
        self.merger = MossVLVisionPatchMerger(
            config=config, 
            num_deepstack_features=num_deepstack_features
        )

        self.gradient_checkpointing = False

    def rot_pos_emb(self, grid_thw: torch.Tensor) -> torch.Tensor:
        merge_size = self.spatial_merge_size
        max_hw = int(grid_thw[:, 1:].max().item())
        freq_table = self.rotary_pos_emb(max_hw)
        device = freq_table.device

        total_tokens = int(torch.prod(grid_thw, dim=1).sum().item())
        pos_ids = torch.empty((total_tokens, 2), dtype=torch.long, device=device)

        offset = 0
        for num_frames, height, width in grid_thw:
            merged_h, merged_w = height // merge_size, width // merge_size

            block_rows = torch.arange(merged_h, device=device)
            block_cols = torch.arange(merged_w, device=device)
            intra_row = torch.arange(merge_size, device=device)
            intra_col = torch.arange(merge_size, device=device)

            row_idx = block_rows[:, None, None, None] * merge_size + intra_row[None, None, :, None]
            col_idx = block_cols[None, :, None, None] * merge_size + intra_col[None, None, None, :]

            row_idx = row_idx.expand(merged_h, merged_w, merge_size, merge_size).reshape(-1)
            col_idx = col_idx.expand(merged_h, merged_w, merge_size, merge_size).reshape(-1)

            coords = torch.stack((row_idx, col_idx), dim=-1)

            if num_frames > 1:
                coords = coords.repeat(num_frames, 1)

            num_tokens = coords.shape[0]
            pos_ids[offset : offset + num_tokens] = coords
            offset += num_tokens

        embeddings = freq_table[pos_ids]
        embeddings = embeddings.flatten(1)
        return embeddings

    def fast_pos_embed_interpolate(self, grid_thw):
        grid_ts, grid_hs, grid_ws = grid_thw[:, 0], grid_thw[:, 1], grid_thw[:, 2]

        idx_list = [[] for _ in range(4)]
        weight_list = [[] for _ in range(4)]

        for t, h, w in zip(grid_ts, grid_hs, grid_ws):
            h_idxs = torch.linspace(0, self.num_grid_per_side - 1, h)
            w_idxs = torch.linspace(0, self.num_grid_per_side - 1, w)

            h_idxs_floor = h_idxs.int()
            w_idxs_floor = w_idxs.int()
            h_idxs_ceil = (h_idxs.int() + 1).clip(max=self.num_grid_per_side - 1)
            w_idxs_ceil = (w_idxs.int() + 1).clip(max=self.num_grid_per_side - 1)

            dh = h_idxs - h_idxs_floor
            dw = w_idxs - w_idxs_floor

            base_h = h_idxs_floor * self.num_grid_per_side
            base_h_ceil = h_idxs_ceil * self.num_grid_per_side

            indices = [
                (base_h[None].T + w_idxs_floor[None]).flatten(),
                (base_h[None].T + w_idxs_ceil[None]).flatten(),
                (base_h_ceil[None].T + w_idxs_floor[None]).flatten(),
                (base_h_ceil[None].T + w_idxs_ceil[None]).flatten(),
            ]

            weights = [
                ((1 - dh)[None].T * (1 - dw)[None]).flatten(),
                ((1 - dh)[None].T * dw[None]).flatten(),
                (dh[None].T * (1 - dw)[None]).flatten(),
                (dh[None].T * dw[None]).flatten(),
            ]

            for i in range(4):
                idx_list[i].extend(indices[i].tolist())
                weight_list[i].extend(weights[i].tolist())

        idx_tensor = torch.tensor(idx_list, dtype=torch.long, device=self.pos_embed.weight.device)
        weight_tensor = torch.tensor(
            weight_list, dtype=self.pos_embed.weight.dtype, device=self.pos_embed.weight.device
        )
        pos_embeds = self.pos_embed(idx_tensor) * weight_tensor[:, :, None]
        patch_pos_embeds = pos_embeds[0] + pos_embeds[1] + pos_embeds[2] + pos_embeds[3]

        patch_pos_embeds = patch_pos_embeds.split([h * w for h, w in zip(grid_hs, grid_ws)])

        patch_pos_embeds_permute = []
        merge_size = self.config.spatial_merge_size
        for pos_embed, t, h, w in zip(patch_pos_embeds, grid_ts, grid_hs, grid_ws):
            pos_embed = pos_embed.repeat(t, 1)
            pos_embed = (
                pos_embed.view(t, h // merge_size, merge_size, w // merge_size, merge_size, -1)
                .permute(0, 1, 3, 2, 4, 5)
                .flatten(0, 4)
            )
            patch_pos_embeds_permute.append(pos_embed)
        patch_pos_embeds = torch.cat(patch_pos_embeds_permute)
        return patch_pos_embeds

    def forward(
        self, 
        hidden_states: torch.Tensor, 
        grid_thw: torch.Tensor, 
        **kwargs
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: input tensor
            grid_thw: [num_images, 3] tensor with (t, h, w) for each image
        Returns:
            hidden_states: [num_tokens, out_hidden_size] - packed hidden states
        """
        hidden_states = self.patch_embed(hidden_states)

        pos_embeds = self.fast_pos_embed_interpolate(grid_thw)
        hidden_states = hidden_states + pos_embeds

        rotary_pos_emb = self.rot_pos_emb(grid_thw)

        seq_len, _ = hidden_states.size()
        hidden_states = hidden_states.reshape(seq_len, -1)
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())

        cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
            dim=0,
            dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
        )
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

        # DeepStack: 收集不同层的视觉特征
        deepstack_features = []
        for layer_idx, blk in enumerate(self.blocks):
            hidden_states = blk(
                hidden_states,
                cu_seqlens=cu_seqlens,
                position_embeddings=position_embeddings,
                **kwargs,
            )
            # 如果当前层在 deepstack 索引中，保存特征
            if layer_idx in self.deepstack_visual_indexes:
                deepstack_features.append(hidden_states)

        # Merger: 从 hidden_size * (1 + num_deepstack) 映射到 out_hidden_size
        hidden_states = self.merger(hidden_states, deepstack_features)

        return hidden_states






@auto_docstring(
    custom_intro="""
    The MossVL Text Model with self-attention and cross-attention layers for vision-language interaction.
    """
)
class MossVLTextModel(MossVLPreTrainedModel):
    config: MossVLTextConfig
    _no_split_modules = ["MossVLSelfAttentionDecoderLayer", "MossVLCrossAttentionDecoderLayer"]


    def __init__(self, config: MossVLTextConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        
        # Store cross_attention_layers for use in forward pass
        self.cross_attention_layers = config.cross_attention_layers
        
        # Create layers: self-attention or cross-attention at specified indices
        self.layers = nn.ModuleList()
        for layer_idx in range(config.num_hidden_layers):
            if layer_idx in config.cross_attention_layers:
                # Cross attention layer
                self.layers.append(
                    MossVLCrossAttentionDecoderLayer(config, layer_idx)
                )
            else:
                # Self attention layer
                self.layers.append(
                    MossVLSelfAttentionDecoderLayer(config, layer_idx)
                )
        
        self.norm = MossVLTextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = MossVLTextRotaryEmbedding(config=config)
        self.gradient_checkpointing = False

        self.post_init()


    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        full_text_row_masked_out_mask: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        cross_attention_states: Optional[torch.Tensor] = None,
        cross_attention_mask: Optional[torch.Tensor] = None,
        vision_position_ids: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        vision_cache_position: Optional[torch.LongTensor] = None, 
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Union[tuple, BaseModelOutputWithPast]:
        """
        Args:
            full_text_row_masked_out_mask (`Tuple[torch.Tensor, torch.Tensor]`, *optional*):
                Mask for full text rows that should be masked out in attention computation.
            cross_attention_states (`torch.Tensor`, *optional*):
                Vision features to be used in cross-attention layers. Shape: `(batch_size, vision_seq_len, hidden_size)`.
            cross_attention_mask (`torch.Tensor`, *optional*):
                Attention mask for cross-attention between text and vision. Shape: `(batch_size, 1, text_seq_len, vision_seq_len)`.
            vision_position_ids (`torch.LongTensor`, *optional*):
                Position IDs for vision tokens used in cross-attention. Shape: `(batch_size, vision_seq_len)`.
            vision_cache_position (`torch.LongTensor`, *optional*):
                Cache position for vision tokens. Shape: `(vision_seq_len,)`.
        """
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if use_cache and past_key_values is None and not torch.jit.is_tracing():
            past_key_values = DynamicCache(config=self.config)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )


        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        # dual-version: transformers 5.x renamed input_embeds→inputs_embeds and
        # dropped cache_position; 4.57 (this checkpoint's native line, per
        # config.json transformers_version 4.57.3) uses the old kwargs.
        try:
            attention_mask = create_causal_mask(
                config=self.config,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                position_ids=position_ids,
            )
        except TypeError:
            attention_mask = create_causal_mask(
                config=self.config,
                input_embeds=inputs_embeds,
                attention_mask=attention_mask,
                cache_position=cache_position,
                past_key_values=past_key_values,
                position_ids=position_ids,
            )

        hidden_states = inputs_embeds

        # Compute text position embeddings (for self-attention and cross-attention query)
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # Compute vision position embeddings (for cross-attention key/value) if needed
        vision_position_embeddings = None

        if vision_cache_position is None:
            # TODO：use cache_position now
            vision_cache_position = cache_position

        if cross_attention_states is not None:
            if vision_position_ids is not None:
                vision_position_embeddings = self.rotary_emb(cross_attention_states, vision_position_ids)


        for idx, decoder_layer in enumerate(self.layers):
            # For text-only path we should skip cross attention layers.
            # Let's check if the layer is cross attention layer and if we have cross attention states
            # or cached cross attention states.
            is_cross_attention_layer = idx in self.cross_attention_layers
            is_cross_attention_cache_empty = past_key_values is None or (
                past_key_values is not None and past_key_values.get_seq_length(idx) == 0
            )

            if is_cross_attention_layer and cross_attention_states is None and is_cross_attention_cache_empty:
                continue

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                full_text_row_masked_out_mask=full_text_row_masked_out_mask,
                past_key_values=past_key_values,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                cross_attention_states=cross_attention_states,
                cross_attention_mask=cross_attention_mask,
                vision_position_ids=vision_position_ids,
                vision_cache_position=vision_cache_position,
                vision_position_embeddings=vision_position_embeddings,
                **kwargs,
            )
            hidden_states = layer_outputs

        hidden_states = self.norm(hidden_states)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )


@auto_docstring(
    custom_intro="""
    The MossVL model which consists of a vision encoder (from Qwen3VL) and a language model with cross-attention layers.
    """
)
class MossVLModel(MossVLPreTrainedModel):
    base_model_prefix = ""
    config: MossVLConfig
    _no_split_modules = ["MossVLSelfAttentionDecoderLayer", "MossVLCrossAttentionDecoderLayer", "MossVLVisionBlock"]
    _checkpoint_conversion_mapping = {}
    accepts_loss_kwargs = False
    def __init__(self, config):
        super().__init__(config)
        self.visual = MossVLVisionModel._from_config(config.vision_config)
        self.language_model = MossVLTextModel._from_config(config.text_config)
        self.vision_token_info = None  # cache vision_token_info here for decode stage
        self.rope_deltas = None  # cache position deltas for decode stage 

        # Learnable Separator Token: inserted after each image/frame's vision tokens
        # Initialized from LLM's separator_token_init_id embedding
        self.separator_token = nn.Parameter(
            torch.zeros(config.vision_config.out_hidden_size)
        )

        self.post_init()
    


    def convert_packed_to_batch(
        self,
        hidden_states: torch.Tensor,
        grid_thw: torch.Tensor,
        media_nums_per_sample: Optional[List[int]],
    ) -> Tuple[torch.Tensor, List[dict]]:
        """
        Convert packed vision tokens to batched format with separator tokens.
        
        For each image: inserts 1 separator token after the vision tokens.
        For each video: inserts 1 separator token after EACH frame's vision tokens.
        
        Note: media_nums_per_sample counts each video as 1 media item, 
        but each frame in a video gets its own separator token.
        """
        
        # Calculate number of tokens per media after spatial merge
        tokens_per_media = (grid_thw[:, 0] * grid_thw[:, 1] * grid_thw[:, 2]) // (self.visual.spatial_merge_size ** 2)
        hidden_size = hidden_states.shape[-1]
        
        # If media_nums_per_sample is not provided, assume batch size = 1
        if media_nums_per_sample is None:
            batch_size = 1
            media_nums_per_sample = [grid_thw.shape[0]]
        else:
            batch_size = len(media_nums_per_sample)
        
        # Optimization for batch_size = 1 (common in inference)
        if batch_size == 1:
            # 1. Calculate total length (pure math, fast)
            total_len = 0
            for i in range(grid_thw.shape[0]):
                num_tokens = tokens_per_media[i].item()
                num_frames = grid_thw[i, 0].item()
                total_len += num_tokens + num_frames # + separators
            
            # 2. Handle Padding alignment
            pad_multiple = self.config.vision_seq_pad_multiple
            if total_len % pad_multiple != 0:
                max_seq_len = (total_len + pad_multiple - 1) // pad_multiple * pad_multiple
            else:
                max_seq_len = total_len
            
            # 3. Pre-allocate final tensor
            batched_hidden_states = torch.zeros(
                1, max_seq_len, hidden_size,
                dtype=hidden_states.dtype,
                device=hidden_states.device
            )
            
            # 4. Vectorized fill
            sample_info = {
                'medias': [],
                'total_length': total_len,
                'pad_start': total_len,
                'pad_end': max_seq_len
            }
            
            token_offset = 0
            current_seq_len = 0
            separator_embedding = self.separator_token.to(hidden_states.dtype)
            
            # Iterate through all medias in this single sample
            for media_idx in range(grid_thw.shape[0]):
                num_tokens = tokens_per_media[media_idx].item()
                t, h, w = grid_thw[media_idx].tolist()
                num_frames = t
                tokens_per_frame = num_tokens // num_frames
                
                # --- Vectorized processing start ---
                # Extract vision tokens: (num_tokens, hidden)
                media_vision_tokens = hidden_states[token_offset : token_offset + num_tokens]
                
                # Reshape to (num_frames, tokens_per_frame, hidden)
                media_vision_tokens = media_vision_tokens.view(num_frames, tokens_per_frame, hidden_size)
                              
                # Directly write to destination without creating intermediate large tensors
                chunk_len = num_frames * (tokens_per_frame + 1)
                
                # Get view of the target area: (num_frames, tokens_per_frame + 1, hidden)
                target_view = batched_hidden_states[0, current_seq_len : current_seq_len + chunk_len]
                target_view = target_view.view(num_frames, tokens_per_frame + 1, hidden_size)
                
                # 1. Fill vision tokens
                target_view[:, :tokens_per_frame].copy_(media_vision_tokens)
                
                # 2. Fill separators (Broadcast assignment)
                # separator_embedding is (hidden,), automatically broadcasts to (num_frames, hidden)
                target_view[:, tokens_per_frame] = separator_embedding
                
                # --- Vectorized processing end ---
                
                sample_info['medias'].append({
                    'start': current_seq_len,
                    'end': current_seq_len + chunk_len,
                    'length': chunk_len,
                    'num_frames': num_frames,
                    'grid_h': h,
                    'grid_w': w,
                    'vision_tokens_per_frame': tokens_per_frame,
                    'has_separator': True,
                })
                
                current_seq_len += chunk_len
                token_offset += num_tokens
            
            vision_token_info = [sample_info]
            
            return batched_hidden_states, vision_token_info

        # Calculate tokens per sample including separator tokens
        # For images: +1 separator per image
        # For videos: +num_frames separators per video (one after each frame)
        tokens_per_sample = []
        media_idx = 0
        for num_medias_in_sample in media_nums_per_sample:
            sample_tokens = 0
            for i in range(num_medias_in_sample):
                num_tokens = tokens_per_media[media_idx + i].item()
                num_frames = grid_thw[media_idx + i, 0].item()
                sample_tokens += num_tokens + num_frames  # +num_frames separator tokens
            tokens_per_sample.append(sample_tokens)
            media_idx += num_medias_in_sample
        
        max_seq_len = max(tokens_per_sample)
        pad_multiple = self.config.vision_seq_pad_multiple
        if max_seq_len % pad_multiple != 0:
            max_seq_len = (max_seq_len + pad_multiple - 1) // pad_multiple * pad_multiple
        
        # Initialize batched output with zeros (for padding)
        batched_hidden_states = torch.zeros(
            batch_size, max_seq_len, hidden_size,
            dtype=hidden_states.dtype,
            device=hidden_states.device
        )
        
        # Get separator token (learnable parameter)
        separator_embedding = self.separator_token.to(hidden_states.dtype)
        
        # Track token positions for each sample
        vision_token_info = []
        
        # Split packed tensor and fill batched output
        token_offset = 0
        media_idx = 0
        
        for sample_idx, num_medias_in_sample in enumerate(media_nums_per_sample):
            sample_info = {
                'medias': [],  # List of dicts for each media in this sample
                'total_length': tokens_per_sample[sample_idx],
                'pad_start': tokens_per_sample[sample_idx],
                'pad_end': max_seq_len
            }
            
            seq_offset = 0  # Offset within this sample's sequence
            
            # Process each image/video in this sample
            for _ in range(num_medias_in_sample):
                num_tokens = tokens_per_media[media_idx].item()
                
                t, h, w = grid_thw[media_idx].tolist()
                num_frames = t
                tokens_per_frame = num_tokens // num_frames
                
                # Record start position for this media
                media_start = seq_offset
                
                # Vectorized handling of frames
                # Extract vision tokens for this media: (num_tokens, hidden)
                media_vision_tokens = hidden_states[token_offset : token_offset + num_tokens]
                
                # Reshape to (num_frames, tokens_per_frame, hidden)
                media_vision_tokens = media_vision_tokens.view(num_frames, tokens_per_frame, hidden_size)
                
                # Create separators: (num_frames, 1, hidden)
                separators = separator_embedding.view(1, 1, hidden_size).expand(num_frames, 1, hidden_size)
                
                # Concatenate: (num_frames, tokens_per_frame + 1, hidden)
                media_tokens_with_sep = torch.cat([media_vision_tokens, separators], dim=1)
                
                # Flatten: (num_frames * (tokens_per_frame + 1), hidden)
                media_tokens_with_sep = media_tokens_with_sep.view(-1, hidden_size)
                
                # Assign to batched tensor
                media_length_with_sep = media_tokens_with_sep.shape[0]
                batched_hidden_states[sample_idx, seq_offset : seq_offset + media_length_with_sep] = media_tokens_with_sep
                
                seq_offset += media_length_with_sep
                
                # Total tokens for this media = vision_tokens + num_separators
                media_length = num_tokens + num_frames
                
                # Record this image/video's position within the sample
                # Note: length now includes separator tokens
                sample_info['medias'].append({
                    'start': media_start,
                    'end': media_start + media_length,
                    'length': media_length,
                    'num_frames': num_frames,  # 1 for image, >1 for video
                    'grid_h': h,
                    'grid_w': w,
                    'vision_tokens_per_frame': tokens_per_frame,  # Actual vision tokens per frame (excluding separator)
                    'has_separator': True,  # Flag indicating separator tokens are included
                })
                
                token_offset += num_tokens
                media_idx += 1
            
            vision_token_info.append(sample_info)
        
        return batched_hidden_states, vision_token_info

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def set_decoder(self, decoder):
        self.language_model = decoder

    def get_decoder(self):
        return self.language_model

    def _expand_cross_attention_mask(
        self,
        cross_attention_mask: torch.Tensor,
        vision_token_info: List[dict],
        target_dtype: torch.dtype,
    ) -> torch.Tensor:
        """
        Expand cross_attention_mask from (B, 1, T, N_frames) to (B, 1, T, N_tokens).
        
        Args:
            cross_attention_mask (`torch.Tensor` of shape `(batch_size, 1, text_seq_len, num_frames)`):
                Coarse attention mask where each frame corresponds to one column.
                Can be bool (True=masked) or float (min_value=masked).
            vision_token_info (`List[dict]`):
                Precomputed token info that includes actual token counts after ViT padding.
                Must be provided (either from prefill computation or from cache).
                Each dict contains 'medias' list with 'length', 'num_frames', and 'vision_tokens_per_frame'.
            target_dtype (`torch.dtype`):
                Target dtype for the output mask (typically inputs_embeds.dtype).
        
        Returns:
            `torch.Tensor` of shape `(batch_size, 1, text_seq_len, total_vision_tokens)`:
                Fine-grained attention mask where each vision token has its own column.
                Masked positions have min_value, unmasked positions have 0.0.
        
        Note:
            - vision_token_info contains the actual token counts after ViT padding (pad to multiple of 8)
            - Separator tokens are treated as part of the same frame, sharing the same mask
        """
        if vision_token_info is None:
            raise ValueError(
                "vision_token_info must be provided to _expand_cross_attention_mask. "
                "This should be cached from prefill stage or computed during current forward pass."
            )
        
        batch_size = cross_attention_mask.shape[0]
        
        # Determine target vision length (should be consistent across batch, but take max to be safe)
        max_vision_len = 0
        if vision_token_info:
             max_vision_len = max([info.get('pad_end', 0) for info in vision_token_info])

        if max_vision_len == 0:
             return None

        # Convert bool mask to float mask if needed
        if cross_attention_mask.dtype == torch.bool:
            # True = masked, False = visible
            # Convert to float: True -> min_value, False -> 0.0
            min_value = torch.finfo(target_dtype).min
            float_mask = torch.zeros_like(cross_attention_mask, dtype=target_dtype)
            float_mask.masked_fill_(cross_attention_mask, min_value)
            cross_attention_mask = float_mask
        else:
            # Already float, ensure it's the right dtype
            cross_attention_mask = cross_attention_mask.to(dtype=target_dtype)

        # Pre-allocate final mask with min_dtype (masked)
        # This is memory efficient and handles padding automatically
        min_dtype = torch.finfo(target_dtype).min
        final_mask = torch.full(
            (batch_size, 1, cross_attention_mask.shape[2], max_vision_len),
            min_dtype,
            dtype=target_dtype,
            device=cross_attention_mask.device
        )
        
        for i in range(batch_size):
            medias = vision_token_info[i]['medias']
            if not medias:
                continue
            
            # Collect repetition counts for all frames in this sample
            repeats = []
            for media in medias:
                num_frames = media.get('num_frames', 1)
                length = media['length']
                has_separator = media.get('has_separator', False)
                
                # Determine tokens per frame (including separator)
                if has_separator:
                    vision_tokens_per_frame = media.get('vision_tokens_per_frame', (length // num_frames) - 1)
                    tokens_per_frame_with_sep = vision_tokens_per_frame + 1
                else:
                    tokens_per_frame_with_sep = length // num_frames
                
                # In convert_packed_to_batch we enforce strictly regular frames
                # so we can assume all frames have the same number of tokens
                repeats.extend([tokens_per_frame_with_sep] * num_frames)
            
            num_valid_frames = len(repeats)
            if num_valid_frames == 0:
                continue
                
            # If cross_attention_mask has more frames (e.g. padded), slice it
            # If it has fewer (shouldn't happen), slice repeats
            valid_mask_frames = min(num_valid_frames, cross_attention_mask.shape[-1])
            if valid_mask_frames < num_valid_frames:
                 repeats = repeats[:valid_mask_frames]
            
            # Extract valid columns for this sample
            # (1, text_len, valid_mask_frames)
            source_mask = cross_attention_mask[i, :, :, :valid_mask_frames]
            
            # Convert repeats to tensor
            repeats_tensor = torch.tensor(repeats, device=cross_attention_mask.device)
            
            # Expand using repeat_interleave
            # output shape: (1, text_len, sum(repeats))
            expanded_mask = source_mask.repeat_interleave(repeats_tensor, dim=-1)
            
            # Assign to final_mask
            num_tokens = expanded_mask.shape[-1]
            if num_tokens > max_vision_len:
                num_tokens = max_vision_len
                expanded_mask = expanded_mask[..., :num_tokens]
                
            final_mask[i, :, :, :num_tokens] = expanded_mask
            
        return final_mask

    def compute_position_ids(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        """
        Compute 3D position IDs for text tokens with special handling for image tokens.
        
        Rules:
        - Regular text tokens: increment position (x, x, x) -> (x+1, x+1, x+1)
        - Image token: gets (t, t, t) where t = previous_text_position + 1
        - After processing vision tokens, next text token starts at max(vision_bottom_right) + 1
        
        In decode stage, uses cached rope_deltas to quickly compute new positions.
        
        Args:
            input_ids: (batch_size, seq_len)
            attention_mask: (batch_size, seq_len), optional
            cache_position: (seq_len,), position in cache
            
        Returns:
            position_ids: (3, batch_size, seq_len)
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        image_token_id = self.config.image_token_id
        
        # Decode stage: use cached rope_deltas for fast computation
        if cache_position is not None and cache_position[0] != 0 and self.rope_deltas is not None:
            # In decode, position = cache_position + rope_deltas
            # rope_deltas is per-sample: (batch_size,)
            position_ids = torch.arange(seq_len, device=device, dtype=torch.long)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)  # (batch, seq_len)
            
            # Add cache_position offset
            if cache_position is not None:
                position_ids = position_ids + cache_position[0]
            
            # Add rope_deltas (position offset due to vision tokens)
            # self.rope_deltas shape: (batch_size,), need to unsqueeze for broadcasting
            position_ids = position_ids + self.rope_deltas.unsqueeze(1)  # (batch, seq_len)
            
            # Expand to 3D: (3, batch, seq_len)
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)
            
            return position_ids
        
        # Prefill stage: compute full position_ids with image token awareness
        # Vectorized implementation
        
        # 1. Identify token types
        is_image_token = (input_ids == image_token_id)
        if attention_mask is not None:
            is_padding = (attention_mask == 0)
        else:
            is_padding = torch.zeros_like(input_ids, dtype=torch.bool)
            
        is_regular_token = ~(is_image_token | is_padding)
        
        # 2. Calculate position increments
        # Regular tokens increment position by 1
        # Image tokens do not increment position (they reuse the "current" position counter)
        # Padding tokens do not increment
        
        # cumulative sum of regular tokens gives the position index
        # We want 0-based index for the first regular token
        # cumsum: [1, 2, 2, 3] -> positions: [0, 1, 2, 2]
        # For image token at index i, we want count of regular tokens before i.
        # This is exactly (cumsum - 1) if the token itself is regular? No.
        
        # Let's use the logic: position[i] = sum(is_regular[:i])
        # We can achieve this by cumsum(is_regular) - is_regular
        
        cumulative_regular = is_regular_token.long().cumsum(dim=1)
        
        # For regular token: position = cumsum - 1 (since it's inclusive) => 0, 1, 2...
        # For image token: position = cumsum (since it's not included in cumsum, cumsum is count of prev regulars)
        # Wait, if is_regular[i] is 0, cumsum[i] == cumsum[i-1].
        # So for image token, position = cumsum[i] is correct.
        # For regular token, position = cumsum[i] - 1 is correct.
        
        # Combine: position = cumsum - is_regular.long()
        base_position_ids = cumulative_regular - is_regular_token.long()
        
        # Apply padding mask (set padding positions to 0)
        base_position_ids = base_position_ids.masked_fill(is_padding, 0)
        
        # Expand to 3D: (3, batch, seq_len)
        position_ids = base_position_ids.unsqueeze(0).expand(3, -1, -1).clone()
        
        return position_ids

    def compute_vision_position_ids(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        vision_token_info: List[dict],
        cross_attention_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute 3D position IDs for vision tokens (including separator tokens) and update text position_ids.
        Vectorized implementation for improved efficiency.
        
        Position encoding rules:
        - For text: if not image token, increment position (t-1, t-1, t-1) -> (t, t, t) -> ...
        - For vision: top-left is (t, t, t), increases towards bottom-right to (t, t+h-1, t+w-1)
        - Separator Token after each frame: (x, x, x) where x = max(t+h-1, t+w-1) + 1 = max(t+h, t+w)
        - Image token in text: also gets position (x, x, x) - same as separator
        - Next text token after image: starts at (x+1, x+1, x+1)
        
        Args:
            input_ids: (batch_size, seq_len)
            position_ids: (3, batch_size, seq_len) - will be updated in place
            vision_token_info: metadata about vision tokens (now includes separator positions)
            cross_attention_states: (batch_size, max_vision_seq_len, hidden_size)
            attention_mask: (batch_size, seq_len), optional
            
        Returns:
            vision_pos_ids: (3, batch_size, max_vision_seq_len)
            position_ids: (3, batch_size, seq_len) - updated
            rope_deltas: (batch_size,) - position offset due to vision tokens
        """
        batch_size, max_vision_seq_len, _ = cross_attention_states.shape
        device = cross_attention_states.device
        image_token_id = self.config.image_token_id
        merge_size = self.visual.spatial_merge_size

        # 1. Gather all frame metadata
        # We need to flatten the nested vision_token_info structure to align with image tokens in input_ids
        
        # Find all image tokens in text: (num_occurrences, 2) -> [batch_idx, seq_idx]
        image_token_indices = (input_ids == image_token_id).nonzero()
        
        # Flatten vision_token_info to parallel lists
        # We assume the order of medias in vision_token_info matches the appearance of image tokens in input_ids
        flat_eff_h = []
        flat_eff_w = []
        flat_vis_starts = []
        flat_batch_indices = []
        
        # Processing metadata on CPU (fast enough for typical batch sizes)
        for b_idx, info in enumerate(vision_token_info):
            medias = info.get('medias', [])
            for media in medias:
                num_frames = media['num_frames']
                h, w = media['grid_h'], media['grid_w']
                eh, ew = h // merge_size, w // merge_size
                start = media['start']
                tok_per_frame = media['vision_tokens_per_frame']
                stride = tok_per_frame + 1 # +1 for separator
                
                # Generate entries for all frames in this media
                for f in range(num_frames):
                    flat_eff_h.append(eh)
                    flat_eff_w.append(ew)
                    flat_vis_starts.append(start + f * stride)
                    flat_batch_indices.append(b_idx)

        # Pre-allocate output
        vision_pos_ids = torch.zeros(
            (3, batch_size, max_vision_seq_len), 
            dtype=torch.long, 
            device=device
        )

        # Handle case where no image tokens or info
        if len(flat_eff_h) == 0 or len(image_token_indices) == 0:
            rope_deltas = position_ids.max(dim=0).values.max(dim=-1).values + 1 - input_ids.shape[1]
            return vision_pos_ids, position_ids, rope_deltas

        # Align lengths (handle truncation if text has fewer tokens or vice versa)
        num_matches = min(len(flat_eff_h), len(image_token_indices))
        
        # Convert to tensors
        flat_eff_h = torch.tensor(flat_eff_h[:num_matches], device=device, dtype=torch.long)
        flat_eff_w = torch.tensor(flat_eff_w[:num_matches], device=device, dtype=torch.long)
        flat_vis_starts = torch.tensor(flat_vis_starts[:num_matches], device=device, dtype=torch.long)
        
        # Get corresponding text positions
        target_indices = image_token_indices[:num_matches]
        batch_rows = target_indices[:, 0]
        text_cols = target_indices[:, 1]
        
        # 2. Compute Shifts and Update Position IDs
        
        # Calculate max dimensions for each image token: separator_pos = t + max(h, w)
        # Shift amount for subsequent tokens = max(h, w) + 1
        max_hw = torch.maximum(flat_eff_h, flat_eff_w)
        shifts = max_hw + 1
        
        # Create a shift map to apply cumulative shifts
        shift_map = torch.zeros((batch_size, input_ids.shape[1]), dtype=torch.long, device=device)
        shift_map[batch_rows, text_cols] = shifts
        
        # Calculate cumulative shifts along sequence
        cum_shifts = shift_map.cumsum(dim=1)
        
        # Calculate t_vals (start position for each vision grid)
        # t_val = original_pos + shifts_before_this_image
        # cum_shifts at image index includes the image's own shift, so we subtract it
        orig_pos = position_ids[0, batch_rows, text_cols]
        shifts_before = cum_shifts[batch_rows, text_cols] - shifts
        t_vals = orig_pos + shifts_before
        
        # Update text position_ids
        # All tokens get shifted by cum_shifts
        # Image tokens specifically need to be at t_val + max_hw (which is t_val + shift - 1)
        # Our cum_shift update gives: orig_pos + shifts_before + shift = t_val + shift
        # So we subtract 1 from image tokens
        
        # Apply global shift
        # Note: position_ids is (3, B, L), cum_shifts is (B, L). Expand to match.
        new_pos_ids = position_ids + cum_shifts.unsqueeze(0)
        
        # Correct image tokens (subtract 1)
        # We can use boolean mask for efficient update
        img_token_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        img_token_mask[batch_rows, text_cols] = True
        new_pos_ids[:, img_token_mask] -= 1
        
        # Ensure padding positions remain 0 (if attention_mask provided)
        if attention_mask is not None:
             # Assuming padding is 0 in attention_mask
             padding_mask = (attention_mask == 0).unsqueeze(0)
             new_pos_ids.masked_fill_(padding_mask, 0)
        
        # Update position_ids in-place
        position_ids.copy_(new_pos_ids)
        
        # 3. Populate Vision Pos IDs
        # Group frames by size (eff_h, eff_w) to vectorize grid generation
        # This is efficient because typically there are few distinct aspect ratios
        unique_shapes = torch.unique(torch.stack([flat_eff_h, flat_eff_w], dim=1), dim=0)
        
        for shape in unique_shapes:
            eh, ew = shape[0].item(), shape[1].item()
            
            # Mask for frames of this shape
            mask = (flat_eff_h == eh) & (flat_eff_w == ew)
            
            sub_t_vals = t_vals[mask]
            sub_batch_rows = batch_rows[mask]
            sub_vis_starts = flat_vis_starts[mask]
            
            num_frames_sub = sub_t_vals.shape[0]
            if num_frames_sub == 0: continue
                
            # Generate grids: (num_frames, eh, ew)
            # y ranges 0..eh-1, x ranges 0..ew-1
            # positions: t + y, t + x
            
            y_grid = torch.arange(eh, device=device).view(1, eh, 1).expand(num_frames_sub, -1, ew)
            x_grid = torch.arange(ew, device=device).view(1, 1, ew).expand(num_frames_sub, eh, -1)
            t_grid = sub_t_vals.view(-1, 1, 1).expand(-1, eh, ew)
            
            h_grid = t_grid + y_grid
            w_grid = t_grid + x_grid
            
            # Flatten to assign
            flat_t = t_grid.reshape(-1)
            flat_h = h_grid.reshape(-1)
            flat_w = w_grid.reshape(-1)
            
            # Calculate destination indices in vision_pos_ids
            # (batch, seq_pos)
            tokens_per_frame = eh * ew
            
            # Offsets for each token in the frame 0..N-1
            seq_offsets = torch.arange(tokens_per_frame, device=device).unsqueeze(0) 
            # Add start index: (num_frames, 1) + (1, tokens) -> (num_frames, tokens)
            abs_seq_offsets = seq_offsets + sub_vis_starts.unsqueeze(1)
            
            flat_seq_inds = abs_seq_offsets.reshape(-1)
            flat_batch_inds = sub_batch_rows.unsqueeze(1).expand(-1, tokens_per_frame).reshape(-1)
            
            # Clip to max_vision_seq_len
            valid_mask = flat_seq_inds < max_vision_seq_len
            
            if valid_mask.any():
                final_b = flat_batch_inds[valid_mask]
                final_s = flat_seq_inds[valid_mask]
                
                vision_pos_ids[0, final_b, final_s] = flat_t[valid_mask]
                vision_pos_ids[1, final_b, final_s] = flat_h[valid_mask]
                vision_pos_ids[2, final_b, final_s] = flat_w[valid_mask]
                
        # 4. Handle Separator Tokens
        # Position: t_val + max(eh, ew)
        sep_vals = t_vals + max_hw
        # Index: start + tokens_per_frame = start + eh*ew
        sep_indices = flat_vis_starts + (flat_eff_h * flat_eff_w)
        
        valid_sep_mask = sep_indices < max_vision_seq_len
        
        if valid_sep_mask.any():
            final_b = batch_rows[valid_sep_mask]
            final_s = sep_indices[valid_sep_mask]
            vals = sep_vals[valid_sep_mask]
            
            vision_pos_ids[0, final_b, final_s] = vals
            vision_pos_ids[1, final_b, final_s] = vals
            vision_pos_ids[2, final_b, final_s] = vals
            
        # 5. Compute Rope Deltas
        # rope_deltas[batch_idx] = max_pos + 1 - seq_len
        
        # Use updated position_ids
        # Max pos in each batch - take max across all 3 position dimensions
        # position_ids shape: (3, batch_size, seq_len)
        # We need rope_deltas shape: (batch_size,)
        max_pos = position_ids.max(dim=0).values.max(dim=-1).values  # (batch_size,)
        rope_deltas = max_pos + 1 - input_ids.shape[1]  # (batch_size,)
        
        return vision_pos_ids, position_ids, rope_deltas

    def get_vision_features(
        self, 
        pixel_values: torch.FloatTensor, 
        grid_thw: Optional[torch.LongTensor] = None,
        media_nums_per_sample: Optional[List[int]] = None
    ):
        """
        Args:
            pixel_values: vision pixel values (images and videos merged)
            grid_thw: [num_media, 3] tensor with (t, h, w) for each media item
            media_nums_per_sample: List indicating how many media items each sample has
        Returns:
            vision_embeds: [batch_size, max_seq_len, hidden_size]
            vision_token_info: List[Dict] with media positions and padding info for each sample
        """
        pixel_values = pixel_values.type(self.visual.dtype)
        hidden_states = self.visual(
            pixel_values, 
            grid_thw=grid_thw
        )
        vision_embeds, vision_token_info = self.convert_packed_to_batch(
            hidden_states, 
            grid_thw, 
            media_nums_per_sample
        )
        return vision_embeds, vision_token_info



    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        grid_thw: Optional[torch.LongTensor] = None,
        media_nums_per_sample: Optional[List[int]] = None,
        vision_position_ids: Optional[torch.LongTensor] = None,
        cross_attention_mask: Optional[torch.Tensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        vision_cache_position: Optional[torch.LongTensor] = None,
        full_vision_token_info: Optional[List[dict]] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[tuple, BaseModelOutputWithPast]:
        """
        Args:
            grid_thw (`torch.LongTensor` of shape `(num_media, 3)`, *optional*):
                Grid size for each media item in (temporal, height, width) format. Each row contains `[t, h, w]`
                representing the number of temporal, height, and width patches for a media item (image or video).
            media_nums_per_sample (`List[int]`, *optional*):
                List indicating how many media items each sample in the batch has. For example, `[2, 1, 3]` means
                the first sample has 2 media items, the second has 1, and the third has 3.
            vision_position_ids (`torch.LongTensor` of shape `(batch_size, vision_seq_len)`, *optional*):
                Position IDs for vision tokens used in cross-attention. These are computed from text position IDs
                based on the positions of image/video tokens in the input text.
            cross_attention_mask (`torch.Tensor` of shape `(batch_size, 1, text_seq_len, vision_seq_len)`, *optional*):
                Attention mask for cross-attention between text and vision. Controls which vision tokens each text
                token can attend to, enforcing causal visibility for video frames.
            vision_cache_position (`torch.LongTensor`, *optional*):
                Cache positions (in vision token units) at which to write the new vision key/value states into the
                cross-attention KV cache. Used by real-time generation to incrementally append new frames.
            full_vision_token_info (`List[dict]`, *optional*):
                Cumulative `vision_token_info` covering all frames in the cross-attention KV cache (history +
                current). When provided, it is used to expand the frame-level `cross_attention_mask` to token-level.
                The current-turn-only metadata returned by `get_vision_features` is still used internally for the
                newly-encoded `cross_attention_states`. Used by real-time generation.
        """
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        # Process vision features (images and videos are already merged by processor)
        cross_attention_states = None
        current_vision_token_info = None

        if pixel_values is not None:
            # Determine batch size
            batch_size = inputs_embeds.shape[0]

            # Get default media_nums_per_sample if not provided
            if media_nums_per_sample is None:
                # Assume all media belong to first sample if batch_size=1, otherwise raise error
                if batch_size == 1:
                    media_nums_per_sample = [grid_thw.shape[0]]
                else:
                    raise ValueError("media_nums_per_sample must be provided when batch_size > 1")

            # Process all vision inputs together through VIT
            # pixel_values and grid_thw are already ordered by appearance in text
            vision_chunked_length = kwargs.pop("vision_chunked_length", None)  # offline path needs this; None keeps behavior identical to the original
            vision_embeds, current_vision_token_info = self.get_vision_features_chunked(
                pixel_values, grid_thw, media_nums_per_sample,
                vision_chunked_length=vision_chunked_length,
            )

            # vision_embeds: [batch_size, max_seq_len, hidden_size]
            cross_attention_states = vision_embeds.to(inputs_embeds.device, inputs_embeds.dtype)

            # Cache vision_token_info for decode stage (prefill only)
            # In real-time mode the caller manages cumulative metadata via `full_vision_token_info`, so we
            # only refresh `self.vision_token_info` when the caller is NOT managing it externally.
            if full_vision_token_info is None:
                self.vision_token_info = current_vision_token_info

        # Pick the metadata used for cross-attention mask expansion:
        # - real-time: caller-supplied cumulative `full_vision_token_info`
        # - offline: cached `self.vision_token_info`
        mask_vision_token_info = (
            full_vision_token_info if full_vision_token_info is not None else self.vision_token_info
        )

        # Generate 3D position IDs for text if not provided
        if position_ids is None:
            # Compute position IDs with image token awareness
            # In decode stage, this uses cached rope_deltas for fast computation
            position_ids = self.compute_position_ids(
                input_ids=input_ids,
                attention_mask=attention_mask,
                cache_position=cache_position,
            )

        # Compute cross_attention_mask, vision_position_ids, and full_text_row_masked_out_mask
        full_text_row_masked_out_mask = None

        if cross_attention_mask is not None:
            # Expand mask from frame-level to token-level
            # The processor outputs coarse masks (bool or float) where each frame has one column,
            # we need to expand to fine-grained masks where each vision token has its own column
            # This function also converts bool to float with correct min/max values
            cross_attention_mask = self._expand_cross_attention_mask(
                cross_attention_mask,
                mask_vision_token_info,
                target_dtype=inputs_embeds.dtype
            )

            # Handle full_text_row_masked_out_mask logic
            if cross_attention_mask is not None:
                negative_inf_value = torch.finfo(cross_attention_mask.dtype).min
                full_text_row_masked_out_mask = (
                    (cross_attention_mask != negative_inf_value).any(dim=-1).type_as(cross_attention_mask)[..., None]
                )
                cross_attention_mask = cross_attention_mask * full_text_row_masked_out_mask



        if vision_position_ids is None and cross_attention_states is not None and input_ids is not None:
            # `compute_vision_position_ids` assumes the image_pad tokens in `input_ids` align with the medias in
            # `current_vision_token_info` (offline prefill semantics). Real-time callers pre-compute
            # `vision_position_ids` so this branch is skipped.
            vision_position_ids, position_ids, rope_deltas = self.compute_vision_position_ids(
                input_ids,
                position_ids,
                current_vision_token_info,
                cross_attention_states,
                attention_mask
            )

            # Cache rope_deltas for decode stage (only in prefill)
            # rope_deltas = max_position - sequence_length
            # This allows fast position computation in decode: position = cache_position + rope_deltas
            if cache_position is not None and cache_position[0] == 0:
                self.rope_deltas = rope_deltas



        outputs = self.language_model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            vision_cache_position=vision_cache_position,
            cross_attention_states=cross_attention_states,
            cross_attention_mask=cross_attention_mask,
            vision_position_ids=vision_position_ids,
            full_text_row_masked_out_mask=full_text_row_masked_out_mask,
            **kwargs,
        )

        return MossVLModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            vision_token_info=mask_vision_token_info,
            rope_deltas=self.rope_deltas,
        )

    def get_vision_features_chunked(
        self,
        pixel_values: torch.FloatTensor,
        grid_thw: Optional[torch.LongTensor] = None,
        media_nums_per_sample: Optional[List[int]] = None,
        vision_chunked_length: Optional[int] = None,
    ):
        """
        Chunk the visual encoder forward by media items, then reuse the same
        packed-to-batch conversion logic. This keeps output semantics identical
        to `get_vision_features(...)` while reducing prefill memory pressure.
        """
        if (
            vision_chunked_length is None
            or vision_chunked_length <= 0
            or grid_thw is None
            or grid_thw.shape[0] <= vision_chunked_length
        ):
            return self.get_vision_features(pixel_values, grid_thw, media_nums_per_sample)

        pixel_values = pixel_values.type(self.visual.dtype)
        token_counts = (grid_thw[:, 0] * grid_thw[:, 1] * grid_thw[:, 2]).tolist()

        hidden_state_chunks = []
        token_offset = 0
        for media_start in range(0, grid_thw.shape[0], vision_chunked_length):
            media_end = min(media_start + vision_chunked_length, grid_thw.shape[0])
            chunk_grid_thw = grid_thw[media_start:media_end]
            chunk_token_count = sum(token_counts[media_start:media_end])
            chunk_pixel_values = pixel_values[token_offset:token_offset + chunk_token_count]
            token_offset += chunk_token_count

            hidden_state_chunks.append(
                self.visual(
                    chunk_pixel_values,
                    grid_thw=chunk_grid_thw,
                )
            )

        hidden_states = torch.cat(hidden_state_chunks, dim=0)
        vision_embeds, vision_token_info = self.convert_packed_to_batch(
            hidden_states,
            grid_thw,
            media_nums_per_sample,
        )
        return vision_embeds, vision_token_info





class MossVLRealtimeSession:
    """Thread-backed deployment wrapper around `real_time_generate`.

    The session owns the realtime queues and exposes a small imperative API:
    push frames, push prompts, and poll generated text chunks. Video capture and
    decoding stay outside the model, so service code can feed PIL-compatible
    frames from camera, stream, or file readers.
    """

    def __init__(
        self,
        model: "MossVLForConditionalGeneration",
        processor: Any,
        *,
        initial_prompt: str = "",
        system_prompt: Optional[str] = None,
        frame_queue_size: int = 256,
        max_tokens_per_turn: int = 86400,
        **generate_kwargs: Any,
    ) -> None:
        if frame_queue_size < 1:
            raise ValueError("frame_queue_size must be at least 1")
        if max_tokens_per_turn < 1:
            raise ValueError("max_tokens_per_turn must be at least 1")

        self.model = model
        self.processor = processor
        self.initial_prompt = str(initial_prompt or "")
        self.system_prompt = system_prompt
        self.max_tokens_per_turn = int(max_tokens_per_turn)
        self.generate_kwargs = dict(generate_kwargs)

        self._frame_queue: queue.Queue = queue.Queue(maxsize=frame_queue_size)
        self._prompt_queue: queue.Queue = queue.Queue()
        self._output_queue: queue.Queue = queue.Queue()
        self._thread: Optional[threading.Thread] = None
        self._done = threading.Event()
        self._closed = threading.Event()
        self._state_lock = threading.Lock()
        self._started_at: Optional[float] = None
        self._last_timestamp: Optional[float] = None
        self._error: Optional[BaseException] = None

    @property
    def active(self) -> bool:
        return bool(self._thread is not None and self._thread.is_alive() and not self._done.is_set())

    @property
    def pending_frames(self) -> int:
        return self._frame_queue.qsize()

    def start(self) -> "MossVLRealtimeSession":
        with self._state_lock:
            if self._closed.is_set():
                raise RuntimeError("Cannot start a closed realtime session")
            if self._thread is not None:
                return self
            if self.model.continue_generating:
                raise RuntimeError("This model already has an active realtime generation loop")

            self._started_at = time.monotonic()
            self._thread = threading.Thread(
                target=self._run,
                name="mossvl-realtime",
                daemon=True,
            )
            self._thread.start()
        return self

    def _run(self) -> None:
        try:
            system_prompt = self.system_prompt
            if system_prompt is None:
                system_prompt = (
                    "You are a helpful AI assistant specializing in real-time video analysis. "
                    "The video streams to you frame by frame. At every frame, you decide independently "
                    "whether to respond or stay silent — output `<|silence|>` when nothing relevant has happened, "
                    "and respond when the visual content warrants it."
                )
            initial_messages = [
                {"role": "system", "content": str(system_prompt)},
                {"role": "user", "content": self.initial_prompt},
            ]
            initial_input_ids = self.processor.tokenizer.apply_chat_template(
                initial_messages,
                add_generation_prompt=True,
                tokenize=True,
                return_tensors="pt",
            ).to(self.model.device)
            initial_attention_mask = torch.ones_like(initial_input_ids)
            prefill_len = initial_input_ids.shape[1]
            prefill_positions = (
                torch.arange(prefill_len, dtype=torch.long, device=self.model.device)
                .view(1, 1, prefill_len)
                .expand(3, 1, prefill_len)
                .contiguous()
            )

            self.model.start_real_time_generate()
            try:
                self.model._real_time_generate(
                    new_video_frames=self._frame_queue,
                    new_prompts=self._prompt_queue,
                    output_text_queue=self._output_queue,
                    processor=self.processor,
                    inputs=initial_input_ids,
                    attention_mask=initial_attention_mask,
                    position_ids=prefill_positions,
                    realtime_next_position=prefill_len,
                    full_vision_token_info=None,
                    max_tokens_per_turn=self.max_tokens_per_turn,
                    **self.generate_kwargs,
                )
            finally:
                self.model.stop_real_time_generate()
        except BaseException as exc:
            self._error = exc
        finally:
            self._done.set()

    @staticmethod
    def _event_frame_timestamps(event: Any) -> List[float]:
        if isinstance(event, dict):
            event_type = event.get("type")
            if event_type == "batch":
                timestamps: List[float] = []
                for child in event.get("events") or []:
                    timestamps.extend(MossVLRealtimeSession._event_frame_timestamps(child))
                return timestamps
            if event_type == "frame":
                timestamp = event.get("timestamp")
                return [float(timestamp)] if timestamp is not None else []
            return []
        if isinstance(event, (tuple, list)) and len(event) >= 2:
            return [float(event[1])]
        return []

    @staticmethod
    def _event_has_prompt(event: Any) -> bool:
        if isinstance(event, dict):
            event_type = event.get("type")
            if event_type == "batch":
                return any(MossVLRealtimeSession._event_has_prompt(child) for child in event.get("events") or [])
            return event_type == "prompt"
        return False

    def push_event(self, event: Any, *, drop_oldest: bool = True) -> bool:
        """Translate a board-style event into the model's original frame/prompt queues."""
        if self._closed.is_set():
            raise RuntimeError("Realtime session is closed")
        if not isinstance(event, dict):
            image, timestamp = event
            return self.push_frame(image, timestamp=float(timestamp), drop_oldest=drop_oldest)

        event_type = event.get("type")
        if event_type == "batch":
            dropped = False
            for child in event.get("events") or []:
                dropped = self.push_event(child, drop_oldest=drop_oldest) or dropped
            return dropped
        if event_type == "frame":
            image = event.get("image", event.get("frame"))
            timestamp = event.get("timestamp")
            return self.push_frame(image, timestamp=timestamp, drop_oldest=drop_oldest)
        if event_type == "prompt":
            self.push_prompt(event.get("prompt", ""))
            return False
        raise ValueError(f"Unsupported realtime event type: {event_type!r}")

    def push_prompt_frame(
        self,
        prompt: str,
        image: Any,
        timestamp: Optional[float] = None,
        *,
        drop_oldest: bool = True,
    ) -> bool:
        """Queue an aligned frame and user turn through the original model queues."""
        dropped = self.push_frame(image, timestamp=timestamp, drop_oldest=drop_oldest)
        self.push_prompt(prompt)
        return dropped

    def push_frame(
        self,
        image: Any,
        timestamp: Optional[float] = None,
        *,
        drop_oldest: bool = True,
    ) -> bool:
        """Append one frame and return whether an older queued frame was dropped."""
        if self._closed.is_set():
            raise RuntimeError("Realtime session is closed")
        if image is None:
            raise ValueError("image must not be None")

        if self._started_at is None:
            self.start()
        if timestamp is None:
            timestamp = time.monotonic() - self._started_at
        timestamp = float(timestamp)

        with self._state_lock:
            if self._last_timestamp is not None and timestamp < self._last_timestamp:
                raise ValueError(
                    f"Frame timestamps must be non-decreasing: {timestamp} < {self._last_timestamp}"
                )
            self._last_timestamp = timestamp

        item = (image, timestamp)
        try:
            self._frame_queue.put_nowait(item)
            return False
        except queue.Full:
            if not drop_oldest:
                raise

        dropped = False
        while True:
            try:
                self._frame_queue.get_nowait()
                dropped = True
            except queue.Empty:
                pass
            try:
                self._frame_queue.put_nowait(item)
                return dropped
            except queue.Full:
                continue

    def push_prompt(self, prompt: str) -> None:
        if self._closed.is_set():
            raise RuntimeError("Realtime session is closed")
        prompt = str(prompt or "")
        if not prompt:
            raise ValueError("prompt must not be empty")
        if self._thread is None:
            self.start()
        self._prompt_queue.put_nowait(prompt)

    def poll_output(self, timeout: float = 0.0) -> Optional[str]:
        if timeout < 0:
            raise ValueError("timeout must be non-negative")
        if self._thread is None:
            self.start()

        try:
            if timeout == 0:
                item = self._output_queue.get_nowait()
            else:
                item = self._output_queue.get(timeout=timeout)
        except queue.Empty:
            self._raise_if_failed()
            return None
        return str(item)

    def stream_outputs(self, poll_interval: float = 0.1) -> Iterator[str]:
        if poll_interval <= 0:
            raise ValueError("poll_interval must be positive")
        if self._thread is None:
            self.start()

        while not self._done.is_set() or not self._output_queue.empty():
            item = self.poll_output(timeout=poll_interval)
            if item is not None:
                yield item
        self._raise_if_failed()

    def _raise_if_failed(self) -> None:
        if self._done.is_set() and self._error is not None:
            raise RuntimeError("MOSS-VL realtime generation failed") from self._error

    def close(self, timeout: Optional[float] = 5.0) -> None:
        self._closed.set()
        if self._thread is None:
            self._done.set()
            return

        self.model.stop_real_time_generate()
        self._thread.join(timeout=timeout)
        if self._thread.is_alive():
            raise TimeoutError("Realtime generation did not stop before the timeout")
        self._raise_if_failed()

    def __enter__(self) -> "MossVLRealtimeSession":
        return self.start()

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()




@auto_docstring(
    custom_intro="""
    The MossVL model with a language modeling head on top, for conditional generation tasks.
    Combines Qwen3VL vision encoder with LLM via cross-attention layers.
    """
)
class MossVLForConditionalGeneration(MossVLPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]
    config: MossVLConfig
    _checkpoint_conversion_mapping = {}
    accepts_loss_kwargs = False
    def __init__(self, config):
        super().__init__(config)
        self.model = MossVLModel(config)
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)

        # Real-time generation control flag. The realtime sample loop polls this every step;
        # `stop_real_time_generate()` flips it to False to break out of the loop.
        self.continue_generating = False

        self.post_init()

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def set_decoder(self, decoder):
        self.model.set_decoder(decoder)

    def get_decoder(self):
        return self.model.get_decoder()


    def get_vision_features(
        self, 
        pixel_values: torch.FloatTensor, 
        grid_thw: Optional[torch.LongTensor] = None,
        media_nums_per_sample: Optional[List[int]] = None
    ):
        """
        Get vision features for images and videos (merged).
        
        Args:
            pixel_values: vision pixel values (images and videos merged)
            grid_thw: [num_media, 3] tensor with (t, h, w) for each media item
            media_nums_per_sample: List indicating how many media items each sample has
        Returns:
            vision_embeds: [batch_size, max_seq_len, hidden_size]
            vision_token_info: List[Dict] with media positions and padding info for each sample
        """
        return self.model.get_vision_features(pixel_values, grid_thw, media_nums_per_sample)

    @property
    def language_model(self):
        return self.model.language_model

    @property
    def visual(self):
        return self.model.visual

    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        grid_thw: Optional[torch.LongTensor] = None,
        media_nums_per_sample: Optional[List[int]] = None,
        vision_position_ids: Optional[torch.LongTensor] = None,
        cross_attention_mask: Optional[torch.Tensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        vision_chunked_length: Optional[int] = None,  # offline path needs this; None keeps behavior identical to the original
        vision_cache_position: Optional[torch.LongTensor] = None,
        full_vision_token_info: Optional[List[dict]] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[tuple, CausalLMOutputWithPast]:
        """
        Args:
            grid_thw (`torch.LongTensor` of shape `(num_media, 3)`, *optional*):
                Grid size for each media item in (temporal, height, width) format. Each row contains `[t, h, w]`
                representing the number of temporal, height, and width patches for a media item (image or video).
            media_nums_per_sample (`List[int]`, *optional*):
                List indicating how many media items each sample in the batch has. For example, `[2, 1, 3]` means
                the first sample has 2 media items, the second has 1, and the third has 3.
            vision_position_ids (`torch.LongTensor` of shape `(batch_size, vision_seq_len)`, *optional*):
                Position IDs for vision tokens used in cross-attention. These are computed from text position IDs
                based on the positions of image/video tokens in the input text.
            cross_attention_mask (`torch.Tensor` of shape `(batch_size, 1, text_seq_len, vision_seq_len)`, *optional*):
                Attention mask for cross-attention between text and vision. Controls which vision tokens each text
                token can attend to, enforcing causal visibility for video frames.
            vision_cache_position / full_vision_token_info: see `MossVLModel.forward`. Used by real-time generation
                to incrementally append new frames to the cross-attention KV cache.
        """
        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            grid_thw=grid_thw,
            media_nums_per_sample=media_nums_per_sample,
            position_ids=position_ids,
            attention_mask=attention_mask,
            vision_position_ids=vision_position_ids,
            cross_attention_mask=cross_attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            vision_chunked_length=vision_chunked_length,
            vision_cache_position=vision_cache_position,
            full_vision_token_info=full_vision_token_info,
            **kwargs,
        )

        hidden_states = outputs[0]

        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.text_config.vocab_size)

        return MossVLCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            vision_token_info=outputs.vision_token_info,
            rope_deltas=outputs.rope_deltas,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        pixel_values=None,
        grid_thw=None,
        media_nums_per_sample=None,  # One video is one meida.
        vision_position_ids=None,
        cross_attention_mask=None,
        vision_chunked_length=None,  # offline path needs this; None keeps behavior identical to the original
        **kwargs,
    ):
        """
        Prepare inputs for generation.
        
        Note: Currently only supports offline visual understanding, meaning all multimodal 
        content must be provided before generation starts. We don't support adding new 
        images/videos during generation (streaming mode).
        
        Args:
            media_nums_per_sample: One video counts as one media item (regardless of frame count)
        """
        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            position_ids=position_ids,
            pixel_values=pixel_values,
            grid_thw=grid_thw,
            media_nums_per_sample=media_nums_per_sample,
            use_cache=use_cache,
            **kwargs,
        )


        # For decoding stage, if position_ids are generated by GenerationMixin (2D),
        # we can set them to None to let forward recompute them from cache_position.
        model_inputs["position_ids"] = None
        
        # Handle cross attention mask
        if cross_attention_mask is not None:
            # Slice to current sequence length on text dimension (dim=2)
            # Shape: [batch, 1, text_len, vision_len] -> [batch, 1, cache_len, vision_len]
            cross_attention_mask = cross_attention_mask[:, :, -cache_position.shape[0]:, :]
            model_inputs["cross_attention_mask"] = cross_attention_mask

        # Vision inputs are only needed in prefill stage (cache_position[0] == 0)
        # In decode stage, vision features are retrieved from cross attention cache
        if cache_position[0] != 0:
            model_inputs["pixel_values"] = None
            model_inputs["grid_thw"] = None 
            model_inputs["media_nums_per_sample"] = None
            model_inputs["vision_position_ids"] = None
            
        else:
            # In prefill stage, include all vision-related inputs
            model_inputs["vision_position_ids"] = vision_position_ids

        model_inputs["vision_chunked_length"] = vision_chunked_length

        return model_inputs

    def _update_model_kwargs_for_generation(self, outputs, model_kwargs, is_encoder_decoder, **kwargs):
        """
        Update model kwargs for generation, extending cross_attention_mask for the newly generated token.
        
        In offline mode (all multimodal content provided before generation):
        - Each newly generated token should have the same cross_attention_mask pattern as the previous token
        - This ensures all generated tokens can attend to all vision tokens that were visible before
        """
        cross_attention_mask_prev = model_kwargs.get("cross_attention_mask", None)
        
        model_kwargs = super()._update_model_kwargs_for_generation(
            outputs=outputs,
            model_kwargs=model_kwargs,
            is_encoder_decoder=is_encoder_decoder,
            **kwargs,
        )
        
        # Extend cross_attention_mask for the new token
        # Copy the last token's mask pattern for the newly generated token
        if cross_attention_mask_prev is not None:
            model_kwargs["cross_attention_mask"] = torch.cat(
                [cross_attention_mask_prev, cross_attention_mask_prev[:, :, -1:, :]],
                dim=2  # Concatenate along text sequence dimension
            )

        return model_kwargs

    # =====================================================================
    # Real-time generation
    #
    # The methods below extend HuggingFace `generate()` with a streaming variant: at every decoded
    # token we drain external queues for new prompts / new (image, timestamp) frames, splice them
    # into the running context, and continue. Constraints (mirroring the VideoMllama reference):
    #   - batch size 1 only;
    #   - greedy / sampling only — no beam search, static cache, assisted generation, synced GPUs;
    #   - stops when `self.continue_generating` flips to False (not via official stopping criteria).
    # =====================================================================

    def create_realtime_session(
        self,
        processor: Any,
        *,
        initial_prompt: str = "",
        system_prompt: Optional[str] = None,
        frame_queue_size: int = 256,
        max_tokens_per_turn: int = 86400,
        **generate_kwargs: Any,
    ) -> MossVLRealtimeSession:
        """Create a deployment-friendly realtime session wrapper."""
        return MossVLRealtimeSession(
            self,
            processor,
            initial_prompt=initial_prompt,
            system_prompt=system_prompt,
            frame_queue_size=frame_queue_size,
            max_tokens_per_turn=max_tokens_per_turn,
            **generate_kwargs,
        )

    def online_generate(
        self,
        processor,
        new_queries: "queue.Queue[dict]",
        output_text_queue: "queue.Queue[str]",
        frame_queue_size: int = 256,
        max_tokens_per_turn: int = 86400,
        **generate_kwargs,
    ) -> None:
        """Queue-style realtime inference wrapper for service deployment.

        This mirrors the board backend event protocol: a query can carry frames,
        prompts, or a prompt aligned with a frame. When prompt and frame appear
        in the same query, they are pushed as one `batch(frame, prompt)` event so
        the model resumes from the assistant prompt while attending to that frame.
        """
        session: Optional[MossVLRealtimeSession] = None

        def _ensure_session(query: Dict[str, Any]) -> MossVLRealtimeSession:
            nonlocal session
            if session is not None and not query.get("reset_session") and not query.get("clear_history"):
                return session

            if session is not None:
                session.close()

            session_generate_kwargs = dict(generate_kwargs)
            session_generate_kwargs.update(dict(query.get("generate_kwargs") or {}))
            session = self.create_realtime_session(
                processor,
                initial_prompt=query.get("initial_prompt", ""),
                system_prompt=query.get("system_prompt"),
                frame_queue_size=int(query.get("frame_queue_size", frame_queue_size)),
                max_tokens_per_turn=int(query.get("max_tokens_per_turn", max_tokens_per_turn)),
                **session_generate_kwargs,
            )
            session.start()
            return session

        def _drain_outputs(raise_errors: bool = True) -> Optional[Exception]:
            if session is None:
                return None
            while True:
                try:
                    item = session.poll_output(timeout=0.0)
                except Exception as exc:
                    if raise_errors:
                        raise
                    return exc
                if item is None:
                    break
                output_text_queue.put(item)
            return None

        def _normalize_frame_item(item: Any) -> Tuple[Any, Optional[float]]:
            if isinstance(item, dict):
                image = item.get("image", item.get("frame"))
                timestamp = item.get("timestamp", item.get("time"))
                return image, None if timestamp is None else float(timestamp)
            if isinstance(item, (tuple, list)) and len(item) >= 2:
                image, timestamp = item[0], item[1]
                return image, None if timestamp is None else float(timestamp)
            return item, None

        try:
            while True:
                try:
                    query = new_queries.get(timeout=0.05)
                except queue.Empty:
                    _drain_outputs()
                    continue

                if not isinstance(query, dict):
                    continue

                if (
                    query.get("stop_online_generate")
                    or query.get("stop_realtime_generate")
                    or query.get("stop_generation")
                ):
                    break

                current_session = _ensure_session(query)
                drop_oldest = bool(query.get("drop_oldest", True))

                if isinstance(query.get("event"), dict):
                    current_session.push_event(query["event"], drop_oldest=drop_oldest)
                    _drain_outputs()
                    continue
                if isinstance(query.get("events"), list):
                    current_session.push_event({"type": "batch", "events": query["events"]}, drop_oldest=drop_oldest)
                    _drain_outputs()
                    continue

                prompt = query.get("prompt", query.get("text"))
                prompt = None if prompt is None else str(prompt)

                frame_items: List[Any] = []
                if "frames" in query and query["frames"] is not None:
                    frames = query["frames"]
                    if isinstance(frames, (list, tuple)):
                        frame_items.extend(frames)
                    else:
                        frame_items.append(frames)
                if "image" in query:
                    frame_items.append({"image": query.get("image"), "timestamp": query.get("timestamp")})
                if "frame" in query:
                    frame_items.append({"frame": query.get("frame"), "timestamp": query.get("timestamp")})

                if prompt and frame_items:
                    events = []
                    for frame_item in frame_items:
                        image, timestamp = _normalize_frame_item(frame_item)
                        if image is None:
                            raise ValueError("frame/image must not be None")
                        if timestamp is None:
                            if current_session._started_at is None:
                                current_session.start()
                            timestamp = time.monotonic() - current_session._started_at
                        events.append({"type": "frame", "image": image, "timestamp": float(timestamp)})
                    events.append({"type": "prompt", "prompt": prompt})
                    current_session.push_event({"type": "batch", "events": events}, drop_oldest=drop_oldest)
                else:
                    for frame_item in frame_items:
                        image, timestamp = _normalize_frame_item(frame_item)
                        current_session.push_frame(image, timestamp=timestamp, drop_oldest=drop_oldest)
                    if prompt:
                        current_session.push_prompt(prompt)

                _drain_outputs()
        except Exception as exc:
            output_text_queue.put(f"[ERROR] {exc}")
        finally:
            if session is not None:
                close_error = None
                try:
                    session.close()
                except Exception as exc:
                    close_error = exc
                drain_error = _drain_outputs(raise_errors=False)
                if close_error is not None:
                    output_text_queue.put(f"[ERROR] {close_error}")
                elif drain_error is not None:
                    output_text_queue.put(f"[ERROR] {drain_error}")

    def start_real_time_generate(self):
        self.continue_generating = True

    def stop_real_time_generate(self):
        gc.collect()
        try:
            import torch_npu  # noqa: F401
            if torch.npu.is_available():
                torch.npu.empty_cache()
        except ImportError:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        self.continue_generating = False

    @staticmethod
    def _compute_realtime_mrope_for_segment(
        new_input_ids: torch.LongTensor,
        new_grid_thw: Optional[torch.LongTensor],
        start_position: int,
        image_token_id: int,
        merge_size: int,
    ):
        """
        Compute 3D MRoPE positions for a single real-time text segment that may contain new
        `<|image_pad|>` tokens, plus the vision-token positions for each new frame.

        This mirrors the offline logic in `MossVLModel.compute_position_ids` +
        `compute_vision_position_ids` but applies it incrementally so that mid-decode insertion of
        image tokens does not depend on the offline `rope_deltas` fast path.

        Args:
            new_input_ids: (1, segment_len) — only the new tokens being appended in this turn.
            new_grid_thw: (num_new_frames, 3) — (t, h, w) per frame; aligned in order with the
                `<|image_pad|>` occurrences in `new_input_ids`. None if no new frames this turn.
            start_position: the MRoPE position to assign to the first new text token (= one past
                the position assigned to the previously-generated token, before any new frame
                grid shifts).
            image_token_id: id of `<|image_pad|>`.
            merge_size: vision spatial merge size (eh = grid_h // merge_size, ew = grid_w // merge_size).

        Returns:
            text_position_ids: (3, 1, segment_len) — final positions for the new text tokens.
            vision_position_ids: (3, 1, sum_frame_vision_lens) or None — positions for the new
                vision tokens (each frame contributes eh*ew + 1 entries: the grid then the
                separator). None when there are no new frames.
            next_position: int — the MRoPE position the next text token should use.
            grid_summary: list of (eh, ew) per new frame, in order (useful for tests).
        """
        device = new_input_ids.device
        seq_len = new_input_ids.shape[1]
        text_pos = torch.zeros((3, 1, seq_len), dtype=torch.long, device=device)
        vision_pos_chunks = []  # list of (3, eh*ew + 1) tensors
        grid_summary = []

        cur_pos = start_position
        frame_idx = 0
        num_new_frames = new_grid_thw.shape[0] if new_grid_thw is not None else 0

        for t_idx in range(seq_len):
            token_id = int(new_input_ids[0, t_idx].item())
            if token_id == image_token_id and frame_idx < num_new_frames:
                grid = new_grid_thw[frame_idx]
                grid_h = int(grid[1].item())
                grid_w = int(grid[2].item())
                eh = grid_h // merge_size
                ew = grid_w // merge_size
                max_hw = max(eh, ew)
                grid_summary.append((eh, ew))

                # Vision grid top-left = cur_pos
                y = torch.arange(eh, device=device, dtype=torch.long)
                x = torch.arange(ew, device=device, dtype=torch.long)
                yy = y.view(eh, 1).expand(eh, ew).reshape(-1)
                xx = x.view(1, ew).expand(eh, ew).reshape(-1)
                grid_t = torch.full((eh * ew,), cur_pos, dtype=torch.long, device=device)
                grid_h_pos = grid_t + yy
                grid_w_pos = grid_t + xx
                frame_grid_pos = torch.stack([grid_t, grid_h_pos, grid_w_pos], dim=0)  # (3, eh*ew)

                # Separator position (same as image_pad's text position after shifting)
                sep_val = cur_pos + max_hw
                sep_pos = torch.full((3, 1), sep_val, dtype=torch.long, device=device)

                # Concatenate frame grid and separator: (3, eh*ew + 1)
                frame_chunk = torch.cat([frame_grid_pos, sep_pos], dim=1)
                vision_pos_chunks.append(frame_chunk)

                # The `<|image_pad|>` text token itself takes the shifted position (= separator pos).
                text_pos[0, 0, t_idx] = sep_val
                text_pos[1, 0, t_idx] = sep_val
                text_pos[2, 0, t_idx] = sep_val

                # The next regular text token starts past the grid: cur_pos + max_hw + 1.
                cur_pos = sep_val + 1
                frame_idx += 1
            else:
                text_pos[0, 0, t_idx] = cur_pos
                text_pos[1, 0, t_idx] = cur_pos
                text_pos[2, 0, t_idx] = cur_pos
                cur_pos += 1

        if frame_idx != num_new_frames:
            raise ValueError(
                f"Mismatch between number of new frames ({num_new_frames}) and `<|image_pad|>` tokens "
                f"found in this segment ({frame_idx}). Check the text/image alignment in "
                f"`_update_model_kwargs_for_real_time_generation`."
            )

        if vision_pos_chunks:
            vision_position_ids = torch.cat(vision_pos_chunks, dim=1).unsqueeze(1)  # (3, 1, total_vision_len)
        else:
            vision_position_ids = None

        return text_pos, vision_position_ids, cur_pos, grid_summary

    def prepare_inputs_for_real_time_generation(
        self,
        input_ids=None,
        inputs_embeds=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        use_cache=True,
        cache_position=None,
        pixel_values=None,
        grid_thw=None,
        media_nums_per_sample=None,
        vision_position_ids=None,
        vision_cache_position=None,
        cross_attention_mask=None,
        full_vision_token_info=None,
        logits_to_keep=None,
        **kwargs,
    ):
        """
        Build the kwargs for a single forward pass of `_real_time_sample`.

        Important differences from `prepare_inputs_for_generation`:
          - Vision inputs (`pixel_values`, `grid_thw`, ...) are *not* dropped during decode. New
            frames can arrive mid-decode and must be encoded by the vision tower.
          - `position_ids` is already 3D MRoPE and managed by `_update_model_kwargs_for_real_time_generation`;
            we slice it to the new tokens here instead of letting `MossVLModel.forward` recompute it.
          - `cross_attention_mask` is kept frame-level (B, 1, T, num_frames). We slice on the text
            dimension to the new tokens; expansion to token-level happens inside `MossVLModel.forward`
            using `full_vision_token_info` (cumulative metadata).
        """
        if past_key_values is not None and input_ids is not None and cache_position is not None:
            # Slice input_ids to only the unprocessed tokens.
            if input_ids.shape[1] != cache_position.shape[0]:
                input_ids = input_ids[:, cache_position]

        model_inputs = {
            "input_ids": input_ids.clone(memory_format=torch.contiguous_format) if input_ids is not None else None,
            "inputs_embeds": None,
        }

        model_inputs["attention_mask"] = attention_mask
        model_inputs["cache_position"] = cache_position
        model_inputs["use_cache"] = use_cache
        model_inputs["past_key_values"] = past_key_values

        # Slice 3D MRoPE position_ids: (3, B, total_text_len) -> (3, B, num_new_tokens).
        if position_ids is not None and cache_position is not None:
            num_new = cache_position.shape[0]
            if position_ids.shape[-1] != num_new:
                position_ids = position_ids[..., -num_new:].contiguous()
        model_inputs["position_ids"] = position_ids

        # Slice cross_attention_mask on the text dim to the new tokens only.
        if cross_attention_mask is not None and cache_position is not None:
            num_new = cache_position.shape[0]
            if cross_attention_mask.shape[2] != num_new:
                cross_attention_mask = cross_attention_mask[:, :, -num_new:, :].contiguous()
        model_inputs["cross_attention_mask"] = cross_attention_mask

        # Vision inputs flow through unchanged. None of these are dropped based on cache_position
        # — new frames can arrive mid-decode.
        model_inputs["pixel_values"] = pixel_values
        model_inputs["grid_thw"] = grid_thw
        model_inputs["media_nums_per_sample"] = media_nums_per_sample
        model_inputs["vision_position_ids"] = vision_position_ids
        model_inputs["vision_cache_position"] = vision_cache_position
        model_inputs["full_vision_token_info"] = full_vision_token_info

        if logits_to_keep is not None:
            model_inputs["logits_to_keep"] = logits_to_keep

        return model_inputs

    def _update_model_kwargs_for_real_time_generation(
        self,
        outputs,
        input_ids,
        model_kwargs,
        should_wait_for_new_input: bool,
        new_video_frames: queue.Queue,
        new_prompts: queue.Queue,
        output_text_queue: queue.Queue,
        token_buffer: deque,
        processor,
        **kwargs,
    ):
        """
        After the model emitted `next_token`, drain the realtime queues, splice the new content
        into the running context, and update every piece of cumulative state needed for the next
        forward pass: input_ids, attention_mask, position_ids, cache_position, vision_position_ids,
        vision_cache_position, full_vision_token_info, cross_attention_mask, pixel_values, grid_thw.

        Notes on the design:
          - pure-text turns bypass the processor's image branch (avoid fake-image cache pollution);
          - each new frame is wrapped in `<|vision_start|>...<|vision_end|>` so SFT label-masking
            rules apply directly;
          - MRoPE positions are computed incrementally instead of via `rope_deltas`.
        """
        device = input_ids.device

        # ----- 1. Drain external queues. Wait here if we're in <|silence|> with nothing pending. -----
        frames_to_process = []      # list of (PIL.Image, float timestamp)
        prompts_to_process = []
        while True:
            while not new_video_frames.empty():
                try:
                    frames_to_process.append(new_video_frames.get_nowait())
                except queue.Empty:
                    break
            while not new_prompts.empty():
                try:
                    prompts_to_process.append(new_prompts.get_nowait())
                    if output_text_queue is not None:
                        output_text_queue.put("<|round_start|>")
                    token_buffer.clear()
                except queue.Empty:
                    break

            if self.continue_generating and should_wait_for_new_input and not frames_to_process and not prompts_to_process:
                # Busy-wait — matches VideoMllama reference. Caller controls cadence via
                # `max_tokens_per_turn` sleeping in `_real_time_sample`.
                continue
            break

        # Sort frames by timestamp; the realtime contract guarantees monotone-nondecreasing
        # timestamps across calls, so this only normalizes intra-batch ordering.
        if frames_to_process:
            frames_to_process.sort(key=lambda item: item[1])
        frame_images = [img for img, _ in frames_to_process]
        frame_timestamps = [ts for _, ts in frames_to_process]

        # ----- 2. Build the new text segment. -----
        # Chat turn boundaries: when a new user prompt arrives, close any open assistant turn
        # and open user/assistant turns. Frames after a prompt sit inside the same user turn.
        text_to_append = ""
        if prompts_to_process:
            for prompt in prompts_to_process:
                text_to_append += (
                    "<|im_end|>\n<|im_start|>user\n"
                    f"{prompt}"
                    "<|im_end|>\n<|im_start|>assistant\n"
                )
            # SFT-format alignment: every assistant turn in training data starts with
            # <|silence|> as the first token. When a prompt is drained alongside frames
            # in the same cycle, model would skip emitting that silence (the prompt
            # splice ends in <|im_start|>assistant\n and frame tokens are appended
            # directly after). Insert <|silence|> manually to match training distribution.
            # Confirmed with project team: original inference code had this; restoring it.
                if frame_images:
                    text_to_append += "<|silence|>"
                    if output_text_queue is not None:
                        output_text_queue.put("<|silence|>")

        if frame_images:
            # Each frame is its own self-contained vision region.
            # Use `<|image|>` placeholder (processor will replace with `<|image_pad|>`).
            for ts in frame_timestamps:
                text_to_append += (
                    f"<|vision_start|><|time_start|>{ts:.1f} seconds<|time_end|><|image|><|vision_end|>"
                )

        # ----- 3. Tokenize the new text (and process new frames if any). -----
        # Branch to avoid the processor's fake-image fallback on pure-text turns.
        new_pixel_values = None
        new_grid_thw = None
        new_media_nums_per_sample = None
        new_input_ids = None
        new_attention_mask = None

        if frame_images:
            new_inputs = processor(
                text=text_to_append,
                images=frame_images,
                add_special_tokens=False,
                return_tensors="pt",
            )
            new_input_ids = new_inputs["input_ids"].to(device)
            new_attention_mask = new_inputs["attention_mask"].to(device)
            new_pixel_values = new_inputs["pixel_values"].to(device)
            new_grid_thw = new_inputs["grid_thw"].to(device)
            new_media_nums_per_sample = new_inputs.get("media_nums_per_sample", None)
        elif text_to_append:
            # Pure-text turn: go through the tokenizer directly so the processor never sees
            # `images=None` and never injects a blank image into the cache.
            tok_out = processor.tokenizer(
                text_to_append,
                add_special_tokens=False,
                return_tensors="pt",
            )
            new_input_ids = tok_out["input_ids"].to(device)
            new_attention_mask = tok_out["attention_mask"].to(device)
        # else: nothing to splice this turn — the just-generated token is the only new content.

        # ----- 4. Update past_key_values from outputs. -----
        # transformers >= 4.55 removed `_extract_past_from_model_output`. Modern caches are
        # surfaced as `outputs.past_key_values` directly. MossVL uses the standard name.
        if hasattr(outputs, "past_key_values") and outputs.past_key_values is not None:
            model_kwargs["past_key_values"] = outputs.past_key_values

        assert input_ids.shape[0] == 1, "Real-time generation only supports batch size 1."

        # ----- 5. Append the new tokens onto input_ids. -----
        # input_ids already includes the just-generated token (appended in `_real_time_sample`
        # right before this call). We now also append any drained prompts / frame placeholders.
        num_new_tokens = 1  # the generated token
        if new_input_ids is not None:
            input_ids = torch.cat([input_ids, new_input_ids], dim=1)
            num_new_tokens += new_input_ids.shape[1]

        # ----- 6. attention_mask: extend with ones for all new positions. -----
        if "attention_mask" in model_kwargs and model_kwargs["attention_mask"] is not None:
            attn = model_kwargs["attention_mask"]
            extra = attn.new_ones((attn.shape[0], num_new_tokens))
            model_kwargs["attention_mask"] = torch.cat([attn, extra], dim=-1)

        # ----- 7. MRoPE positions for this turn. -----
        # `realtime_next_position` is the MRoPE position the *generated* token should take. It is
        # initialized by `real_time_generate` from the prefill positions and only advances here.
        next_pos = int(model_kwargs.pop("realtime_next_position"))
        merge_size = self.model.visual.spatial_merge_size
        image_token_id = self.config.image_token_id

        # Position for the just-generated token (one regular text token; no frame here).
        generated_pos = torch.full((3, 1, 1), next_pos, dtype=torch.long, device=device)
        next_pos += 1

        # Position for the spliced new segment (prompt + frame wrappers).
        if new_input_ids is not None:
            segment_text_pos, segment_vision_pos, next_pos, _ = self._compute_realtime_mrope_for_segment(
                new_input_ids=new_input_ids,
                new_grid_thw=new_grid_thw,
                start_position=next_pos,
                image_token_id=image_token_id,
                merge_size=merge_size,
            )
        else:
            segment_text_pos = None
            segment_vision_pos = None

        if segment_text_pos is not None:
            new_text_pos = torch.cat([generated_pos, segment_text_pos], dim=-1)  # (3, 1, num_new_tokens)
        else:
            new_text_pos = generated_pos

        past_position_ids = model_kwargs.get("position_ids", None)
        if past_position_ids is not None:
            model_kwargs["position_ids"] = torch.cat([past_position_ids, new_text_pos], dim=-1)
        else:
            model_kwargs["position_ids"] = new_text_pos
        model_kwargs["realtime_next_position"] = next_pos

        # ----- 8. cache_position: new tokens only (use_cache=True path). -----
        past_cache_position = model_kwargs.pop("cache_position")
        new_cache_position = torch.arange(
            past_cache_position[-1].item() + 1,
            past_cache_position[-1].item() + 1 + num_new_tokens,
            dtype=past_cache_position.dtype,
            device=device,
        )
        if model_kwargs.get("use_cache", True):
            model_kwargs["cache_position"] = new_cache_position
        else:
            model_kwargs["cache_position"] = torch.cat([past_cache_position, new_cache_position])

        # ----- 9. Vision-side state (only updated when new frames came in). -----
        if frame_images:
            # 9a. Accumulate vision_token_info.
            # Replicate `convert_packed_to_batch`'s layout math without running the vision tower —
            # the layout only depends on grid_thw + media_nums_per_sample. The real vision tower
            # call happens inside `MossVLModel.forward` on the next step; we just need the metadata
            # now so we can extend the cumulative state correctly.
            #
            # `vision_seq_pad_multiple` controls the optional tail padding `convert_packed_to_batch`
            # appends to align the vision sequence length for fast attention kernels. In the
            # released Moss-VL-8B-Realtime config it is 1, so `new_pad_end == actual_new_len`
            # and everything below collapses to a plain contiguous append. The branch is kept
            # forward-compatible for configs that set it > 1: in that case the new batch's
            # `vision_cache_position` deliberately starts at the previous actual total (not the
            # previous pad_end) so this batch's head overwrites the previous batch's padding tail,
            # keeping the cumulative K/V layout gap-free for `_expand_cross_attention_mask`
            # (which packs medias contiguously and doesn't honor `media['start']` offsets).
            tokens_per_media = (new_grid_thw[:, 0] * new_grid_thw[:, 1] * new_grid_thw[:, 2]) // (merge_size ** 2)
            actual_new_len = int((tokens_per_media + new_grid_thw[:, 0]).sum().item())
            pad_multiple = self.config.vision_seq_pad_multiple
            if pad_multiple > 1 and actual_new_len % pad_multiple != 0:
                new_pad_end = (actual_new_len + pad_multiple - 1) // pad_multiple * pad_multiple
            else:
                # pad_multiple == 1 → no padding ever; or pad_multiple > 1 and already aligned.
                new_pad_end = actual_new_len

            # Build per-frame media records (one media per frame in real-time mode).
            # Offsets are relative to the START of this turn's new vision tokens; we shift them
            # by the cumulative actual_total before merging into the cumulative info.
            past_full_info = model_kwargs.get("full_vision_token_info", None)
            if past_full_info is None:
                past_actual_total = 0
                cum_medias = []
            else:
                past_actual_total = int(past_full_info[0].get("total_length", 0))
                cum_medias = list(past_full_info[0].get("medias", []))

            local_seq_offset = 0
            for i in range(new_grid_thw.shape[0]):
                t_i = int(new_grid_thw[i, 0].item())
                h_i = int(new_grid_thw[i, 1].item())
                w_i = int(new_grid_thw[i, 2].item())
                tokens_i = (h_i * w_i // (merge_size ** 2)) * t_i
                media_length = tokens_i + t_i  # +t_i separators (one per frame)
                cum_medias.append({
                    "start": past_actual_total + local_seq_offset,
                    "end": past_actual_total + local_seq_offset + media_length,
                    "length": media_length,
                    "num_frames": t_i,
                    "grid_h": h_i,
                    "grid_w": w_i,
                    "vision_tokens_per_frame": tokens_i // t_i,
                    "has_separator": True,
                })
                local_seq_offset += media_length

            new_full_info = [{
                "medias": cum_medias,
                "total_length": past_actual_total + actual_new_len,
                "pad_start": past_actual_total + actual_new_len,
                # pad_end = current K/V cache size. With pad_multiple=1 this equals total_length.
                "pad_end": past_actual_total + new_pad_end,
            }]
            model_kwargs["full_vision_token_info"] = new_full_info

            # 9b. vision_cache_position for this turn's new states. With pad_multiple=1 this is
            # simply `arange(past_total, past_total + actual_new_len)` — a plain contiguous append.
            # For pad_multiple > 1, see the note in 9a: starting at `past_actual_total` makes the
            # head of this batch overlap and overwrite the previous batch's padding tail.
            model_kwargs["vision_cache_position"] = torch.arange(
                past_actual_total,
                past_actual_total + new_pad_end,
                dtype=torch.long,
                device=device,
            )

            # 9c. vision_position_ids: per-call (only for the new vision states).
            # `segment_vision_pos` already has length `actual_new_len = sum_per_frame(eh*ew + 1)`.
            # When pad_multiple=1 (current config) this matches `new_pad_end` exactly and the
            # right-pad below is a no-op. The padding branch only matters if pad_multiple > 1.
            if segment_vision_pos is None:
                segment_vision_pos = torch.zeros((3, 1, 0), dtype=torch.long, device=device)
            pad_len = new_pad_end - segment_vision_pos.shape[-1]
            if pad_len > 0:
                # Padded slots have no real position — fill with zeros. They get masked out by
                # cross_attention_mask, so the RoPE values applied to them don't affect output.
                pad_tensor = torch.zeros((3, 1, pad_len), dtype=torch.long, device=device)
                segment_vision_pos = torch.cat([segment_vision_pos, pad_tensor], dim=-1)
            model_kwargs["vision_position_ids"] = segment_vision_pos

            # 9d. New pixel inputs for forward.
            model_kwargs["pixel_values"] = new_pixel_values
            model_kwargs["grid_thw"] = new_grid_thw
            model_kwargs["media_nums_per_sample"] = new_media_nums_per_sample
        else:
            # No new frames: don't pass any vision inputs to forward — cross-attention layers
            # will read keys/values from the cache directly. (See `MossVLTextCrossAttention.forward`.)
            model_kwargs["pixel_values"] = None
            model_kwargs["grid_thw"] = None
            model_kwargs["media_nums_per_sample"] = None
            model_kwargs["vision_position_ids"] = None
            model_kwargs["vision_cache_position"] = None
            # `full_vision_token_info` stays as-is in model_kwargs (cumulative state).

        # ----- 10. cross_attention_mask: rebuild cumulatively from input_ids. -----
        # The mask is (B=1, 1, total_text_len, total_num_frames). Each text token at position t can
        # attend to frame i iff the count of `<|image_pad|>` tokens at positions [0, t] is > i
        # (i.e., frame i has already appeared in the text). This is the same causal rule the
        # processor uses (see `_create_cross_attention_mask`).
        full_info = model_kwargs.get("full_vision_token_info", None)
        if full_info is not None and full_info[0]["medias"]:
            total_num_frames = sum(media["num_frames"] for media in full_info[0]["medias"])
            is_image_pad = (input_ids == image_token_id)
            cum_image_tokens = is_image_pad.cumsum(dim=1)
            frame_indices = torch.arange(total_num_frames, device=device)
            visible = cum_image_tokens.unsqueeze(-1) > frame_indices  # (1, T, num_frames)
            mask = (~visible).unsqueeze(1)  # (1, 1, T, num_frames)
            model_kwargs["cross_attention_mask"] = mask
        else:
            model_kwargs["cross_attention_mask"] = None

        return input_ids, model_kwargs

    @torch.no_grad()
    def _real_time_generate(
        self,
        new_video_frames: queue.Queue,
        new_prompts: queue.Queue,
        output_text_queue: queue.Queue,
        processor,
        inputs: Optional[torch.Tensor] = None,
        generation_config: Optional[GenerationConfig] = None,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        prefix_allowed_tokens_fn=None,
        synced_gpus: Optional[bool] = None,
        assistant_model=None,
        streamer=None,
        negative_prompt_ids: Optional[torch.Tensor] = None,
        negative_prompt_attention_mask: Optional[torch.Tensor] = None,
        max_tokens_per_turn: int = 86400,
        **kwargs,
    ) -> Union[torch.LongTensor, GenerateDecoderOnlyOutput]:
        """
        Custom `generate()` for real-time streaming. Structure mirrors HuggingFace
        `GenerationMixin.generate()` but only supports the subset we need (see class docstring).
        """
        # NOTE: `_validate_model_class()` and `_validate_assistant()` were removed from
        # GenerationMixin in transformers >= 4.55 (consolidated into `generate()` itself).
        # We don't call them here for forward compat with 4.57+.
        tokenizer = kwargs.pop("tokenizer", None)
        assistant_tokenizer = kwargs.pop("assistant_tokenizer", None)

        generation_config, model_kwargs = self._prepare_generation_config(generation_config, **kwargs)
        # `realtime_next_position` is purely internal state for the realtime loop — it doesn't appear in
        # `forward` or `prepare_inputs_for_generation`, so we pop it before validation and restore it after.
        realtime_next_position = model_kwargs.pop("realtime_next_position", None)
        self._validate_model_kwargs(model_kwargs.copy())
        if realtime_next_position is not None:
            model_kwargs["realtime_next_position"] = realtime_next_position

        if synced_gpus is None:
            synced_gpus = False
        assert not synced_gpus, "Real-time generation does not support synced_gpus."

        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()

        accepts_attention_mask = "attention_mask" in set(inspect.signature(self.forward).parameters.keys())
        requires_attention_mask = "encoder_outputs" not in model_kwargs
        kwargs_has_attention_mask = model_kwargs.get("attention_mask", None) is not None

        inputs_tensor, model_input_name, model_kwargs = self._prepare_model_inputs(
            inputs, generation_config.bos_token_id, model_kwargs
        )
        batch_size = inputs_tensor.shape[0]
        assert batch_size == 1, "Real-time generation only supports batch size 1."

        device = inputs_tensor.device
        self._prepare_special_tokens(generation_config, kwargs_has_attention_mask, device=device)

        if not kwargs_has_attention_mask and requires_attention_mask and accepts_attention_mask:
            model_kwargs["attention_mask"] = self._prepare_attention_mask_for_generation(
                inputs_tensor, generation_config, model_kwargs
            )

        if not self.config.is_encoder_decoder:
            input_ids = inputs_tensor if model_input_name == "input_ids" else model_kwargs.pop("input_ids")
        else:
            raise NotImplementedError("Encoder-decoder models are not supported for real-time generation.")

        input_ids_length = input_ids.shape[-1]
        has_default_max_length = kwargs.get("max_length") is None and generation_config.max_length is not None
        has_default_min_length = kwargs.get("min_length") is None and generation_config.min_length is not None
        generation_config = self._prepare_generated_length(
            generation_config=generation_config,
            has_default_max_length=has_default_max_length,
            has_default_min_length=has_default_min_length,
            model_input_name=model_input_name,
            inputs_tensor=inputs_tensor,
            input_ids_length=input_ids_length,
        )

        # Use `logits_to_keep=1` (Transformers 4.57+) to avoid computing the whole logit matrix.
        if "logits_to_keep" not in model_kwargs:
            model_kwargs["logits_to_keep"] = 1

        # In transformers 4.57 `_prepare_cache_for_generation` takes (generation_config,
        # model_kwargs, generation_mode, batch_size, max_cache_length) — no `assistant_model`,
        # no `device`. We need `generation_mode` upfront, so compute it before the cache prep.
        generation_mode = generation_config.get_generation_mode(assistant_model=assistant_model)
        if generation_mode not in (GenerationMode.SAMPLE, GenerationMode.GREEDY_SEARCH):
            raise NotImplementedError(
                f"Generation mode {generation_mode} is not supported for real-time generation."
            )
        max_cache_length = generation_config.max_length
        self._prepare_cache_for_generation(
            generation_config, model_kwargs, generation_mode, batch_size, max_cache_length
        )
        if isinstance(model_kwargs.get("past_key_values"), StaticCache):
            raise NotImplementedError("StaticCache is not supported for real-time generation.")

        logits_processor = self._get_logits_processor(
            generation_config=generation_config,
            input_ids_seq_length=input_ids_length,
            encoder_input_ids=inputs_tensor,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            logits_processor=logits_processor,
            device=device,
            model_kwargs=model_kwargs,
            negative_prompt_ids=negative_prompt_ids,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
        )
        # transformers 4.57: `_get_stopping_criteria` no longer accepts arbitrary kwargs
        # (used to take `attention_mask`, `negative_prompt_*`). Pass only the supported args.
        stopping_criteria = self._get_stopping_criteria(
            generation_config=generation_config,
            stopping_criteria=stopping_criteria,
            tokenizer=tokenizer,
        )

        model_kwargs["use_cache"] = generation_config.use_cache

        input_ids, model_kwargs = self._expand_inputs_for_generation(
            input_ids=input_ids,
            expand_size=generation_config.num_return_sequences,
            is_encoder_decoder=self.config.is_encoder_decoder,
            **model_kwargs,
        )

        return self._real_time_sample(
            input_ids=input_ids,
            new_video_frames=new_video_frames,
            new_prompts=new_prompts,
            output_text_queue=output_text_queue,
            processor=processor,
            logits_processor=logits_processor,
            stopping_criteria=stopping_criteria,
            generation_config=generation_config,
            streamer=streamer,
            max_tokens_per_turn=max_tokens_per_turn,
            **model_kwargs,
        )

    def _real_time_sample(
        self,
        input_ids: torch.LongTensor,
        new_video_frames: queue.Queue,
        new_prompts: queue.Queue,
        output_text_queue: queue.Queue,
        processor,
        logits_processor: LogitsProcessorList,
        stopping_criteria: StoppingCriteriaList,
        generation_config: GenerationConfig,
        streamer=None,
        max_tokens_per_turn: int = 86400,
        **model_kwargs,
    ) -> Union[torch.LongTensor, GenerateDecoderOnlyOutput]:
        """
        Main streaming loop. One token per iteration; after each token we drain the realtime
        queues, splice any new content into the running state, and continue.
        """
        do_sample = generation_config.do_sample
        return_dict_in_generate = generation_config.return_dict_in_generate

        # Token bookkeeping for streaming decode.
        token_buffer: deque = deque()
        silence_token_id = processor.tokenizer.convert_tokens_to_ids("<|silence|>")
        # `<|...|>` may or may not exist in this tokenizer; fall back to -1 if absent so the
        # suppression / streaming checks become no-ops.
        ellipsis_candidates = processor.tokenizer.convert_tokens_to_ids(["<|...|>"])
        ellipsis_token_id = ellipsis_candidates[0] if ellipsis_candidates and ellipsis_candidates[0] is not None else -1
        invalid_candidates = processor.tokenizer.convert_tokens_to_ids(["�"])
        invalid_token_id = invalid_candidates[0] if invalid_candidates and invalid_candidates[0] is not None else -1

        # Bootstrap cache_position from the prefill input_ids length.
        if hasattr(self, '_get_initial_cache_position'):
            model_kwargs = self._get_initial_cache_position(input_ids.shape[1], input_ids.device, model_kwargs)
        else:
            model_kwargs["cache_position"] = torch.arange(input_ids.shape[1], device=input_ids.device)

        is_prefill = True
        current_token_start_time = time.time()

        while True:
            if not self.continue_generating:
                break

            model_inputs = self.prepare_inputs_for_real_time_generation(input_ids, **model_kwargs)
            outputs = self(**model_inputs, return_dict=True)
            is_prefill = False  # noqa: F841 — kept for parity with VideoMllama; we never branch on it.

            next_token_logits = outputs.logits[:, -1, :].clone().float().to(input_ids.device)
            next_token_scores = logits_processor(input_ids, next_token_logits)

            # Suppress `<|...|>` so `<|silence|>` is the unique idle signal (matches VideoMllama).
            if ellipsis_token_id >= 0:
                next_token_scores[0, ellipsis_token_id] = float("-inf")
            probs = F.softmax(next_token_scores, dim=-1)

            # Silence threshold — only commit to `<|silence|>` when the model is highly confident.
            # Silence threshold pinned to 0.0. Some upstream code defaulted to 0.6,
            # which suppressed legitimate <|silence|> emits whose prob fell below the
            # threshold — visible as streaming "should-be-silence but model speaks
            # anyway" artifacts. Raising this is not recommended.
            silence_threshold = 0.0
            silence_prob = probs[0, silence_token_id].item() if silence_token_id is not None and silence_token_id >= 0 else 0.0
            if silence_token_id is not None and silence_token_id >= 0 and silence_prob < silence_threshold:
                probs[0, silence_token_id] = 0.0
                probs = probs / probs.sum(dim=-1, keepdim=True)

            if do_sample:
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                next_tokens = torch.argmax(probs, dim=-1)

            # Stream decoded text to the output queue, accumulating multi-byte fragments.
            if output_text_queue is not None:
                current_token = int(next_tokens.item())
                token_buffer.append(current_token)
                buf_list = list(token_buffer)
                decoded_text = processor.tokenizer.decode(
                    buf_list,
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                )
                is_silence_or_ellipsis = current_token in (silence_token_id, ellipsis_token_id)
                is_complete_text = bool(decoded_text) and decoded_text[-1] != "�"
                is_invalid_complete = (
                    bool(decoded_text)
                    and decoded_text[-1] == "�"
                    and buf_list[-1] == invalid_token_id
                )
                if is_silence_or_ellipsis or is_complete_text or is_invalid_complete:
                    output_text_queue.put(decoded_text)
                    token_buffer.clear()

            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            if streamer is not None:
                streamer.put(next_tokens.cpu())

            # Pace ourselves to at most `max_tokens_per_turn` tokens / second.
            cost = time.time() - current_token_start_time
            wait = 1.0 / max_tokens_per_turn - cost
            if wait > 0:
                time.sleep(wait)
            current_token_start_time = time.time()

            should_wait_for_new_input = next_tokens.item() == silence_token_id

            input_ids, model_kwargs = self._update_model_kwargs_for_real_time_generation(
                outputs=outputs,
                input_ids=input_ids,
                model_kwargs=model_kwargs,
                should_wait_for_new_input=should_wait_for_new_input,
                new_video_frames=new_video_frames,
                new_prompts=new_prompts,
                output_text_queue=output_text_queue,
                token_buffer=token_buffer,
                processor=processor,
            )

            # Free large intermediate tensors before the next iteration.
            del outputs

        if streamer is not None:
            streamer.end()

        if return_dict_in_generate:
            return GenerateDecoderOnlyOutput(
                sequences=input_ids,
                past_key_values=model_kwargs.get("past_key_values"),
            )
        return input_ids

    def real_time_generate(
        self,
        new_video_frames: queue.Queue,
        new_prompts: queue.Queue,
        output_text_queue: queue.Queue,
        processor,
        max_tokens_per_turn: int = 86400,
        **generate_kwargs,
    ):
        """
        Top-level real-time generation entry point. Mirrors VideoMllama's `real_time_generate`.

        Args:
            new_video_frames: queue of `(PIL.Image, timestamp_seconds)` tuples (real-time frames).
            new_prompts: queue of user prompt strings.
            output_text_queue: queue the generated text fragments are pushed into.
            processor: the `MossVLProcessor` instance.
            max_tokens_per_turn: pace cap (tokens/sec). Default matches VideoMllama (~86400).
            **generate_kwargs: forwarded to the underlying `_real_time_generate`.
        """
        system_prompt = (
            "You are a helpful AI assistant. You perceive and understand the surrounding environment in real-time "
            "through the camera and interact with the user. Whether or not the user actively asks questions, you "
            "continuously observe and analyze visual information, maintaining awareness of the environment.\n\n"
            "Core Abilities:\n"
            "- Continuous Perception: Always observe and analyze the visual information captured by the camera in "
            "real-time. This perception does not stop even when there is no user interaction.\n"
            "- Observation-Based Answers: All answers must be based on visual observations, including both historical "
            "and real-time data. Do not make guesses without evidence from visual input.\n"
            "- Dynamic Adjustment: When you observe changes relevant to the user's question, promptly update and "
            "adjust your answers to ensure timeliness and accuracy.\n\n"
            "Interaction Rules:\n"
            "- Always base your answers on observed visual information.\n"
            "- If you notice significant changes related to the user's question, proactively and promptly update your answer.\n"
            "- When you are unable to answer or have finished answering, output `<|silence|>` directly.\n\n"
            "Your goal is to be the user's reliable \"eyes\", helping them understand and perceive the world around them."
        )
        # TODO(realtime-sft): the empty user content here is a MVP placeholder. Production / eval
        #   runs MUST replace it with the fixed user prompt used in the realtime SFT training data
        #   (otherwise the train/inference distribution diverges — affects the <|silence|> threshold,
        #   first-token distribution, and visual sensitivity). Source of the prompt string should be
        #   referenced explicitly here once the SFT schema is finalized.
        initial_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": ""},  # TEMP: see the TODO above — do not ship as-is.
        ]
        # Pure-text prefill: go through `processor.tokenizer` to bypass the processor's fake-image
        # fallback.
        initial_input_ids = processor.tokenizer.apply_chat_template(
            initial_messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
        ).to(self.device)
        initial_attention_mask = torch.ones_like(initial_input_ids)

        # Seed 3D MRoPE positions for prefill: every prefill token is a regular text token
        # (no images, no padding), so position_ids[:, :, i] = i, broadcast over the 3 MRoPE axes.
        prefill_len = initial_input_ids.shape[1]
        prefill_positions = (
            torch.arange(prefill_len, dtype=torch.long, device=self.device)
            .view(1, 1, prefill_len)
            .expand(3, 1, prefill_len)
            .contiguous()
        )

        self.start_real_time_generate()
        try:
            return self._real_time_generate(
                new_video_frames=new_video_frames,
                new_prompts=new_prompts,
                output_text_queue=output_text_queue,
                processor=processor,
                inputs=initial_input_ids,
                attention_mask=initial_attention_mask,
                position_ids=prefill_positions,
                # `realtime_next_position` is the MRoPE position the FIRST generated token will use.
                # Prefill positions are 0..prefill_len-1, so the next position is prefill_len.
                realtime_next_position=prefill_len,
                full_vision_token_info=None,
                max_tokens_per_turn=max_tokens_per_turn,
                **generate_kwargs,
            )
        finally:
            self.stop_real_time_generate()

    # The offline path expects an `_offline_processor_lock` instance attribute that
    # the streaming `__init__` does not create. Provide it lazily via a property so
    # that `__init__` stays untouched (works for already-instantiated objects too).
    @property
    def _offline_processor_lock(self):
        lk = self.__dict__.get("_offline_processor_lock_obj")
        if lk is None:
            import threading as _th
            lk = _th.RLock()
            self.__dict__["_offline_processor_lock_obj"] = lk
        return lk

    @classmethod
    def build_offline_prepare_helper(cls):
        helper = cls.__new__(cls)
        helper.__dict__["_offline_processor_lock_obj"] = threading.RLock()
        return helper

    @staticmethod
    def _offline_flatten_content_with_vision_tokens(content) -> str:
        if isinstance(content, str):
            return content
        if not isinstance(content, list):
            return str(content) if content else ""

        parts = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "image" or "image" in item:
                    parts.append("<|image|>")
                elif item.get("type") == "video" or "video" in item:
                    parts.append("<|video|>")
                if "text" in item:
                    parts.append(str(item["text"]))
            elif isinstance(item, str):
                parts.append(item)
        return "".join(parts)

    @staticmethod
    def _offline_sanitize_prompt_text(processor, text: Any) -> str:
        if text is None:
            return ""

        sanitized = str(text)
        replacements = [
            (getattr(processor, "image_placeholder", None), ""),
            (getattr(processor, "video_placeholder", None), ""),
            (getattr(processor, "image_token", None), ""),
            (getattr(processor, "video_token", None), ""),
        ]
        for needle, replacement in replacements:
            if needle:
                sanitized = sanitized.replace(needle, replacement)
        return sanitized.lstrip("\n")

    def _offline_sanitize_message_content(self, processor, content: Any) -> Any:
        if isinstance(content, str):
            return self._offline_sanitize_prompt_text(processor, content)
        if not isinstance(content, list):
            return content

        sanitized_items = []
        for item in content:
            if isinstance(item, dict):
                item_copy = dict(item)
                if "text" in item_copy:
                    item_copy["text"] = self._offline_sanitize_prompt_text(processor, item_copy.get("text"))
                sanitized_items.append(item_copy)
            elif isinstance(item, str):
                sanitized_items.append(self._offline_sanitize_prompt_text(processor, item))
            else:
                sanitized_items.append(item)
        return sanitized_items

    def _offline_prepare_messages(self, processor, query: Dict[str, Any]) -> List[Dict[str, Any]]:
        messages = query.get("messages")
        if messages:
            prepared_messages = []
            for message in messages:
                if not isinstance(message, dict):
                    continue
                message_copy = dict(message)
                message_copy["content"] = self._offline_sanitize_message_content(
                    processor,
                    message_copy.get("content", ""),
                )
                prepared_messages.append(message_copy)
            if prepared_messages:
                return prepared_messages

        prompt = self._offline_sanitize_prompt_text(processor, query.get("prompt", ""))
        images = list(query.get("images") or [])
        videos = list(query.get("videos") or [])

        content = []
        for image in images:
            content.append({"type": "image", "image": image})
        for video in videos:
            content.append({"type": "video", "video": video})
        if prompt:
            content.append({"type": "text", "text": prompt.lstrip("\n")})

        if not content:
            content = [{"type": "text", "text": ""}]

        return [{"role": "user", "content": content}]

    def _offline_prepare_input_text(self, processor, messages: List[Dict[str, Any]]) -> str:
        processed_messages = []
        for message in messages:
            message_copy = dict(message)
            message_copy["content"] = self._offline_flatten_content_with_vision_tokens(
                message_copy.get("content", "")
            )
            processed_messages.append(message_copy)
        return processor.apply_chat_template(
            processed_messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    @staticmethod
    def _offline_collect_media(messages: List[Dict[str, Any]]) -> tuple[List[Any], List[Any]]:
        all_images: List[Any] = []
        all_videos: List[Any] = []

        for message in messages:
            content = message.get("content")
            if isinstance(content, list):
                for item in content:
                    if not isinstance(item, dict):
                        continue
                    if item.get("type") == "image" or "image" in item:
                        image = item.get("image") or item.get("image_url")
                        if image is not None:
                            all_images.append(image)
                    elif item.get("type") == "video" or "video" in item:
                        video = item.get("video")
                        if video is not None:
                            all_videos.append(video)

        return all_images, all_videos

    def _offline_build_processor_kwargs(
        self,
        input_text: Union[str, List[str]],
        all_images: List[Any],
        all_videos: List[Any],
        media_kwargs: Dict[str, Any],
    ) -> Dict[str, Any]:
        processor_kwargs: Dict[str, Any] = {
            "text": input_text,
            "images": all_images or None,
            "videos": all_videos or None,
            "return_tensors": "pt",
            "padding": False,
        }

        if media_kwargs.get("min_pixels") is not None:
            processor_kwargs["min_pixels"] = media_kwargs["min_pixels"]
        if media_kwargs.get("max_pixels") is not None:
            processor_kwargs["max_pixels"] = media_kwargs["max_pixels"]
        if media_kwargs.get("video_fps") is not None:
            processor_kwargs["video_fps"] = media_kwargs["video_fps"]

        min_frames = media_kwargs.get("min_frames", media_kwargs.get("video_minlen"))
        max_frames = media_kwargs.get("max_frames", media_kwargs.get("video_maxlen"))
        if min_frames is not None:
            processor_kwargs["min_frames"] = min_frames
        if max_frames is not None:
            processor_kwargs["max_frames"] = max_frames

        return processor_kwargs

    def _offline_run_processor(self, processor, processor_kwargs: Dict[str, Any], media_kwargs: Dict[str, Any]):
        image_proc = getattr(processor, "image_processor", None)
        video_proc = getattr(processor, "video_processor", None)
        modified_multi_image = False
        modified_video = False

        with self._offline_processor_lock:
            try:
                multi_image_max_pixels = media_kwargs.get("multi_image_max_pixels")
                if multi_image_max_pixels is not None and image_proc is not None:
                    orig_multi_image_max_pixels = getattr(image_proc, "multi_image_max_pixels", None)
                    image_proc.multi_image_max_pixels = multi_image_max_pixels
                    modified_multi_image = True

                video_max_pixels = media_kwargs.get("video_max_pixels")
                if video_max_pixels is not None and video_proc is not None:
                    orig_video_max_pixels = getattr(video_proc, "video_max_pixels", None)
                    video_proc.video_max_pixels = video_max_pixels
                    modified_video = True

                inputs = processor(**processor_kwargs)
            finally:
                if modified_multi_image and image_proc is not None:
                    image_proc.multi_image_max_pixels = orig_multi_image_max_pixels
                if modified_video and video_proc is not None:
                    video_proc.video_max_pixels = orig_video_max_pixels

        return inputs

    def _offline_move_inputs_to_devices(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        moved_inputs = dict(inputs)
        text_device = self.get_input_embeddings().weight.device
        vision_device = self.visual.patch_embed.proj.weight.device
        vision_input_keys = {"pixel_values", "grid_thw"}

        for key, value in list(moved_inputs.items()):
            if not isinstance(value, torch.Tensor):
                continue

            target_device = vision_device if key in vision_input_keys else text_device
            moved_value = value.to(target_device)
            if moved_value.dtype == torch.float32:
                moved_value = moved_value.to(torch.bfloat16)
            moved_inputs[key] = moved_value

        return moved_inputs

    @staticmethod
    def _offline_build_call_kwargs(generate_kwargs: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        normalized_generate_kwargs = dict(generate_kwargs or {})
        max_new_tokens = normalized_generate_kwargs.pop("max_new_tokens", 1024)
        temperature = normalized_generate_kwargs.pop("temperature", 1.0)
        top_k = normalized_generate_kwargs.pop("top_k", 50)
        top_p = normalized_generate_kwargs.pop("top_p", 1.0)
        repetition_penalty = normalized_generate_kwargs.pop("repetition_penalty", 1.0)
        do_sample = normalized_generate_kwargs.pop("do_sample", False)
        vision_chunked_length = normalized_generate_kwargs.pop("vision_chunked_length", None)

        if temperature is None:
            temperature = 1.0
        if temperature <= 0:
            temperature = 1.0
            do_sample = False

        return dict(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            do_sample=do_sample,
            vision_chunked_length=vision_chunked_length,
            **normalized_generate_kwargs,
        )

    def offline_prepare_query_cpu(
        self,
        processor,
        query: Dict[str, Any],
        session_messages: Optional[List[Dict[str, Any]]] = None,
        *,
        padding: bool = False,
    ) -> Dict[str, Any]:
        current_session = session_messages or []
        if query.get("reset_session") or query.get("clear_history"):
            current_session = []

        working_messages = self._offline_build_session_messages(
            processor,
            query,
            current_session,
        )
        input_text = self._offline_prepare_input_text(processor, working_messages)
        all_images, all_videos = self._offline_collect_media(working_messages)
        media_kwargs = dict(query.get("media_kwargs") or {})
        processor_kwargs = self._offline_build_processor_kwargs(
            input_text,
            all_images,
            all_videos,
            media_kwargs,
        )
        processor_kwargs["padding"] = padding
        inputs_cpu = self._offline_run_processor(processor, processor_kwargs, media_kwargs)

        return {
            "inputs_cpu": inputs_cpu,
            "input_text": input_text,
            "working_messages": working_messages,
            "call_kwargs": self._offline_build_call_kwargs(query.get("generate_kwargs")),
        }

    def _offline_prepare_inputs(self, processor, query: Dict[str, Any]):
        prepared = self.offline_prepare_query_cpu(processor, query)
        inputs = self._offline_move_inputs_to_devices(prepared["inputs_cpu"])
        return inputs, prepared["input_text"]

    def offline_generate_from_prepared(self, processor, prepared: Dict[str, Any]) -> Dict[str, Any]:
        inputs = self._offline_move_inputs_to_devices(prepared["inputs_cpu"])
        input_seq_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            outputs = self.generate(
                **inputs,
                **prepared["call_kwargs"],
            )

        generated_tokens = outputs[:, input_seq_len:]
        decoded_texts = processor.batch_decode(generated_tokens, skip_special_tokens=True)
        text = decoded_texts[0] if decoded_texts else ""

        return {
            "text": text,
            "input_text": prepared["input_text"],
            "messages": prepared["working_messages"],
        }

    def _offline_build_session_messages(
        self,
        processor,
        query: Dict[str, Any],
        session_messages: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        has_explicit_messages = bool(query.get("messages"))
        if has_explicit_messages and not query.get("append_messages_to_session", False):
            base_messages: List[Dict[str, Any]] = []
        else:
            base_messages = [dict(message) for message in session_messages]

        turn_messages = self._offline_prepare_messages(processor, query)
        has_system_message = any(
            isinstance(message, dict) and message.get("role") == "system"
            for message in (base_messages + turn_messages)
        )

        should_add_system_prompt = (
            query.get("use_default_system_prompt", False)
            or query.get("system_prompt") is not None
            or query.get("system_prompt_type") is not None
            or query.get("thinking_mode") is not None
        )

        if not base_messages and not has_system_message and should_add_system_prompt:
            system_prompt = self._offline_resolve_system_prompt(query, turn_messages)
            if system_prompt is not None:
                base_messages.append({"role": "system", "content": system_prompt})

        return base_messages + turn_messages

    @staticmethod
    def _offline_query_contains_video(query: Dict[str, Any], messages: List[Dict[str, Any]]) -> bool:
        if query.get("videos"):
            return True

        for message in messages:
            content = message.get("content") if isinstance(message, dict) else None
            if isinstance(content, list) and any(
                isinstance(item, dict) and (item.get("type") == "video" or "video" in item)
                for item in content
            ):
                return True
        return False

    @staticmethod
    def _offline_normalize_thinking_mode(value: Optional[str]) -> str:
        if value is None:
            return "no_thinking"

        normalized = _OFFLINE_THINKING_MODE_ALIASES.get(str(value).strip().lower())
        if normalized is None:
            allowed = ", ".join(sorted(set(_OFFLINE_THINKING_MODE_ALIASES.values())))
            raise ValueError(f"Unsupported thinking_mode: {value!r}. Supported values: {allowed}")
        return normalized

    @staticmethod
    def _offline_normalize_system_prompt_type(value: Optional[str], has_video: bool) -> str:
        if value is None:
            return "video" if has_video else "text_image"

        normalized_key = str(value).strip().lower().replace("/", "_").replace(" ", "_")
        while "__" in normalized_key:
            normalized_key = normalized_key.replace("__", "_")

        normalized = _OFFLINE_SYSTEM_PROMPT_TYPE_ALIASES.get(normalized_key)
        if normalized is None:
            allowed = ", ".join(sorted(set(_OFFLINE_SYSTEM_PROMPT_TYPE_ALIASES.values())))
            raise ValueError(f"Unsupported system_prompt_type: {value!r}. Supported values: {allowed}")
        return normalized

    def _offline_resolve_system_prompt(
        self,
        query: Dict[str, Any],
        turn_messages: List[Dict[str, Any]],
    ) -> Optional[str]:
        explicit_system_prompt = query.get("system_prompt")
        if explicit_system_prompt is not None:
            return str(explicit_system_prompt)

        has_video = self._offline_query_contains_video(query, turn_messages)
        thinking_mode = self._offline_normalize_thinking_mode(query.get("thinking_mode"))
        system_prompt_type = self._offline_normalize_system_prompt_type(
            query.get("system_prompt_type"),
            has_video=has_video,
        )
        return _OFFLINE_SYSTEM_PROMPTS[thinking_mode][system_prompt_type]

    @staticmethod
    def _offline_finalize_session_messages(
        working_messages: List[Dict[str, Any]],
        assistant_text: str,
    ) -> List[Dict[str, Any]]:
        next_messages = [dict(message) for message in working_messages]
        next_messages.append({"role": "assistant", "content": assistant_text})
        return next_messages

    def _offline_prepare_generation(self, processor, query: Dict[str, Any]):
        prepared = self.offline_prepare_query_cpu(processor, query)
        inputs = self._offline_move_inputs_to_devices(prepared["inputs_cpu"])
        return inputs, prepared["input_text"], prepared["call_kwargs"]

    @staticmethod
    def _offline_normalize_shared_mapping(
        values: List[Dict[str, Any]],
        mapping_name: str,
    ) -> Dict[str, Any]:
        normalized_values = [dict(value or {}) for value in values]
        if not normalized_values:
            return {}

        all_keys = set()
        for value in normalized_values:
            all_keys.update(value.keys())

        merged: Dict[str, Any] = {}
        mismatched_keys: List[str] = []
        for key in sorted(all_keys):
            unique_values = {repr(value.get(key)) for value in normalized_values}
            if len(unique_values) > 1:
                mismatched_keys.append(key)
            else:
                merged[key] = normalized_values[0].get(key)

        if mismatched_keys:
            mismatch_text = ", ".join(mismatched_keys)
            raise ValueError(
                f"All batch queries must share the same {mapping_name}. "
                f"Mismatched keys: {mismatch_text}"
            )
        return merged

    def _offline_prepare_batch_generation(
        self,
        processor,
        queries: List[Dict[str, Any]],
        session_states: Optional[List[List[Dict[str, Any]]]] = None,
    ):
        if not queries:
            raise ValueError("`queries` must contain at least one query.")

        if session_states is None:
            session_states = [[] for _ in queries]
        elif len(session_states) != len(queries):
            raise ValueError("`session_states` must have the same length as `queries`.")

        working_messages_list: List[List[Dict[str, Any]]] = []
        input_texts: List[str] = []
        all_images_per_query: List[List[Any]] = []
        all_videos_per_query: List[List[Any]] = []

        for query, session_state in zip(queries, session_states):
            if not isinstance(query, dict):
                raise TypeError("Each batch query must be a dict.")
            if query.get("stop_offline_generate"):
                raise ValueError("`stop_offline_generate` is not supported in offline_batch_generate.")
            if query.get("stream_output", query.get("stream", False)):
                raise ValueError("Streaming is not supported in offline_batch_generate.")
            if query.get("cancel_current_generate") or query.get("stop_generation"):
                raise ValueError("Cancel / stop controls are not supported in offline_batch_generate.")

            current_session = [] if query.get("reset_session") or query.get("clear_history") else session_state
            working_messages = self._offline_build_session_messages(
                processor,
                query,
                current_session,
            )
            working_messages_list.append(working_messages)
            input_texts.append(self._offline_prepare_input_text(processor, working_messages))

            all_images, all_videos = self._offline_collect_media(working_messages)
            all_images_per_query.append(all_images)
            all_videos_per_query.append(all_videos)

        media_kwargs = self._offline_normalize_shared_mapping(
            [query.get("media_kwargs") or {} for query in queries],
            mapping_name="media_kwargs",
        )
        processor_kwargs = self._offline_build_processor_kwargs(
            input_text=input_texts,
            all_images=[image for images in all_images_per_query for image in images],
            all_videos=[video for videos in all_videos_per_query for video in videos],
            media_kwargs=media_kwargs,
        )
        processor_kwargs["padding"] = True

        tokenizer = getattr(processor, "tokenizer", None)
        orig_padding_side = None

        if tokenizer is not None and hasattr(tokenizer, "padding_side"):
            orig_padding_side = tokenizer.padding_side
            tokenizer.padding_side = "left"
        try:
            inputs = self._offline_run_processor(processor, processor_kwargs, media_kwargs)
        finally:
            if tokenizer is not None and orig_padding_side is not None:
                tokenizer.padding_side = orig_padding_side

        inputs = self._offline_move_inputs_to_devices(inputs)

        generate_kwargs = self._offline_normalize_shared_mapping(
            [query.get("generate_kwargs") or {} for query in queries],
            mapping_name="generate_kwargs",
        )
        call_kwargs = self._offline_build_call_kwargs(generate_kwargs)
        return inputs, input_texts, working_messages_list, call_kwargs

    def offline_batch_generate(
        self,
        processor,
        queries: List[Dict[str, Any]],
        session_states: Optional[List[List[Dict[str, Any]]]] = None,
        vision_chunked_length: int = 64,
    ) -> Dict[str, Any]:
        """
        Batch offline generation for multiple independent samples.

        This method supports:
        - batched single-turn generation
        - batched multi-turn continuation through `session_states`

        It intentionally does not support queue-style controls such as:
        - `stream_output`
        - `cancel_current_generate`
        - `stop_generation`
        - `stop_offline_generate`
        """
        if not queries:
            return {"results": [], "session_states": []}

        prepared_queries = [dict(query) for query in queries]
        for query in prepared_queries:
            generate_kwargs = query.setdefault("generate_kwargs", {})
            generate_kwargs.setdefault("vision_chunked_length", vision_chunked_length)
        if session_states is None:
            session_states = [[] for _ in prepared_queries]
        elif len(session_states) != len(prepared_queries):
            raise ValueError("`session_states` must have the same length as `queries`.")

        tokenizer = getattr(processor, "tokenizer", None)
        bucketed_indices: Dict[Any, List[int]] = {}
        for index, (query, session_state) in enumerate(zip(prepared_queries, session_states)):
            current_session = [] if query.get("reset_session") or query.get("clear_history") else session_state
            working_messages = self._offline_build_session_messages(processor, query, current_session)
            input_text = self._offline_prepare_input_text(processor, working_messages)

            if tokenizer is not None:
                token_ids = tokenizer(input_text, add_special_tokens=False)["input_ids"]
                bucket_key = len(token_ids)
            else:
                bucket_key = len(input_text)
            bucketed_indices.setdefault(bucket_key, []).append(index)

        results: List[Optional[Dict[str, Any]]] = [None] * len(prepared_queries)
        next_session_states: List[Optional[List[Dict[str, Any]]]] = [None] * len(prepared_queries)

        for bucket_indices in bucketed_indices.values():
            bucket_queries = [prepared_queries[index] for index in bucket_indices]
            bucket_session_states = [session_states[index] for index in bucket_indices]
            inputs, input_texts, working_messages_list, call_kwargs = self._offline_prepare_batch_generation(
                processor,
                bucket_queries,
                session_states=bucket_session_states,
            )

            with torch.no_grad():
                outputs = self.generate(
                    **inputs,
                    **call_kwargs,
                )

            input_seq_len = inputs["input_ids"].shape[1]
            generated_tokens = outputs[:, input_seq_len:]
            decoded_texts = processor.batch_decode(generated_tokens, skip_special_tokens=True)

            for local_index, (query, input_text, working_messages, text) in enumerate(
                zip(bucket_queries, input_texts, working_messages_list, decoded_texts)
            ):
                original_index = bucket_indices[local_index]
                if query.get("persist_session", True):
                    next_session_state = self._offline_finalize_session_messages(working_messages, text)
                else:
                    next_session_state = working_messages
                next_session_states[original_index] = next_session_state
                results[original_index] = {
                    "index": original_index,
                    "text": text,
                    "input_text": input_text,
                    "messages": working_messages,
                }

        return {
            "results": [item for item in results if item is not None],
            "session_states": [item for item in next_session_states if item is not None],
        }

    def _offline_generate_one(self, processor, query: Dict[str, Any]) -> str:
        working_messages = self._offline_build_session_messages(processor, query, [])
        generation_query = dict(query)
        generation_query["messages"] = working_messages
        inputs, _, call_kwargs = self._offline_prepare_generation(processor, generation_query)

        with torch.no_grad():
            outputs = self.generate(
                **inputs,
                **call_kwargs,
            )

        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        return processor.decode(new_tokens, skip_special_tokens=True)

    @staticmethod
    def _offline_capture_processor_attrs(target, overrides: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if target is None or not overrides:
            return None
        return {name: copy.deepcopy(getattr(target, name)) for name in overrides}

    @staticmethod
    def _offline_apply_processor_attrs(target, overrides: Optional[Dict[str, Any]]) -> None:
        if target is None or not overrides:
            return
        for name, value in overrides.items():
            setattr(target, name, copy.deepcopy(value))

    @staticmethod
    def _offline_restore_processor_attrs(target, snapshot: Optional[Dict[str, Any]]) -> None:
        if target is None or snapshot is None:
            return
        for name, value in snapshot.items():
            setattr(target, name, copy.deepcopy(value))

    def _offline_generate_one_with_processor_overrides(
        self,
        processor,
        query: Dict[str, Any],
        image_processor_overrides: Optional[Dict[str, Any]] = None,
        video_processor_overrides: Optional[Dict[str, Any]] = None,
    ) -> str:
        image_proc = getattr(processor, "image_processor", None)
        video_proc = getattr(processor, "video_processor", None)
        image_snapshot = self._offline_capture_processor_attrs(image_proc, image_processor_overrides)
        video_snapshot = self._offline_capture_processor_attrs(video_proc, video_processor_overrides)

        with self._offline_processor_lock:
            try:
                self._offline_apply_processor_attrs(image_proc, image_processor_overrides)
                self._offline_apply_processor_attrs(video_proc, video_processor_overrides)
                return self._offline_generate_one(processor, query)
            finally:
                self._offline_restore_processor_attrs(image_proc, image_snapshot)
                self._offline_restore_processor_attrs(video_proc, video_snapshot)

    def offline_image_generate(
        self,
        processor,
        prompt: str,
        image: Any,
        *,
        shortest_edge: int = 4096,
        longest_edge: int = 16777216,
        multi_image_max_pixels: int = 201326592,
        patch_size: int = 16,
        temporal_patch_size: int = 1,
        merge_size: int = 2,
        image_mean: Optional[Union[List[float], Tuple[float, ...]]] = (0.5, 0.5, 0.5),
        image_std: Optional[Union[List[float], Tuple[float, ...]]] = (0.5, 0.5, 0.5),
        max_new_tokens: int = 1024,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
        do_sample: bool = False,
        vision_chunked_length: int = 64,
        thinking_mode: Optional[str] = None,
        system_prompt_type: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ) -> str:
        """
        Single-image offline generation with explicit image preprocessor defaults.

        The default values mirror `preprocessor_config.json` so README examples can
        surface the full image preprocessing setup without requiring a batch wrapper.
        """
        query: Dict[str, Any] = {
            "prompt": prompt,
            "images": [image],
            "videos": [],
            "media_kwargs": {
                "min_pixels": shortest_edge,
                "max_pixels": longest_edge,
                "multi_image_max_pixels": multi_image_max_pixels,
            },
            "generate_kwargs": {
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "top_k": top_k,
                "top_p": top_p,
                "repetition_penalty": repetition_penalty,
                "do_sample": do_sample,
                "vision_chunked_length": vision_chunked_length,
            },
        }
        if thinking_mode is not None:
            query["thinking_mode"] = thinking_mode
        if system_prompt_type is not None:
            query["system_prompt_type"] = system_prompt_type
        if system_prompt is not None:
            query["system_prompt"] = system_prompt

        image_processor_overrides = {
            "size": {"shortest_edge": shortest_edge, "longest_edge": longest_edge},
            "multi_image_max_pixels": multi_image_max_pixels,
            "patch_size": patch_size,
            "temporal_patch_size": temporal_patch_size,
            "merge_size": merge_size,
            "image_mean": list(image_mean) if image_mean is not None else None,
            "image_std": list(image_std) if image_std is not None else None,
        }
        return self._offline_generate_one_with_processor_overrides(
            processor,
            query,
            image_processor_overrides=image_processor_overrides,
        )

    def offline_video_generate(
        self,
        processor,
        prompt: str,
        video: Any,
        *,
        shortest_edge: int = 4096,
        longest_edge: int = 16777216,
        video_max_pixels: int = 201326592,
        patch_size: int = 16,
        temporal_patch_size: int = 1,
        merge_size: int = 2,
        video_fps: float = 1.0,
        min_frames: int = 1,
        max_frames: int = 256,
        num_extract_threads: int = 4,
        image_mean: Optional[Union[List[float], Tuple[float, ...]]] = (0.5, 0.5, 0.5),
        image_std: Optional[Union[List[float], Tuple[float, ...]]] = (0.5, 0.5, 0.5),
        max_new_tokens: int = 1024,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
        do_sample: bool = False,
        vision_chunked_length: int = 64,
        thinking_mode: Optional[str] = None,
        system_prompt_type: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ) -> str:
        """
        Single-video offline generation with explicit video preprocessor defaults.

        The default values mirror `video_preprocessor_config.json` so README examples
        can show a standalone video entry point with the effective preprocessing knobs.
        """
        query: Dict[str, Any] = {
            "prompt": prompt,
            "images": [],
            "videos": [video],
            "media_kwargs": {
                "min_pixels": shortest_edge,
                "max_pixels": longest_edge,
                "video_max_pixels": video_max_pixels,
                "video_fps": video_fps,
                "min_frames": min_frames,
                "max_frames": max_frames,
            },
            "generate_kwargs": {
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "top_k": top_k,
                "top_p": top_p,
                "repetition_penalty": repetition_penalty,
                "do_sample": do_sample,
                "vision_chunked_length": vision_chunked_length,
            },
        }
        if thinking_mode is not None:
            query["thinking_mode"] = thinking_mode
        if system_prompt_type is not None:
            query["system_prompt_type"] = system_prompt_type
        if system_prompt is not None:
            query["system_prompt"] = system_prompt

        video_processor_overrides = {
            "size": {"shortest_edge": shortest_edge, "longest_edge": longest_edge},
            "video_max_pixels": video_max_pixels,
            "patch_size": patch_size,
            "temporal_patch_size": temporal_patch_size,
            "merge_size": merge_size,
            "video_fps": video_fps,
            "min_frames": min_frames,
            "max_frames": max_frames,
            "num_extract_threads": num_extract_threads,
            "image_mean": list(image_mean) if image_mean is not None else None,
            "image_std": list(image_std) if image_std is not None else None,
        }
        return self._offline_generate_one_with_processor_overrides(
            processor,
            query,
            video_processor_overrides=video_processor_overrides,
        )

    def offline_generate(
        self,
        processor,
        new_queries: "queue.Queue[dict]",
        output_text_queue: "queue.Queue[str]",
        vision_chunked_length: int = 64,
    ) -> None:
        """
        HF-style offline inference wrapper aligned with the previous backend output path.

        This method intentionally reuses the checkpoint's existing processor and
        `generate()` flow so that outputs stay consistent with the old external
        backend inference implementation.

        Supported query keys include:
        - `prompt` / `messages`
        - `images` / `videos`
        - `media_kwargs` / `generate_kwargs`
        - `thinking_mode` (`no_thinking` or `deep_thinking`, plus compatible aliases)
        - `system_prompt_type` (`text_image` or `video`, plus compatible aliases)
        - `system_prompt` for an explicit override
        - `stream_output` / `stream`
        - `reset_session` / `clear_history`
        - `cancel_current_generate` / `stop_generation` / `stop_offline_generate`
        """
        buffered_queries: List[Dict[str, Any]] = []
        session_messages: List[Dict[str, Any]] = []

        while True:
            if buffered_queries:
                query = buffered_queries.pop(0)
            else:
                query = new_queries.get()
            if not isinstance(query, dict):
                continue

            if query.get("stop_offline_generate"):
                break

            if query.get("reset_session") or query.get("clear_history"):
                session_messages = []

            try:
                generate_kwargs = query.setdefault("generate_kwargs", {})
                generate_kwargs.setdefault("vision_chunked_length", vision_chunked_length)
                working_messages = self._offline_build_session_messages(
                    processor,
                    query,
                    session_messages,
                )

                generation_query = dict(query)
                generation_query["messages"] = working_messages
                inputs, input_text, call_kwargs = self._offline_prepare_generation(processor, generation_query)

                stream_output = bool(query.get("stream_output", query.get("stream", False)))
                cancel_event = threading.Event()
                stopping_criteria = StoppingCriteriaList([_OfflineCancelStoppingCriteria(cancel_event)])
                generation_state: Dict[str, Any] = {}

                if stream_output:
                    output_text_queue.put("<|round_start|>")
                    streamer = _OfflineQueueStreamer(getattr(processor, "tokenizer", processor), output_text_queue)
                else:
                    streamer = None

                def _run_generation():
                    try:
                        with torch.no_grad():
                            generation_state["outputs"] = self.generate(
                                **inputs,
                                stopping_criteria=stopping_criteria,
                                streamer=streamer,
                                **call_kwargs,
                            )
                    except Exception as exc:
                        generation_state["exception"] = exc

                worker = threading.Thread(target=_run_generation, daemon=True)
                worker.start()

                stop_conversation_after_turn = False
                while worker.is_alive():
                    try:
                        control_query = new_queries.get(timeout=0.1)
                    except queue.Empty:
                        continue

                    if not isinstance(control_query, dict):
                        continue

                    if control_query.get("cancel_current_generate") or control_query.get("stop_generation"):
                        cancel_event.set()
                        stop_conversation_after_turn = stop_conversation_after_turn or control_query.get("stop_offline_generate", False)
                        continue

                    if control_query.get("stop_offline_generate"):
                        cancel_event.set()
                        stop_conversation_after_turn = True
                        continue

                    buffered_queries.append(control_query)

                worker.join()
                was_cancelled = cancel_event.is_set()

                if "exception" in generation_state:
                    raise generation_state["exception"]

                if stream_output and streamer is not None:
                    text = "".join(streamer.collected_chunks)
                else:
                    outputs = generation_state["outputs"]
                    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
                    text = processor.decode(new_tokens, skip_special_tokens=True)
                    output_text_queue.put(text)

                if query.get("persist_session", True) and (not was_cancelled or query.get("persist_cancelled_turn", False)):
                    session_messages = self._offline_finalize_session_messages(working_messages, text)

                output_text_queue.put("<|round_end|>")

                if stop_conversation_after_turn:
                    break
            except Exception as exc:
                output_text_queue.put(f"[ERROR] {exc}")
                output_text_queue.put("<|round_end|>")


__all__ = [
    "MossVLRealtimeSession",
    "MossVLVisionModel",
    "MossVLForConditionalGeneration",
    "MossVLModel",
    "MossVLPreTrainedModel",
    "MossVLTextModel",
]
