# coding=utf-8
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
try:
    # NPU-only acceleration stack (optional): guarded so this checkpoint also
    # loads on CUDA/CPU-only machines. cache_compile below is likewise gated
    # on torch.npu availability — GPU machines run the plain eager methods.
    import torch_npu  # noqa: F401
    import torchair as tng
    from torchair.configs.compiler_config import CompilerConfig
    _TORCHAIR_AVAILABLE = True
except ImportError:
    tng = None
    CompilerConfig = None
    _TORCHAIR_AVAILABLE = False
import torch.nn as nn
import torch.utils.checkpoint
from transformers.activations import ACT2FN
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.gpt2.configuration_gpt2 import GPT2Config

try:
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import pad_input, unpad_input

    _FLASH_ATTN_AVAILABLE = True
except Exception:
    flash_attn_func = None
    flash_attn_varlen_func = None
    pad_input = None
    unpad_input = None
    _FLASH_ATTN_AVAILABLE = False


@dataclass
class PackedSequenceMetadata:
    cu_seqlens: torch.Tensor
    max_seqlen: int
    indices: Optional[torch.Tensor] = None
    batch_size: Optional[int] = None
    seq_len: Optional[int] = None


class MossTTSNanoGPT2RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, base: float = 10000.0) -> None:
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"RoPE head_dim must be even, got {dim}")
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        # repair metadata: transformers' from_pretrained (meta-device loading)
        # can clobber non-persistent buffers with zeros/garbage — the runtime's
        # post-load repair and forward() below use these to restore the
        # correct values (see the CPU adaptation's "cicici noise" root cause).
        self._base = float(base)
        self._dim = int(dim)

    def _compute_inv_freq(self) -> torch.Tensor:
        return 1.0 / (self._base ** (torch.arange(0, self._dim, 2, dtype=torch.float32, device="cpu") / self._dim))

    def forward(
        self,
        position_ids: torch.LongTensor,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # NOTE: no in-forward inv_freq guard — a data-dependent .item() branch
        # breaks torchair's fullgraph compile (gb0170). transformers 4.57.6
        # does not clobber non-persistent buffers on load (the 5.x behavior
        # that motivated the CPU-side guard), and the runtime's post-load
        # repair remains as a belt-and-braces via the _base/_dim attrs above.
        if position_ids.ndim == 1:
            position_ids = position_ids.unsqueeze(0)
        freqs = torch.einsum("bs,d->bsd", position_ids.to(device=device, dtype=self.inv_freq.dtype), self.inv_freq)
        # repeat_interleave(2, dim=-1) rewritten as stack+flatten: the bundled
        # torchair (torch_npu 2.11) has no GE converter for aten.repeat_interleave
        cos_f, sin_f = freqs.cos(), freqs.sin()
        cos = torch.stack((cos_f, cos_f), dim=-1).flatten(start_dim=-2).unsqueeze(2).to(dtype=dtype)
        sin = torch.stack((sin_f, sin_f), dim=-1).flatten(start_dim=-2).unsqueeze(2).to(dtype=dtype)
        return cos, sin


def rotate_half(hidden_states: torch.Tensor) -> torch.Tensor:
    even = hidden_states[..., ::2]
    odd = hidden_states[..., 1::2]
    return torch.stack((-odd, even), dim=-1).reshape_as(hidden_states)


def apply_rotary_pos_emb(
    hidden_states: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    return (hidden_states * cos) + (rotate_half(hidden_states) * sin)


class MossTTSNanoGPT2MLP(nn.Module):
    def __init__(self, config: GPT2Config) -> None:
        super().__init__()
        hidden_size = int(config.hidden_size)
        inner_size = int(config.n_inner or 4 * hidden_size)
        self.fc_in = nn.Linear(hidden_size, inner_size)
        self.fc_out = nn.Linear(inner_size, hidden_size)
        self.act = ACT2FN[config.activation_function]
        self.dropout = nn.Dropout(config.resid_pdrop)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc_in(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.fc_out(hidden_states)
        return self.dropout(hidden_states)


class MossTTSNanoGPT2Attention(nn.Module):
    def __init__(self, config: GPT2Config, layer_idx: int, attn_implementation: str) -> None:
        super().__init__()
        hidden_size = int(config.hidden_size)
        num_heads = int(config.num_attention_heads)
        if hidden_size % num_heads != 0:
            raise ValueError(f"hidden_size={hidden_size} must be divisible by num_attention_heads={num_heads}")

        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.embed_dim = hidden_size
        self.layer_idx = layer_idx
        self.attn_implementation = attn_implementation
        self.attn_dropout = float(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        self.scale_attn_weights = bool(getattr(config, "scale_attn_weights", True))
        self.scale_attn_by_inverse_layer_idx = bool(getattr(config, "scale_attn_by_inverse_layer_idx", False))
        self.position_embedding_type = str(getattr(config, "position_embedding_type", "absolute")).lower()
        if self.position_embedding_type not in {"absolute", "rope"}:
            raise ValueError(f"Unsupported position_embedding_type={self.position_embedding_type!r}")

        self.c_attn = nn.Linear(hidden_size, 3 * hidden_size)
        self.c_proj = nn.Linear(hidden_size, hidden_size)
        self.rotary_emb = None
        if self.position_embedding_type == "rope":
            self.rotary_emb = MossTTSNanoGPT2RotaryEmbedding(
                self.head_dim,
                base=float(getattr(config, "rope_base", 10000.0)),
            )

    def _split_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.ndim == 3:
            batch_size, seq_len, _ = tensor.shape
            return tensor.view(batch_size, seq_len, self.num_heads, self.head_dim)
        if tensor.ndim == 2:
            total_tokens, _ = tensor.shape
            return tensor.view(total_tokens, self.num_heads, self.head_dim)
        raise ValueError(f"Unsupported tensor rank for attention split: {tensor.ndim}")

    def _merge_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.ndim == 4:
            batch_size, seq_len, _, _ = tensor.shape
            return tensor.reshape(batch_size, seq_len, self.embed_dim)
        if tensor.ndim == 3:
            total_tokens, _, _ = tensor.shape
            return tensor.reshape(total_tokens, self.embed_dim)
        raise ValueError(f"Unsupported tensor rank for attention merge: {tensor.ndim}")

    def _causal_attention_mask(
        self,
        attention_mask: Optional[torch.Tensor],
        query_length: int,
        key_length: int,
        device: torch.device,
    ) -> torch.Tensor:
        query_positions = torch.arange(query_length, device=device, dtype=torch.long)
        query_positions = query_positions + max(key_length - query_length, 0)
        key_positions = torch.arange(key_length, device=device, dtype=torch.long)
        causal = key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)
        causal = causal.unsqueeze(0).unsqueeze(0)
        if attention_mask is None:
            return causal
        key_mask = attention_mask[:, None, None, :].to(dtype=torch.bool)
        return causal & key_mask

    def _eager_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        scale = 1.0
        if self.scale_attn_weights:
            scale /= self.head_dim ** 0.5
        if self.scale_attn_by_inverse_layer_idx:
            scale /= float(self.layer_idx + 1)

        scores = torch.matmul(query, key.transpose(-1, -2)) * scale
        causal_mask = self._causal_attention_mask(
            attention_mask=attention_mask,
            query_length=query.shape[-2],
            key_length=key.shape[-2],
            device=query.device,
        )
        scores = scores.masked_fill(~causal_mask, torch.finfo(scores.dtype).min)
        probs = torch.softmax(scores, dim=-1)
        if self.training and self.attn_dropout > 0:
            probs = torch.dropout(probs, self.attn_dropout, train=True)
        output = torch.matmul(probs, value)
        return output.transpose(1, 2).contiguous()

    def _sdpa_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        # torch_npu 2.11 dispatches F.scaled_dot_product_attention to
        # npu_fusion_attention_v3, whose torchair GE converter is unavailable on
        # CANN 9.0.0 — run the math path instead (_eager_attention is
        # semantically identical: same causal & key mask, same scale).
        return self._eager_attention(query=query, key=key, value=value, attention_mask=attention_mask)

    def _sdpa_attention_prefill(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        # math path (same npu_fusion_attention_v3 GE-converter workaround);
        # keeps the original's padded-query-row zeroing on the (B,S,N,D) output.
        query_attention_mask = None
        if attention_mask is not None:
            query_length = query.shape[1]
            query_attention_mask = attention_mask[:, -query_length:].to(dtype=torch.bool, device=query.device)
        output = self._eager_attention(query=query, key=key, value=value, attention_mask=attention_mask)
        if query_attention_mask is not None:
            output = output.masked_fill(~query_attention_mask[:, :, None, None], 0.0)
        return output

    def _flash_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        packed_metadata: Optional[PackedSequenceMetadata],
    ) -> torch.Tensor:
        if not _FLASH_ATTN_AVAILABLE:
            raise ImportError("flash_attn is not installed, but attn_implementation='flash_attention_2' was requested.")
        if query.device.type != "cuda":
            raise ValueError("flash_attention_2 requires CUDA tensors.")
        if query.dtype not in (torch.float16, torch.bfloat16):
            raise ValueError(
                f"flash_attention_2 requires fp16/bf16 tensors, but received dtype={query.dtype}."
            )

        dropout_p = self.attn_dropout if self.training else 0.0
        if packed_metadata is not None:
            if packed_metadata.indices is not None:
                query = query.reshape(-1, self.num_heads, self.head_dim).index_select(0, packed_metadata.indices)
                key = key.reshape(-1, self.num_heads, self.head_dim).index_select(0, packed_metadata.indices)
                value = value.reshape(-1, self.num_heads, self.head_dim).index_select(0, packed_metadata.indices)
            output = flash_attn_varlen_func(
                query,
                key,
                value,
                packed_metadata.cu_seqlens,
                packed_metadata.cu_seqlens,
                packed_metadata.max_seqlen,
                packed_metadata.max_seqlen,
                dropout_p=dropout_p,
                causal=True,
            )
            if packed_metadata.indices is None:
                return output
            return pad_input(
                output,
                packed_metadata.indices,
                packed_metadata.batch_size,
                packed_metadata.seq_len,
            )

        if attention_mask is None or bool(attention_mask.all()):
            return flash_attn_func(
                query,
                key,
                value,
                dropout_p=dropout_p,
                causal=True,
            )

        unpadded_query, indices, cu_seqlens, max_seqlen, _ = unpad_input(query, attention_mask)
        unpadded_key, _, _, _, _ = unpad_input(key, attention_mask)
        unpadded_value, _, _, _, _ = unpad_input(value, attention_mask)
        output = flash_attn_varlen_func(
            unpadded_query,
            unpadded_key,
            unpadded_value,
            cu_seqlens,
            cu_seqlens,
            max_seqlen,
            max_seqlen,
            dropout_p=dropout_p,
            causal=True,
        )
        return pad_input(output, indices, query.shape[0], query.shape[1])

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        packed_metadata: Optional[PackedSequenceMetadata] = None,
        layer_past: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        prefill: bool = False,
    ) -> tuple[torch.Tensor, Optional[tuple[torch.Tensor, torch.Tensor]]]:
        qkv = self.c_attn(hidden_states)
        query, key, value = qkv.split(self.embed_dim, dim=-1)
        query = self._split_heads(query)
        key = self._split_heads(key)
        value = self._split_heads(value)

        if self.rotary_emb is not None:
            if position_ids is None:
                raise ValueError("position_ids must be provided when position_embedding_type='rope'.")
            cos, sin = self.rotary_emb(
                position_ids.to(device=query.device),
                device=query.device,
                dtype=query.dtype,
            )
            query = apply_rotary_pos_emb(query, cos, sin)
            key = apply_rotary_pos_emb(key, cos, sin)

        if layer_past is not None:
            past_key, past_value = layer_past
            index = position_ids[0, 0].view(1, 1, 1, 1).expand_as(key)
            past_key.scatter_(1, index, key)
            past_value.scatter_(1, index, value)
            key = past_key
            value = past_value

        present = (key, value) if use_cache else None

        if prefill:
            attn_output = self._sdpa_attention_prefill(
                query=query,
                key=key,
                value=value,
                attention_mask=attention_mask,
            )
        elif self.attn_implementation == "flash_attention_2" and layer_past is None:
            attn_output = self._flash_attention(
                query=query,
                key=key,
                value=value,
                attention_mask=attention_mask,
                packed_metadata=packed_metadata,
            )
        elif self.attn_implementation == "sdpa":
            attn_output = self._sdpa_attention(
                query=query,
                key=key,
                value=value,
                attention_mask=attention_mask,
            )
        else:
            attn_output = self._eager_attention(
                query=query,
                key=key,
                value=value,
                attention_mask=attention_mask,
            )

        attn_output = self._merge_heads(attn_output)
        attn_output = self.c_proj(attn_output)
        return self.resid_dropout(attn_output), present


class MossTTSNanoGPT2Block(nn.Module):
    def __init__(self, config: GPT2Config, layer_idx: int, attn_implementation: str) -> None:
        super().__init__()
        hidden_size = int(config.hidden_size)
        self.ln_1 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.attn = MossTTSNanoGPT2Attention(config, layer_idx=layer_idx, attn_implementation=attn_implementation)
        self.ln_2 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.mlp = MossTTSNanoGPT2MLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        packed_metadata: Optional[PackedSequenceMetadata] = None,
        layer_past: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        prefill: bool = False,
    ) -> tuple[torch.Tensor, Optional[tuple[torch.Tensor, torch.Tensor]]]:
        attn_output, present = self.attn(
            self.ln_1(hidden_states),
            attention_mask=attention_mask,
            position_ids=position_ids,
            packed_metadata=packed_metadata,
            layer_past=layer_past,
            use_cache=use_cache,
            prefill=prefill,
        )
        hidden_states = hidden_states + attn_output
        hidden_states = hidden_states + self.mlp(self.ln_2(hidden_states))
        return hidden_states, present


class MossTTSNanoGPT2Model(nn.Module):
    def __init__(self, config: GPT2Config, attn_implementation: str = "eager") -> None:
        super().__init__()
        self.config = config
        self.attn_implementation = attn_implementation
        self.position_embedding_type = str(getattr(config, "position_embedding_type", "absolute")).lower()
        if self.position_embedding_type not in {"absolute", "rope"}:
            raise ValueError(f"Unsupported position_embedding_type={self.position_embedding_type!r}")
        hidden_size = int(config.hidden_size)
        self.wte = nn.Embedding(config.vocab_size, hidden_size)
        self.wpe = nn.Embedding(config.n_positions, hidden_size) if self.position_embedding_type == "absolute" else nn.Identity()
        self.drop = nn.Dropout(config.embd_pdrop)
        self.h = nn.ModuleList(
            [MossTTSNanoGPT2Block(config, layer_idx=index, attn_implementation=attn_implementation) for index in range(config.n_layer)]
        )
        self.ln_f = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.gradient_checkpointing = False
        self._reset_parameters()

        # torchair graph-compiled prefill/decode (NPU only): CUDA/CPU machines
        # bind the plain eager methods instead — the caller
        # (modeling_moss_tts_nano.inference_stream) only requires the two
        # attributes to exist.
        if _TORCHAIR_AVAILABLE and torch.npu.is_available():
            _cfg = CompilerConfig()
            _cfg.experimental_config.frozen_parameter = True
            _cfg.experimental_config.tiling_schedule_optimize = True
            self.cached_prefill = tng.inference.cache_compile(self.prefill, config=_cfg)
            self.cached_decode = tng.inference.cache_compile(self.decode, config=_cfg)
        else:
            self.cached_prefill = self.prefill
            self.cached_decode = self.decode

    def _reset_parameters(self) -> None:
        init_std = float(self.config.initializer_range)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=init_std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=init_std)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    @staticmethod
    def _normalize_num_sequences(
        cu_seqlens: torch.Tensor,
        num_sequences: Optional[torch.Tensor],
        device: torch.device,
    ) -> torch.Tensor:
        if cu_seqlens.ndim == 1:
            cu_seqlens = cu_seqlens.unsqueeze(0)
        if num_sequences is None:
            counts = []
            for boundary in cu_seqlens:
                diffs = boundary[1:] - boundary[:-1]
                counts.append(int((diffs > 0).sum().item()))
            return torch.tensor(counts, dtype=torch.int32, device=device)
        if num_sequences.ndim == 0:
            return num_sequences.unsqueeze(0)
        return num_sequences

    @staticmethod
    def build_packed_position_ids(
        attention_mask: Optional[torch.Tensor],
        cu_seqlens: torch.Tensor,
        num_sequences: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if cu_seqlens.ndim == 1:
            cu_seqlens = cu_seqlens.unsqueeze(0)
        batch_size, seq_len = cu_seqlens.shape[0], cu_seqlens.shape[1] - 1
        device = cu_seqlens.device
        position_ids = torch.zeros((batch_size, seq_len), dtype=torch.long, device=device)
        counts = MossTTSNanoGPT2Model._normalize_num_sequences(cu_seqlens, num_sequences, device=device)
        for batch_index in range(batch_size):
            sequence_count = int(counts[batch_index].item())
            boundaries = cu_seqlens[batch_index, : sequence_count + 1].tolist()
            for start, end in zip(boundaries[:-1], boundaries[1:]):
                start = int(start)
                end = int(end)
                if end > start:
                    position_ids[batch_index, start:end] = torch.arange(end - start, device=device)
        if attention_mask is not None:
            position_ids = position_ids * attention_mask.to(dtype=position_ids.dtype)
        return position_ids

    @staticmethod
    def build_packed_metadata(
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        num_sequences: Optional[torch.Tensor],
    ) -> PackedSequenceMetadata:
        if cu_seqlens.ndim == 1:
            cu_seqlens = cu_seqlens.unsqueeze(0)
        device = hidden_states.device
        counts = MossTTSNanoGPT2Model._normalize_num_sequences(cu_seqlens, num_sequences, device=device)
        flat_indices = []
        cumulative = [0]
        max_seqlen = 0
        seq_len = hidden_states.shape[1]

        for batch_index in range(hidden_states.shape[0]):
            sequence_count = int(counts[batch_index].item())
            boundaries = cu_seqlens[batch_index, : sequence_count + 1].tolist()
            for start, end in zip(boundaries[:-1], boundaries[1:]):
                start = int(start)
                end = int(end)
                if end <= start:
                    continue
                segment_indices = batch_index * seq_len + torch.arange(start, end, device=device)
                flat_indices.append(segment_indices)
                cumulative.append(cumulative[-1] + (end - start))
                max_seqlen = max(max_seqlen, end - start)

        if not flat_indices:
            raise ValueError("cu_seqlens did not describe any non-empty packed sequences.")

        indices = torch.cat(flat_indices, dim=0)
        return PackedSequenceMetadata(
            cu_seqlens=torch.tensor(cumulative, dtype=torch.int32, device=device),
            max_seqlen=max_seqlen,
            indices=indices,
            batch_size=hidden_states.shape[0],
            seq_len=hidden_states.shape[1],
        )

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[tuple[tuple[torch.Tensor, torch.Tensor], ...]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: bool = True,
        cu_seqlens: Optional[torch.Tensor] = None,
        num_sequences: Optional[torch.Tensor] = None,
    ) -> BaseModelOutputWithPast:
        del input_ids, output_attentions

        if inputs_embeds is None:
            raise ValueError("inputs_embeds must be provided.")

        use_cache = bool(use_cache)
        if use_cache and cu_seqlens is not None:
            raise ValueError("use_cache=True is not supported together with cu_seqlens packing.")

        hidden_states = inputs_embeds
        if attention_mask is None:
            attention_mask = torch.ones(hidden_states.shape[:2], dtype=torch.bool, device=hidden_states.device)
        else:
            attention_mask = attention_mask.to(dtype=torch.bool, device=hidden_states.device)
        query_attention_mask = attention_mask[:, -hidden_states.shape[1] :]

        packed_metadata = None
        if position_ids is None:
            if cu_seqlens is not None:
                position_ids = self.build_packed_position_ids(
                    attention_mask=attention_mask,
                    cu_seqlens=cu_seqlens.to(device=hidden_states.device),
                    num_sequences=num_sequences.to(device=hidden_states.device) if num_sequences is not None else None,
                )
            elif attention_mask is not None:
                position_ids = attention_mask.long().cumsum(dim=-1) - 1
                position_ids = position_ids.masked_fill(~attention_mask, 0)
                position_ids = position_ids[:, -hidden_states.shape[1] :]
            else:
                past_length = 0
                if past_key_values is not None and len(past_key_values) > 0:
                    past_length = past_key_values[0][0].shape[1]
                position_ids = torch.arange(hidden_states.shape[1], device=hidden_states.device, dtype=torch.long)
                position_ids = position_ids + past_length
                position_ids = position_ids.unsqueeze(0).expand(hidden_states.shape[0], -1)

        if cu_seqlens is not None and self.attn_implementation == "flash_attention_2":
            packed_metadata = self.build_packed_metadata(
                hidden_states=hidden_states,
                cu_seqlens=cu_seqlens.to(device=hidden_states.device),
                num_sequences=num_sequences.to(device=hidden_states.device) if num_sequences is not None else None,
            )

        if self.position_embedding_type == "absolute":
            hidden_states = hidden_states + self.wpe(position_ids)
        hidden_states = self.drop(hidden_states)
        hidden_states = hidden_states * query_attention_mask.unsqueeze(-1).to(dtype=hidden_states.dtype)

        all_hidden_states = () if output_hidden_states else None
        presents = [] if use_cache else None
        for layer_index, block in enumerate(self.h):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:
                if use_cache:
                    raise ValueError("use_cache=True is not supported when gradient checkpointing is enabled during training.")

                def custom_forward(*inputs):
                    output, _ = block(
                        inputs[0],
                        attention_mask=inputs[1],
                        position_ids=inputs[2],
                        packed_metadata=packed_metadata,
                        layer_past=None,
                        use_cache=False,
                    )
                    return output

                hidden_states = torch.utils.checkpoint.checkpoint(
                    custom_forward,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    use_reentrant=False,
                )
                present = None
            else:
                hidden_states, present = block(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    packed_metadata=packed_metadata,
                    layer_past=None if past_key_values is None else past_key_values[layer_index],
                    use_cache=use_cache,
                )
            hidden_states = hidden_states * query_attention_mask.unsqueeze(-1).to(dtype=hidden_states.dtype)
            if presents is not None:
                presents.append(present)

        hidden_states = self.ln_f(hidden_states)
        hidden_states = hidden_states * query_attention_mask.unsqueeze(-1).to(dtype=hidden_states.dtype)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return (hidden_states, tuple(presents) if presents is not None else None, all_hidden_states, None)

        from transformers.modeling_outputs import BaseModelOutputWithPast  # torchair kernel-namespace safe
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=tuple(presents) if presents is not None else None,
            hidden_states=all_hidden_states,
            attentions=None,
        )

    def decode(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[tuple[tuple[torch.Tensor, torch.Tensor], ...]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: bool = True,
        cu_seqlens: Optional[torch.Tensor] = None,
        num_sequences: Optional[torch.Tensor] = None,
    ) -> BaseModelOutputWithPast:
        del input_ids, output_attentions

        if inputs_embeds is None:
            raise ValueError("inputs_embeds must be provided.")

        use_cache = bool(use_cache)
        if use_cache and cu_seqlens is not None:
            raise ValueError("use_cache=True is not supported together with cu_seqlens packing.")

        hidden_states = inputs_embeds
        attention_mask = attention_mask.to(dtype=torch.bool, device=hidden_states.device)

        packed_metadata = None

        hidden_states = self.drop(hidden_states)

        all_hidden_states = () if output_hidden_states else None
        presents = [] if use_cache else None
        for layer_index, block in enumerate(self.h):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            hidden_states, present = block(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                packed_metadata=packed_metadata,
                layer_past=None if past_key_values is None else past_key_values[layer_index],
                use_cache=use_cache,
            )
            if presents is not None:
                presents.append(present)

        hidden_states = self.ln_f(hidden_states)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return (hidden_states, tuple(presents) if presents is not None else None, all_hidden_states, None)

        from transformers.modeling_outputs import BaseModelOutputWithPast  # torchair kernel-namespace safe
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=tuple(presents) if presents is not None else None,
            hidden_states=all_hidden_states,
            attentions=None,
        )

    def prefill(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[tuple[tuple[torch.Tensor, torch.Tensor], ...]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: bool = True,
        cu_seqlens: Optional[torch.Tensor] = None,
        num_sequences: Optional[torch.Tensor] = None,
    ) -> BaseModelOutputWithPast:
        del input_ids, output_attentions

        if inputs_embeds is None:
            raise ValueError("inputs_embeds must be provided.")

        hidden_states = inputs_embeds

        assert attention_mask is not None
        attention_mask = attention_mask.to(dtype=torch.bool, device=hidden_states.device)
        query_attention_mask = attention_mask[:, -hidden_states.shape[1] :]

        packed_metadata = None
        position_ids = attention_mask.long().cumsum(dim=-1) - 1
        position_ids = position_ids.masked_fill(~attention_mask, 0)
        position_ids = position_ids[:, -hidden_states.shape[1] :]

        hidden_states = self.drop(hidden_states)
        hidden_states = hidden_states * query_attention_mask.unsqueeze(-1).to(dtype=hidden_states.dtype)

        all_hidden_states = () if output_hidden_states else None
        presents = [] if use_cache else None
        for layer_index, block in enumerate(self.h):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            hidden_states, present = block(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                packed_metadata=packed_metadata,
                layer_past=None if past_key_values is None else past_key_values[layer_index],
                use_cache=use_cache,
                prefill=True
            )
            hidden_states = hidden_states * query_attention_mask.unsqueeze(-1).to(dtype=hidden_states.dtype)
            if presents is not None:
                presents.append(present)

        hidden_states = self.ln_f(hidden_states)
        hidden_states = hidden_states * query_attention_mask.unsqueeze(-1).to(dtype=hidden_states.dtype)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return (hidden_states, tuple(presents) if presents is not None else None, all_hidden_states, None)

        from transformers.modeling_outputs import BaseModelOutputWithPast  # torchair kernel-namespace safe
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=tuple(presents) if presents is not None else None,
            hidden_states=all_hidden_states,
            attentions=None,
        )

    def forward_local(
        self,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
    ) -> BaseModelOutputWithPast:
        if inputs_embeds is None:
            raise ValueError("inputs_embeds must be provided.")

        hidden_states = inputs_embeds
        if attention_mask is None:
            attention_mask = torch.ones(hidden_states.shape[:2], dtype=torch.bool, device=hidden_states.device)
        else:
            attention_mask = attention_mask.to(dtype=torch.bool, device=hidden_states.device)
        query_attention_mask = attention_mask[:, -hidden_states.shape[1] :]

        packed_metadata = None
        position_ids = None
        cu_seqlens = None
        output_hidden_states = None
        use_cache = None
        past_key_values = None
        return_dict = True
        if position_ids is None:
            if cu_seqlens is not None:
                position_ids = self.build_packed_position_ids(
                    attention_mask=attention_mask,
                    cu_seqlens=cu_seqlens.to(device=hidden_states.device),
                    num_sequences=num_sequences.to(device=hidden_states.device) if num_sequences is not None else None,
                )
            elif attention_mask is not None:
                position_ids = attention_mask.long().cumsum(dim=-1) - 1
                position_ids = position_ids.masked_fill(~attention_mask, 0)
                position_ids = position_ids[:, -hidden_states.shape[1] :]
            else:
                past_length = 0
                if past_key_values is not None and len(past_key_values) > 0:
                    past_length = past_key_values[0][0].shape[1]
                position_ids = torch.arange(hidden_states.shape[1], device=hidden_states.device, dtype=torch.long)
                position_ids = position_ids + past_length
                position_ids = position_ids.unsqueeze(0).expand(hidden_states.shape[0], -1)

        if cu_seqlens is not None and self.attn_implementation == "flash_attention_2":
            packed_metadata = self.build_packed_metadata(
                hidden_states=hidden_states,
                cu_seqlens=cu_seqlens.to(device=hidden_states.device),
                num_sequences=num_sequences.to(device=hidden_states.device) if num_sequences is not None else None,
            )

        if self.position_embedding_type == "absolute":
            hidden_states = hidden_states + self.wpe(position_ids)
        hidden_states = self.drop(hidden_states)
        hidden_states = hidden_states * query_attention_mask.unsqueeze(-1).to(dtype=hidden_states.dtype)

        all_hidden_states = () if output_hidden_states else None
        presents = [] if use_cache else None
        for layer_index, block in enumerate(self.h):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:
                if use_cache:
                    raise ValueError("use_cache=True is not supported when gradient checkpointing is enabled during training.")

                def custom_forward(*inputs):
                    output, _ = block(
                        inputs[0],
                        attention_mask=inputs[1],
                        position_ids=inputs[2],
                        packed_metadata=packed_metadata,
                        layer_past=None,
                        use_cache=False,
                    )
                    return output

                hidden_states = torch.utils.checkpoint.checkpoint(
                    custom_forward,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    use_reentrant=False,
                )
                present = None
            else:
                hidden_states, present = block(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    packed_metadata=packed_metadata,
                    layer_past=None if past_key_values is None else past_key_values[layer_index],
                    use_cache=use_cache,
                )
            hidden_states = hidden_states * query_attention_mask.unsqueeze(-1).to(dtype=hidden_states.dtype)
            if presents is not None:
                presents.append(present)

        hidden_states = self.ln_f(hidden_states)
        hidden_states = hidden_states * query_attention_mask.unsqueeze(-1).to(dtype=hidden_states.dtype)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return (hidden_states, tuple(presents) if presents is not None else None, all_hidden_states, None)

        from transformers.modeling_outputs import BaseModelOutputWithPast  # torchair kernel-namespace safe
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=tuple(presents) if presents is not None else None,
            hidden_states=all_hidden_states,
            attentions=None,
        )
