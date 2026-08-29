# coding=utf-8
from __future__ import annotations

import json
import logging
import os
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn as nn
import torchaudio
from transformers import AutoModel, AutoTokenizer
from transformers.modeling_outputs import ModelOutput
from transformers.modeling_utils import PreTrainedModel
from transformers.models.gpt2.configuration_gpt2 import GPT2Config

from .configuration_moss_tts_nano import MossTTSNanoConfig
from .gpt2_decoder import MossTTSNanoGPT2Block, MossTTSNanoGPT2Model
from .prompting import (
    build_assistant_prompt_prefix,
    build_prompt_token_ids,
    build_user_prompt_after_reference,
    build_user_prompt_prefix,
)
from .tokenization_moss_tts_nano import MossTTSNanoSentencePieceTokenizer


@dataclass
class MossTTSNanoOutput(ModelOutput):
    global_hidden_states: Optional[torch.FloatTensor] = None
    past_key_values: Optional[tuple[tuple[torch.Tensor, torch.Tensor], ...]] = None
    hidden_states: Optional[tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[tuple[torch.FloatTensor, ...]] = None


@dataclass
class MossTTSNanoGenerationOutput(ModelOutput):
    audio_token_ids: torch.LongTensor
    prompt_input_ids: Optional[torch.LongTensor] = None


MOSS_AUDIO_TOKENIZER_NANO_TYPE = "moss-audio-tokenizer-nano"
DEFAULT_MOSS_AUDIO_TOKENIZER_PRETRAINED_NAME_OR_PATH = "OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano"
DEFAULT_VOICE_CLONE_MAX_TEXT_TOKENS = 50
DEFAULT_VOICE_CLONE_MAX_MEMORY_PER_SAMPLE_GB = 1.0
DEFAULT_VOICE_CLONE_INTER_CHUNK_PAUSE_SHORT_SECONDS = 0.40
DEFAULT_VOICE_CLONE_INTER_CHUNK_PAUSE_LONG_SECONDS = 0.24
_SENTENCE_END_PUNCTUATION = frozenset(".!?。！？；;")
_CLAUSE_SPLIT_PUNCTUATION = frozenset(",，、；;：:")
_CLOSING_PUNCTUATION = frozenset("\"'”’)]}）】》」』")


class MossTTSNanoPreTrainedModel(PreTrainedModel):
    config_class = MossTTSNanoConfig
    base_model_prefix = "transformer"
    supports_gradient_checkpointing = False
    _no_split_modules = ["MossTTSNanoGPT2Block"]
    _supports_flash_attn_2 = True
    _supports_sdpa = True


class MossTTSNanoForCausalLM(MossTTSNanoPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"local_transformer\.wte\.weight"]

    def __init__(self, config: MossTTSNanoConfig) -> None:
        super().__init__(config)
        config.gpt2_config.pad_token_id = config.pad_token_id
        config.gpt2_config._attn_implementation = config.attn_implementation

        self.transformer = MossTTSNanoGPT2Model(
            config.gpt2_config,
            attn_implementation=config.attn_implementation,
        )
        hidden_size = config.gpt2_config.hidden_size
        init_std = config.gpt2_config.initializer_range

        self.audio_embeddings = nn.ModuleList(
            [
                nn.Embedding(int(config.audio_codebook_sizes[index]), hidden_size)
                for index in range(config.n_vq)
            ]
        )
        self.text_lm_head = nn.Linear(hidden_size, config.gpt2_config.vocab_size, bias=False)
        self.audio_lm_heads = nn.ModuleList(
            [
                nn.Linear(hidden_size, int(config.audio_codebook_sizes[index]), bias=False)
                for index in range(config.n_vq)
            ]
        )

        local_gpt2_config = config.gpt2_config.to_dict()
        local_gpt2_config["n_layer"] = int(config.local_transformer_layers)
        local_gpt2_config["n_positions"] = config.n_vq + 1
        local_gpt2_config["n_ctx"] = config.n_vq + 1
        self.local_transformer = MossTTSNanoGPT2Model(
            GPT2Config(**local_gpt2_config),
            attn_implementation=str(config.local_transformer_attn_implementation),
        )
        self.local_transformer.wte = nn.Identity()

        for module in list(self.audio_embeddings) + [self.text_lm_head] + list(self.audio_lm_heads):
            if hasattr(module, "weight") and module.weight is not None:
                nn.init.normal_(module.weight, mean=0.0, std=init_std)

        self._tied_weights_keys = tuple(self.all_tied_weights_keys.keys())
        self.tie_weights()

    @property
    def all_tied_weights_keys(self) -> dict[str, str]:
        tied_weights = {"text_lm_head.weight": "transformer.wte.weight"}
        tied_weights.update(
            {
                f"audio_lm_heads.{index}.weight": f"audio_embeddings.{index}.weight"
                for index in range(self.config.n_vq)
            }
        )
        return tied_weights

    def tie_weights(self, *args, **kwargs) -> None:
        del args, kwargs
        self.text_lm_head.weight = self.transformer.wte.weight
        for embedding, lm_head in zip(self.audio_embeddings, self.audio_lm_heads):
            lm_head.weight = embedding.weight

    def get_input_embeddings(self) -> nn.Embedding:
        return self.transformer.wte

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        self.transformer.wte = value
        self.tie_weights()

    def _build_inputs_embeds(self, input_ids: torch.LongTensor) -> torch.FloatTensor:
        if input_ids.ndim != 3 or input_ids.shape[-1] != self.config.n_vq + 1:
            raise ValueError(
                f"Expected input_ids shape [batch, seq, {self.config.n_vq + 1}], got {tuple(input_ids.shape)}"
            )

        text_ids = input_ids[..., 0]
        inputs_embeds = self.transformer.wte(text_ids)

        for channel_index, embedding in enumerate(self.audio_embeddings):
            channel_ids = input_ids[..., channel_index + 1]
            valid_mask = channel_ids.ne(self.config.audio_pad_token_id)
            invalid_mask = valid_mask & ((channel_ids < 0) | (channel_ids >= embedding.num_embeddings))
            if invalid_mask.any():
                invalid_token_ids = channel_ids[invalid_mask]
                raise ValueError(
                    "Found out-of-range audio token ids for channel "
                    f"{channel_index}: min={int(invalid_token_ids.min().item())} "
                    f"max={int(invalid_token_ids.max().item())} "
                    f"codebook_size={embedding.num_embeddings} "
                    f"audio_pad_token_id={self.config.audio_pad_token_id}"
                )
            safe_ids = channel_ids.masked_fill(~valid_mask, 0)
            audio_embeds = embedding(safe_ids)
            audio_embeds = audio_embeds * valid_mask.unsqueeze(-1)
            inputs_embeds = inputs_embeds + audio_embeds

        return inputs_embeds

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[tuple[tuple[torch.Tensor, torch.Tensor], ...]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ):
        labels = kwargs.pop("labels", None)
        if labels is not None:
            raise NotImplementedError("This open-source package is inference-only and does not support training forward.")
        if kwargs:
            ignored = ", ".join(sorted(kwargs.keys()))
            logging.debug("ignoring unsupported forward kwargs: %s", ignored)

        return_dict = self.config.use_return_dict if return_dict is None else return_dict
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("Either input_ids or inputs_embeds must be provided.")
            inputs_embeds = self._build_inputs_embeds(input_ids)

        outputs = self.transformer(
            input_ids=None,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            position_ids=None,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            cu_seqlens=None,
            num_sequences=None,
        )

        if not return_dict:
            return (
                outputs.last_hidden_state,
                outputs.past_key_values,
                outputs.hidden_states,
                outputs.attentions,
            )

        return MossTTSNanoOutput(
            global_hidden_states=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def _build_text_rows(
        self,
        token_ids: list[int],
        device: torch.device,
    ) -> torch.LongTensor:
        rows = torch.full(
            (len(token_ids), self.config.n_vq + 1),
            self.config.audio_pad_token_id,
            dtype=torch.long,
            device=device,
        )
        if token_ids:
            rows[:, 0] = torch.tensor(token_ids, dtype=torch.long, device=device)
        return rows

    def _encode_text(self, tokenizer, text: str) -> list[int]:
        try:
            return list(tokenizer.encode(text, add_special_tokens=False))
        except TypeError:
            return list(tokenizer.encode(text))

    @staticmethod
    def _contains_cjk(text: str) -> bool:
        return any(
            "\u4e00" <= ch <= "\u9fff"
            or "\u3400" <= ch <= "\u4dbf"
            or "\u3040" <= ch <= "\u30ff"
            or "\uac00" <= ch <= "\ud7af"
            for ch in str(text)
        )

    @staticmethod
    def _prepare_text_for_sentence_chunking(text: str) -> str:
        normalized_text = str(text).strip()
        if normalized_text == "":
            raise ValueError("Text prompt cannot be empty.")

        normalized_text = normalized_text.replace("\n", " ").replace("\r", " ")
        while "  " in normalized_text:
            normalized_text = normalized_text.replace("  ", " ")

        contains_cjk = MossTTSNanoForCausalLM._contains_cjk(normalized_text)
        if contains_cjk:
            if normalized_text[-1] not in _SENTENCE_END_PUNCTUATION:
                normalized_text = normalized_text + "。"
            return normalized_text

        if not normalized_text[0].isupper():
            normalized_text = normalized_text[0].upper() + normalized_text[1:]
        if normalized_text[-1].isalnum():
            normalized_text = normalized_text + "."
        if len(normalized_text.split()) < 5:
            normalized_text = " " * 8 + normalized_text
        return normalized_text

    @staticmethod
    def _split_text_by_punctuation(text: str, punctuation: set[str] | frozenset[str]) -> list[str]:
        sentences: list[str] = []
        current_chars: list[str] = []
        text = str(text)
        index = 0
        while index < len(text):
            char = text[index]
            current_chars.append(char)
            if char in punctuation:
                lookahead = index + 1
                while lookahead < len(text) and text[lookahead] in _CLOSING_PUNCTUATION:
                    current_chars.append(text[lookahead])
                    lookahead += 1
                sentence = "".join(current_chars).strip()
                if sentence:
                    sentences.append(sentence)
                current_chars = []
                while lookahead < len(text) and text[lookahead].isspace():
                    lookahead += 1
                index = lookahead
                continue
            index += 1

        tail = "".join(current_chars).strip()
        if tail:
            sentences.append(tail)
        return sentences

    def _count_text_tokens(self, text_tokenizer, text: str) -> int:
        return len(self._encode_text(text_tokenizer, text))

    def _split_text_by_token_budget(
        self,
        text_tokenizer,
        text: str,
        max_tokens: int,
    ) -> list[str]:
        remaining_text = str(text).strip()
        if remaining_text == "":
            return []

        pieces: list[str] = []
        preferred_boundary_chars = _CLAUSE_SPLIT_PUNCTUATION | _SENTENCE_END_PUNCTUATION | frozenset({" "})
        while remaining_text:
            if self._count_text_tokens(text_tokenizer, remaining_text) <= int(max_tokens):
                pieces.append(remaining_text)
                break

            low = 1
            high = len(remaining_text)
            best_prefix_length = 1
            while low <= high:
                middle = (low + high) // 2
                candidate = remaining_text[:middle].strip()
                if not candidate:
                    low = middle + 1
                    continue
                if self._count_text_tokens(text_tokenizer, candidate) <= int(max_tokens):
                    best_prefix_length = middle
                    low = middle + 1
                else:
                    high = middle - 1

            cut_index = best_prefix_length
            prefix = remaining_text[:best_prefix_length]
            preferred_index = -1
            for scan_index in range(len(prefix) - 1, max(-1, len(prefix) - 25), -1):
                if prefix[scan_index] in preferred_boundary_chars:
                    preferred_index = scan_index + 1
                    break
            if preferred_index > 0:
                cut_index = preferred_index

            piece = remaining_text[:cut_index].strip()
            if not piece:
                piece = remaining_text[:best_prefix_length].strip()
                cut_index = best_prefix_length
            pieces.append(piece)
            remaining_text = remaining_text[cut_index:].strip()
        return pieces

    @staticmethod
    def _join_sentence_parts(left: str, right: str) -> str:
        if not left:
            return right
        if not right:
            return left
        if MossTTSNanoForCausalLM._contains_cjk(left) or MossTTSNanoForCausalLM._contains_cjk(right):
            return left + right
        return left + " " + right

    def _split_text_into_best_sentences(
        self,
        text_tokenizer,
        text: str,
        max_tokens: int,
    ) -> list[str]:
        if int(max_tokens) <= 0:
            return [str(text)]

        prepared_text = self._prepare_text_for_sentence_chunking(text)
        sentence_candidates = self._split_text_by_punctuation(prepared_text, punctuation=_SENTENCE_END_PUNCTUATION)
        if not sentence_candidates:
            sentence_candidates = [prepared_text.strip()]

        sentence_slices: list[tuple[int, str]] = []
        for sentence_text in sentence_candidates:
            normalized_sentence = sentence_text.strip()
            if not normalized_sentence:
                continue
            sentence_token_count = self._count_text_tokens(text_tokenizer, normalized_sentence)
            if sentence_token_count <= int(max_tokens):
                sentence_slices.append((sentence_token_count, normalized_sentence))
                continue

            clause_candidates = self._split_text_by_punctuation(
                normalized_sentence,
                punctuation=_CLAUSE_SPLIT_PUNCTUATION,
            )
            if len(clause_candidates) <= 1:
                clause_candidates = [normalized_sentence]

            for clause_text in clause_candidates:
                normalized_clause = clause_text.strip()
                if not normalized_clause:
                    continue
                clause_token_count = self._count_text_tokens(text_tokenizer, normalized_clause)
                if clause_token_count <= int(max_tokens):
                    sentence_slices.append((clause_token_count, normalized_clause))
                    continue
                for piece in self._split_text_by_token_budget(
                    text_tokenizer=text_tokenizer,
                    text=normalized_clause,
                    max_tokens=max_tokens,
                ):
                    normalized_piece = piece.strip()
                    if normalized_piece:
                        sentence_slices.append(
                            (self._count_text_tokens(text_tokenizer, normalized_piece), normalized_piece)
                        )

        chunks: list[str] = []
        current_chunk = ""
        current_chunk_token_count = 0
        for sentence_token_count, sentence_text in sentence_slices:
            if current_chunk == "":
                current_chunk = sentence_text
                current_chunk_token_count = sentence_token_count
                continue
            if current_chunk_token_count + sentence_token_count > int(max_tokens):
                chunks.append(current_chunk.strip())
                current_chunk = sentence_text
                current_chunk_token_count = sentence_token_count
            else:
                current_chunk = self._join_sentence_parts(current_chunk, sentence_text)
                current_chunk_token_count = self._count_text_tokens(text_tokenizer, current_chunk)

        if current_chunk:
            chunks.append(current_chunk.strip())
        return chunks or [prepared_text.strip()]

    @staticmethod
    def _estimate_voice_clone_inter_chunk_pause_seconds(text_chunk: str) -> float:
        return (
            DEFAULT_VOICE_CLONE_INTER_CHUNK_PAUSE_SHORT_SECONDS
            if len(str(text_chunk).strip().split()) <= 4
            else DEFAULT_VOICE_CLONE_INTER_CHUNK_PAUSE_LONG_SECONDS
        )

    def _concat_voice_clone_waveform_chunks(
        self,
        waveform_chunks: list[torch.FloatTensor],
        text_chunks: list[str],
        sample_rate: int,
    ) -> torch.FloatTensor:
        if not waveform_chunks:
            return torch.zeros((1, 0), dtype=torch.float32)
        if len(waveform_chunks) != len(text_chunks):
            raise ValueError("waveform_chunks and text_chunks must have the same length.")
        if len(waveform_chunks) == 1:
            return waveform_chunks[0]

        segments: list[torch.FloatTensor] = []
        for chunk_index, waveform_chunk in enumerate(waveform_chunks):
            segments.append(waveform_chunk)
            if chunk_index >= len(waveform_chunks) - 1:
                continue
            pause_seconds = self._estimate_voice_clone_inter_chunk_pause_seconds(text_chunks[chunk_index])
            pause_samples = max(0, int(round(float(sample_rate) * pause_seconds)))
            if pause_samples > 0:
                silence = torch.zeros((waveform_chunk.shape[0], pause_samples), dtype=waveform_chunk.dtype)
                segments.append(silence)
        return torch.cat(segments, dim=-1)

    @staticmethod
    def _resolve_inference_mode(
        mode: str,
        has_prompt_text: bool,
        has_prompt_audio: bool,
    ) -> str:
        normalized_mode = str(mode or "continuation").strip().lower() or "continuation"
        if normalized_mode not in {"continuation", "voice_clone"}:
            raise ValueError(f"Unsupported inference mode {mode!r}.")
        if normalized_mode == "voice_clone":
            if not has_prompt_audio:
                raise ValueError("voice_clone mode requires prompt_audio_path.")
            if has_prompt_text:
                raise ValueError("voice_clone mode does not accept prompt_text.")
        elif has_prompt_text != has_prompt_audio:
            raise ValueError(
                "continuation mode accepts either target text only, or prompt_text and prompt_audio_path together."
            )
        return normalized_mode

    def _resolve_inference_nq(self, nq: Optional[int] = None) -> int:
        if nq is None:
            return int(self.config.n_vq)
        resolved_nq = int(nq)
        if resolved_nq < 1 or resolved_nq > int(self.config.n_vq):
            raise ValueError(f"nq must be in [1, {self.config.n_vq}], got {resolved_nq}.")
        return resolved_nq

    def _mask_unused_audio_channels(
        self,
        audio_token_ids: torch.LongTensor,
        nq: int,
    ) -> torch.LongTensor:
        tensor = torch.as_tensor(audio_token_ids, dtype=torch.long)
        if tensor.shape[-1] != self.config.n_vq:
            raise ValueError(
                f"Expected audio token ids with trailing dim {self.config.n_vq}, got {tuple(tensor.shape)}"
            )
        if nq < self.config.n_vq:
            tensor = tensor.clone()
            tensor[..., nq:] = self.config.audio_pad_token_id
        return tensor

    def _build_audio_prefix_rows(
        self,
        prompt_audio_codes: torch.LongTensor,
        slot_token_id: int,
        device: torch.device,
    ) -> torch.LongTensor:
        rows = torch.full(
            (int(prompt_audio_codes.shape[0]), self.config.n_vq + 1),
            self.config.audio_pad_token_id,
            dtype=torch.long,
            device=device,
        )
        if rows.shape[0] > 0:
            rows[:, 0] = int(slot_token_id)
            rows[:, 1:] = prompt_audio_codes
        return rows

    def build_inference_input_ids(
        self,
        text: str,
        text_tokenizer,
        mode: str = "continuation",
        prompt_text: Optional[str] = None,
        prompt_audio_codes: Optional[torch.LongTensor] = None,
        device: Optional[Union[str, torch.device]] = None,
    ) -> tuple[torch.LongTensor, torch.BoolTensor]:
        resolved_device = self._resolve_device(device)
        resolved_mode = self._resolve_inference_mode(
            mode=mode,
            has_prompt_text=prompt_text is not None,
            has_prompt_audio=prompt_audio_codes is not None,
        )

        if resolved_mode == "voice_clone":
            assert prompt_audio_codes is not None
            text_token_ids = self._encode_text(text_tokenizer, text)
            prompt_token_ids = build_user_prompt_prefix(text_tokenizer, self.config) + [self.config.audio_start_token_id]
            suffix_token_ids = (
                [self.config.audio_end_token_id]
                + build_user_prompt_after_reference(text_tokenizer)
                + text_token_ids
                + build_assistant_prompt_prefix(text_tokenizer, self.config)
                + [self.config.audio_start_token_id]
            )
            sections = [
                self._build_text_rows(prompt_token_ids, device=resolved_device),
                self._build_audio_prefix_rows(
                    prompt_audio_codes=prompt_audio_codes.to(resolved_device),
                    slot_token_id=self.config.audio_user_slot_token_id,
                    device=resolved_device,
                ),
                self._build_text_rows(suffix_token_ids, device=resolved_device),
            ]
            input_ids = torch.cat(sections, dim=0).unsqueeze(0)
            attention_mask = torch.ones(input_ids.shape[:2], dtype=torch.bool, device=resolved_device)
            return input_ids, attention_mask

        effective_text = text if prompt_text is None else prompt_text + text
        prompt_token_ids = build_prompt_token_ids( #list of int
            tokenizer=text_tokenizer,
            config=self.config,
            text_token_ids=self._encode_text(text_tokenizer, effective_text),
        )
        sections = [
            self._build_text_rows(prompt_token_ids, device=resolved_device),
            self._build_text_rows([self.config.audio_start_token_id], device=resolved_device),
        ]
        if prompt_audio_codes is not None:
            sections.append(
                self._build_audio_prefix_rows(
                    prompt_audio_codes=prompt_audio_codes.to(resolved_device),
                    slot_token_id=self.config.audio_assistant_slot_token_id,
                    device=resolved_device,
                )
            )
        input_ids = torch.cat(sections, dim=0).unsqueeze(0)
        attention_mask = torch.ones(input_ids.shape[:2], dtype=torch.bool, device=resolved_device)
        return input_ids, attention_mask

    def _left_pad_inference_batch(
        self,
        input_id_batches: list[torch.LongTensor],
        attention_mask_batches: list[torch.BoolTensor],
        device: torch.device,
    ) -> tuple[torch.LongTensor, torch.BoolTensor]:
        if not input_id_batches:
            raise ValueError("input_id_batches must not be empty.")
        if len(input_id_batches) != len(attention_mask_batches):
            raise ValueError("input_id_batches and attention_mask_batches must have the same length.")

        batch_size = len(input_id_batches)
        max_seq_len = max(int(batch.shape[1]) for batch in input_id_batches)
        row_width = self.config.n_vq + 1

        padded_input_ids = torch.full(
            (batch_size, max_seq_len, row_width),
            self.config.audio_pad_token_id,
            dtype=torch.long,
            device=device,
        )
        padded_input_ids[:, :, 0] = self.config.pad_token_id
        padded_attention_mask = torch.zeros((batch_size, max_seq_len), dtype=torch.bool, device=device)

        for batch_index, (input_ids, attention_mask) in enumerate(zip(input_id_batches, attention_mask_batches)):
            normalized_input_ids = input_ids.squeeze(0).to(device=device, dtype=torch.long)
            normalized_attention_mask = attention_mask.squeeze(0).to(device=device, dtype=torch.bool)
            seq_len = int(normalized_input_ids.shape[0])
            padded_input_ids[batch_index, -seq_len:, :] = normalized_input_ids
            padded_attention_mask[batch_index, -seq_len:] = normalized_attention_mask

        return padded_input_ids, padded_attention_mask

    def _trim_generated_audio_token_ids(
        self,
        audio_token_ids: torch.LongTensor,
        effective_nq: int,
    ) -> torch.LongTensor:
        tensor = self._mask_unused_audio_channels(audio_token_ids, nq=effective_nq)
        if tensor.ndim != 2:
            raise ValueError(f"Expected a 2D audio token tensor, got {tuple(tensor.shape)}")
        valid_rows = tensor[:, :effective_nq].ne(self.config.audio_pad_token_id).any(dim=-1)
        if not bool(valid_rows.any()):
            return tensor[:0]
        last_valid_index = int(torch.nonzero(valid_rows, as_tuple=False)[-1].item()) + 1
        return tensor[:last_valid_index]

    def _resolve_voice_clone_chunk_batch_size(
        self,
        *,
        resolved_device: torch.device,
        chunk_count: int,
        max_memory_per_sample_gb: float,
    ) -> int:
        if chunk_count <= 1 or max_memory_per_sample_gb <= 0 or resolved_device.type != "cuda":
            return 1
        if not hasattr(torch.cuda, "mem_get_info"):
            return 1
        try:
            free_bytes, _ = torch.cuda.mem_get_info(resolved_device)
        except Exception:
            return 1
        bytes_per_sample = int(float(max_memory_per_sample_gb) * (1024**3))
        if bytes_per_sample <= 0:
            return 1
        usable_free_bytes = max(0, int(free_bytes * 0.9))
        batch_size = max(1, usable_free_bytes // bytes_per_sample)
        resolved_batch_size = max(1, min(int(chunk_count), int(batch_size)))
        logging.info(
            "voice_clone chunk batching device=%s free_gb=%.2f max_memory_per_sample_gb=%.2f resolved_batch_size=%d chunk_count=%d",
            resolved_device,
            float(free_bytes) / float(1024**3),
            float(max_memory_per_sample_gb),
            resolved_batch_size,
            int(chunk_count),
        )
        return resolved_batch_size

    @staticmethod
    def _resolve_requested_batch_size_limit(requested_batch_size: Optional[int]) -> Optional[int]:
        if requested_batch_size is None:
            return None
        resolved_batch_size = int(requested_batch_size)
        if resolved_batch_size <= 0:
            return None
        return max(1, resolved_batch_size)

    def _resolve_effective_voice_clone_batch_sizes(
        self,
        *,
        resolved_device: torch.device,
        chunk_count: int,
        max_memory_per_sample_gb: float,
        requested_tts_max_batch_size: Optional[int] = None,
        requested_codec_max_batch_size: Optional[int] = None,
        realtime_streaming: bool = False,
    ) -> tuple[int, int]:
        effective_tts_batch_size = self._resolve_voice_clone_chunk_batch_size(
            resolved_device=resolved_device,
            chunk_count=chunk_count,
            max_memory_per_sample_gb=max_memory_per_sample_gb,
        )
        requested_tts_limit = self._resolve_requested_batch_size_limit(requested_tts_max_batch_size)
        requested_codec_limit = self._resolve_requested_batch_size_limit(requested_codec_max_batch_size)

        if requested_tts_limit is not None:
            effective_tts_batch_size = min(effective_tts_batch_size, requested_tts_limit)
        if realtime_streaming and requested_codec_limit is not None:
            effective_tts_batch_size = min(effective_tts_batch_size, requested_codec_limit)

        effective_tts_batch_size = max(1, min(int(chunk_count), int(effective_tts_batch_size)))

        if realtime_streaming:
            effective_codec_batch_size = effective_tts_batch_size
        elif requested_codec_limit is None:
            effective_codec_batch_size = 1
        else:
            effective_codec_batch_size = max(1, min(int(requested_codec_limit), int(effective_tts_batch_size)))

        return int(effective_tts_batch_size), int(effective_codec_batch_size)

    def _generate_audio_token_ids_with_fallback(
        self,
        *,
        prompt_input_ids: torch.LongTensor,
        attention_mask: torch.BoolTensor,
        effective_nq: int,
        max_new_frames: int,
        do_sample: bool,
        text_temperature: float,
        text_top_p: float,
        text_top_k: int,
        audio_temperature: float,
        audio_top_p: float,
        audio_top_k: int,
        audio_repetition_penalty: float,
        use_kv_cache: bool,
        resolved_device: torch.device,
    ) -> torch.LongTensor:
        try:
            generation = self.generate(
                input_ids=prompt_input_ids,
                attention_mask=attention_mask,
                nq=effective_nq,
                max_new_frames=max_new_frames,
                do_sample=do_sample,
                text_temperature=text_temperature,
                text_top_p=text_top_p,
                text_top_k=text_top_k,
                audio_temperature=audio_temperature,
                audio_top_p=audio_top_p,
                audio_top_k=audio_top_k,
                audio_repetition_penalty=audio_repetition_penalty,
                use_kv_cache=use_kv_cache,
                return_dict_in_generate=True,
            )
        except (RuntimeError, ValueError) as exc:
            if not self._is_generation_stability_error(exc):
                raise
            self._apply_inference_stability_fallback(resolved_device)
            generation = self.generate(
                input_ids=prompt_input_ids,
                attention_mask=attention_mask,
                nq=effective_nq,
                max_new_frames=max_new_frames,
                do_sample=do_sample,
                text_temperature=text_temperature,
                text_top_p=text_top_p,
                text_top_k=text_top_k,
                audio_temperature=audio_temperature,
                audio_top_p=audio_top_p,
                audio_top_k=audio_top_k,
                audio_repetition_penalty=audio_repetition_penalty,
                use_kv_cache=use_kv_cache,
                return_dict_in_generate=True,
            )
        return self._mask_unused_audio_channels(generation.audio_token_ids, nq=effective_nq)

    def _decode_audio_token_ids_to_waveform(
        self,
        *,
        audio_tokenizer,
        audio_token_ids: torch.LongTensor,
        target_sample_rate: int,
        effective_nq: int,
        resolved_device: torch.device,
    ) -> tuple[torch.FloatTensor, int]:
        decoded = self._call_audio_decode(
            audio_tokenizer=audio_tokenizer,
            audio_token_ids=audio_token_ids.to(resolved_device),
            sample_rate=target_sample_rate,
            nq=effective_nq,
        )
        return self._extract_waveform_and_sample_rate(decoded, fallback_sample_rate=target_sample_rate)

    def _decode_audio_token_id_batch_to_waveforms(
        self,
        *,
        audio_tokenizer,
        audio_token_id_batches: Sequence[torch.LongTensor],
        target_sample_rate: int,
        effective_nq: int,
        resolved_device: torch.device,
    ) -> tuple[list[torch.FloatTensor], int]:
        if not audio_token_id_batches:
            return [], target_sample_rate

        if len(audio_token_id_batches) == 1:
            waveform, sample_rate = self._decode_audio_token_ids_to_waveform(
                audio_tokenizer=audio_tokenizer,
                audio_token_ids=audio_token_id_batches[0],
                target_sample_rate=target_sample_rate,
                effective_nq=effective_nq,
                resolved_device=resolved_device,
            )
            return [waveform], sample_rate

        decode_codes = [
            self._prepare_audio_codes_for_decode(audio_token_ids.to(resolved_device), nq=effective_nq)
            for audio_token_ids in audio_token_id_batches
        ]
        batch_decode_fn = getattr(audio_tokenizer, "batch_decode", None)
        if batch_decode_fn is None:
            raise AttributeError("audio_tokenizer must provide a batch_decode method.")

        try:
            with self._audio_tokenizer_inference_context(audio_tokenizer, resolved_device):
                decode_output = batch_decode_fn(
                    decode_codes,
                    num_quantizers=effective_nq,
                    chunk_duration=None,
                )
            return self._extract_batch_waveforms_and_sample_rate(
                decode_output,
                fallback_sample_rate=target_sample_rate,
                batch_size=len(audio_token_id_batches),
            )
        except Exception:
            logging.warning(
                "batched audio decode failed; falling back to per-chunk decode for batch_size=%d",
                len(audio_token_id_batches),
                exc_info=True,
            )
            waveform_rows: list[torch.FloatTensor] = []
            sample_rate = target_sample_rate
            for audio_token_ids in audio_token_id_batches:
                waveform_row, sample_rate = self._decode_audio_token_ids_to_waveform(
                    audio_tokenizer=audio_tokenizer,
                    audio_token_ids=audio_token_ids,
                    target_sample_rate=target_sample_rate,
                    effective_nq=effective_nq,
                    resolved_device=resolved_device,
                )
                waveform_rows.append(waveform_row)
            return waveform_rows, sample_rate

    def _build_generation_row(
        self,
        batch_size: int,
        device: torch.device,
        audio_token_ids: torch.LongTensor,
    ) -> torch.LongTensor:
        row = torch.full(
            (batch_size, 1, self.config.n_vq + 1),
            self.config.audio_pad_token_id,
            dtype=torch.long,
            device=device,
        )
        row[:, :, 0] = self.config.audio_assistant_slot_token_id
        row[:, :, 1:] = audio_token_ids.unsqueeze(1)
        return row

    @staticmethod
    def _compute_stream_lead_seconds(
        emitted_samples_total: int,
        sample_rate: int,
        first_audio_emitted_at: Optional[float],
    ) -> float:
        if first_audio_emitted_at is None or sample_rate <= 0:
            return 0.0
        elapsed_seconds = max(0.0, time.monotonic() - first_audio_emitted_at)
        emitted_seconds = float(emitted_samples_total) / float(sample_rate)
        return emitted_seconds - elapsed_seconds

    @staticmethod
    def _resolve_stream_decode_frame_budget(
        *,
        emitted_samples_total: int,
        sample_rate: int,
        first_audio_emitted_at: Optional[float],
    ) -> int:
        lead_seconds = MossTTSNanoForCausalLM._compute_stream_lead_seconds(
            emitted_samples_total=emitted_samples_total,
            sample_rate=sample_rate,
            first_audio_emitted_at=first_audio_emitted_at,
        )
        if first_audio_emitted_at is None or lead_seconds < 0.20:
            return 1
        if lead_seconds < 0.55:
            return 2
        if lead_seconds < 1.10:
            return 4
        return 8

    def _sample_next_token(
        self,
        logits: torch.FloatTensor,
        do_sample: bool,
        temperature: float,
        top_k: Optional[int],
        top_p: Optional[float],
        previous_token_ids: Optional[torch.LongTensor] = None,
        repetition_penalty: float = 1.0,
    ) -> torch.LongTensor:
        scores = self._apply_repetition_penalty(
            logits=logits,
            previous_token_ids=previous_token_ids,
            repetition_penalty=repetition_penalty,
        )
        if not do_sample:
            return scores.argmax(dim=-1)
        if temperature <= 0:
            raise ValueError("temperature must be positive when do_sample=True")

        scores = scores / temperature
        if top_k is not None and top_k > 0:
            top_k = min(top_k, scores.shape[-1])
            threshold = torch.topk(scores, top_k, dim=-1).values[..., -1, None]
            scores = scores.masked_fill(scores < threshold, float("-inf"))

        if top_p is not None and 0.0 < top_p < 1.0:
            sorted_scores, sorted_indices = torch.sort(scores, descending=True, dim=-1)
            sorted_probs = torch.softmax(sorted_scores, dim=-1)
            sorted_cumsum = torch.cumsum(sorted_probs, dim=-1)
            sorted_remove = sorted_cumsum > top_p
            sorted_remove[..., 1:] = sorted_remove[..., :-1].clone()
            sorted_remove[..., 0] = False
            sorted_scores = sorted_scores.masked_fill(sorted_remove, float("-inf"))
            scores = torch.full_like(scores, float("-inf"))
            scores.scatter_(dim=-1, index=sorted_indices, src=sorted_scores)

        probs = torch.softmax(scores, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)

    @staticmethod
    def _ensure_finite_generation_logits(logits: torch.FloatTensor, name: str) -> None:
        if torch.isfinite(logits).all():
            return
        finite_mask = torch.isfinite(logits)
        finite_logits = logits[finite_mask]
        min_value = float(finite_logits.min().item()) if finite_logits.numel() > 0 else float("nan")
        max_value = float(finite_logits.max().item()) if finite_logits.numel() > 0 else float("nan")
        raise RuntimeError(
            f"Non-finite {name} during generation: dtype={logits.dtype} "
            f"shape={tuple(logits.shape)} finite={int(finite_mask.sum().item())}/{int(logits.numel())} "
            f"min={min_value} max={max_value}"
        )

    def _apply_repetition_penalty(
        self,
        logits: torch.FloatTensor,
        previous_token_ids: Optional[torch.LongTensor],
        repetition_penalty: float,
    ) -> torch.FloatTensor:
        if repetition_penalty <= 0:
            raise ValueError("repetition_penalty must be positive")
        if repetition_penalty == 1.0 or previous_token_ids is None:
            return logits

        token_ids = torch.as_tensor(previous_token_ids, device=logits.device, dtype=torch.long)
        if token_ids.ndim == 1:
            token_ids = token_ids.unsqueeze(0)
        elif token_ids.ndim > 2:
            token_ids = token_ids.reshape(token_ids.shape[0], -1)

        scores = logits.clone()
        vocab_size = scores.shape[-1]
        for batch_index in range(scores.shape[0]):
            valid_token_ids = token_ids[batch_index]
            valid_token_ids = valid_token_ids[(valid_token_ids >= 0) & (valid_token_ids < vocab_size)]
            if valid_token_ids.numel() == 0:
                continue
            unique_token_ids = torch.unique(valid_token_ids)
            token_scores = scores[batch_index].index_select(0, unique_token_ids)
            token_scores = torch.where(
                token_scores < 0,
                token_scores * repetition_penalty,
                token_scores / repetition_penalty,
            )
            scores[batch_index].scatter_(0, unique_token_ids, token_scores)
        return scores

    def _sample_next_assistant_text_token(
        self,
        logits: torch.FloatTensor,
        do_sample: bool,
        temperature: float,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> torch.LongTensor:
        candidate_ids = torch.tensor(
            [
                self.config.audio_assistant_slot_token_id,
                self.config.audio_end_token_id,
            ],
            dtype=torch.long,
            device=logits.device,
        )
        candidate_logits = logits.index_select(dim=-1, index=candidate_ids)
        sampled_indices = self._sample_next_token(
            logits=candidate_logits,
            do_sample=do_sample,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )
        return candidate_ids[sampled_indices]

    def _resolve_device(self, device: Optional[Union[str, torch.device]] = None) -> torch.device:
        return torch.device(device) if device is not None else next(self.parameters()).device

    @staticmethod
    def _looks_like_hf_tokenizer_dir(candidate_path: Path) -> bool:
        if not candidate_path.is_dir():
            return False
        if (candidate_path / "tokenizer.model").is_file():
            return True
        if (candidate_path / "tokenizer.json").is_file():
            return True
        if (candidate_path / "tokenizer_config.json").is_file() and (
            (candidate_path / "vocab.json").is_file()
            or (candidate_path / "merges.txt").is_file()
            or (candidate_path / "special_tokens_map.json").is_file()
        ):
            return True
        return False

    @staticmethod
    def _looks_like_hf_repo_id(candidate: str) -> bool:
        stripped = candidate.strip()
        if not stripped:
            return False
        if stripped.startswith((os.sep, ".", "~")):
            return False
        if "\\" in stripped:
            return False
        parts = stripped.split("/")
        return len(parts) == 2 and all(part.strip() for part in parts)

    @staticmethod
    def _existing_local_path(raw_path: Union[str, Path]) -> Optional[Path]:
        candidate_path = Path(raw_path).expanduser()
        if not candidate_path.exists():
            return None
        return candidate_path.resolve()

    def _resolve_text_tokenizer_path(self, raw_path: Union[str, Path]) -> str:
        local_candidate_path = self._existing_local_path(raw_path)
        if local_candidate_path is None:
            raw_source = str(raw_path).strip()
            if self._looks_like_hf_repo_id(raw_source):
                return raw_source
            raise FileNotFoundError(f"Tokenizer path does not exist: {raw_source}")

        candidate_path = local_candidate_path
        if candidate_path.is_file() and candidate_path.suffix == ".model":
            return str(candidate_path)
        if candidate_path.is_dir():
            if (candidate_path / "tokenizer.model").is_file():
                return str(candidate_path)
            if self._looks_like_hf_tokenizer_dir(candidate_path):
                return str(candidate_path)
            hf_dir = candidate_path / "hf_tokenizer"
            if self._looks_like_hf_tokenizer_dir(hf_dir):
                return str(hf_dir)
            sentencepiece_model = candidate_path / "sentencepiece" / "mossttsnano_spm_bpe.model"
            if sentencepiece_model.is_file():
                return str(sentencepiece_model)
            final_summary_path = candidate_path / "final_summary.json"
            if final_summary_path.is_file():
                final_summary = json.loads(final_summary_path.read_text(encoding="utf-8"))
                latest_hf_dir = final_summary.get("latest_hf_tokenizer_dir")
                if latest_hf_dir:
                    latest_hf_path = Path(str(latest_hf_dir))
                    if self._looks_like_hf_tokenizer_dir(latest_hf_path):
                        return str(latest_hf_path.resolve())
        raise ValueError(
            "Could not resolve a tokenizer from the provided path. Expected a tokenizer dir, experiment dir, or SentencePiece .model file."
        )

    def _load_resolved_text_tokenizer(self, resolved_path: str, cache_dir: str):
        local_path = self._existing_local_path(resolved_path)
        load_source = str(local_path) if local_path is not None else str(resolved_path)
        if local_path is not None and local_path.is_file() and local_path.suffix == ".model":
            return MossTTSNanoSentencePieceTokenizer(vocab_file=str(local_path))
        try:
            load_kwargs: dict[str, object] = {
                "trust_remote_code": True,
                "use_fast": bool(self.config.tokenizer_use_fast),
                "cache_dir": cache_dir,
            }
            if local_path is not None:
                load_kwargs["local_files_only"] = True
            return AutoTokenizer.from_pretrained(
                load_source,
                **load_kwargs,
            )
        except Exception:
            if local_path is not None:
                model_path = local_path / "tokenizer.model"
                if model_path.is_file():
                    return MossTTSNanoSentencePieceTokenizer(vocab_file=str(model_path))
            raise

    @staticmethod
    def _resolve_hf_cache_dir() -> str:
        cache_dir = Path(__file__).resolve().parent / ".cache" / "huggingface"
        cache_dir.mkdir(parents=True, exist_ok=True)
        return str(cache_dir)

    @staticmethod
    def _patch_hf_dynamic_module_cache_dir(cache_dir: str) -> None:
        import transformers.dynamic_module_utils as dynamic_module_utils

        modules_cache_dir = str(Path(cache_dir) / "modules")
        Path(modules_cache_dir).mkdir(parents=True, exist_ok=True)
        os.environ["HF_MODULES_CACHE"] = modules_cache_dir
        dynamic_module_utils.HF_MODULES_CACHE = modules_cache_dir

    def _resolve_default_text_tokenizer_path(self) -> str:
        candidates: list[Union[str, Path]] = []

        raw_name_or_path = getattr(self.config, "_name_or_path", None)
        if raw_name_or_path:
            candidates.append(str(raw_name_or_path).strip())

        raw_model_name_or_path = getattr(self, "name_or_path", None)
        if raw_model_name_or_path:
            candidates.append(str(raw_model_name_or_path).strip())

        candidates.append(Path(__file__).resolve().parent)

        checked: set[str] = set()
        for candidate in candidates:
            raw_candidate = str(candidate).strip()
            if not raw_candidate:
                continue
            try:
                resolved_candidate = self._resolve_text_tokenizer_path(raw_candidate)
            except (FileNotFoundError, ValueError):
                continue
            if resolved_candidate in checked:
                continue
            checked.add(resolved_candidate)
            return resolved_candidate

        for candidate in candidates:
            raw_candidate = str(candidate).strip()
            if raw_candidate:
                return raw_candidate

        return str(Path(__file__).resolve().parent)

    def _load_text_tokenizer(self, text_tokenizer=None, text_tokenizer_path: Optional[str] = None):
        if text_tokenizer is not None:
            return text_tokenizer

        resolved_path = (
            self._resolve_text_tokenizer_path(text_tokenizer_path)
            if text_tokenizer_path is not None
            else self._resolve_default_text_tokenizer_path()
        )
        normalized_path = str(resolved_path)
        cached = getattr(self, "_cached_text_tokenizer", None)
        cached_path = getattr(self, "_cached_text_tokenizer_path", None)
        if cached is not None and cached_path == normalized_path:
            return cached

        cache_dir = self._resolve_hf_cache_dir()
        self._patch_hf_dynamic_module_cache_dir(cache_dir)
        tokenizer = self._load_resolved_text_tokenizer(resolved_path=resolved_path, cache_dir=cache_dir)
        if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        self._cached_text_tokenizer = tokenizer
        self._cached_text_tokenizer_path = normalized_path
        return tokenizer

    @staticmethod
    def _normalize_audio_tokenizer_type(audio_tokenizer_type: Optional[str]) -> Optional[str]:
        if audio_tokenizer_type is None:
            return None
        normalized = str(audio_tokenizer_type).strip().lower()
        if not normalized:
            return None
        if normalized == MOSS_AUDIO_TOKENIZER_NANO_TYPE:
            return MOSS_AUDIO_TOKENIZER_NANO_TYPE
        raise ValueError(
            "Unsupported audio tokenizer type. "
            f"The open-source package only supports '{MOSS_AUDIO_TOKENIZER_NANO_TYPE}'."
        )

    def _resolve_audio_tokenizer_type(self, audio_tokenizer_type: Optional[str]) -> str:
        explicit_type = self._normalize_audio_tokenizer_type(audio_tokenizer_type)
        if explicit_type is not None:
            return explicit_type
        config_type = self._normalize_audio_tokenizer_type(getattr(self.config, "audio_tokenizer_type", None))
        return MOSS_AUDIO_TOKENIZER_NANO_TYPE if config_type is None else config_type

    @staticmethod
    def _set_decoder_attention_implementation(decoder, attn_implementation: str) -> None:
        decoder.attn_implementation = str(attn_implementation)
        if getattr(decoder, "config", None) is not None:
            decoder.config._attn_implementation = str(attn_implementation)
        for block in getattr(decoder, "h", []):
            block.attn.attn_implementation = str(attn_implementation)

    def _set_attention_implementation(
        self,
        attn_implementation: str,
        local_attn_implementation: Optional[str] = None,
    ) -> None:
        resolved_global = str(attn_implementation)
        resolved_local = resolved_global if local_attn_implementation is None else str(local_attn_implementation)
        self.config.attn_implementation = resolved_global
        self.config.gpt2_config._attn_implementation = resolved_global
        self._set_decoder_attention_implementation(self.transformer, resolved_global)
        self.config.local_transformer_attn_implementation = resolved_local
        self._set_decoder_attention_implementation(self.local_transformer, resolved_local)

    @staticmethod
    def _select_fallback_attention_implementation(device: torch.device) -> str:
        return "sdpa" if device.type == "cuda" else "eager"

    @staticmethod
    def _is_generation_stability_error(exc: Exception) -> bool:
        message = str(exc)
        return any(
            marker in message
            for marker in (
                "Non-finite",
                "device-side assert triggered",
                "probability tensor contains either",
                "flash_attention_2 requires fp16/bf16 tensors",
            )
        )

    def _apply_inference_stability_fallback(self, device: torch.device) -> None:
        fallback_attn = self._select_fallback_attention_implementation(device)
        if next(self.parameters()).dtype != torch.float32:
            self.to(device=device, dtype=torch.float32)
        self._set_attention_implementation(fallback_attn)
        logging.warning(
            "retrying inference with dtype=float32 attn_implementation=%s due to numerical instability",
            fallback_attn,
        )

    def _load_audio_tokenizer(
        self,
        audio_tokenizer=None,
        audio_tokenizer_type: Optional[str] = None,
        audio_tokenizer_pretrained_name_or_path: Optional[str] = None,
        device: Optional[Union[str, torch.device]] = None,
    ):
        if audio_tokenizer is not None:
            return audio_tokenizer

        resolved_type = self._resolve_audio_tokenizer_type(audio_tokenizer_type=audio_tokenizer_type)
        if resolved_type != MOSS_AUDIO_TOKENIZER_NANO_TYPE:
            raise ValueError(
                f"Unsupported audio tokenizer type {resolved_type!r}; expected '{MOSS_AUDIO_TOKENIZER_NANO_TYPE}'."
            )

        resolved_pretrained_name_or_path = (
            audio_tokenizer_pretrained_name_or_path
            or getattr(self.config, "audio_tokenizer_pretrained_name_or_path", None)
            or DEFAULT_MOSS_AUDIO_TOKENIZER_PRETRAINED_NAME_OR_PATH
        )
        candidate_path = Path(str(resolved_pretrained_name_or_path)).expanduser()
        if candidate_path.exists():
            load_source = str(candidate_path.resolve())
            load_kwargs: dict[str, object] = {
                "trust_remote_code": True,
                "local_files_only": True,
                "force_download": True,
            }
            cache_key = f"{resolved_type}|{load_source}"
        else:
            load_source = str(resolved_pretrained_name_or_path)
            load_kwargs = {
                "trust_remote_code": True,
            }
            cache_key = f"{resolved_type}|{load_source}"

        cached = getattr(self, "_cached_audio_tokenizer", None)
        cached_path = getattr(self, "_cached_audio_tokenizer_path", None)
        if cached is not None and cached_path == cache_key:
            tokenizer = cached
        else:
            tokenizer = AutoModel.from_pretrained(load_source, **load_kwargs)
            if hasattr(tokenizer, "eval"):
                tokenizer.eval()
            self._cached_audio_tokenizer = tokenizer
            self._cached_audio_tokenizer_path = cache_key

        resolved_device = self._resolve_device(device)
        return tokenizer.to(resolved_device) if hasattr(tokenizer, "to") else tokenizer

    @staticmethod
    def _extract_tensor_candidate(output: Any) -> Any:
        if torch.is_tensor(output) or isinstance(output, np.ndarray):
            return output
        for attr_name in ("audio_codes", "audio_token_ids", "codes", "tokens", "input_ids"):
            value = getattr(output, attr_name, None)
            if value is not None:
                return value
        if isinstance(output, dict):
            for key in ("audio_codes", "audio_token_ids", "codes", "tokens", "input_ids"):
                if key in output:
                    return output[key]
            if len(output) == 1:
                return next(iter(output.values()))
        if isinstance(output, (list, tuple)) and output:
            if len(output) == 2 and isinstance(output[1], (int, float)):
                return output[0]
            return MossTTSNanoForCausalLM._extract_tensor_candidate(output[0])
        raise TypeError(f"Unsupported audio tokenizer output type: {type(output)!r}")

    @staticmethod
    def _extract_audio_code_length(output: Any) -> Optional[int]:
        for attr_name in ("audio_codes_lengths", "audio_token_ids_lengths", "codes_lengths", "lengths"):
            candidate = getattr(output, attr_name, None)
            if candidate is not None:
                lengths = torch.as_tensor(candidate).reshape(-1)
                if lengths.numel() > 0:
                    return int(lengths[0].item())
        if isinstance(output, dict):
            for key in ("audio_codes_lengths", "audio_token_ids_lengths", "codes_lengths", "lengths"):
                if key in output:
                    lengths = torch.as_tensor(output[key]).reshape(-1)
                    if lengths.numel() > 0:
                        return int(lengths[0].item())
        if isinstance(output, (list, tuple)) and len(output) >= 2:
            candidate = output[1]
            if torch.is_tensor(candidate) or isinstance(candidate, np.ndarray):
                lengths = torch.as_tensor(candidate).reshape(-1)
                if lengths.numel() > 0:
                    return int(lengths[0].item())
            if isinstance(candidate, (int, float)):
                return int(candidate)
        return None

    def _normalize_audio_codes(self, audio_codes: Any) -> torch.LongTensor:
        code_length = self._extract_audio_code_length(audio_codes)
        tensor = torch.as_tensor(self._extract_tensor_candidate(audio_codes))
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(-1)
        if tensor.ndim == 3:
            if tensor.shape[1] == 1 and tensor.shape[0] >= self.config.n_vq:
                tensor = tensor[: self.config.n_vq, 0, :].transpose(0, 1)
            elif tensor.shape[0] == 1:
                tensor = tensor[0]
            elif tensor.shape[1] == self.config.n_vq:
                tensor = tensor.transpose(1, 2)[0]
            elif tensor.shape[-1] == self.config.n_vq:
                tensor = tensor[0]
            else:
                raise ValueError(f"Unable to normalize audio codes with shape {tuple(tensor.shape)}")

        if tensor.ndim != 2:
            raise ValueError(f"Expected audio codes with 2 dims after normalization, got {tuple(tensor.shape)}")
        if tensor.shape[-1] != self.config.n_vq and tensor.shape[0] == self.config.n_vq:
            tensor = tensor.transpose(0, 1)
        elif tensor.shape[-1] != self.config.n_vq and tensor.shape[0] > self.config.n_vq:
            tensor = tensor[: self.config.n_vq].transpose(0, 1)
        elif tensor.shape[-1] > self.config.n_vq:
            tensor = tensor[:, : self.config.n_vq]
        if tensor.shape[-1] != self.config.n_vq:
            raise ValueError(
                f"Expected normalized audio codes with trailing dim {self.config.n_vq}, got {tuple(tensor.shape)}"
            )
        if code_length is not None:
            tensor = tensor[:code_length]
        return tensor.to(dtype=torch.long)

    def _extract_waveform_and_sample_rate(
        self,
        decode_output: Any,
        fallback_sample_rate: int,
    ) -> tuple[torch.FloatTensor, int]:
        sample_rate = fallback_sample_rate
        waveform = decode_output
        waveform_length = None

        for key in ("sample_rate", "sampling_rate"):
            value = getattr(decode_output, key, None)
            if value is not None:
                sample_rate = int(value)
                break
        for key in ("waveform", "audio", "wav", "samples"):
            value = getattr(decode_output, key, None)
            if value is not None:
                waveform = value
                break
        for key in ("audio_lengths", "waveform_lengths", "lengths"):
            value = getattr(decode_output, key, None)
            if value is not None:
                lengths = torch.as_tensor(value).reshape(-1)
                if lengths.numel() > 0:
                    waveform_length = int(lengths[0].item())
                    break

        if isinstance(decode_output, dict):
            for key in ("sample_rate", "sampling_rate"):
                if key in decode_output:
                    sample_rate = int(decode_output[key])
                    break
            for key in ("waveform", "audio", "wav", "samples"):
                if key in decode_output:
                    waveform = decode_output[key]
                    break
            for key in ("audio_lengths", "waveform_lengths", "lengths"):
                if key in decode_output:
                    lengths = torch.as_tensor(decode_output[key]).reshape(-1)
                    if lengths.numel() > 0:
                        waveform_length = int(lengths[0].item())
                        break
        elif isinstance(decode_output, (list, tuple)) and decode_output:
            if len(decode_output) == 2 and isinstance(decode_output[1], (int, float)):
                waveform = decode_output[0]
                sample_rate = int(decode_output[1])
            else:
                waveform = decode_output[0]

        waveform_tensor = torch.as_tensor(waveform, dtype=torch.float32)
        if waveform_tensor.ndim == 3 and waveform_tensor.shape[0] == 1:
            waveform_tensor = waveform_tensor[0]
        if waveform_tensor.ndim == 2 and waveform_tensor.shape[0] > waveform_tensor.shape[1]:
            waveform_tensor = waveform_tensor.transpose(0, 1)
        if waveform_tensor.ndim == 1:
            waveform_tensor = waveform_tensor.unsqueeze(0)
        if waveform_tensor.ndim != 2:
            raise ValueError(f"Expected decoded waveform with 2 dims, got {tuple(waveform_tensor.shape)}")
        if waveform_length is not None:
            waveform_tensor = waveform_tensor[..., : max(0, waveform_length)]
        return waveform_tensor.cpu(), sample_rate

    def _call_audio_encode(
        self,
        audio_tokenizer,
        waveform: torch.FloatTensor,
        sample_rate: int,
    ) -> Any:
        del sample_rate
        batch_encode_fn = getattr(audio_tokenizer, "batch_encode", None)
        if batch_encode_fn is None:
            raise AttributeError("audio_tokenizer must provide a batch_encode method.")

        waveform_tensor = torch.as_tensor(waveform, dtype=torch.float32, device=self._resolve_device(waveform.device))
        if waveform_tensor.ndim == 1:
            waveform_tensor = waveform_tensor.unsqueeze(0)
        if waveform_tensor.ndim != 2:
            raise ValueError(
                f"MOSS audio tokenizer encode expects waveform shaped like (C, T), got {tuple(waveform_tensor.shape)}"
            )

        with self._audio_tokenizer_inference_context(audio_tokenizer, waveform_tensor.device):
            return batch_encode_fn([waveform_tensor], chunk_duration=None)

    def _call_audio_decode(
        self,
        audio_tokenizer,
        audio_token_ids: torch.LongTensor,
        sample_rate: int,
        nq: Optional[int] = None,
    ) -> Any:
        del sample_rate
        batch_decode_fn = getattr(audio_tokenizer, "batch_decode", None)
        if batch_decode_fn is None:
            raise AttributeError("audio_tokenizer must provide a batch_decode method.")

        effective_nq = self._resolve_inference_nq(nq)
        decode_codes = self._prepare_audio_codes_for_decode(audio_token_ids, nq=effective_nq)
        with self._audio_tokenizer_inference_context(audio_tokenizer, decode_codes.device):
            return batch_decode_fn([decode_codes], num_quantizers=effective_nq, chunk_duration=None)

    def _extract_batch_waveforms_and_sample_rate(
        self,
        decode_output: Any,
        fallback_sample_rate: int,
        batch_size: int,
    ) -> tuple[list[torch.FloatTensor], int]:
        sample_rate = fallback_sample_rate
        audio = decode_output
        audio_lengths = None

        for key in ("sample_rate", "sampling_rate"):
            value = getattr(decode_output, key, None)
            if value is not None:
                sample_rate = int(value)
                break
        for key in ("waveform", "audio", "wav", "samples"):
            value = getattr(decode_output, key, None)
            if value is not None:
                audio = value
                break
        for key in ("audio_lengths", "waveform_lengths", "lengths"):
            value = getattr(decode_output, key, None)
            if value is not None:
                audio_lengths = value
                break

        if isinstance(decode_output, dict):
            for key in ("sample_rate", "sampling_rate"):
                if key in decode_output:
                    sample_rate = int(decode_output[key])
                    break
            for key in ("waveform", "audio", "wav", "samples"):
                if key in decode_output:
                    audio = decode_output[key]
                    break
            for key in ("audio_lengths", "waveform_lengths", "lengths"):
                if key in decode_output:
                    audio_lengths = decode_output[key]
                    break

        audio_tensor = torch.as_tensor(audio, dtype=torch.float32)
        if audio_tensor.ndim == 2:
            audio_tensor = audio_tensor.unsqueeze(0)
        if audio_tensor.ndim != 3:
            raise ValueError(f"Expected batched decoded audio with 3 dims, got {tuple(audio_tensor.shape)}")
        if audio_tensor.shape[0] != int(batch_size):
            raise ValueError(
                f"Expected decoded batch size {batch_size}, got audio tensor shape {tuple(audio_tensor.shape)}"
            )

        if audio_lengths is None:
            lengths_tensor = torch.full(
                (batch_size,),
                int(audio_tensor.shape[-1]),
                device=audio_tensor.device,
                dtype=torch.long,
            )
        else:
            lengths_tensor = torch.as_tensor(audio_lengths, dtype=torch.long, device=audio_tensor.device).reshape(-1)
            if lengths_tensor.numel() != int(batch_size):
                raise ValueError(f"Expected {batch_size} decoded audio lengths, got {int(lengths_tensor.numel())}")

        waveform_rows: list[torch.FloatTensor] = []
        for row_index in range(batch_size):
            row_length = max(0, int(lengths_tensor[row_index].item()))
            waveform_rows.append(audio_tensor[row_index, :, :row_length].detach().cpu())
        return waveform_rows, sample_rate

    @staticmethod
    def _resolve_audio_tokenizer_downsample_rate(audio_tokenizer) -> int:
        for holder in (audio_tokenizer, getattr(audio_tokenizer, "config", None)):
            if holder is None:
                continue
            for attr_name in ("downsample_rate", "hop_length", "frame_size"):
                value = getattr(holder, attr_name, None)
                if value is not None:
                    return int(value)
            sampling_rate = getattr(holder, "sampling_rate", None)
            frame_rate = getattr(holder, "frame_rate", None)
            if sampling_rate is not None and frame_rate not in (None, 0):
                return int(round(float(sampling_rate) / float(frame_rate)))
        raise ValueError("audio_tokenizer.downsample_rate is required for prompt-audio decoding.")

    def _resolve_audio_tokenizer_sample_rate(self, audio_tokenizer) -> int:
        for holder in (audio_tokenizer, getattr(audio_tokenizer, "config", None)):
            if holder is None:
                continue
            for attr_name in ("sampling_rate", "sample_rate"):
                value = getattr(holder, attr_name, None)
                if value is not None:
                    return int(value)
        return int(self.config.audio_tokenizer_sample_rate)

    @staticmethod
    def _resolve_audio_tokenizer_channels(audio_tokenizer) -> int:
        for holder in (audio_tokenizer, getattr(audio_tokenizer, "config", None)):
            if holder is None:
                continue
            for attr_name in ("number_channels", "channels_numbers", "audio_channels", "channels", "num_channels"):
                value = getattr(holder, attr_name, None)
                if value is not None:
                    return int(value)
        return 1

    @staticmethod
    def _audio_tokenizer_inference_context(audio_tokenizer, device: Union[str, torch.device]):
        del audio_tokenizer, device
        return nullcontext()

    def _prepare_audio_codes_for_decode(
        self,
        audio_token_ids: torch.LongTensor,
        nq: Optional[int] = None,
    ) -> torch.LongTensor:
        effective_nq = self._resolve_inference_nq(nq)
        tensor = torch.as_tensor(audio_token_ids, dtype=torch.long)
        if tensor.ndim == 2:
            if tensor.shape[-1] == self.config.n_vq and tensor.shape[0] != self.config.n_vq:
                return tensor[:, :effective_nq].transpose(0, 1).contiguous()
            if tensor.shape[0] == self.config.n_vq:
                return tensor[:effective_nq].contiguous()
        elif tensor.ndim == 3:
            if tensor.shape[-1] == self.config.n_vq:
                return tensor[..., :effective_nq].permute(2, 0, 1).contiguous()
            if tensor.shape[0] == self.config.n_vq:
                return tensor[:effective_nq].contiguous()
        raise ValueError(
            f"Expected generated audio token ids shaped like (T, {self.config.n_vq}) or ({self.config.n_vq}, T); got {tuple(tensor.shape)}"
        )

    def _load_reference_audio(
        self,
        reference_audio_path: Union[str, Path],
        target_sample_rate: int,
        target_channels: int,
    ) -> tuple[torch.FloatTensor, int]:
        waveform, sample_rate = torchaudio.load(str(reference_audio_path))
        waveform = waveform.to(torch.float32)
        if sample_rate != target_sample_rate:
            waveform = torchaudio.functional.resample(waveform, sample_rate, target_sample_rate)
            sample_rate = target_sample_rate
        current_channels = int(waveform.shape[0])
        if current_channels == target_channels:
            return waveform, sample_rate
        if current_channels == 1 and target_channels > 1:
            return waveform.repeat(target_channels, 1), sample_rate
        if current_channels > 1 and target_channels == 1:
            return waveform.mean(dim=0, keepdim=True), sample_rate
        raise ValueError(f"Unsupported reference audio channel conversion: {current_channels} -> {target_channels}")

    def _decode_local_last_hidden_state(
        self,
        local_inputs_embeds: torch.FloatTensor,
    ) -> torch.FloatTensor:
        local_attention_mask = torch.ones(
            local_inputs_embeds.shape[:2],
            dtype=torch.bool,
            device=local_inputs_embeds.device,
        )
        local_outputs = self.local_transformer.forward_local(
            attention_mask=local_attention_mask,
            inputs_embeds=local_inputs_embeds,
        )
        return local_outputs.last_hidden_state[:, -1, :]

    def _decode_local(
        self,
        local_inputs_embeds: torch.FloatTensor,
        current_local_input: torch.FloatTensor,
        effective_nq: int,
        do_sample: bool,
        audio_temperature: float,
        audio_top_k: int,
        audio_top_p: float,
        audio_repetition_penalty: float,
    ) -> [torch.IntTensor]:
        local_dtype = self.local_transformer.ln_f.weight.dtype
        next_frame_tokens = []
        for channel_index in range(effective_nq):
            local_inputs_embeds = torch.cat([local_inputs_embeds, current_local_input.unsqueeze(1)], dim=1)
            local_hidden_states = self._decode_local_last_hidden_state(local_inputs_embeds)
            channel_logits = self.audio_lm_heads[channel_index](local_hidden_states)
            #self._ensure_finite_generation_logits(channel_logits, f"audio logits[{channel_index}]")
            channel_token = self._sample_next_token(
                logits=channel_logits,
                do_sample=do_sample,
                temperature=audio_temperature,
                top_k=audio_top_k,
                top_p=audio_top_p,
                previous_token_ids=None,
                repetition_penalty=audio_repetition_penalty,
            )
            next_frame_tokens.append(channel_token)
            current_local_input = self.audio_embeddings[channel_index](channel_token).to(dtype=local_dtype)
        return next_frame_tokens

    def _iter_generation_events(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        nq: Optional[int] = None,
        max_new_frames: int = 300,
        do_sample: bool = False,
        text_temperature: float = 1.5,
        text_top_p: float = 1.0,
        text_top_k: int = 50,
        audio_temperature: float = 1.7,
        audio_top_p: float = 0.8,
        audio_top_k: int = 25,
        audio_repetition_penalty: float = 1.0,
        use_kv_cache: bool = True,
        return_dict_in_generate: bool = True,
        max_prompt_len: int = 200,
        max_total_len: int = 400,
    ) -> Iterator[dict[str, Any]]:
        if input_ids.ndim == 2:
            input_ids = input_ids.unsqueeze(0)
            raise ValueError(f"Expected input_ids with 3 dims, got shape {tuple(input_ids.shape)}")
        if attention_mask is None:
            attention_mask = torch.ones(input_ids.shape[:2], dtype=torch.bool, device=input_ids.device)
        elif attention_mask.ndim == 1:
            attention_mask = attention_mask.unsqueeze(0)

        effective_nq = self._resolve_inference_nq(nq)
        batch_size = input_ids.shape[0]
        current_input_ids = input_ids
        current_attention_mask = attention_mask.to(device=input_ids.device)
        current_model_input_ids = current_input_ids
        generated_frames = []
        finished = torch.zeros(batch_size, dtype=torch.bool, device=input_ids.device)
        past_key_values = None
        local_dtype = self.local_transformer.ln_f.weight.dtype

        max_new_frames = min(max_new_frames, max_total_len - max_prompt_len)
        for step_index in range(max_new_frames):
            generated_audio_history = torch.stack(generated_frames, dim=1) if generated_frames else None
            
            from datetime import datetime
            import logging
            now = datetime.now()
            time_str = now.strftime("%Y-%m-%d %H:%M:%S.") + str(now.microsecond)[:3]
            logging.info("begin build inputs embeds time: %s, pid: %s", time_str, str(os.getpid()))
            
            global_inputs_embeds = self._build_inputs_embeds(current_model_input_ids)
            
            now = datetime.now()
            time_str = now.strftime("%Y-%m-%d %H:%M:%S.") + str(now.microsecond)[:3]
            logging.info("begin transformer time: %s, pid: %s", time_str, str(os.getpid()))
            
            if step_index == 0:
                actual_len = global_inputs_embeds.shape[1]
                logging.info("actual_len: %s, pid: %s", str(actual_len), str(os.getpid()))
                assert actual_len <= max_prompt_len
                num_padding = max_prompt_len - actual_len
                padding_mask = torch.zeros(
                        (global_inputs_embeds.shape[0], num_padding), 
                        dtype=torch.bool, 
                        device=current_attention_mask.device
                        )
                padding_embeds = torch.zeros(
                        (global_inputs_embeds.shape[0], num_padding, self.transformer.config.n_embd), 
                        dtype=torch.bfloat16, 
                        device=global_inputs_embeds.device
                        )
                current_attention_mask = torch.cat((current_attention_mask,padding_mask), dim=1) 
                global_inputs_embeds = torch.cat((global_inputs_embeds,padding_embeds), dim=1)
        
                global_outputs = self.transformer.cached_prefill(input_ids=None,
                    past_key_values=past_key_values,
                    attention_mask=current_attention_mask,
                    position_ids=None,
                    inputs_embeds=global_inputs_embeds,
                    use_cache=use_kv_cache,
                    output_attentions=False,
                    output_hidden_states=False,
                    return_dict=True,
                    cu_seqlens=None,
                    num_sequences=None,
                )
                current_attention_mask = current_attention_mask[:,:actual_len]
                global_outputs.last_hidden_state = global_outputs.last_hidden_state[:,:actual_len]
                past_key_values = []
                for i in range(len(global_outputs.past_key_values)):
                    tmp = []
                    tmp.append(torch.cat((global_outputs.past_key_values[i][0][:,:actual_len], 
                        torch.zeros([global_inputs_embeds.shape[0], 
                            max_total_len - actual_len, 
                            self.transformer.config.n_head, 
                            self.transformer.config.n_embd // self.transformer.config.n_head], 
                            dtype=torch.bfloat16, 
                            device=global_outputs.past_key_values[i][0].device)), 
                        dim=1))
                    tmp.append(torch.cat((global_outputs.past_key_values[i][1][:,:actual_len], 
                        torch.zeros([global_inputs_embeds.shape[0], 
                            max_total_len - actual_len, 
                            self.transformer.config.n_head, 
                            self.transformer.config.n_embd // self.transformer.config.n_head], 
                            dtype=torch.bfloat16, 
                            device=global_outputs.past_key_values[i][1].device)), 
                        dim=1))
                    past_key_values.append(tmp)
                global_outputs.past_key_values = past_key_values
                kv_len = actual_len
            else:
                position_ids = torch.tensor([[kv_len]], dtype=torch.int64, device=global_inputs_embeds.device)
                num_padding = max_total_len - kv_len - 1 
                padding_mask = torch.zeros((global_inputs_embeds.shape[0], num_padding), dtype=torch.bool, device=current_attention_mask.device)
                current_attention_mask_tmp = torch.cat((current_attention_mask, padding_mask), dim=1)
                global_outputs = self.transformer.cached_decode(
                    input_ids=None,
                    past_key_values=past_key_values,
                    attention_mask=current_attention_mask_tmp,
                    position_ids=position_ids,
                    inputs_embeds=global_inputs_embeds,
                    use_cache=use_kv_cache,
                    output_attentions=False,
                    output_hidden_states=False,
                    return_dict=True,
                    cu_seqlens=None,
                    num_sequences=None,
                )
                kv_len += 1
            now = datetime.now()
            time_str = now.strftime("%Y-%m-%d %H:%M:%S.") + str(now.microsecond)[:3]
            logging.info("finish transformer time: %s, pid: %s", time_str, str(os.getpid()))
            global_hidden_states = global_outputs.last_hidden_state[:, -1, :].to(dtype=local_dtype)
            local_inputs_embeds = global_hidden_states.unsqueeze(1)
            local_hidden_states = self._decode_local_last_hidden_state(local_inputs_embeds)
            text_logits = self.text_lm_head(local_hidden_states)
            self._ensure_finite_generation_logits(text_logits, "text logits")
            next_text_tokens = self._sample_next_assistant_text_token(
                logits=text_logits,
                do_sample=do_sample,
                temperature=text_temperature,
                top_k=text_top_k,
                top_p=text_top_p,
            )
            should_continue = next_text_tokens.eq(self.config.audio_assistant_slot_token_id) & ~finished
            finished = finished | next_text_tokens.eq(self.config.audio_end_token_id)
            if not should_continue.any():
                break

            next_frame_tokens = []
            current_local_input = self.transformer.wte(next_text_tokens).to(dtype=local_dtype)
            now = datetime.now()
            time_str = now.strftime("%Y-%m-%d %H:%M:%S.") + str(now.microsecond)[:3]
            logging.info("begin local transformer time: %s, pid: %s", time_str, str(os.getpid()))
            next_frame_tokens = self._decode_local(
                    local_inputs_embeds, 
                    current_local_input, 
                    effective_nq, 
                    do_sample, 
                    audio_temperature, 
                    audio_top_k, 
                    audio_top_p,
                    audio_repetition_penalty,
                )
            now = datetime.now()
            time_str = now.strftime("%Y-%m-%d %H:%M:%S.") + str(now.microsecond)[:3]
            logging.info("finish local transformer time: %s, pid: %s", time_str, str(os.getpid()))
            next_frame_prefix = torch.stack(next_frame_tokens, dim=-1)
            if effective_nq < self.config.n_vq:
                next_frame = torch.full(
                    (batch_size, self.config.n_vq),
                    self.config.audio_pad_token_id,
                    dtype=next_frame_prefix.dtype,
                    device=next_frame_prefix.device,
                )
                next_frame[:, :effective_nq] = next_frame_prefix
            else:
                next_frame = next_frame_prefix
            padded_next_frame = next_frame.masked_fill(~should_continue.unsqueeze(-1), self.config.audio_pad_token_id)
            generated_frames.append(padded_next_frame)

            next_row = self._build_generation_row(
                batch_size=batch_size,
                device=input_ids.device,
                audio_token_ids=padded_next_frame,
            )
            if (~should_continue).any():
                next_row[~should_continue, 0, 0] = self.config.pad_token_id
                next_row[~should_continue, 0, 1:] = self.config.audio_pad_token_id

            current_input_ids = torch.cat([current_input_ids, next_row], dim=1)
            current_attention_mask = torch.cat([current_attention_mask, should_continue.unsqueeze(1)], dim=1)
            if use_kv_cache:
                current_model_input_ids = next_row
                past_key_values = global_outputs.past_key_values
            else:
                current_model_input_ids = current_input_ids

            yield {
                "type": "frame",
                "step_index": int(step_index),
                "audio_token_ids": padded_next_frame.detach().clone(),
                "active_mask": should_continue.detach().clone(),
                "finished_mask": finished.detach().clone(),
            }

        if generated_frames:
            audio_token_ids = torch.stack(generated_frames, dim=1)
        else:
            audio_token_ids = torch.empty((batch_size, 0, self.config.n_vq), dtype=torch.long, device=input_ids.device)

        if not return_dict_in_generate:
            yield {"type": "final", "audio_token_ids": audio_token_ids}
            return
        yield {
            "type": "final",
            "generation": MossTTSNanoGenerationOutput(audio_token_ids=audio_token_ids, prompt_input_ids=input_ids),
        }

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        nq: Optional[int] = None,
        max_new_frames: int = 300,
        do_sample: bool = False,
        text_temperature: float = 1.5,
        text_top_p: float = 1.0,
        text_top_k: int = 50,
        audio_temperature: float = 1.7,
        audio_top_p: float = 0.8,
        audio_top_k: int = 25,
        audio_repetition_penalty: float = 1.0,
        use_kv_cache: bool = True,
        return_dict_in_generate: bool = True,
    ):
        final_output: Any = None
        for event in self._iter_generation_events(
            input_ids=input_ids,
            attention_mask=attention_mask,
            nq=nq,
            max_new_frames=max_new_frames,
            do_sample=do_sample,
            text_temperature=text_temperature,
            text_top_p=text_top_p,
            text_top_k=text_top_k,
            audio_temperature=audio_temperature,
            audio_top_p=audio_top_p,
            audio_top_k=audio_top_k,
            audio_repetition_penalty=audio_repetition_penalty,
            use_kv_cache=use_kv_cache,
            return_dict_in_generate=return_dict_in_generate,
        ):
            if event["type"] != "final":
                continue
            final_output = event.get("generation", event.get("audio_token_ids"))
        if final_output is None:
            raise RuntimeError("Generation finished without producing a final output.")
        return final_output

    @torch.no_grad()
    def generate_stream(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        nq: Optional[int] = None,
        max_new_frames: int = 300,
        do_sample: bool = False,
        text_temperature: float = 1.5,
        text_top_p: float = 1.0,
        text_top_k: int = 50,
        audio_temperature: float = 1.7,
        audio_top_p: float = 0.8,
        audio_top_k: int = 25,
        audio_repetition_penalty: float = 1.0,
        use_kv_cache: bool = True,
        return_dict_in_generate: bool = True,
        max_prompt_len: int = 200,
        max_total_len: int = 400,
    ) -> Iterator[dict[str, Any]]:
        yield from self._iter_generation_events(
            input_ids=input_ids,
            attention_mask=attention_mask,
            nq=nq,
            max_new_frames=max_new_frames,
            do_sample=do_sample,
            text_temperature=text_temperature,
            text_top_p=text_top_p,
            text_top_k=text_top_k,
            audio_temperature=audio_temperature,
            audio_top_p=audio_top_p,
            audio_top_k=audio_top_k,
            audio_repetition_penalty=audio_repetition_penalty,
            use_kv_cache=use_kv_cache,
            return_dict_in_generate=return_dict_in_generate,
            max_prompt_len=max_prompt_len,
            max_total_len=max_total_len,
        )

    @torch.no_grad()
    def inference_stream(
        self,
        text: str,
        output_audio_path: Union[str, Path],
        mode: str = "continuation",
        prompt_text: Optional[str] = None,
        prompt_audio_path: Optional[Union[str, Path]] = None,
        reference_audio_path: Optional[Union[str, Path]] = None,
        prompt_audio_codes=None,
        text_tokenizer=None,
        text_tokenizer_path: Optional[str] = None,
        audio_tokenizer=None,
        audio_tokenizer_type: Optional[str] = None,
        audio_tokenizer_pretrained_name_or_path: Optional[str] = None,
        device: Optional[Union[str, torch.device]] = None,
        nq: Optional[int] = None,
        max_new_frames: int = 300,
        do_sample: bool = False,
        text_temperature: float = 1.5,
        text_top_p: float = 1.0,
        text_top_k: int = 50,
        audio_temperature: float = 1.7,
        audio_top_p: float = 0.8,
        audio_top_k: int = 25,
        audio_repetition_penalty: float = 1.0,
        use_kv_cache: bool = True,
        voice_clone_max_text_tokens: int = DEFAULT_VOICE_CLONE_MAX_TEXT_TOKENS,
        voice_clone_max_memory_per_sample_gb: float = DEFAULT_VOICE_CLONE_MAX_MEMORY_PER_SAMPLE_GB,
        tts_max_batch_size: int = 0,
        codec_max_batch_size: int = 0,
        max_prompt_len: Optional[int] = None,
        max_total_len: Optional[int] = None,
    ) -> Iterator[dict[str, Any]]:

        # env-overridable padding buckets (static-graph shape contract): the
        # compiled graph reuses itself only when prefill pads to the same
        # max_prompt_len every call, so callers must NOT vary this per request.
        # None → the class defaults in _iter_generation_events (200/400).
        import os as _os
        if max_prompt_len is None:
            max_prompt_len = int(_os.getenv("MOSS_TTS_NANO_MAX_PROMPT_LEN", "200"))
        if max_total_len is None:
            max_total_len = int(_os.getenv("MOSS_TTS_NANO_MAX_TOTAL_LEN", "400"))

        from datetime import datetime
        import logging
        now = datetime.now()
        time_str = now.strftime("%Y-%m-%d %H:%M:%S.") + str(now.microsecond)[:3]
        logging.info("real begin time: %s, pid: %s", time_str, str(os.getpid()))

        resolved_device = self._resolve_device(device)
        effective_nq = self._resolve_inference_nq(nq)
        was_training = self.training

        text_tokenizer = self._load_text_tokenizer(
            text_tokenizer=text_tokenizer,
            text_tokenizer_path=text_tokenizer_path,
        )
        audio_tokenizer = self._load_audio_tokenizer(
            audio_tokenizer=audio_tokenizer,
            audio_tokenizer_type=audio_tokenizer_type,
            audio_tokenizer_pretrained_name_or_path=audio_tokenizer_pretrained_name_or_path,
            device=resolved_device,
        )

        target_sample_rate = self._resolve_audio_tokenizer_sample_rate(audio_tokenizer)
        target_channels = self._resolve_audio_tokenizer_channels(audio_tokenizer)
        effective_prompt_audio_path = prompt_audio_path or reference_audio_path
        resolved_mode = self._resolve_inference_mode(
            mode=mode,
            has_prompt_text=prompt_text is not None,
            # cached prompt_audio_codes make voice_clone valid without a path
            has_prompt_audio=(effective_prompt_audio_path is not None) or (prompt_audio_codes is not None),
        )
        if reference_audio_path is not None and prompt_audio_path is None:
            logging.warning(
                "reference_audio_path=%s is treated as prompt_audio_path for backward compatibility.",
                reference_audio_path,
            )

        if prompt_audio_codes is None:
            if effective_prompt_audio_path is not None:
                waveform, sample_rate = self._load_reference_audio(
                    effective_prompt_audio_path,
                    target_sample_rate,
                    target_channels,
                )
                encoded = self._call_audio_encode(
                    audio_tokenizer=audio_tokenizer,
                    waveform=waveform.to(resolved_device),
                    sample_rate=sample_rate,
                )
                prompt_audio_codes = self._mask_unused_audio_channels(
                    self._normalize_audio_codes(encoded),
                    nq=effective_nq,
                ).to(resolved_device)

        voice_clone_text_chunks = [text]

        voice_clone_chunk_batch_size = 1
        voice_clone_codec_batch_size = 1

        generated_audio_token_chunks: list[torch.LongTensor] = []
        emitted_waveform_segments: list[torch.FloatTensor] = []
        decoded_sample_rate: Optional[int] = None
        emitted_samples_total = 0
        first_audio_emitted_at: Optional[float] = None
        streaming_reset_fn = getattr(audio_tokenizer, "_reset_batch_decode_streaming_state", None)

        try:
            for batch_start in range(0, len(voice_clone_text_chunks), voice_clone_chunk_batch_size):
                batch_chunks = voice_clone_text_chunks[batch_start : batch_start + voice_clone_chunk_batch_size]
                batch_prompt_input_ids: list[torch.LongTensor] = []
                batch_attention_masks: list[torch.BoolTensor] = []
                
                for text_chunk in batch_chunks:
                    prompt_input_ids, attention_mask = self.build_inference_input_ids(
                        text=text_chunk,
                        text_tokenizer=text_tokenizer,
                        mode=resolved_mode,
                        prompt_text=prompt_text,
                        prompt_audio_codes=prompt_audio_codes,
                        device=resolved_device,
                    )
                    batch_prompt_input_ids.append(prompt_input_ids)
                    batch_attention_masks.append(attention_mask)

                now = datetime.now()
                time_str = now.strftime("%Y-%m-%d %H:%M:%S.") + str(now.microsecond)[:3]
                logging.info("finish text preprocess time: %s, pid: %s", time_str, str(os.getpid()))

                batched_prompt_input_ids, batched_attention_mask = self._left_pad_inference_batch(
                    input_id_batches=batch_prompt_input_ids,
                    attention_mask_batches=batch_attention_masks,
                    device=resolved_device,
                )

                row_states = [
                    {
                        "pending_decode_frames": [],
                        "decoded_audio_segments": [],
                        "generation_complete": False,
                        "pause_emitted": False,
                    }
                    for _ in batch_chunks
                ]
                batch_emit_index = 0
                codec_stream_started = False

                if resolved_mode == "continuation" and prompt_audio_codes is not None:
                    prompt_decode_codes = self._prepare_audio_codes_for_decode(prompt_audio_codes, nq=effective_nq)
                    _ = audio_tokenizer.batch_decode(
                        [prompt_decode_codes],
                        num_quantizers=effective_nq,
                        streaming=True,
                        max_batch_size=1,
                        reset_stream=True,
                    )
                    codec_stream_started = True

                def _emit_ready_segments() -> Iterator[dict[str, Any]]:
                    nonlocal batch_emit_index, emitted_samples_total, first_audio_emitted_at
                    active_sample_rate = decoded_sample_rate or target_sample_rate
                    while batch_emit_index < len(batch_chunks):
                        state = row_states[batch_emit_index]
                        decoded_segments = state["decoded_audio_segments"]
                        if decoded_segments:
                            next_segment = decoded_segments.pop(0)
                            if next_segment.numel() == 0 or int(next_segment.shape[-1]) <= 0:
                                continue
                            emitted_waveform_segments.append(next_segment)
                            if first_audio_emitted_at is None:
                                first_audio_emitted_at = time.monotonic()
                            emitted_samples_total += int(next_segment.shape[-1])
                            yield {
                                "type": "audio",
                                "waveform": next_segment,
                                "sample_rate": active_sample_rate,
                                "chunk_index": batch_start + batch_emit_index,
                                "is_pause": False,
                                "emitted_audio_seconds": float(emitted_samples_total) / float(active_sample_rate),
                                "lead_seconds": self._compute_stream_lead_seconds(
                                    emitted_samples_total=emitted_samples_total,
                                    sample_rate=active_sample_rate,
                                    first_audio_emitted_at=first_audio_emitted_at,
                                ),
                            }
                            continue

                        if not state["generation_complete"]:
                            break

                        if (
                            resolved_mode == "voice_clone"
                            and len(voice_clone_text_chunks) > 1
                            and not state["pause_emitted"]
                            and (batch_start + batch_emit_index) < len(voice_clone_text_chunks) - 1
                        ):
                            state["pause_emitted"] = True
                            pause_seconds = self._estimate_voice_clone_inter_chunk_pause_seconds(
                                voice_clone_text_chunks[batch_start + batch_emit_index]
                            )
                            pause_samples = max(0, int(round(float(active_sample_rate) * pause_seconds)))
                            if pause_samples > 0:
                                silence = torch.zeros((target_channels, pause_samples), dtype=torch.float32)
                                emitted_waveform_segments.append(silence)
                                if first_audio_emitted_at is None:
                                    first_audio_emitted_at = time.monotonic()
                                emitted_samples_total += int(silence.shape[-1])
                                yield {
                                    "type": "audio",
                                    "waveform": silence,
                                    "sample_rate": active_sample_rate,
                                    "chunk_index": batch_start + batch_emit_index,
                                    "is_pause": True,
                                    "emitted_audio_seconds": float(emitted_samples_total) / float(active_sample_rate),
                                    "lead_seconds": self._compute_stream_lead_seconds(
                                        emitted_samples_total=emitted_samples_total,
                                        sample_rate=active_sample_rate,
                                        first_audio_emitted_at=first_audio_emitted_at,
                                    ),
                                }
                            batch_emit_index += 1
                            continue

                        batch_emit_index += 1

                def _maybe_decode_pending(force: bool) -> Iterator[dict[str, Any]]:
                    nonlocal codec_stream_started, decoded_sample_rate
                    pending_counts = [len(state["pending_decode_frames"]) for state in row_states]
                    total_pending = sum(pending_counts)
                    if total_pending <= 0:
                        return

                    active_sample_rate = decoded_sample_rate or target_sample_rate
                    head_pending = pending_counts[batch_emit_index] if batch_emit_index < len(batch_chunks) else 0
                    lead_seconds = self._compute_stream_lead_seconds(
                        emitted_samples_total=emitted_samples_total,
                        sample_rate=active_sample_rate,
                        first_audio_emitted_at=first_audio_emitted_at,
                    )
                    decode_budget = self._resolve_stream_decode_frame_budget(
                        emitted_samples_total=emitted_samples_total,
                        sample_rate=active_sample_rate,
                        first_audio_emitted_at=first_audio_emitted_at,
                    )

                    if not force:
                        should_decode = False
                        if first_audio_emitted_at is None and head_pending > 0:
                            should_decode = True
                        elif head_pending > 0 and lead_seconds < 0.45:
                            should_decode = True
                        elif max(pending_counts) >= decode_budget:
                            should_decode = True
                        elif lead_seconds < 0.0 and total_pending > 0:
                            should_decode = True
                        if not should_decode:
                            return

                    decode_window = max(pending_counts) if force else max(1, decode_budget)
                    empty_codes = torch.empty((effective_nq, 0), dtype=torch.long, device=resolved_device)
                    codes_list: list[torch.Tensor] = []
                    for state in row_states:
                        take_count = min(len(state["pending_decode_frames"]), decode_window)
                        if take_count <= 0:
                            codes_list.append(empty_codes)
                            continue
                        frame_rows = state["pending_decode_frames"][:take_count]
                        del state["pending_decode_frames"][:take_count]
                        frame_tensor = torch.cat(frame_rows, dim=0).to(device=resolved_device, dtype=torch.long)
                        codes_list.append(frame_tensor[:, :effective_nq].transpose(0, 1).contiguous())

                    now = datetime.now()
                    time_str = now.strftime("%Y-%m-%d %H:%M:%S.") + str(now.microsecond)[:3]
                    logging.info("audio_tokenizer.batch_decode begin time: %s, pid: %s", time_str, str(os.getpid()))
                    decode_output = audio_tokenizer.batch_decode(
                        codes_list,
                        num_quantizers=effective_nq,
                        streaming=True,
                        max_batch_size=(voice_clone_codec_batch_size if not codec_stream_started else None),
                        reset_stream=not codec_stream_started,
                    )
                    now = datetime.now()
                    time_str = now.strftime("%Y-%m-%d %H:%M:%S.") + str(now.microsecond)[:3]
                    logging.info("audio_tokenizer.batch_decode finish time: %s, pid: %s", time_str, str(os.getpid()))
                    codec_stream_started = True
                    waveform_rows, current_sample_rate = self._extract_batch_waveforms_and_sample_rate(
                        decode_output,
                        fallback_sample_rate=target_sample_rate,
                        batch_size=len(batch_chunks),
                    )
                    if decoded_sample_rate is None:
                        decoded_sample_rate = current_sample_rate
                    elif decoded_sample_rate != current_sample_rate:
                        raise ValueError(
                            f"Decoded sample rates differ across streaming decode calls: {decoded_sample_rate} vs {current_sample_rate}"
                        )

                    for row_index, waveform_row in enumerate(waveform_rows):
                        if waveform_row.numel() == 0 or int(waveform_row.shape[-1]) <= 0:
                            continue
                        row_states[row_index]["decoded_audio_segments"].append(waveform_row)

                    yield from _emit_ready_segments()

                try:
                    final_generation = None
                    for event in self.generate_stream(
                        input_ids=batched_prompt_input_ids,
                        attention_mask=batched_attention_mask,
                        nq=effective_nq,
                        max_new_frames=max_new_frames,
                        do_sample=do_sample,
                        text_temperature=text_temperature,
                        text_top_p=text_top_p,
                        text_top_k=text_top_k,
                        audio_temperature=audio_temperature,
                        audio_top_p=audio_top_p,
                        audio_top_k=audio_top_k,
                        audio_repetition_penalty=audio_repetition_penalty,
                        use_kv_cache=use_kv_cache,
                        return_dict_in_generate=True,
                        max_prompt_len=max_prompt_len,
                        max_total_len=max_total_len,
                    ):
                        if event["type"] == "frame":
                            frame_audio_token_ids = event["audio_token_ids"]
                            active_mask = event["active_mask"]
                            finished_mask = event["finished_mask"]
                            for row_index in range(len(batch_chunks)):
                                if bool(active_mask[row_index].item()):
                                    row_states[row_index]["pending_decode_frames"].append(
                                        frame_audio_token_ids[row_index : row_index + 1].detach().clone()
                                    )
                                if bool(finished_mask[row_index].item()):
                                    row_states[row_index]["generation_complete"] = True
                            yield from _maybe_decode_pending(force=True) #False)
                            continue
                        final_generation = event.get("generation")

                    if final_generation is None:
                        raise RuntimeError("Streaming generation finished without a final output.")

                    for state in row_states:
                        state["generation_complete"] = True

                    yield from _maybe_decode_pending(force=True)
                    yield from _emit_ready_segments()

                    batched_audio_token_ids = self._mask_unused_audio_channels(final_generation.audio_token_ids, nq=effective_nq)
                    for sample_index in range(len(batch_chunks)):
                        generated_audio_token_chunks.append(
                            self._trim_generated_audio_token_ids(
                                batched_audio_token_ids[sample_index],
                                effective_nq=effective_nq,
                            )
                        )
                finally:
                    if codec_stream_started and callable(streaming_reset_fn):
                        streaming_reset_fn()

            if generated_audio_token_chunks:
                audio_token_ids = torch.cat(generated_audio_token_chunks, dim=0)
            else:
                audio_token_ids = torch.empty((0, self.config.n_vq), dtype=torch.long, device=resolved_device)

            if emitted_waveform_segments:
                waveform = torch.cat(emitted_waveform_segments, dim=-1)
            else:
                waveform = torch.zeros((target_channels, 0), dtype=torch.float32)

            decoded_sample_rate = decoded_sample_rate or target_sample_rate
            output_path = Path(output_audio_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            torchaudio.save(str(output_path), waveform, decoded_sample_rate)

            # RESTORED from the stock checkpoint (tianjingcheng's edit dropped
            # this closing event; the vendored runtime treats its absence as a
            # failed job and discards the loaded model state):
            yield {
                "type": "result",
                "audio_path": str(output_path),
                "sample_rate": decoded_sample_rate,
                "audio_token_ids": audio_token_ids.detach().cpu(),
                "waveform": waveform,
                "reference_audio_token_ids": None if prompt_audio_codes is None else prompt_audio_codes.detach().cpu(),
                "voice_clone_text_chunks": voice_clone_text_chunks,
                "voice_clone_chunk_batch_size": int(voice_clone_chunk_batch_size),
                "voice_clone_codec_batch_size": int(voice_clone_codec_batch_size),
            }

        finally:
            if was_training:
                self.train()

    @torch.no_grad()
    def inference(
        self,
        text: str,
        output_audio_path: Union[str, Path],
        mode: str = "continuation",
        prompt_text: Optional[str] = None,
        prompt_audio_path: Optional[Union[str, Path]] = None,
        reference_audio_path: Optional[Union[str, Path]] = None,
        text_tokenizer=None,
        text_tokenizer_path: Optional[str] = None,
        audio_tokenizer=None,
        audio_tokenizer_type: Optional[str] = None,
        audio_tokenizer_pretrained_name_or_path: Optional[str] = None,
        device: Optional[Union[str, torch.device]] = None,
        nq: Optional[int] = None,
        max_new_frames: int = 300,
        do_sample: bool = False,
        text_temperature: float = 1.5,
        text_top_p: float = 1.0,
        text_top_k: int = 50,
        audio_temperature: float = 1.7,
        audio_top_p: float = 0.8,
        audio_top_k: int = 25,
        audio_repetition_penalty: float = 1.0,
        use_kv_cache: bool = True,
        voice_clone_max_text_tokens: int = DEFAULT_VOICE_CLONE_MAX_TEXT_TOKENS,
        voice_clone_max_memory_per_sample_gb: float = DEFAULT_VOICE_CLONE_MAX_MEMORY_PER_SAMPLE_GB,
        tts_max_batch_size: int = 0,
        codec_max_batch_size: int = 0,
    ) -> dict[str, Any]:
        resolved_device = self._resolve_device(device)
        effective_nq = self._resolve_inference_nq(nq)
        if next(self.parameters()).device != resolved_device:
            self.to(resolved_device)

        was_training = self.training
        self.eval()

        text_tokenizer = self._load_text_tokenizer(
            text_tokenizer=text_tokenizer,
            text_tokenizer_path=text_tokenizer_path,
        )
        audio_tokenizer = self._load_audio_tokenizer(
            audio_tokenizer=audio_tokenizer,
            audio_tokenizer_type=audio_tokenizer_type,
            audio_tokenizer_pretrained_name_or_path=audio_tokenizer_pretrained_name_or_path,
            device=resolved_device,
        )

        target_sample_rate = self._resolve_audio_tokenizer_sample_rate(audio_tokenizer)
        target_channels = self._resolve_audio_tokenizer_channels(audio_tokenizer)
        effective_prompt_audio_path = prompt_audio_path or reference_audio_path
        resolved_mode = self._resolve_inference_mode(
            mode=mode,
            has_prompt_text=prompt_text is not None,
            # cached prompt_audio_codes make voice_clone valid without a path
            has_prompt_audio=(effective_prompt_audio_path is not None) or (prompt_audio_codes is not None),
        )
        if reference_audio_path is not None and prompt_audio_path is None:
            logging.warning(
                "reference_audio_path=%s is treated as prompt_audio_path for backward compatibility.",
                reference_audio_path,
            )

        prompt_audio_codes = None
        if effective_prompt_audio_path is not None:
            waveform, sample_rate = self._load_reference_audio(
                effective_prompt_audio_path,
                target_sample_rate,
                target_channels,
            )
            encoded = self._call_audio_encode(
                audio_tokenizer=audio_tokenizer,
                waveform=waveform.to(resolved_device),
                sample_rate=sample_rate,
            )
            prompt_audio_codes = self._mask_unused_audio_channels(
                self._normalize_audio_codes(encoded),
                nq=effective_nq,
            ).to(resolved_device)

        if resolved_mode == "voice_clone":
            split_voice_clone_text_chunks = self._split_text_into_best_sentences(
                text_tokenizer=text_tokenizer,
                text=text,
                max_tokens=voice_clone_max_text_tokens,
            )
            voice_clone_text_chunks = split_voice_clone_text_chunks if len(split_voice_clone_text_chunks) > 1 else [text]
        else:
            voice_clone_text_chunks = [text]

        generated_audio_token_chunks: list[torch.LongTensor] = []
        decoded_waveform_chunks: list[torch.FloatTensor] = []
        decoded_sample_rate: Optional[int] = None

        if resolved_mode == "voice_clone" and len(voice_clone_text_chunks) > 1:
            voice_clone_chunk_batch_size, voice_clone_codec_batch_size = self._resolve_effective_voice_clone_batch_sizes(
                resolved_device=resolved_device,
                chunk_count=len(voice_clone_text_chunks),
                max_memory_per_sample_gb=float(voice_clone_max_memory_per_sample_gb),
                requested_tts_max_batch_size=tts_max_batch_size,
                requested_codec_max_batch_size=codec_max_batch_size,
                realtime_streaming=False,
            )
        else:
            voice_clone_chunk_batch_size = 1
            voice_clone_codec_batch_size = 1

        for batch_start in range(0, len(voice_clone_text_chunks), voice_clone_chunk_batch_size):
            batch_chunks = voice_clone_text_chunks[batch_start : batch_start + voice_clone_chunk_batch_size]
            batch_prompt_input_ids: list[torch.LongTensor] = []
            batch_attention_masks: list[torch.BoolTensor] = []
            for text_chunk in batch_chunks:
                prompt_input_ids, attention_mask = self.build_inference_input_ids(
                    text=text_chunk,
                    text_tokenizer=text_tokenizer,
                    mode=resolved_mode,
                    prompt_text=prompt_text,
                    prompt_audio_codes=prompt_audio_codes,
                    device=resolved_device,
                )
                batch_prompt_input_ids.append(prompt_input_ids)
                batch_attention_masks.append(attention_mask)

            batched_prompt_input_ids, batched_attention_mask = self._left_pad_inference_batch(
                input_id_batches=batch_prompt_input_ids,
                attention_mask_batches=batch_attention_masks,
                device=resolved_device,
            )
            batched_audio_token_ids = self._generate_audio_token_ids_with_fallback(
                prompt_input_ids=batched_prompt_input_ids,
                attention_mask=batched_attention_mask,
                effective_nq=effective_nq,
                max_new_frames=max_new_frames,
                do_sample=do_sample,
                text_temperature=text_temperature,
                text_top_p=text_top_p,
                text_top_k=text_top_k,
                audio_temperature=audio_temperature,
                audio_top_p=audio_top_p,
                audio_top_k=audio_top_k,
                audio_repetition_penalty=audio_repetition_penalty,
                use_kv_cache=use_kv_cache,
                resolved_device=resolved_device,
            )

            batch_audio_token_chunks: list[torch.LongTensor] = []
            for sample_index in range(len(batch_chunks)):
                audio_token_ids = self._trim_generated_audio_token_ids(
                    batched_audio_token_ids[sample_index],
                    effective_nq=effective_nq,
                )
                generated_audio_token_chunks.append(audio_token_ids)
                batch_audio_token_chunks.append(audio_token_ids)

            if resolved_mode == "voice_clone" and len(voice_clone_text_chunks) > 1:
                for codec_batch_start in range(0, len(batch_audio_token_chunks), voice_clone_codec_batch_size):
                    codec_audio_token_batches = batch_audio_token_chunks[
                        codec_batch_start : codec_batch_start + voice_clone_codec_batch_size
                    ]
                    decoded_waveforms, current_sample_rate = self._decode_audio_token_id_batch_to_waveforms(
                        audio_tokenizer=audio_tokenizer,
                        audio_token_id_batches=codec_audio_token_batches,
                        target_sample_rate=target_sample_rate,
                        effective_nq=effective_nq,
                        resolved_device=resolved_device,
                    )
                    if decoded_sample_rate is None:
                        decoded_sample_rate = current_sample_rate
                    elif decoded_sample_rate != current_sample_rate:
                        raise ValueError(
                            f"Decoded sample rates differ across voice_clone chunks: {decoded_sample_rate} vs {current_sample_rate}"
                        )
                    decoded_waveform_chunks.extend(decoded_waveforms)

        if generated_audio_token_chunks:
            audio_token_ids = torch.cat(generated_audio_token_chunks, dim=0)
        else:
            audio_token_ids = torch.empty((0, self.config.n_vq), dtype=torch.long, device=resolved_device)

        if resolved_mode == "voice_clone" and len(voice_clone_text_chunks) > 1:
            waveform = (
                self._concat_voice_clone_waveform_chunks(
                    waveform_chunks=decoded_waveform_chunks,
                    text_chunks=voice_clone_text_chunks,
                    sample_rate=decoded_sample_rate,
                )
                if decoded_waveform_chunks
                else torch.zeros((target_channels, 0), dtype=torch.float32)
            )
        else:
            decode_audio_token_ids = audio_token_ids
            prompt_waveform_prefix_samples = 0
            if resolved_mode == "continuation" and prompt_audio_codes is not None:
                decode_audio_token_ids = torch.cat([prompt_audio_codes, audio_token_ids], dim=0)
                prompt_waveform_prefix_samples = (
                    int(prompt_audio_codes.shape[0]) * self._resolve_audio_tokenizer_downsample_rate(audio_tokenizer)
                )

            waveform, decoded_sample_rate = self._decode_audio_token_ids_to_waveform(
                audio_tokenizer=audio_tokenizer,
                audio_token_ids=decode_audio_token_ids,
                target_sample_rate=target_sample_rate,
                effective_nq=effective_nq,
                resolved_device=resolved_device,
            )
            if prompt_waveform_prefix_samples > 0:
                if decoded_sample_rate != target_sample_rate:
                    prompt_waveform_prefix_samples = int(
                        round(prompt_waveform_prefix_samples * float(decoded_sample_rate) / float(target_sample_rate))
                    )
                prompt_waveform_prefix_samples = min(prompt_waveform_prefix_samples, int(waveform.shape[-1]))
                waveform = waveform[:, prompt_waveform_prefix_samples:]

        assert decoded_sample_rate is not None

        output_path = Path(output_audio_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torchaudio.save(str(output_path), waveform, decoded_sample_rate)

        if was_training:
            self.train()

        return {
            "audio_path": str(output_path),
            "sample_rate": decoded_sample_rate,
            "audio_token_ids": audio_token_ids.detach().cpu(),
            "waveform": waveform,
            "reference_audio_token_ids": None if prompt_audio_codes is None else prompt_audio_codes.detach().cpu(),
            "voice_clone_text_chunks": voice_clone_text_chunks,
            "voice_clone_chunk_batch_size": int(voice_clone_chunk_batch_size),
            "voice_clone_codec_batch_size": int(voice_clone_codec_batch_size),
        }
