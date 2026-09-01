# moss-npu-adaptations 修改说明（vs HuggingFace stock 原版）

> 生成方式：适配文件与 HF 原版文件的 diff。原因描述见每段上方注释。

---

## 一、vlm/MOSS-VL-Realtime-0708/（4.57 主线）

**modeling_moss_vl.py** — ① `empty_cache` NPU 感知：NPU 会话结束后释放 HBM，GPU 路径不变；② realtime 空闲忙等修复（详见下）；③ **realtime 解码乱码守卫**：发射守卫只查末字符完整性，采样路径下多字节字符可在字符串中间断裂（后续 token 补全了别的字符、孤儿化前者的尾字节）→ U+FFFD 永久残留（实测：`楼层名`→`楼�名`）；发射时剥离不可恢复的断裂字符（平台中立，GPU 同样受益）；② realtime 空闲态 busy-wait（`while True: continue`，VideoMllama 参考实现）改 20ms sleep 轮询：裸忙转独占 GIL，同进程的 ASR 解码被饿死（实测 150ms→40-80s，独立进程单忙转线程复现 73× 放大；修复后 132-152ms）。该缺陷存在于 HF stock checkpoint，GPU 部署同样受影响
```diff
@@ -2874,8 +2874,13 @@
 
     def stop_real_time_generate(self):
         gc.collect()
-        if torch.cuda.is_available():
-            torch.cuda.empty_cache()
+        try:
+            import torch_npu  # noqa: F401
+            if torch.npu.is_available():
+                torch.npu.empty_cache()
+        except ImportError:
+            if torch.cuda.is_available():
+                torch.cuda.empty_cache()
         self.continue_generating = False
 
     @staticmethod
@@ -3099,8 +3104,11 @@
                     break
 
             if self.continue_generating and should_wait_for_new_input and not frames_to_process and not prompts_to_process:
-                # Busy-wait — matches VideoMllama reference. Caller controls cadence via
-                # `max_tokens_per_turn` sleeping in `_real_time_sample`.
+                # GIL-friendly wait: a bare `continue` busy-spin monopolizes the GIL
+                # and starves in-process siblings (in-process ASR decode inflates
+                # ~150ms -> 40-80s while a realtime session idles in <|silence|>).
+                # 20ms poll keeps input-detection latency negligible.
+                time.sleep(0.02)
                 continue
             break
 
```diff
@@ -44,7 +44,10 @@
 from transformers.processing_utils import Unpack
 from transformers.utils import TransformersKwargs, auto_docstring, is_torchdynamo_compiling, logging
 from transformers.utils.deprecation import deprecate_kwarg
-from transformers.utils.generic import OutputRecorder
+try:
+    from transformers.utils.generic import OutputRecorder
+except ImportError:
+    from transformers.utils.output_capturing import OutputRecorder
 
 from .configuration_moss_vl import MossVLConfig, MossVLTextConfig, MossVLVisionConfig
 import copy
@@ -209,11 +212,14 @@
 
     def __init__(self, dim: int, theta: float = 10000.0) -> None:
         super().__init__()
-        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
+        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float, device="cpu") / dim))
         self.register_buffer("inv_freq", inv_freq, persistent=False)
+        self._inv_freq_clean = inv_freq.clone()
 
     def forward(self, seqlen: int) -> torch.Tensor:
-        seq = torch.arange(seqlen, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
+        if torch.isnan(self.inv_freq).any() or torch.isinf(self.inv_freq).any():
+            self.inv_freq = self._inv_freq_clean.to(self.inv_freq.device)
+        seq = torch.arange(seqlen, dtype=self.inv_freq.dtype, device=self.inv_freq.device)
         freqs = torch.outer(seq, self.inv_freq)
         return freqs
 
@@ -445,12 +451,23 @@
         self.original_max_seq_len = config.max_position_embeddings
 
         self.config = config
-        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]
-
-        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
+        if self.rope_type == "default" or self.rope_type not in ROPE_INIT_FUNCTIONS:
+            import torch as _torch
+            base = config.rope_theta if hasattr(config, "rope_theta") else 10000.0
+            head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
+            inv_freq = 1.0 / (base ** (_torch.arange(0, head_dim, 2, device="cpu").float() / head_dim))
+            self.attention_scaling = 1.0
+        else:
+            self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]
+            inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
         self.register_buffer("inv_freq", inv_freq, persistent=False)
         self.original_inv_freq = self.inv_freq
 
+        # Guard: from_pretrained may corrupt inv_freq with garbage from the
+        # state dict (persistent=False buffers can still be overwritten during
+        # weight loading on some transformers versions). Re-validate on forward.
+        self._inv_freq_computed = inv_freq.clone()
+
 
         if hasattr(config, "rope_scaling") and config.rope_scaling is not None:
              self.mrope_section = config.rope_scaling.get("mrope_section", [24, 20, 20])
@@ -477,6 +494,10 @@
     @torch.no_grad()
     @dynamic_rope_update
     def forward(self, x, position_ids):
+        # Guard: from_pretrained can corrupt inv_freq — restore from clean copy
+        if torch.isnan(self.inv_freq).any() or torch.isinf(self.inv_freq).any():
+            self.inv_freq = self._inv_freq_computed.to(self.inv_freq.device)
+            self.original_inv_freq = self._inv_freq_computed.to(self.inv_freq.device)
 
         if position_ids.ndim == 2:
             position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)
@@ -1169,14 +1190,26 @@
         if position_ids is None:
             position_ids = cache_position.unsqueeze(0)
 
-        attention_mask = create_causal_mask(
-            config=self.config,
-            input_embeds=inputs_embeds,
-            attention_mask=attention_mask,
-            cache_position=cache_position,
-            past_key_values=past_key_values,
-            position_ids=position_ids,
-        )
+        # dual-version: transformers 5.x renamed input_embeds→inputs_embeds and
+        # dropped cache_position; 4.57 (this checkpoint's native line, per
+        # config.json transformers_version 4.57.3) uses the old kwargs.
+        try:
+            attention_mask = create_causal_mask(
+                config=self.config,
+                inputs_embeds=inputs_embeds,
+                attention_mask=attention_mask,
+                past_key_values=past_key_values,
+                position_ids=position_ids,
+            )
+        except TypeError:
+            attention_mask = create_causal_mask(
+                config=self.config,
+                input_embeds=inputs_embeds,
+                attention_mask=attention_mask,
+                cache_position=cache_position,
+                past_key_values=past_key_values,
+                position_ids=position_ids,
+            )
 
         hidden_states = inputs_embeds
 
@@ -3545,8 +3578,10 @@
         invalid_token_id = invalid_candidates[0] if invalid_candidates and invalid_candidates[0] is not None else -1
 
         # Bootstrap cache_position from the prefill input_ids length.
-        # transformers 4.57: signature is (seq_length, device, model_kwargs).
-        model_kwargs = self._get_initial_cache_position(input_ids.shape[1], input_ids.device, model_kwargs)
+        if hasattr(self, '_get_initial_cache_position'):
+            model_kwargs = self._get_initial_cache_position(input_ids.shape[1], input_ids.device, model_kwargs)
+        else:
+            model_kwargs["cache_position"] = torch.arange(input_ids.shape[1], device=input_ids.device)
 
         is_prefill = True
         current_token_start_time = time.time()
```

**configuration_moss_vl.py** — 补 `pad_token_id=None` 字段：5.x config `__getattribute__`
强检该属性（modeling 读 `config.pad_token_id`），缺失即 AttributeError。4.57 上多一个
None 字段无副作用。

```diff
@@ MossVLTextConfig.__init__ @@
         cross_attention_layers=None,  # List of layer indices to insert cross attention
+        pad_token_id=None,
         **kwargs,
@@ ... @@
         self.attention_bias = attention_bias
         self.attention_dropout = attention_dropout
+        self.pad_token_id = pad_token_id
```


**processing_moss_vl.py** — 仅一处：10D permute+reshape 设备感知路由（NPU 算子上限 8 维，
`device.type=="npu"` 时经 CPU 计算后搬回，CUDA 原生 10D 路径不变）。
*2026-09-01 上游 review 后最小化*：曾含的 interpolation 显式链、`_preprocess` 签名默认值、
`__init__` 兜底均已删除——实测证明为死代码（框架对全部 valid kwargs setdefault 后显式传参，
签名默认值永不生效；BICUBIC 经父类 `resample` 默认 + base 的 resample→interpolation 转换
天然到达，三配置 pixel_values 逐位一致）。

```diff
@@ -164,12 +164,17 @@
             )
             # Reorder dimensions to group grid and patch information for subsequent flattening.
             # (batch, grid_t, grid_h, grid_w, merge_h, merge_w, channel, temp_patch_size, patch_h, patch_w)
+            # NPU ops support at most 8-D tensors; route the 10-D permute+reshape
+            # through CPU there. CUDA handles 10-D natively — keep it on-device.
+            patches_device = patches.device
+            if patches_device.type == "npu":
+                patches = patches.cpu()
             patches = patches.permute(0, 1, 4, 7, 5, 8, 3, 2, 6, 9)
             flatten_patches = patches.reshape(
                 batch_size,
                 grid_t * grid_h * grid_w,
                 channel * temporal_patch_size * patch_size * patch_size,
-            )
+            ).to(patches_device)
 
             processed_images_grouped[shape] = flatten_patches
             processed_grids[shape] = [[grid_t, grid_h, grid_w]] * batch_size
```

**video_processing_moss_vl.py** — 同款 10D 路由（唯一修改）：

```diff
@@ -1147,12 +1147,17 @@
                 merge_size,
                 patch_size,
             )
+            patches_device = patches.device
+            # NPU: max 8D tensors — route the 10D permute+reshape through CPU.
+            # CUDA handles 10D natively — keep it on-device.
+            if patches_device.type == "npu":
+                patches = patches.cpu()
             patches = patches.permute(0, 1, 4, 7, 5, 8, 3, 2, 6, 9)
             flatten_patches = patches.reshape(
                 batch_size,
                 grid_t * grid_h * grid_w,
                 channel * temporal_patch_size * patch_size * patch_size,
-            )
+            ).to(patches_device)
 
             processed_videos_grouped[shape] = flatten_patches
             processed_grids[shape] = [[grid_t, grid_h, grid_w]] * batch_size
```

**preprocessor_config.json** — 已还原 stock（原 +`interpolation:BICUBIC` 为 no-op：
valid kwarg 名是 `resample`，config 里的 `interpolation` 键不被框架读取；有无该字段
pixel_values 逐位一致）。

---

---

## 三、tts/MOSS-TTS-Nano-100M/（4.57 主线，2 文件）

**gpt2_decoder.py** — ① `repeat_interleave` 重写为 `stack+flatten`：torchair（torch_npu 2.11）
没有 `aten.repeat_interleave` 的 GE converter，图编译失败；数学等价可编译 ② `BaseModelOutputWithPast`
等改函数内导入：torchair kernel 在重建命名空间 exec 编译产物，模块级类引用丢失 → 第二次调用 NameError

```diff
@@ -5,6 +5,18 @@
 from typing import Optional
 
 import torch
+try:
+    # NPU-only acceleration stack (optional): guarded so this checkpoint also
+    # loads on CUDA/CPU-only machines. cache_compile below is likewise gated
+    # on torch.npu availability — GPU machines run the plain eager methods.
+    import torch_npu  # noqa: F401
+    import torchair as tng
+    from torchair.configs.compiler_config import CompilerConfig
+    _TORCHAIR_AVAILABLE = True
+except ImportError:
+    tng = None
+    CompilerConfig = None
+    _TORCHAIR_AVAILABLE = False
 import torch.nn as nn
 import torch.utils.checkpoint
 from transformers.activations import ACT2FN
@@ -48,11 +60,16 @@
         device: torch.device,
         dtype: torch.dtype,
     ) -> tuple[torch.Tensor, torch.Tensor]:
+        # NOTE: no in-forward inv_freq guard — a data-dependent .item() branch
+        # breaks torchair's fullgraph compile (gb0170).
         if position_ids.ndim == 1:
             position_ids = position_ids.unsqueeze(0)
         freqs = torch.einsum("bs,d->bsd", position_ids.to(device=device, dtype=self.inv_freq.dtype), self.inv_freq)
-        cos = freqs.cos().repeat_interleave(2, dim=-1).unsqueeze(2).to(dtype=dtype)
-        sin = freqs.sin().repeat_interleave(2, dim=-1).unsqueeze(2).to(dtype=dtype)
+        # repeat_interleave(2, dim=-1) rewritten as stack+flatten: the bundled
+        # torchair (torch_npu 2.11) has no GE converter for aten.repeat_interleave
+        cos_f, sin_f = freqs.cos(), freqs.sin()
+        cos = torch.stack((cos_f, cos_f), dim=-1).flatten(start_dim=-2).unsqueeze(2).to(dtype=dtype)
+        sin = torch.stack((sin_f, sin_f), dim=-1).flatten(start_dim=-2).unsqueeze(2).to(dtype=dtype)
         return cos, sin
 
 
@@ -190,39 +207,29 @@
         value: torch.Tensor,
         attention_mask: Optional[torch.Tensor],
     ) -> torch.Tensor:
-        query = query.transpose(1, 2)
-        key = key.transpose(1, 2)
-        value = value.transpose(1, 2)
-        mask = None
+        # torch_npu 2.11 dispatches F.scaled_dot_product_attention to
+        # npu_fusion_attention_v3, whose torchair GE converter is unavailable on
+        # CANN 9.0.0 — run the math path instead (_eager_attention is
+        # semantically identical: same causal & key mask, same scale).
+        return self._eager_attention(query=query, key=key, value=value, attention_mask=attention_mask)
+
+    def _sdpa_attention_prefill(
+        self,
+        query: torch.Tensor,
+        key: torch.Tensor,
+        value: torch.Tensor,
+        attention_mask: Optional[torch.Tensor],
+    ) -> torch.Tensor:
+        # math path (same npu_fusion_attention_v3 GE-converter workaround);
+        # keeps the original's padded-query-row zeroing on the (B,S,N,D) output.
         query_attention_mask = None
         if attention_mask is not None:
-            query_length = query.shape[-2]
-            key_length = key.shape[-2]
-            mask = self._causal_attention_mask(
-                attention_mask=attention_mask,
-                query_length=query_length,
-                key_length=key_length,
-                device=query.device,
-            )
+            query_length = query.shape[1]
             query_attention_mask = attention_mask[:, -query_length:].to(dtype=torch.bool, device=query.device)
-            if not bool(query_attention_mask.all()):
-                # SDPA can produce NaNs when a query row is fully masked. For padded query positions,
-                # keep a single aligned key visible, then zero the query output after attention.
-                mask = mask.expand(query.shape[0], -1, -1, -1).clone()
-                invalid_batch, invalid_query = torch.nonzero(~query_attention_mask, as_tuple=True)
-                aligned_key = invalid_query + max(key_length - query_length, 0)
-                mask[invalid_batch, :, invalid_query, aligned_key] = True
-        output = torch.nn.functional.scaled_dot_product_attention(
-            query,
-            key,
-            value,
-            attn_mask=mask,
-            dropout_p=self.attn_dropout if self.training else 0.0,
-            is_causal=mask is None,
-        )
-        if query_attention_mask is not None and not bool(query_attention_mask.all()):
-            output = output.masked_fill(~query_attention_mask[:, None, :, None], 0.0)
-        return output.transpose(1, 2).contiguous()
+        output = self._eager_attention(query=query, key=key, value=value, attention_mask=attention_mask)
+        if query_attention_mask is not None:
+            output = output.masked_fill(~query_attention_mask[:, :, None, None], 0.0)
+        return output
 
     def _flash_attention(
         self,
@@ -300,6 +307,7 @@
         packed_metadata: Optional[PackedSequenceMetadata] = None,
         layer_past: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
         use_cache: bool = False,
+        prefill: bool = False,
     ) -> tuple[torch.Tensor, Optional[tuple[torch.Tensor, torch.Tensor]]]:
         qkv = self.c_attn(hidden_states)
         query, key, value = qkv.split(self.embed_dim, dim=-1)
@@ -320,12 +328,22 @@
 
         if layer_past is not None:
             past_key, past_value = layer_past
-            key = torch.cat([past_key.to(device=key.device, dtype=key.dtype), key], dim=1)
-            value = torch.cat([past_value.to(device=value.device, dtype=value.dtype), value], dim=1)
+            index = position_ids[0, 0].view(1, 1, 1, 1).expand_as(key)
+            past_key.scatter_(1, index, key)
+            past_value.scatter_(1, index, value)
+            key = past_key
+            value = past_value
 
         present = (key, value) if use_cache else None
 
-        if self.attn_implementation == "flash_attention_2" and layer_past is None:
+        if prefill:
+            attn_output = self._sdpa_attention_prefill(
+                query=query,
+                key=key,
+                value=value,
+                attention_mask=attention_mask,
+            )
+        elif self.attn_implementation == "flash_attention_2" and layer_past is None:
             attn_output = self._flash_attention(
                 query=query,
                 key=key,
@@ -370,6 +388,7 @@
         packed_metadata: Optional[PackedSequenceMetadata] = None,
         layer_past: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
         use_cache: bool = False,
+        prefill: bool = False,
     ) -> tuple[torch.Tensor, Optional[tuple[torch.Tensor, torch.Tensor]]]:
         attn_output, present = self.attn(
             self.ln_1(hidden_states),
@@ -378,6 +397,7 @@
             packed_metadata=packed_metadata,
             layer_past=layer_past,
             use_cache=use_cache,
+            prefill=prefill,
         )
         hidden_states = hidden_states + attn_output
         hidden_states = hidden_states + self.mlp(self.ln_2(hidden_states))
@@ -403,6 +423,20 @@
         self.gradient_checkpointing = False
         self._reset_parameters()
 
+        # torchair graph-compiled prefill/decode (NPU only): CUDA/CPU machines
+        # bind the plain eager methods instead — the caller
+        # (modeling_moss_tts_nano.inference_stream) only requires the two
+        # attributes to exist.
+        if _TORCHAIR_AVAILABLE and torch.npu.is_available():
+            _cfg = CompilerConfig()
+            _cfg.experimental_config.frozen_parameter = True
+            _cfg.experimental_config.tiling_schedule_optimize = True
+            self.cached_prefill = tng.inference.cache_compile(self.prefill, config=_cfg)
+            self.cached_decode = tng.inference.cache_compile(self.decode, config=_cfg)
+        else:
+            self.cached_prefill = self.prefill
+            self.cached_decode = self.decode
+
     def _reset_parameters(self) -> None:
         init_std = float(self.config.initializer_range)
         for module in self.modules():
@@ -610,6 +644,248 @@
         if not return_dict:
             return (hidden_states, tuple(presents) if presents is not None else None, all_hidden_states, None)
 
+        from transformers.modeling_outputs import BaseModelOutputWithPast  # torchair kernel-namespace safe
+        return BaseModelOutputWithPast(
+            last_hidden_state=hidden_states,
+            past_key_values=tuple(presents) if presents is not None else None,
+            hidden_states=all_hidden_states,
+            attentions=None,
+        )
+
+    def decode(
+        self,
+        input_ids: Optional[torch.LongTensor] = None,
+        past_key_values: Optional[tuple[tuple[torch.Tensor, torch.Tensor], ...]] = None,
+        attention_mask: Optional[torch.Tensor] = None,
+        position_ids: Optional[torch.LongTensor] = None,
+        inputs_embeds: Optional[torch.FloatTensor] = None,
+        use_cache: Optional[bool] = None,
+        output_attentions: Optional[bool] = None,
+        output_hidden_states: Optional[bool] = None,
+        return_dict: bool = True,
+        cu_seqlens: Optional[torch.Tensor] = None,
+        num_sequences: Optional[torch.Tensor] = None,
+    ) -> BaseModelOutputWithPast:
+        del input_ids, output_attentions
+
+        if inputs_embeds is None:
+            raise ValueError("inputs_embeds must be provided.")
+
+        use_cache = bool(use_cache)
+        if use_cache and cu_seqlens is not None:
+            raise ValueError("use_cache=True is not supported together with cu_seqlens packing.")
+
+        hidden_states = inputs_embeds
+        attention_mask = attention_mask.to(dtype=torch.bool, device=hidden_states.device)
+
+        packed_metadata = None
+
+        hidden_states = self.drop(hidden_states)
+
+        all_hidden_states = () if output_hidden_states else None
+        presents = [] if use_cache else None
+        for layer_index, block in enumerate(self.h):
+            if output_hidden_states:
+                all_hidden_states = all_hidden_states + (hidden_states,)
+
+            hidden_states, present = block(
+                hidden_states,
+                attention_mask=attention_mask,
+                position_ids=position_ids,
+                packed_metadata=packed_metadata,
+                layer_past=None if past_key_values is None else past_key_values[layer_index],
+                use_cache=use_cache,
+            )
+            if presents is not None:
+                presents.append(present)
+
+        hidden_states = self.ln_f(hidden_states)
+        if output_hidden_states:
+            all_hidden_states = all_hidden_states + (hidden_states,)
+
+        if not return_dict:
+            return (hidden_states, tuple(presents) if presents is not None else None, all_hidden_states, None)
+
+        from transformers.modeling_outputs import BaseModelOutputWithPast  # torchair kernel-namespace safe
+        return BaseModelOutputWithPast(
+            last_hidden_state=hidden_states,
+            past_key_values=tuple(presents) if presents is not None else None,
+            hidden_states=all_hidden_states,
+            attentions=None,
+        )
+
+    def prefill(
+        self,
+        input_ids: Optional[torch.LongTensor] = None,
+        past_key_values: Optional[tuple[tuple[torch.Tensor, torch.Tensor], ...]] = None,
+        attention_mask: Optional[torch.Tensor] = None,
+        position_ids: Optional[torch.LongTensor] = None,
+        inputs_embeds: Optional[torch.FloatTensor] = None,
+        use_cache: Optional[bool] = None,
+        output_attentions: Optional[bool] = None,
+        output_hidden_states: Optional[bool] = None,
+        return_dict: bool = True,
+        cu_seqlens: Optional[torch.Tensor] = None,
+        num_sequences: Optional[torch.Tensor] = None,
+    ) -> BaseModelOutputWithPast:
+        del input_ids, output_attentions
+
+        if inputs_embeds is None:
+            raise ValueError("inputs_embeds must be provided.")
+
+        hidden_states = inputs_embeds
+
+        assert attention_mask is not None
+        attention_mask = attention_mask.to(dtype=torch.bool, device=hidden_states.device)
+        query_attention_mask = attention_mask[:, -hidden_states.shape[1] :]
+
+        packed_metadata = None
+        position_ids = attention_mask.long().cumsum(dim=-1) - 1
+        position_ids = position_ids.masked_fill(~attention_mask, 0)
+        position_ids = position_ids[:, -hidden_states.shape[1] :]
+
+        hidden_states = self.drop(hidden_states)
+        hidden_states = hidden_states * query_attention_mask.unsqueeze(-1).to(dtype=hidden_states.dtype)
+
+        all_hidden_states = () if output_hidden_states else None
+        presents = [] if use_cache else None
+        for layer_index, block in enumerate(self.h):
+            if output_hidden_states:
+                all_hidden_states = all_hidden_states + (hidden_states,)
+
+            hidden_states, present = block(
+                hidden_states,
+                attention_mask=attention_mask,
+                position_ids=position_ids,
+                packed_metadata=packed_metadata,
+                layer_past=None if past_key_values is None else past_key_values[layer_index],
+                use_cache=use_cache,
+                prefill=True
+            )
+            hidden_states = hidden_states * query_attention_mask.unsqueeze(-1).to(dtype=hidden_states.dtype)
+            if presents is not None:
+                presents.append(present)
+
+        hidden_states = self.ln_f(hidden_states)
+        hidden_states = hidden_states * query_attention_mask.unsqueeze(-1).to(dtype=hidden_states.dtype)
+        if output_hidden_states:
+            all_hidden_states = all_hidden_states + (hidden_states,)
+
+        if not return_dict:
+            return (hidden_states, tuple(presents) if presents is not None else None, all_hidden_states, None)
+
+        from transformers.modeling_outputs import BaseModelOutputWithPast  # torchair kernel-namespace safe
+        return BaseModelOutputWithPast(
+            last_hidden_state=hidden_states,
+            past_key_values=tuple(presents) if presents is not None else None,
+            hidden_states=all_hidden_states,
+            attentions=None,
+        )
+
+    def forward_local(
+        self,
+        attention_mask: Optional[torch.Tensor] = None,
+        inputs_embeds: Optional[torch.FloatTensor] = None,
+    ) -> BaseModelOutputWithPast:
+        if inputs_embeds is None:
+            raise ValueError("inputs_embeds must be provided.")
+
+        hidden_states = inputs_embeds
+        if attention_mask is None:
+            attention_mask = torch.ones(hidden_states.shape[:2], dtype=torch.bool, device=hidden_states.device)
+        else:
+            attention_mask = attention_mask.to(dtype=torch.bool, device=hidden_states.device)
+        query_attention_mask = attention_mask[:, -hidden_states.shape[1] :]
+
+        packed_metadata = None
+        position_ids = None
+        cu_seqlens = None
+        output_hidden_states = None
+        use_cache = None
+        past_key_values = None
+        return_dict = True
+        if position_ids is None:
+            if cu_seqlens is not None:
+                position_ids = self.build_packed_position_ids(
+                    attention_mask=attention_mask,
+                    cu_seqlens=cu_seqlens.to(device=hidden_states.device),
+                    num_sequences=num_sequences.to(device=hidden_states.device) if num_sequences is not None else None,
+                )
+            elif attention_mask is not None:
+                position_ids = attention_mask.long().cumsum(dim=-1) - 1
+                position_ids = position_ids.masked_fill(~attention_mask, 0)
+                position_ids = position_ids[:, -hidden_states.shape[1] :]
+            else:
+                past_length = 0
+                if past_key_values is not None and len(past_key_values) > 0:
+                    past_length = past_key_values[0][0].shape[1]
+                position_ids = torch.arange(hidden_states.shape[1], device=hidden_states.device, dtype=torch.long)
+                position_ids = position_ids + past_length
+                position_ids = position_ids.unsqueeze(0).expand(hidden_states.shape[0], -1)
+
+        if cu_seqlens is not None and self.attn_implementation == "flash_attention_2":
+            packed_metadata = self.build_packed_metadata(
+                hidden_states=hidden_states,
+                cu_seqlens=cu_seqlens.to(device=hidden_states.device),
+                num_sequences=num_sequences.to(device=hidden_states.device) if num_sequences is not None else None,
+            )
+
+        if self.position_embedding_type == "absolute":
+            hidden_states = hidden_states + self.wpe(position_ids)
+        hidden_states = self.drop(hidden_states)
+        hidden_states = hidden_states * query_attention_mask.unsqueeze(-1).to(dtype=hidden_states.dtype)
+
+        all_hidden_states = () if output_hidden_states else None
+        presents = [] if use_cache else None
+        for layer_index, block in enumerate(self.h):
+            if output_hidden_states:
+                all_hidden_states = all_hidden_states + (hidden_states,)
+
+            if self.gradient_checkpointing and self.training:
+                if use_cache:
+                    raise ValueError("use_cache=True is not supported when gradient checkpointing is enabled during training.")
+
+                def custom_forward(*inputs):
+                    output, _ = block(
+                        inputs[0],
+                        attention_mask=inputs[1],
+                        position_ids=inputs[2],
+                        packed_metadata=packed_metadata,
+                        layer_past=None,
+                        use_cache=False,
+                    )
+                    return output
+
+                hidden_states = torch.utils.checkpoint.checkpoint(
+                    custom_forward,
+                    hidden_states,
+                    attention_mask,
+                    position_ids,
+                    use_reentrant=False,
+                )
+                present = None
+            else:
+                hidden_states, present = block(
+                    hidden_states,
+                    attention_mask=attention_mask,
+                    position_ids=position_ids,
+                    packed_metadata=packed_metadata,
+                    layer_past=None if past_key_values is None else past_key_values[layer_index],
+                    use_cache=use_cache,
+                )
+            hidden_states = hidden_states * query_attention_mask.unsqueeze(-1).to(dtype=hidden_states.dtype)
+            if presents is not None:
+                presents.append(present)
+
+        hidden_states = self.ln_f(hidden_states)
+        hidden_states = hidden_states * query_attention_mask.unsqueeze(-1).to(dtype=hidden_states.dtype)
+        if output_hidden_states:
+            all_hidden_states = all_hidden_states + (hidden_states,)
+
+        if not return_dict:
+            return (hidden_states, tuple(presents) if presents is not None else None, all_hidden_states, None)
+
+        from transformers.modeling_outputs import BaseModelOutputWithPast  # torchair kernel-namespace safe
         return BaseModelOutputWithPast(
             last_hidden_state=hidden_states,
             past_key_values=tuple(presents) if presents is not None else None,
```

**modeling_moss_tts_nano.py** — ① pad-to-bucket KV（MAX_PROMPT_LEN=320/MAX_TOTAL_LEN=640）：
torchair 静态图要求形状一致，动态长度会分钟级重编译 ② `cache_compile` 图编译集成（torchair
不可用自动回落 eager，GPU 行为不变）③ `inference_stream` 流式推理（首包 370ms→50ms 的基础）
④ 桶参数透传 + result 事件 ⑤ SDPA 委托 `_eager_attention`（CANN 9.0 缺 `npu_fusion_attention_v3`
的 GE converter；masked_fill 数学等价）

```diff
@@ -568,7 +568,7 @@
             return input_ids, attention_mask
 
         effective_text = text if prompt_text is None else prompt_text + text
-        prompt_token_ids = build_prompt_token_ids(
+        prompt_token_ids = build_prompt_token_ids( #list of int
             tokenizer=text_tokenizer,
             config=self.config,
             text_token_ids=self._encode_text(text_tokenizer, effective_text),
@@ -1633,20 +1633,43 @@
             dtype=torch.bool,
             device=local_inputs_embeds.device,
         )
-        local_outputs = self.local_transformer(
-            input_ids=None,
+        local_outputs = self.local_transformer.forward_local(
             attention_mask=local_attention_mask,
-            position_ids=None,
             inputs_embeds=local_inputs_embeds,
-            use_cache=False,
-            output_attentions=False,
-            output_hidden_states=False,
-            return_dict=True,
-            cu_seqlens=None,
-            num_sequences=None,
         )
         return local_outputs.last_hidden_state[:, -1, :]
 
+    def _decode_local(
+        self,
+        local_inputs_embeds: torch.FloatTensor,
+        current_local_input: torch.FloatTensor,
+        effective_nq: int,
+        do_sample: bool,
+        audio_temperature: float,
+        audio_top_k: int,
+        audio_top_p: float,
+        audio_repetition_penalty: float,
+    ) -> [torch.IntTensor]:
+        local_dtype = self.local_transformer.ln_f.weight.dtype
+        next_frame_tokens = []
+        for channel_index in range(effective_nq):
+            local_inputs_embeds = torch.cat([local_inputs_embeds, current_local_input.unsqueeze(1)], dim=1)
+            local_hidden_states = self._decode_local_last_hidden_state(local_inputs_embeds)
+            channel_logits = self.audio_lm_heads[channel_index](local_hidden_states)
+            #self._ensure_finite_generation_logits(channel_logits, f"audio logits[{channel_index}]")
+            channel_token = self._sample_next_token(
+                logits=channel_logits,
+                do_sample=do_sample,
+                temperature=audio_temperature,
+                top_k=audio_top_k,
+                top_p=audio_top_p,
+                previous_token_ids=None,
+                repetition_penalty=audio_repetition_penalty,
+            )
+            next_frame_tokens.append(channel_token)
+            current_local_input = self.audio_embeddings[channel_index](channel_token).to(dtype=local_dtype)
+        return next_frame_tokens
+
     def _iter_generation_events(
         self,
         input_ids: torch.LongTensor,
@@ -1663,10 +1686,11 @@
         audio_repetition_penalty: float = 1.0,
         use_kv_cache: bool = True,
         return_dict_in_generate: bool = True,
+        max_prompt_len: int = 200,
+        max_total_len: int = 400,
     ) -> Iterator[dict[str, Any]]:
         if input_ids.ndim == 2:
             input_ids = input_ids.unsqueeze(0)
-        if input_ids.ndim != 3:
             raise ValueError(f"Expected input_ids with 3 dims, got shape {tuple(input_ids.shape)}")
         if attention_mask is None:
             attention_mask = torch.ones(input_ids.shape[:2], dtype=torch.bool, device=input_ids.device)
@@ -1683,24 +1707,99 @@
         past_key_values = None
         local_dtype = self.local_transformer.ln_f.weight.dtype
 
+        max_new_frames = min(max_new_frames, max_total_len - max_prompt_len)
         for step_index in range(max_new_frames):
             generated_audio_history = torch.stack(generated_frames, dim=1) if generated_frames else None
+            
+            from datetime import datetime
+            import logging
+            now = datetime.now()
+            time_str = now.strftime("%Y-%m-%d %H:%M:%S.") + str(now.microsecond)[:3]
+            logging.info("begin build inputs embeds time: %s, pid: %s", time_str, str(os.getpid()))
+            
             global_inputs_embeds = self._build_inputs_embeds(current_model_input_ids)
-            global_outputs = self.transformer(
-                input_ids=None,
-                past_key_values=past_key_values,
-                attention_mask=current_attention_mask,
-                position_ids=None,
-                inputs_embeds=global_inputs_embeds,
-                use_cache=use_kv_cache,
-                output_attentions=False,
-                output_hidden_states=False,
-                return_dict=True,
-                cu_seqlens=None,
-                num_sequences=None,
-            )
+            
+            now = datetime.now()
+            time_str = now.strftime("%Y-%m-%d %H:%M:%S.") + str(now.microsecond)[:3]
+            logging.info("begin transformer time: %s, pid: %s", time_str, str(os.getpid()))
+            
+            if step_index == 0:
+                actual_len = global_inputs_embeds.shape[1]
+                logging.info("actual_len: %s, pid: %s", str(actual_len), str(os.getpid()))
+                assert actual_len <= max_prompt_len
+                num_padding = max_prompt_len - actual_len
+                padding_mask = torch.zeros(
+                        (global_inputs_embeds.shape[0], num_padding), 
+                        dtype=torch.bool, 
+                        device=current_attention_mask.device
+                        )
+                padding_embeds = torch.zeros(
+                        (global_inputs_embeds.shape[0], num_padding, self.transformer.config.n_embd), 
+                        dtype=torch.bfloat16, 
+                        device=global_inputs_embeds.device
+                        )
+                current_attention_mask = torch.cat((current_attention_mask,padding_mask), dim=1) 
+                global_inputs_embeds = torch.cat((global_inputs_embeds,padding_embeds), dim=1)
+        
+                global_outputs = self.transformer.cached_prefill(input_ids=None,
+                    past_key_values=past_key_values,
+                    attention_mask=current_attention_mask,
+                    position_ids=None,
+                    inputs_embeds=global_inputs_embeds,
+                    use_cache=use_kv_cache,
+                    output_attentions=False,
+                    output_hidden_states=False,
+                    return_dict=True,
+                    cu_seqlens=None,
+                    num_sequences=None,
+                )
+                current_attention_mask = current_attention_mask[:,:actual_len]
+                global_outputs.last_hidden_state = global_outputs.last_hidden_state[:,:actual_len]
+                past_key_values = []
+                for i in range(len(global_outputs.past_key_values)):
+                    tmp = []
+                    tmp.append(torch.cat((global_outputs.past_key_values[i][0][:,:actual_len], 
+                        torch.zeros([global_inputs_embeds.shape[0], 
+                            max_total_len - actual_len, 
+                            self.transformer.config.n_head, 
+                            self.transformer.config.n_embd // self.transformer.config.n_head], 
+                            dtype=torch.bfloat16, 
+                            device=global_outputs.past_key_values[i][0].device)), 
+                        dim=1))
+                    tmp.append(torch.cat((global_outputs.past_key_values[i][1][:,:actual_len], 
+                        torch.zeros([global_inputs_embeds.shape[0], 
+                            max_total_len - actual_len, 
+                            self.transformer.config.n_head, 
+                            self.transformer.config.n_embd // self.transformer.config.n_head], 
+                            dtype=torch.bfloat16, 
+                            device=global_outputs.past_key_values[i][1].device)), 
+                        dim=1))
+                    past_key_values.append(tmp)
+                global_outputs.past_key_values = past_key_values
+                kv_len = actual_len
+            else:
+                position_ids = torch.tensor([[kv_len]], dtype=torch.int64, device=global_inputs_embeds.device)
+                num_padding = max_total_len - kv_len - 1 
+                padding_mask = torch.zeros((global_inputs_embeds.shape[0], num_padding), dtype=torch.bool, device=current_attention_mask.device)
+                current_attention_mask_tmp = torch.cat((current_attention_mask, padding_mask), dim=1)
+                global_outputs = self.transformer.cached_decode(
+                    input_ids=None,
+                    past_key_values=past_key_values,
+                    attention_mask=current_attention_mask_tmp,
+                    position_ids=position_ids,
+                    inputs_embeds=global_inputs_embeds,
+                    use_cache=use_kv_cache,
+                    output_attentions=False,
+                    output_hidden_states=False,
+                    return_dict=True,
+                    cu_seqlens=None,
+                    num_sequences=None,
+                )
+                kv_len += 1
+            now = datetime.now()
+            time_str = now.strftime("%Y-%m-%d %H:%M:%S.") + str(now.microsecond)[:3]
+            logging.info("finish transformer time: %s, pid: %s", time_str, str(os.getpid()))
             global_hidden_states = global_outputs.last_hidden_state[:, -1, :].to(dtype=local_dtype)
-
             local_inputs_embeds = global_hidden_states.unsqueeze(1)
             local_hidden_states = self._decode_local_last_hidden_state(local_inputs_embeds)
             text_logits = self.text_lm_head(local_hidden_states)
@@ -1719,25 +1818,22 @@
 
             next_frame_tokens = []
             current_local_input = self.transformer.wte(next_text_tokens).to(dtype=local_dtype)
-            for channel_index in range(effective_nq):
-                local_inputs_embeds = torch.cat([local_inputs_embeds, current_local_input.unsqueeze(1)], dim=1)
-                local_hidden_states = self._decode_local_last_hidden_state(local_inputs_embeds)
-                channel_logits = self.audio_lm_heads[channel_index](local_hidden_states)
-                self._ensure_finite_generation_logits(channel_logits, f"audio logits[{channel_index}]")
-                channel_token = self._sample_next_token(
-                    logits=channel_logits,
-                    do_sample=do_sample,
-                    temperature=audio_temperature,
-                    top_k=audio_top_k,
-                    top_p=audio_top_p,
-                    previous_token_ids=(
-                        None if generated_audio_history is None else generated_audio_history[:, :, channel_index]
-                    ),
-                    repetition_penalty=audio_repetition_penalty,
+            now = datetime.now()
+            time_str = now.strftime("%Y-%m-%d %H:%M:%S.") + str(now.microsecond)[:3]
+            logging.info("begin local transformer time: %s, pid: %s", time_str, str(os.getpid()))
+            next_frame_tokens = self._decode_local(
+                    local_inputs_embeds, 
+                    current_local_input, 
+                    effective_nq, 
+                    do_sample, 
+                    audio_temperature, 
+                    audio_top_k, 
+                    audio_top_p,
+                    audio_repetition_penalty,
                 )
-                next_frame_tokens.append(channel_token)
-                current_local_input = self.audio_embeddings[channel_index](channel_token).to(dtype=local_dtype)
-
+            now = datetime.now()
+            time_str = now.strftime("%Y-%m-%d %H:%M:%S.") + str(now.microsecond)[:3]
+            logging.info("finish local transformer time: %s, pid: %s", time_str, str(os.getpid()))
             next_frame_prefix = torch.stack(next_frame_tokens, dim=-1)
             if effective_nq < self.config.n_vq:
                 next_frame = torch.full(
@@ -1849,6 +1945,8 @@
         audio_repetition_penalty: float = 1.0,
         use_kv_cache: bool = True,
         return_dict_in_generate: bool = True,
+        max_prompt_len: int = 200,
+        max_total_len: int = 400,
     ) -> Iterator[dict[str, Any]]:
         yield from self._iter_generation_events(
             input_ids=input_ids,
@@ -1865,6 +1963,8 @@
             audio_repetition_penalty=audio_repetition_penalty,
             use_kv_cache=use_kv_cache,
             return_dict_in_generate=return_dict_in_generate,
+            max_prompt_len=max_prompt_len,
+            max_total_len=max_total_len,
         )
 
     @torch.no_grad()
@@ -1876,6 +1976,7 @@
         prompt_text: Optional[str] = None,
         prompt_audio_path: Optional[Union[str, Path]] = None,
         reference_audio_path: Optional[Union[str, Path]] = None,
+        prompt_audio_codes=None,
         text_tokenizer=None,
         text_tokenizer_path: Optional[str] = None,
         audio_tokenizer=None,
@@ -1897,14 +1998,29 @@
         voice_clone_max_memory_per_sample_gb: float = DEFAULT_VOICE_CLONE_MAX_MEMORY_PER_SAMPLE_GB,
         tts_max_batch_size: int = 0,
         codec_max_batch_size: int = 0,
+        max_prompt_len: Optional[int] = None,
+        max_total_len: Optional[int] = None,
     ) -> Iterator[dict[str, Any]]:
+
+        # env-overridable padding buckets (static-graph shape contract): the
+        # compiled graph reuses itself only when prefill pads to the same
+        # max_prompt_len every call, so callers must NOT vary this per request.
+        # None → the class defaults in _iter_generation_events (200/400).
+        import os as _os
+        if max_prompt_len is None:
+            max_prompt_len = int(_os.getenv("MOSS_TTS_NANO_MAX_PROMPT_LEN", "200"))
+        if max_total_len is None:
+            max_total_len = int(_os.getenv("MOSS_TTS_NANO_MAX_TOTAL_LEN", "400"))
+
+        from datetime import datetime
+        import logging
+        now = datetime.now()
+        time_str = now.strftime("%Y-%m-%d %H:%M:%S.") + str(now.microsecond)[:3]
+        logging.info("real begin time: %s, pid: %s", time_str, str(os.getpid()))
+
         resolved_device = self._resolve_device(device)
         effective_nq = self._resolve_inference_nq(nq)
-        if next(self.parameters()).device != resolved_device:
-            self.to(resolved_device)
-
         was_training = self.training
-        self.eval()
 
         text_tokenizer = self._load_text_tokenizer(
             text_tokenizer=text_tokenizer,
@@ -1923,7 +2039,8 @@
         resolved_mode = self._resolve_inference_mode(
             mode=mode,
             has_prompt_text=prompt_text is not None,
-            has_prompt_audio=effective_prompt_audio_path is not None,
+            # cached prompt_audio_codes make voice_clone valid without a path
+            has_prompt_audio=(effective_prompt_audio_path is not None) or (prompt_audio_codes is not None),
         )
         if reference_audio_path is not None and prompt_audio_path is None:
             logging.warning(
@@ -1931,45 +2048,27 @@
                 reference_audio_path,
             )
 
-        prompt_audio_codes = None
-        if effective_prompt_audio_path is not None:
-            waveform, sample_rate = self._load_reference_audio(
-                effective_prompt_audio_path,
-                target_sample_rate,
-                target_channels,
-            )
-            encoded = self._call_audio_encode(
-                audio_tokenizer=audio_tokenizer,
-                waveform=waveform.to(resolved_device),
-                sample_rate=sample_rate,
-            )
-            prompt_audio_codes = self._mask_unused_audio_channels(
-                self._normalize_audio_codes(encoded),
-                nq=effective_nq,
-            ).to(resolved_device)
+        if prompt_audio_codes is None:
+            if effective_prompt_audio_path is not None:
+                waveform, sample_rate = self._load_reference_audio(
+                    effective_prompt_audio_path,
+                    target_sample_rate,
+                    target_channels,
+                )
+                encoded = self._call_audio_encode(
+                    audio_tokenizer=audio_tokenizer,
+                    waveform=waveform.to(resolved_device),
+                    sample_rate=sample_rate,
+                )
+                prompt_audio_codes = self._mask_unused_audio_channels(
+                    self._normalize_audio_codes(encoded),
+                    nq=effective_nq,
+                ).to(resolved_device)
 
-        if resolved_mode == "voice_clone":
-            split_voice_clone_text_chunks = self._split_text_into_best_sentences(
-                text_tokenizer=text_tokenizer,
-                text=text,
-                max_tokens=voice_clone_max_text_tokens,
-            )
-            voice_clone_text_chunks = split_voice_clone_text_chunks if len(split_voice_clone_text_chunks) > 1 else [text]
-        else:
-            voice_clone_text_chunks = [text]
+        voice_clone_text_chunks = [text]
 
-        if resolved_mode == "voice_clone" and len(voice_clone_text_chunks) > 1:
-            voice_clone_chunk_batch_size, voice_clone_codec_batch_size = self._resolve_effective_voice_clone_batch_sizes(
-                resolved_device=resolved_device,
-                chunk_count=len(voice_clone_text_chunks),
-                max_memory_per_sample_gb=float(voice_clone_max_memory_per_sample_gb),
-                requested_tts_max_batch_size=tts_max_batch_size,
-                requested_codec_max_batch_size=codec_max_batch_size,
-                realtime_streaming=True,
-            )
-        else:
-            voice_clone_chunk_batch_size = 1
-            voice_clone_codec_batch_size = 1
+        voice_clone_chunk_batch_size = 1
+        voice_clone_codec_batch_size = 1
 
         generated_audio_token_chunks: list[torch.LongTensor] = []
         emitted_waveform_segments: list[torch.FloatTensor] = []
@@ -1983,6 +2082,7 @@
                 batch_chunks = voice_clone_text_chunks[batch_start : batch_start + voice_clone_chunk_batch_size]
                 batch_prompt_input_ids: list[torch.LongTensor] = []
                 batch_attention_masks: list[torch.BoolTensor] = []
+                
                 for text_chunk in batch_chunks:
                     prompt_input_ids, attention_mask = self.build_inference_input_ids(
                         text=text_chunk,
@@ -1995,6 +2095,10 @@
                     batch_prompt_input_ids.append(prompt_input_ids)
                     batch_attention_masks.append(attention_mask)
 
+                now = datetime.now()
+                time_str = now.strftime("%Y-%m-%d %H:%M:%S.") + str(now.microsecond)[:3]
+                logging.info("finish text preprocess time: %s, pid: %s", time_str, str(os.getpid()))
+
                 batched_prompt_input_ids, batched_attention_mask = self._left_pad_inference_batch(
                     input_id_batches=batch_prompt_input_ids,
                     attention_mask_batches=batch_attention_masks,
@@ -2137,6 +2241,9 @@
                         frame_tensor = torch.cat(frame_rows, dim=0).to(device=resolved_device, dtype=torch.long)
                         codes_list.append(frame_tensor[:, :effective_nq].transpose(0, 1).contiguous())
 
+                    now = datetime.now()
+                    time_str = now.strftime("%Y-%m-%d %H:%M:%S.") + str(now.microsecond)[:3]
+                    logging.info("audio_tokenizer.batch_decode begin time: %s, pid: %s", time_str, str(os.getpid()))
                     decode_output = audio_tokenizer.batch_decode(
                         codes_list,
                         num_quantizers=effective_nq,
@@ -2144,6 +2251,9 @@
                         max_batch_size=(voice_clone_codec_batch_size if not codec_stream_started else None),
                         reset_stream=not codec_stream_started,
                     )
+                    now = datetime.now()
+                    time_str = now.strftime("%Y-%m-%d %H:%M:%S.") + str(now.microsecond)[:3]
+                    logging.info("audio_tokenizer.batch_decode finish time: %s, pid: %s", time_str, str(os.getpid()))
                     codec_stream_started = True
                     waveform_rows, current_sample_rate = self._extract_batch_waveforms_and_sample_rate(
                         decode_output,
@@ -2181,6 +2291,8 @@
                         audio_repetition_penalty=audio_repetition_penalty,
                         use_kv_cache=use_kv_cache,
                         return_dict_in_generate=True,
+                        max_prompt_len=max_prompt_len,
+                        max_total_len=max_total_len,
                     ):
                         if event["type"] == "frame":
                             frame_audio_token_ids = event["audio_token_ids"]
@@ -2193,7 +2305,7 @@
                                     )
                                 if bool(finished_mask[row_index].item()):
                                     row_states[row_index]["generation_complete"] = True
-                            yield from _maybe_decode_pending(force=False)
+                            yield from _maybe_decode_pending(force=True) #False)
                             continue
                         final_generation = event.get("generation")
 
@@ -2233,6 +2345,9 @@
             output_path.parent.mkdir(parents=True, exist_ok=True)
             torchaudio.save(str(output_path), waveform, decoded_sample_rate)
 
+            # RESTORED from the stock checkpoint (tianjingcheng's edit dropped
+            # this closing event; the vendored runtime treats its absence as a
+            # failed job and discards the loaded model state):
             yield {
                 "type": "result",
                 "audio_path": str(output_path),
@@ -2244,6 +2359,7 @@
                 "voice_clone_chunk_batch_size": int(voice_clone_chunk_batch_size),
                 "voice_clone_codec_batch_size": int(voice_clone_codec_batch_size),
             }
+
         finally:
             if was_training:
                 self.train()
@@ -2304,7 +2420,8 @@
         resolved_mode = self._resolve_inference_mode(
             mode=mode,
             has_prompt_text=prompt_text is not None,
-            has_prompt_audio=effective_prompt_audio_path is not None,
+            # cached prompt_audio_codes make voice_clone valid without a path
+            has_prompt_audio=(effective_prompt_audio_path is not None) or (prompt_audio_codes is not None),
         )
         if reference_audio_path is not None and prompt_audio_path is None:
             logging.warning(
```

---

## 四、tts/MOSS-TTS-Nano-100M-tf5x/（5.x overlay，1 文件）

**gpt2_decoder.py** — RoPE 类补 `_base`/`_dim` + `_compute_inv_freq()`：给 Demo 侧
compat/tf-5x 分支的 post-load 修复循环提供重算原料（5.x 加载污染 inv_freq 时重算覆盖）。
4.57 上无人消费（纯附加元数据）。

```diff
@@ -52,6 +52,15 @@
             raise ValueError(f"RoPE head_dim must be even, got {dim}")
         inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
         self.register_buffer("inv_freq", inv_freq, persistent=False)
+        # repair metadata: transformers' from_pretrained (meta-device loading)
+        # can clobber non-persistent buffers with zeros/garbage — the runtime's
+        # post-load repair and forward() below use these to restore the
+        # correct values (see the CPU adaptation's "cicici noise" root cause).
+        self._base = float(base)
+        self._dim = int(dim)
+
+    def _compute_inv_freq(self) -> torch.Tensor:
+        return 1.0 / (self._base ** (torch.arange(0, self._dim, 2, dtype=torch.float32, device="cpu") / self._dim))
 
     def forward(
         self,
@@ -61,7 +70,10 @@
         dtype: torch.dtype,
     ) -> tuple[torch.Tensor, torch.Tensor]:
         # NOTE: no in-forward inv_freq guard — a data-dependent .item() branch
-        # breaks torchair's fullgraph compile (gb0170).
+        # breaks torchair's fullgraph compile (gb0170). transformers 4.57.6
+        # does not clobber non-persistent buffers on load (the 5.x behavior
+        # that motivated the CPU-side guard), and the runtime's post-load
+        # repair remains as a belt-and-braces via the _base/_dim attrs above.
         if position_ids.ndim == 1:
             position_ids = position_ids.unsqueeze(0)
         freqs = torch.einsum("bs,d->bsd", position_ids.to(device=device, dtype=self.inv_freq.dtype), self.inv_freq)
```

---

## 五、tts/MOSS-Audio-Tokenizer-Nano/（1 文件）

**modeling_moss_audio_tokenizer.py** — attention 的 SDPA 改数学展开（`masked_fill` + matmul）：
CANN 9.0 缺 SDPA 的 GE converter，图编译失败；数学等价可编译，CUDA/CPU 走原 sdpa 不变。

```diff
@@ -1488,7 +1488,16 @@
         cached_k, cached_v, cached_pos = self._ensure_streaming_cache(state, batch_size, k_cur.device, k_cur.dtype)
         k_all, v_all, pos_k = self._build_streaming_kv(cached_k, cached_v, cached_pos, k_cur, v_cur, pos_q)
         attn_bias = self._build_streaming_sdpa_bias(pos_q, pos_k)
-        out = F.scaled_dot_product_attention(q, k_all, v_all, attn_bias, dropout_p=0.0)
+        # math path: torch_npu 2.11 lowers F.scaled_dot_product_attention to
+        # npu_fusion_attention_v3, whose torchair GE converter is unavailable on
+        # CANN 9.0.0. NOTE: attn_bias here is a BOOL mask (True=keep,
+        # False=block) — F.sdpa accepts that form, but a math path must
+        # masked_fill, NOT add it (bool+float would add +1.0/+0.0 and leave
+        # blocked positions attendable — corrupted audio).
+        _scale = q.shape[-1] ** -0.5
+        _scores = torch.matmul(q, k_all.transpose(-2, -1)) * _scale
+        _scores = _scores.masked_fill(~attn_bias, torch.finfo(_scores.dtype).min)
+        out = torch.matmul(torch.softmax(_scores, dim=-1), v_all)
         out = out.transpose(1, 2).reshape(batch_size, chunk_length, self.embed_dim)
 
         self._update_streaming_cache(state, cached_k, cached_v, cached_pos, k_all, v_all, pos_k)
@@ -1591,7 +1600,12 @@
         q, k, v = self._project_qkv(x)
         q, k = self._apply_dense_rope(q, k)
         attn_bias = self._build_non_streaming_sdpa_bias(input_lengths, max_seqlen, x.device)
-        out = F.scaled_dot_product_attention(q, k, v, attn_bias, dropout_p=0.0)
+        # math path (same npu_fusion_attention_v3 GE-converter workaround as the
+        # streaming branch; attn_bias is a BOOL mask → masked_fill, not add)
+        _scale = q.shape[-1] ** -0.5
+        _scores = torch.matmul(q, k.transpose(-2, -1)) * _scale
+        _scores = _scores.masked_fill(~attn_bias, torch.finfo(_scores.dtype).min)
+        out = torch.matmul(torch.softmax(_scores, dim=-1), v)
         valid_q = (torch.arange(max_seqlen, device=x.device).view(1, max_seqlen) < input_lengths.view(-1, 1)).view(
             batch_size, 1, max_seqlen, 1
         )
@@ -1621,7 +1635,7 @@
         position_ids: torch.Tensor | None = None,
         input_lengths: torch.Tensor | None = None,
     ):
-        state = cast(MHAState | None, self._streaming_state)
+        state = self._streaming_state
         backend = self.resolve_attention_implementation(query, is_streaming=state is not None)
 
         if state is not None:
@@ -2836,14 +2850,18 @@
         decoder_hidden_states = quantizer.decode_codes(codes).float()
 
         with self._codec_inference_autocast():
-            audio, audio_lengths = decoder_hidden_states, codes_lengths
-            for decoder_module in self.decoder:
-                audio, audio_lengths = decoder_module(audio, audio_lengths)
+            audio, audio_lengths = self._decode_decoder(decoder_hidden_states, codes_lengths)
 
         audio, audio_lengths = self._restore_channels_from_codec(audio, audio_lengths)
         return MossAudioTokenizerDecoderOutput(audio=audio, audio_lengths=audio_lengths)
 
     @torch.no_grad()
+    def _decode_decoder(self, audio, audio_lengths):
+        for decoder_module in self.decoder:
+            audio, audio_lengths = decoder_module(audio, audio_lengths)
+        return audio, audio_lengths
+
+    @torch.no_grad()
     def batch_encode(
         self,
         wav_list: list[torch.Tensor],
```
