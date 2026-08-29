# MOSS NPU Model Adaptations

Drop-in replacements for the `trust_remote_code` code files inside MOSS
model checkpoints, adding **Ascend NPU (910B, CANN 9.0.0) support** while
keeping **CUDA/GPU behavior unchanged** (all NPU paths are guarded by
`torch.npu.is_available()` or optional imports). Weights are not modified.

## What this adds

### VLM — MOSS-VL-Realtime-0708 (`vlm/`)
| Capability | Detail |
|---|---|
| NPU inference | Works on Ascend 910B with eager attention (flash_attn unavailable on NPU) |
| transformers compatibility | Runs on both 4.57 and 5.x (`create_causal_mask` dual-signature) |
| 10D vision processing | Device-aware: CPU round-trip on NPU (8-dim limit), native on CUDA |
| GPU unchanged | `_supports_flash_attn=True` preserved; flash-attn 2 path untouched |

### TTS — MOSS-TTS-Nano-100M (`tts/`)
| Capability | Detail |
|---|---|
| Graph-compiled inference | torchair static graph on prefill/decode (7.4× first-packet speedup) |
| Pad-to-bucket KV | Fixed shapes (configurable 320/640) → static-graph reuse across requests |
| Streaming output | `inference_stream` with incremental audio + per-voice prompt cache |
| Runaway generation guard | Length-aware frame cap (4 frames/char + 32) against EOS-miss loops |
| GPU fallback | CUDA/CPU machines automatically run eager (guarded imports) |

### Audio Tokenizer — MOSS-Audio-Tokenizer-Nano (`tts/`)
| Capability | Detail |
|---|---|
| CANN 9.0 attention | SDPA→math path (equivalent semantics; `npu_fusion_attention_v3` converter unavailable) |

## Performance (910B2C, measured)

| Metric | Stock (CPU) | Adapted (NPU) | Speedup |
|---|---|---|---|
| TTS first packet | 370ms | **~50ms** | 7.4× |
| TTS throughput (RTF) | 0.73 | **0.35** | 2.1× |
| ASR decode (5.6s audio) | 5–10s | **0.1s** | 50–100× |

## Usage

```bash
./apply.sh --vlm /path/to/MOSS-VL-Realtime-0708
./apply.sh --tts /path/to/models/tts
```

## Compatibility matrix

| Platform | Status | Notes |
|---|---|---|
| Ascend 910B (CANN 9.0.0) | ✅ | All features (torchair, bucket KV) |
| CUDA GPU | ✅ | Unchanged from stock (guards fall through to eager/native) |
| CPU only | ✅ | Unchanged from stock |
