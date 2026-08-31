# MOSS-VL-Realtime-0708 — standalone GIL busy-wait fix

**用途**：独立 HF PR 制品（`OpenMOSS-Team/MOSS-VL-Realtime-0708`）。
本目录的 `modeling_moss_vl.py` = **HF stock 原版 + 仅一处修复**（realtime 空闲忙等 → 20ms sleep 轮询），不含任何 NPU 适配内容。

## 为什么独立成 PR

- **平台中立的 stock bug**：忙等独占的是 CPython GIL，与加速器无关——**上游 GPU 部署同样受影响**（同进程任意 Python 线程被饿死）
- **diff 极小**（~6 行）、无任何依赖，reviewer 无需 Ascend 背景
- **证据可量化**：单忙转线程 × 同进程 SenseVoice 解码 = 584ms → 42.6s（73×）；修复后 132-152ms
- **提交顺序**：先提本 PR（快合、建立信用），NPU 适配 PR（`../MOSS-VL-Realtime-0708/`）随后；本 PR 合入后 NPU PR 的 modeling diff 自动缩小为 empty_cache hunk

## 提交

```bash
export HF_TOKEN=hf_xxx
bash ../../Adapt/submit_hf_pr.sh vlm-gil-fix
```

PR 描述素材（实测数据 + 复现方法）见 `../../Adapt/buglist.md` BL-004（内部文档，**提交时按其内容改写，不得引用内部编号**）。
