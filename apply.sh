#!/usr/bin/env bash
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

apply_vlm() {
    local dst="$1"
    [ -d "$dst" ] || { echo "VLM dir not found: $dst"; exit 1; }
    echo "Applying VLM adaptations → $dst"
    cp "$DIR"/vlm/MOSS-VL-Realtime-0708/*.py "$dst/"
    python3 -c "
import json
p = '$dst/preprocessor_config.json'
with open(p) as f: c = json.load(f)
if 'interpolation' not in c:
    c['interpolation'] = 'BICUBIC'
    with open(p, 'w') as f: json.dump(c, f, indent=4)
"
    rm -rf ~/.cache/huggingface/modules/transformers_modules/MOSS_hyphen_VL_hyphen_Realtime_hyphen_0708 2>/dev/null || true
    echo "  ✓ VLM done"
}

apply_tts() {
    local dst="$1"
    [ -d "$dst/MOSS-TTS-Nano-100M" ] || { echo "TTS dir not found"; exit 1; }
    echo "Applying TTS adaptations → $dst"
    cp "$DIR"/tts/MOSS-TTS-Nano-100M/*.py "$dst/MOSS-TTS-Nano-100M/"
    cp "$DIR"/tts/MOSS-Audio-Tokenizer-Nano/*.py "$dst/MOSS-Audio-Tokenizer-Nano/"
    rm -rf ~/.cache/huggingface/modules/transformers_modules/MOSS_hyphen_TTS_hyphen_Nano 2>/dev/null || true
    rm -rf ~/.cache/huggingface/modules/transformers_modules/MOSS_hyphen_Audio_hyphen_Tokenizer_hyphen_Nano 2>/dev/null || true
    echo "  ✓ TTS done"
}

case "${1:-}" in
    --vlm)  apply_vlm "$2" ;;
    --tts)  apply_tts "$2" ;;
    --all)  apply_vlm "$2"; apply_tts "$3" ;;
    *)      echo "Usage: $0 --vlm <dir> | --tts <dir> | --all <vlm> <tts>"; exit 2 ;;
esac
