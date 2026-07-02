#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
PIP_CMD=("$PYTHON_BIN" -m pip)
HF_MIRROR="${HF_MIRROR:-https://hf-mirror.com}"
DATA_OUT_DIR="${DATA_OUT_DIR:-data/processed}"
FORCE_CUDA11_TORCH="${FORCE_CUDA11_TORCH:-auto}"
# 清华源
PIP_MIRROR="${PIP_MIRROR:-https://pypi.tuna.tsinghua.edu.cn/simple}"

log() {
  printf '\n[%s] %s\n' "$(date '+%F %T')" "$*"
}

detect_cuda_version() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "none"
    return
  fi

  local raw
  raw="$(nvidia-smi | sed -n 's/.*CUDA Version: \([0-9.]*\).*/\1/p' | head -n 1)"
  if [[ -z "$raw" ]]; then
    echo "unknown"
  else
    echo "$raw"
  fi
}

should_use_cu113() {
  case "$FORCE_CUDA11_TORCH" in
    1|true|TRUE|yes|YES)
      return 0
      ;;
    0|false|FALSE|no|NO)
      return 1
      ;;
  esac

  local cuda_version
  cuda_version="$(detect_cuda_version)"
  if [[ "$cuda_version" == "none" || "$cuda_version" == "unknown" ]]; then
    return 1
  fi

  local major="${cuda_version%%.*}"
  local minor="${cuda_version#*.}"
  minor="${minor%%.*}"

  if (( major < 11 )); then
    return 0
  fi
  if (( major == 11 && minor <= 4 )); then
    return 0
  fi
  return 1
}

log "Upgrading pip"
"${PIP_CMD[@]}" install -U pip -i "$PIP_MIRROR"

log "Uninstalling existing torch packages"
"${PIP_CMD[@]}" uninstall -y torch torchvision torchaudio || true

if should_use_cu113; then
  log "Installing CUDA 11.3 Torch stack for older GPU drivers"
  # 清华源不支持 cu113 的 extra-index，所以使用官方源并设置超时
  "${PIP_CMD[@]}" install \
    torch==1.12.1+cu113 \
    torchvision==0.13.1+cu113 \
    torchaudio==0.12.1 \
    --extra-index-url https://download.pytorch.org/whl/cu113 \
    --timeout 100
  # numpy 从清华源安装
  "${PIP_CMD[@]}" install "numpy<2" -i "$PIP_MIRROR"
else
  log "Installing project dependencies from requirements.txt"
  "${PIP_CMD[@]}" install --force-reinstall -r requirements.txt -i "$PIP_MIRROR"
fi

log "Installing project dependencies again to ensure editable package and pinned libs are present"
"${PIP_CMD[@]}" install --force-reinstall -r requirements.txt -i "$PIP_MIRROR"

log "Running unit tests"
"$PYTHON_BIN" -m pytest -q

log "Preparing processed datasets via Hugging Face mirror"
HF_ENDPOINT="$HF_MIRROR" "$PYTHON_BIN" scripts/prepare_data.py \
  --out-dir "$DATA_OUT_DIR" \
  --force-download

log "Environment setup complete"
log "Repository: $ROOT_DIR"
log "Processed data: $DATA_OUT_DIR"