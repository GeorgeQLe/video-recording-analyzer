#!/usr/bin/env bash
# Source this to put the locally-installed tools on PATH (no sudo install needed).
# Usage: source env.sh
PROJ="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
export RECORDINGS_TOOLING="$PROJ"

# ffmpeg / ffprobe / tesseract wrappers
export PATH="$PROJ/bin:$PATH"

# tesseract runtime
TESS_LIB="$PROJ/opt/tesseract/usr/lib/x86_64-linux-gnu"
export TESSDATA_PREFIX="$PROJ/opt/tesseract/usr/share/tesseract-ocr/4.00/tessdata"

# faster-whisper GPU runtime libs (cuBLAS + cuDNN wheels installed in the venv)
CUDA_LIBS=""
if [ -d "$PROJ/venv" ]; then
  for d in "$PROJ"/venv/lib/python*/site-packages/nvidia/*/lib; do
    [ -d "$d" ] && CUDA_LIBS="$CUDA_LIBS:$d"
  done
fi

export LD_LIBRARY_PATH="$TESS_LIB${CUDA_LIBS}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
