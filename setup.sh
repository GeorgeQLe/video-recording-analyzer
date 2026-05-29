#!/usr/bin/env bash
# One-time, no-sudo setup for recordings-tooling.
# Installs into this folder: static ffmpeg, Tesseract (extracted from .deb), Python venv.
set -euo pipefail
PROJ="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
cd "$PROJ"
mkdir -p bin vendor opt/tesseract

UBU_MAIN="http://archive.ubuntu.com/ubuntu/pool/main"
UBU_UNI="http://archive.ubuntu.com/ubuntu/pool/universe"

dl() {  # dl <url> <outfile>
  [ -f "vendor/$2" ] || curl -fsSL -o "vendor/$2" "$1"
}

OS="$(uname -s)"; ARCH="$(uname -m)"
if [ "$OS" != "Linux" ] || [ "$ARCH" != "x86_64" ]; then
  echo "NOTE: the no-sudo ffmpeg/tesseract downloads target Linux/x86_64 ($OS/$ARCH detected)."
  echo "      On other platforms, install ffmpeg + tesseract via your package manager;"
  echo "      this script will reuse whatever is already on PATH."
fi

echo "==> ffmpeg"
if command -v ffmpeg >/dev/null 2>&1 && command -v ffprobe >/dev/null 2>&1; then
  echo "  using system ffmpeg: $(command -v ffmpeg)"
elif [ ! -x bin/ffmpeg ] && [ "$OS" = "Linux" ] && [ "$ARCH" = "x86_64" ]; then
  dl "https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz" ffmpeg-static.tar.xz
  tmp="$(mktemp -d)"; tar -xf vendor/ffmpeg-static.tar.xz -C "$tmp"
  ffdir="$(find "$tmp" -maxdepth 1 -type d -name 'ffmpeg-*-static' | head -1)"
  cp "$ffdir/ffmpeg" "$ffdir/ffprobe" bin/; rm -rf "$tmp"
fi
{ [ -x bin/ffmpeg ] && bin/ffmpeg -version || ffmpeg -version; } | head -1

echo "==> tesseract"
# bin/tesseract is a committed wrapper; if a system tesseract exists, prefer it and
# neutralize the wrapper so transcribe.py's PATH fallback picks the system binary.
if command -v tesseract >/dev/null 2>&1 && [ ! -d opt/tesseract/usr/bin ]; then
  echo "  using system tesseract: $(command -v tesseract)"
  rm -f bin/tesseract
elif [ "$OS" = "Linux" ] && [ "$ARCH" = "x86_64" ]; then
  # no-sudo install: core + english data + runtime libs + transitive image libs
  dl "$UBU_UNI/t/tesseract/tesseract-ocr_4.1.1-2.1build1_amd64.deb"                tesseract-ocr.deb
  dl "$UBU_UNI/t/tesseract-lang/tesseract-ocr-eng_4.00%7egit30-7274cfa-1.1_all.deb" tesseract-ocr-eng.deb
  dl "$UBU_UNI/t/tesseract/libtesseract4_4.1.1-2.1build1_amd64.deb"                libtesseract4.deb
  dl "$UBU_UNI/l/leptonlib/liblept5_1.82.0-3build1_amd64.deb"                      liblept5.deb
  dl "$UBU_MAIN/g/giflib/libgif7_5.1.9-2ubuntu0.1_amd64.deb"                       libgif7.deb
  dl "$UBU_MAIN/o/openjpeg2/libopenjp2-7_2.4.0-6ubuntu0.5_amd64.deb"               libopenjp2-7.deb
  dl "$UBU_MAIN/libw/libwebp/libwebpmux3_1.2.2-2ubuntu0.22.04.2_amd64.deb"         libwebpmux3.deb
  dl "$UBU_MAIN/libw/libwebp/libwebp7_1.2.2-2ubuntu0.22.04.2_amd64.deb"            libwebp7.deb
  for d in tesseract-ocr tesseract-ocr-eng libtesseract4 liblept5 libgif7 libopenjp2-7 libwebpmux3 libwebp7; do
    dpkg -x "vendor/$d.deb" opt/tesseract
  done
  chmod +x bin/tesseract 2>/dev/null || true
  if missing="$(ldd opt/tesseract/usr/bin/tesseract 2>/dev/null | grep 'not found' || true)"; [ -n "$missing" ]; then
    echo "WARNING: tesseract still missing libs:"; echo "$missing"
  fi
else
  echo "  ERROR: no system tesseract and no no-sudo build for $OS/$ARCH — install tesseract-ocr." >&2
fi
{ [ -x bin/tesseract ] && bin/tesseract --version || tesseract --version; } 2>&1 | head -1

echo "==> python venv + faster-whisper (GPU)"
if [ ! -x venv/bin/python ]; then
  python3 -m pip install --user -q virtualenv
  python3 -m virtualenv -q venv
fi
venv/bin/pip install -q faster-whisper Pillow
# GPU runtime libs (CUDA 12 / cuDNN 9) — best-effort; CPU-only machines just skip these.
venv/bin/pip install -q nvidia-cublas-cu12 nvidia-cudnn-cu12 \
  || echo "  (no NVIDIA CUDA wheels installed — will run on CPU)"

echo "==> GPU self-test"
# shellcheck disable=SC1091
source "$PROJ/env.sh"
venv/bin/python - <<'PY'
import ctranslate2, faster_whisper
n = ctranslate2.get_cuda_device_count()
print(f"faster-whisper {faster_whisper.__version__} | ctranslate2 {ctranslate2.__version__} | CUDA devices: {n}")
print("GPU available ✓" if n > 0 else "No CUDA device — pipeline will fall back to CPU")
PY

echo
echo "Setup complete. Try:  ./transcribe list"
