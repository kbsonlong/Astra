#!/bin/bash
# Simple download script for Zipformer Bilingual tarball with mirror fallback
set -e
cd "$(dirname "$0")/models"

F="sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20.tar.bz2"
URL1="https://ghproxy.net/https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/${F}"
URL2="https://mirror.ghproxy.com/https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/${F}"
URL3="https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/${F}"

echo "=== Download Zipformer Bilingual (487MB bz2 compressed) ==="
echo "Current size: $(ls -lh "$F" 2>/dev/null | awk '{print $5}')"

for i in 1 2 3; do
  case $i in
    1) URL="$URL1"; echo "--- Try mirror ghproxy.net ---" ;;
    2) URL="$URL2"; echo "--- Try mirror mirror.ghproxy.com ---" ;;
    3) URL="$URL3"; echo "--- Try original GitHub ---" ;;
  esac
  if curl -L --retry 5 --continue-at - --connect-timeout 20 \
            --max-time 3600 -o "$F" "$URL"; then
    echo "Download OK, final size=$(ls -lh "$F" | awk '{print $5}')"
    exit 0
  fi
  echo "Mirror $i failed, try next..."
done

echo "All mirrors failed"
exit 1
