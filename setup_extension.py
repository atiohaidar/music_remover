"""
One-click setup script for the Chrome Extension (chrome_extension/).

Downloads:
  1. onnxruntime-web@1.31.1 JS + WASM runtime files into chrome_extension/lib/
  2. ONNX models (gtcrn_simple.onnx, Kim_Vocal_2.onnx, UVR-MDX-NET-Voc_FT.onnx)
     into chrome_extension/models/ (reusing local models/ if already downloaded).

Usage:
    python setup_extension.py
"""

import os
import shutil
import urllib.request

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
EXT_DIR = os.path.join(ROOT_DIR, "chrome_extension")
LIB_DIR = os.path.join(EXT_DIR, "lib")
EXT_MODELS_DIR = os.path.join(EXT_DIR, "models")
PY_MODELS_DIR = os.path.join(ROOT_DIR, "models")

ORT_VERSION = "1.31.1"
ORT_BASE_URL = f"https://cdn.jsdelivr.net/npm/onnxruntime-web@{ORT_VERSION}/dist"
ORT_FILES = [
    "ort.all.min.js",
    "ort-wasm-simd-threaded.jsep.mjs",
    "ort-wasm-simd-threaded.jsep.wasm",
]

MODELS = {
    "gtcrn_simple.onnx": "https://github.com/k2-fsa/sherpa-onnx/releases/download/speech-enhancement-models/gtcrn_simple.onnx",
    "Kim_Vocal_2.onnx": "https://huggingface.co/seanghay/uvr_models/resolve/main/Kim_Vocal_2.onnx",
    "UVR-MDX-NET-Voc_FT.onnx": "https://huggingface.co/aaatmy/uvr-mdx-net-onnx/resolve/main/UVR-MDX-NET-Voc_FT.onnx",
}


def download_if_missing(url: str, dest: str, min_bytes: int = 1024):
    if os.path.exists(dest) and os.path.getsize(dest) >= min_bytes:
        size_mb = os.path.getsize(dest) / (1024 * 1024)
        print(f"  [SKIP] {os.path.basename(dest)} ({size_mb:.2f} MB)")
        return
    print(f"  [GET]  {os.path.basename(dest)} <- {url}")
    urllib.request.urlretrieve(url, dest)
    size_mb = os.path.getsize(dest) / (1024 * 1024)
    print(f"  [OK]   Saved ({size_mb:.2f} MB)")


def main():
    os.makedirs(LIB_DIR, exist_ok=True)
    os.makedirs(EXT_MODELS_DIR, exist_ok=True)

    print("=== 1/2 Setting up onnxruntime-web (lib/) ===")
    for fname in ORT_FILES:
        download_if_missing(f"{ORT_BASE_URL}/{fname}", os.path.join(LIB_DIR, fname))

    print("\n=== 2/2 Setting up ONNX models (models/) ===")
    for fname, url in MODELS.items():
        dest = os.path.join(EXT_MODELS_DIR, fname)
        local_py = os.path.join(PY_MODELS_DIR, fname)
        if not os.path.exists(dest) and os.path.exists(local_py):
            print(f"  [COPY] {fname} from models/")
            shutil.copy2(local_py, dest)
        else:
            download_if_missing(url, dest, min_bytes=100_000)

    print("\nDone! Load 'chrome_extension/' as an unpacked extension in chrome://extensions")


if __name__ == "__main__":
    main()
