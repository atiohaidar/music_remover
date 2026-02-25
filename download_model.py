"""
Download DTLN ONNX models from the breizhn/DTLN GitHub repository.

Downloads model_1.onnx and model_2.onnx to the models/ directory.
These are the pre-trained DTLN models for speech enhancement.

Usage:
    python download_model.py
"""

import os
import sys
import urllib.request
import urllib.error

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

# DTLN ONNX model URLs from breizhn/DTLN pretrained_model directory
BASE_URL = "https://github.com/breizhn/DTLN/raw/master/pretrained_model"
MODELS = {
    "dtln_1.onnx": f"{BASE_URL}/model_1.onnx",
    "dtln_2.onnx": f"{BASE_URL}/model_2.onnx",
}


def download_file(url: str, dest: str) -> bool:
    """Download a file from URL to destination path."""
    try:
        print(f"  Downloading: {url}")
        urllib.request.urlretrieve(url, dest)
        size_kb = os.path.getsize(dest) / 1024
        print(f"  Saved: {dest} ({size_kb:.1f} KB)")
        return True
    except urllib.error.URLError as e:
        print(f"  Failed: {e}")
        return False
    except Exception as e:
        print(f"  Error: {e}")
        return False


def verify_onnx(filepath: str) -> bool:
    """Basic verification that the file exists and has reasonable size."""
    try:
        size = os.path.getsize(filepath)
        return size > 1000  # At least 1KB
    except Exception:
        return False


def main():
    print("=" * 60)
    print("DTLN ONNX Model Downloader")
    print("=" * 60)

    os.makedirs(MODEL_DIR, exist_ok=True)

    all_ok = True
    for local_name, url in MODELS.items():
        dest = os.path.join(MODEL_DIR, local_name)

        if os.path.exists(dest) and verify_onnx(dest):
            size_kb = os.path.getsize(dest) / 1024
            print(f"\n[SKIP] {local_name} already exists ({size_kb:.1f} KB)")
            continue

        print(f"\n[DOWNLOAD] {local_name}")
        success = download_file(url, dest)

        if not success or not verify_onnx(dest):
            print(f"  ERROR: Failed to download {local_name}")
            all_ok = False
        else:
            print(f"  OK!")

    print("\n" + "=" * 60)
    if all_ok:
        print("All models downloaded successfully!")
        print(f"Model directory: {MODEL_DIR}")
    else:
        print("Some models failed to download.")
        print("Please manually download from:")
        print("  https://github.com/breizhn/DTLN/tree/master/pretrained_model")
        print("Place model_1.onnx and model_2.onnx in the models/ directory")
        print("(rename to dtln_1.onnx and dtln_2.onnx)")
        sys.exit(1)


if __name__ == "__main__":
    main()
