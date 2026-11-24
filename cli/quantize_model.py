"""
Quantize DTLN ONNX models to INT8 for faster inference.

Uses ONNX Runtime dynamic quantization (no calibration data needed).

Usage:
    python quantize_model.py
    python quantize_model.py --model-dir models
"""

import os
import sys
import argparse


def quantize_model(input_path: str, output_path: str):
    """
    Quantize an ONNX model to INT8 using dynamic quantization.
    
    Args:
        input_path: Path to the FP32 ONNX model.
        output_path: Path to save the INT8 ONNX model.
    """
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
    except ImportError:
        print("ERROR: onnxruntime quantization module not found.")
        print("Install with: pip install onnxruntime")
        sys.exit(1)

    print(f"  Input:  {input_path}")
    print(f"  Output: {output_path}")

    quantize_dynamic(
        model_input=input_path,
        model_output=output_path,
        weight_type=QuantType.QUInt8,
        optimize_model=True,
    )

    orig_size = os.path.getsize(input_path) / (1024 * 1024)
    quant_size = os.path.getsize(output_path) / (1024 * 1024)
    reduction = (1 - quant_size / orig_size) * 100

    print(f"  Original:  {orig_size:.2f} MB")
    print(f"  Quantized: {quant_size:.2f} MB")
    print(f"  Reduction: {reduction:.1f}%")


def main():
    parser = argparse.ArgumentParser(description="Quantize DTLN ONNX models to INT8")
    parser.add_argument(
        "--model-dir",
        type=str,
        default="models",
        help="Directory containing ONNX models (default: models)",
    )
    args = parser.parse_args()

    model_dir = args.model_dir

    print("=" * 60)
    print("DTLN ONNX Model Quantization (INT8)")
    print("=" * 60)

    models = [
        ("dtln_1.onnx", "dtln_1_int8.onnx"),
        ("dtln_2.onnx", "dtln_2_int8.onnx"),
    ]

    for orig_name, quant_name in models:
        orig_path = os.path.join(model_dir, orig_name)
        quant_path = os.path.join(model_dir, quant_name)

        if not os.path.exists(orig_path):
            print(f"\n[SKIP] {orig_name} not found. Run download_model.py first.")
            continue

        print(f"\n[QUANTIZE] {orig_name} → {quant_name}")
        try:
            quantize_model(orig_path, quant_path)
            print(f"  ✓ Done")
        except Exception as e:
            print(f"  ✗ Failed: {e}")

    print("\n" + "=" * 60)
    print("Quantization complete!")
    print("Use --quantized flag with music_remover.py to use INT8 models.")
    print("=" * 60)


if __name__ == "__main__":
    main()
