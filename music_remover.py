"""
Real-Time Music Remover CLI

Captures system audio via WASAPI loopback, removes background music
using DTLN speech enhancement model (ONNX), and outputs clean speech
to a target audio device.

Usage:
    python music_remover.py --output "Headphones (Senary Audio)" --latency
    python music_remover.py --list-devices

Architecture:
    System Audio → Virtual Audio Cable → WASAPI Loopback
    → Ring Buffer → DTLN ONNX → Resample+Stereo → Output Buffer → Speakers
"""

import argparse
import signal
import sys
import time
import threading

import numpy as np

from ring_buffer import RingBuffer
from audio_capture import AudioCapture, list_audio_devices
from audio_output import AudioOutput
from inference_engine import DTLNInferenceEngine
from diagnostics import Diagnostics
from resampler import resample_from_16k, mono_to_stereo

try:
    import pyaudiowpatch as pyaudio
except ImportError:
    print("ERROR: PyAudioWPatch is required. Install with: pip install PyAudioWPatch")
    sys.exit(1)


BANNER = r"""
  _   _       __  __           _      
 | \ | | ___ |  \/  |_   _ ___(_) ___ 
 |  \| |/ _ \| |\/| | | | / __| |/ __|
 | |\  | (_) | |  | | |_| \__ \ | (__ 
 |_| \_|\___/|_|  |_|\__,_|___/_|\___|
                                        
  Real-Time Music Remover v1.0
  Speech Enhancement via DTLN + ONNX Runtime
"""


def print_devices():
    """Print all available audio devices."""
    p = pyaudio.PyAudio()
    devices = list_audio_devices(p)
    p.terminate()

    print("\n Available Audio Devices:")
    print("-" * 80)
    print(f"{'Idx':<5} {'Name':<50} {'Ch':<4} {'SR':<8} {'Loop':<6}")
    print("-" * 80)

    for d in devices:
        loop_str = "✓" if d["is_loopback"] else ""
        print(
            f"{d['index']:<5} "
            f"{d['name'][:49]:<50} "
            f"{d['channels']:<4} "
            f"{d['sample_rate']:<8} "
            f"{loop_str:<6}"
        )

    print("-" * 80)
    print("\nTip: Loopback devices (✓) capture system audio.")
    print("     Use --output to specify your real speaker/headphone.\n")


def processing_loop(
    engine: DTLNInferenceEngine,
    input_buffer: RingBuffer,
    output_buffer: RingBuffer,
    diagnostics: Diagnostics,
    stop_event: threading.Event,
    output_sr: int,
    output_channels: int,
):
    """
    Main processing loop: pop from input → inference → batch resample → stereo → push to output.
    Runs in a dedicated thread. Accumulates enhanced audio before resampling
    for better filter quality (fewer edge artifacts).
    """
    model_sr = engine.sample_rate

    # Accumulate enhanced audio for batch resampling (better quality)
    RESAMPLE_BATCH = 2048  # 128ms at 16kHz
    accum_enhanced = np.array([], dtype=np.float32)

    while not stop_event.is_set():
        chunk = input_buffer.pop(timeout=0.02)
        if chunk is None:
            # Flush remaining data
            if len(accum_enhanced) > 0:
                _push_resampled(
                    accum_enhanced, output_buffer,
                    model_sr, output_sr, output_channels,
                )
                accum_enhanced = np.array([], dtype=np.float32)
            continue

        # --- Inference ---
        t0 = time.perf_counter()
        enhanced = engine.process_chunk(chunk)
        dt = time.perf_counter() - t0
        diagnostics.record_inference_time(dt)

        # --- Accumulate for batch resampling ---
        accum_enhanced = np.concatenate([accum_enhanced, enhanced])

        if len(accum_enhanced) >= RESAMPLE_BATCH:
            _push_resampled(
                accum_enhanced, output_buffer,
                model_sr, output_sr, output_channels,
            )
            accum_enhanced = np.array([], dtype=np.float32)


def _push_resampled(
    audio: np.ndarray,
    output_buffer: RingBuffer,
    model_sr: int,
    output_sr: int,
    output_channels: int,
):
    """Resample a large batch, convert to stereo, push to output buffer."""
    if model_sr != output_sr:
        audio = resample_from_16k(audio, output_sr)
    if output_channels >= 2:
        audio = mono_to_stereo(audio).flatten()
    output_buffer.push(audio)


def main():
    parser = argparse.ArgumentParser(
        description="Real-Time Music Remover — Speech Enhancement CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python music_remover.py --list-devices\n"
            '  python music_remover.py --output "Headphones (Senary Audio)" --latency\n'
        ),
    )
    parser.add_argument(
        "--input",
        type=str,
        default="loopback",
        help="Input source: 'loopback' for WASAPI loopback (default: loopback)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="Headphones",
        help="Output device name (default: 'Headphones')",
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="List all audio devices and exit",
    )
    parser.add_argument(
        "--latency",
        action="store_true",
        help="Enable real-time latency diagnostics display",
    )
    parser.add_argument(
        "--quantized",
        action="store_true",
        help="Use INT8 quantized models for faster inference",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="models",
        help="Directory containing ONNX models (default: models)",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=1,
        help="Number of ONNX Runtime inference threads (default: 1)",
    )
    parser.add_argument(
        "--buffer-size",
        type=int,
        default=128,
        help="Ring buffer max chunks (default: 128)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )

    args = parser.parse_args()

    # List devices mode
    if args.list_devices:
        print_devices()
        return

    # Banner
    print(BANNER)

    # === Initialize components ===
    print("[INIT] Loading DTLN ONNX model...")
    try:
        engine = DTLNInferenceEngine(
            model_dir=args.model_dir,
            use_quantized=args.quantized,
            num_threads=args.threads,
        )
        print(f"       Model: {'INT8 quantized' if args.quantized else 'FP32'}")
        print(f"       Frame size: {engine.frame_size} samples ({engine.shift_duration_ms:.1f}ms shift)")
        print(f"       Block size: {engine.block_size} samples ({engine.frame_duration_ms:.1f}ms FFT frame)")
        print(f"       Sample rate: {engine.sample_rate} Hz")
    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
        print("Run: python download_model.py")
        sys.exit(1)

    # Create ring buffers (large for quality)
    input_buffer = RingBuffer(max_chunks=args.buffer_size)
    output_buffer = RingBuffer(max_chunks=args.buffer_size * 4)

    # === Start audio capture (but NOT output yet) ===
    print("\n[INIT] Starting audio capture (WASAPI loopback)...")
    capture = AudioCapture(
        ring_buffer=input_buffer,
        target_sr=engine.sample_rate,
        chunk_samples=512,  # 32ms chunks at 16kHz
    )
    try:
        cap_info = capture.start()
        print(f"       Device: {cap_info['device_name']}")
        print(f"       Native SR: {cap_info['native_sr']} Hz")
        print(f"       Channels: {cap_info['channels']}")
    except RuntimeError as e:
        print(f"\nERROR: {e}")
        sys.exit(1)

    # === Prepare output (but don't start stream yet) ===
    print(f"\n[INIT] Preparing audio output → '{args.output}'...")
    output = AudioOutput(
        ring_buffer=output_buffer,
        device_name=args.output,
    )

    # Diagnostics
    diag = Diagnostics(
        input_buffer=input_buffer,
        output_buffer=output_buffer,
        model_sr=engine.sample_rate,
        frame_size=512,
    )

    # === Start processing thread (fills output buffer) ===
    stop_event = threading.Event()
    proc_thread = threading.Thread(
        target=processing_loop,
        args=(
            engine, input_buffer, output_buffer, diag, stop_event,
            48000,  # Will be updated after output starts
            2,      # stereo
        ),
        daemon=True,
    )
    proc_thread.start()

    # === SUPER PRE-BUFFER: fill 1 second of processed audio before playing ===
    print("\n[INIT] Pre-buffering 1 second of audio (for clean output)...")
    prebuf_start = time.time()
    while output_buffer.size < 5 and (time.time() - prebuf_start) < 3.0:
        time.sleep(0.1)
    print(f"       Pre-buffered {output_buffer.size} chunks")

    # === NOW start output stream ===
    try:
        out_info = output.start()
        print(f"       Device: {out_info['device_name']}")
        print(f"       Native SR: {out_info['native_sr']} Hz")
        print(f"       Channels: {out_info['channels']}")
        print(f"       Host API: {out_info.get('host_api', '?')}")
    except RuntimeError as e:
        print(f"\nERROR: {e}")
        stop_event.set()
        capture.stop()
        sys.exit(1)

    print("\n" + "=" * 60)
    print("  Processing active! Press Ctrl+C to stop.")
    print("=" * 60)

    if not args.latency:
        print("\nTip: Add --latency flag to see real-time performance stats.\n")

    # Small delay to let messages print before diagnostics
    time.sleep(0.3)

    # Start diagnostics display
    if args.latency:
        diag.start()

    # === Main loop (wait for Ctrl+C) ===
    def signal_handler(sig, frame):
        stop_event.set()

    signal.signal(signal.SIGINT, signal_handler)

    try:
        while not stop_event.is_set():
            stop_event.wait(timeout=0.5)
    except KeyboardInterrupt:
        stop_event.set()

    # === Cleanup ===
    print("\n\n[STOP] Shutting down...")
    diag.stop()
    capture.stop()
    output.stop()
    proc_thread.join(timeout=2.0)

    # Print summary
    if args.latency:
        diag.print_summary()

    print("[DONE] Goodbye!\n")


if __name__ == "__main__":
    main()
