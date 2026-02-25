"""
Real-Time Music Remover CLI

Captures system audio via WASAPI loopback, removes background music
using speech enhancement models, and outputs clean speech to a target
audio device.

Supported engines:
  - DTLN (default): 16kHz, ONNX Runtime — lightweight, low latency
  - DeepFilterNet: 48kHz, PyTorch — higher quality noise suppression

Usage:
    python music_remover.py --output "Headphones (Senary Audio)" --latency
    python music_remover.py --engine deepfilter --output "Headphones (Senary Audio)" --latency
    python music_remover.py --list-devices

Architecture:
    System Audio → Virtual Audio Cable → WASAPI Loopback
    → Ring Buffer → Engine (DTLN/DeepFilterNet) → [Resample] → Stereo → Output Buffer → Speakers
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
from diagnostics import Diagnostics
from resampler import resample_from_16k, mono_to_stereo

try:
    import pyaudiowpatch as pyaudio
except ImportError:
    print("ERROR: PyAudioWPatch is required. Install with: pip install PyAudioWPatch")
    sys.exit(1)


def _make_banner(engine_name: str) -> str:
    return rf"""
  __  __           _        ____                                   
 |  \/  |_   _ ___(_) ___  |  _ \ ___ _ __ ___   _____   _____ _ __
 | |\/| | | | / __| |/ __| | |_) / _ \ '_ ` _ \ / _ \ \ / / _ \ '__|
 | |  | | |_| \__ \ | (__  |  _ <  __/ | | | | | (_) \ V /  __/ |   
 |_|  |_|\__,_|___/_|\___| |_| \_\___|_| |_| |_|\___/ \_/ \___|_|   
 
  By Atiohaidar
  Develop With Antigravity
                                         
  Real-Time Music Remover v2.0
  Engine: {engine_name}
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
    engine,
    input_buffer: RingBuffer,
    output_buffer: RingBuffer,
    diagnostics: Diagnostics,
    stop_event: threading.Event,
    output_sr: int,
    output_channels: int,
    filter_strength: list,
):
    """
    Main processing loop: pop from input → inference → dry/wet mix → [resample] → stereo → push.
    Runs in a dedicated thread.
    
    filter_strength is a shared list [float] where filter_strength[0] is the
    dry/wet mix ratio (0.0 = bypass/original, 1.0 = full filter).
    """
    model_sr = engine.sample_rate

    # Accumulate enhanced audio for batch resampling (better quality)
    RESAMPLE_BATCH = 9600 if model_sr == 48000 else 2048
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

        # --- Inference (dry/wet mix now handled inside engine) ---
        t0 = time.perf_counter()
        strength = filter_strength[0]
        enhanced = engine.process_chunk(chunk, strength=strength)
        dt = time.perf_counter() - t0
        diagnostics.record_inference_time(dt)

        # DeepFilterNet may return empty array while buffering
        if len(enhanced) == 0:
            continue

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


def interactive_setup():
    print("\n" + "=" * 60)
    print("  🛠️  Interactive Setup Mode")
    print("=" * 60)

    # 1. Engine selection
    print("\n[1] Pilih Engine Noise Removal:")
    print("    1. DTLN (Cepat, Ringan, 16kHz)")
    print("    2. DeepFilterNet (Kualitas Tinggi, 48kHz, PyTorch) [Default]")
    engine_choice = input("    Masukkan Pilihan (1/2) [2]: ").strip()
    engine = "dtln" if engine_choice == "1" else "deepfilter"

    # 2. Device selection
    print("\n[2] Pilih Output Audio Device (Speakers/Headset):")
    out_device = "Headphones"
    try:
        import pyaudiowpatch as pyaudio
        from audio_capture import list_audio_devices
        p = pyaudio.PyAudio()
        devices = list_audio_devices(p)
        p.terminate()
        
        # Filter parameter maxOutputChannels > 0 and not loopback
        ignore_names = {"Microsoft Sound Mapper - Output", "Primary Sound Driver"}
        valid_devs = [
            d for d in devices 
            if not d["is_loopback"] 
            and d.get("output_channels", 0) > 0
            and d["name"] not in ignore_names
        ]
        
        # Deduplicate device names
        seen_names = set()
        out_devs = []
        for d in valid_devs:
            if d["name"] not in seen_names:
                seen_names.add(d["name"])
                out_devs.append(d)
        
        if out_devs:
            for i, d in enumerate(out_devs):
                print(f"    {i+1}. {d['name'][:50]}")
            
            dev_choice = input(f"    Masukkan Pilihan (1-{len(out_devs)}) [1]: ").strip()
            try:
                idx = int(dev_choice) - 1
                if 0 <= idx < len(out_devs):
                    out_device = out_devs[idx]["name"]
                else:
                    out_device = out_devs[0]["name"]
            except ValueError:
                out_device = out_devs[0]["name"]
        else:
            print("    [!] Tidak ada output yang ditemukan. Menggunakan default 'Headphones'.")
    except Exception as e:
        print(f"    [!] Gagal memuat devices: {e}. Menggunakan default 'Headphones'.")

    # 3. Latency option
    print("\n[3] Tampilkan Diagnostik Latensi Real-time?")
    print("    y. Ya (Tampilkan latensi)")
    print("    n. Tidak (Tampilan bersih) [Default]")
    lat_choice = input("    Masukkan Pilihan (y/n) [n]: ").strip().lower()
    latency = lat_choice == 'y'

    print("\n" + "=" * 60)
    print("  🚀 Memulai aplikasi dengan konfigurasi pilihanmu...")
    print("=" * 60 + "\n")

    class Args:
        pass
    
    args = Args()
    args.engine = engine
    args.output = out_device
    args.latency = latency
    args.input = "loopback"
    args.list_devices = False
    args.quantized = False
    args.model_dir = "models"
    args.threads = 1
    args.buffer_size = 128
    args.verbose = False
    
    return args


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
        "--engine",
        type=str,
        choices=["dtln", "deepfilter"],
        default="dtln",
        help="Speech enhancement engine: 'dtln' (default, 16kHz ONNX) or 'deepfilter' (48kHz, better quality)",
    )
    parser.add_argument(
        "--latency",
        action="store_true",
        help="Enable real-time latency diagnostics display",
    )
    parser.add_argument(
        "--quantized",
        action="store_true",
        help="Use INT8 quantized models for faster inference (DTLN only)",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="models",
        help="Directory containing ONNX models (default: models, DTLN only)",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=1,
        help="Number of ONNX Runtime inference threads (default: 1, DTLN only)",
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

    # Jika user tidak memasukkan argumen di terminal, jalankan interactive mode
    if len(sys.argv) == 1:
        args = interactive_setup()

    # List devices mode
    if args.list_devices:
        print_devices()
        return

    # === Initialize engine ===
    use_deepfilter = args.engine == "deepfilter"

    if use_deepfilter:
        engine_label = "DeepFilterNet (48kHz, PyTorch)"
    else:
        engine_label = "DTLN (16kHz, ONNX Runtime)"

    # Banner
    print(_make_banner(engine_label))

    if use_deepfilter:
        print("[INIT] Loading DeepFilterNet model...")
        try:
            from inference_engine_df import DeepFilterNetEngine
            engine = DeepFilterNetEngine()
            print(f"       Model: DeepFilterNet3")
            print(f"       Chunk size: {engine.frame_size} samples ({engine.shift_duration_ms:.1f}ms)")
            print(f"       Sample rate: {engine.sample_rate} Hz")
        except ImportError as e:
            print(f"\nERROR: DeepFilterNet not installed: {e}")
            print("Run: pip install deepfilternet torch torchaudio")
            sys.exit(1)
        except Exception as e:
            print(f"\nERROR: Failed to load DeepFilterNet: {e}")
            sys.exit(1)
    else:
        print("[INIT] Loading DTLN ONNX model...")
        try:
            from inference_engine import DTLNInferenceEngine
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
            print(f"\n[!] Model DTLN tidak ditemukan: {e}")
            choice = input("    Download model sekarang? (y/n) [y]: ").strip().lower()
            if choice != 'n':
                try:
                    import download_model
                    download_model.main()
                    # Re-attempt loading
                    print("\n[INIT] Mencoba memuat model kembali...")
                    engine = DTLNInferenceEngine(
                        model_dir=args.model_dir,
                        use_quantized=args.quantized,
                        num_threads=args.threads,
                    )
                except Exception as e2:
                    print(f"\nERROR: Gagal memuat model setelah download: {e2}")
                    sys.exit(1)
            else:
                print("Aborting. DTLN requires models to run.")
                sys.exit(1)

    # Create ring buffers (large for quality)
    input_buffer = RingBuffer(max_chunks=args.buffer_size)
    output_buffer = RingBuffer(max_chunks=args.buffer_size * 4)

    # === Start audio capture (but NOT output yet) ===
    # Determine capture chunk size based on engine
    if use_deepfilter:
        capture_chunk = engine.frame_size  # Matches engine HOP_SIZE (12000 samples = 250ms)
    else:
        capture_chunk = 512  # Keep 512 for DTLN (32ms at 16kHz) for buffer stability

    print("\n[INIT] Starting audio capture (WASAPI loopback)...")
    capture = AudioCapture(
        ring_buffer=input_buffer,
        target_sr=engine.sample_rate,
        chunk_samples=capture_chunk,
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
    print(f"\n[INIT] Opening output → '{args.output}'...")
    output = AudioOutput(
        ring_buffer=output_buffer,
        device_name=args.output,
    )

    # Diagnostics
    diag = Diagnostics(
        input_buffer=input_buffer,
        output_buffer=output_buffer,
        model_sr=engine.sample_rate,
        frame_size=capture_chunk,
    )

    # === Shared filter strength (mutable list for thread safety) ===
    filter_strength = [1.0]  # 1.0 = full filter, 0.0 = bypass

    # === Start processing thread (fills output buffer) ===
    stop_event = threading.Event()
    proc_thread = threading.Thread(
        target=processing_loop,
        args=(
            engine, input_buffer, output_buffer, diag, stop_event,
            48000,  # output SR
            2,      # stereo
            filter_strength,
        ),
        daemon=True,
    )
    proc_thread.start()

    # === SUPER PRE-BUFFER: fill 1 second of processed audio before playing ===
    print("\n[INIT] Pre-buffering audio...")
    prebuf_target = 3 if use_deepfilter else 5  # Less chunks needed at 48kHz
    prebuf_start = time.time()
    while output_buffer.size < prebuf_target and (time.time() - prebuf_start) < 5.0:
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
    print("  Processing active!")
    print("=" * 60)
    print("  Controls:")
    print("    +/=  Increase filter strength (+10%)")
    print("    -    Decrease filter strength (-10%)")
    print("    0    Bypass (no filter)")
    print("    9    Max filter (100%)")
    print("    q    Quit")
    print("=" * 60)
    print(f"  Filter strength: {filter_strength[0]*100:.0f}%")

    if not args.latency:
        print("\nTip: Add --latency flag to see real-time performance stats.\n")

    # Small delay to let messages print before diagnostics
    time.sleep(0.3)

    # Start diagnostics display
    if args.latency:
        diag.start()

    # === Main loop with interactive keyboard control ===
    def signal_handler(sig, frame):
        stop_event.set()

    signal.signal(signal.SIGINT, signal_handler)

    try:
        import msvcrt  # Windows non-blocking keyboard input
        while not stop_event.is_set():
            if msvcrt.kbhit():
                key = msvcrt.getch()
                _handle_key(key, filter_strength, stop_event)
            else:
                time.sleep(0.05)
    except ImportError:
        # Non-Windows: fall back to simple wait
        try:
            while not stop_event.is_set():
                stop_event.wait(timeout=0.5)
        except KeyboardInterrupt:
            stop_event.set()
    except KeyboardInterrupt:
        stop_event.set()

    # === Cleanup ===
    _cleanup(diag, capture, output, proc_thread, args)


def _handle_key(key: bytes, filter_strength: list, stop_event):
    """Handle a single keypress for filter strength control."""
    STEP = 0.10  # 10% steps

    if key in (b'+', b'='):
        filter_strength[0] = min(1.0, filter_strength[0] + STEP)
    elif key == b'-':
        filter_strength[0] = max(0.0, filter_strength[0] - STEP)
    elif key == b'0':
        filter_strength[0] = 0.0
    elif key == b'9':
        filter_strength[0] = 1.0
    elif key in (b'q', b'Q', b'\x03'):  # q or Ctrl+C
        stop_event.set()
        return
    else:
        return  # Unknown key, ignore

    pct = filter_strength[0] * 100
    bar = '█' * int(pct / 5) + '░' * (20 - int(pct / 5))
    print(f"\r  Filter: [{bar}] {pct:3.0f}%   ", end="", flush=True)


def _cleanup(diag, capture, output, proc_thread, args):
    """Shutdown all components and print summary."""
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
