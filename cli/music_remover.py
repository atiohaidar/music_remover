"""
Real-Time Music Remover CLI

Captures system audio via WASAPI loopback, removes background music
using speech enhancement models, and outputs clean speech to a target
audio device.

Supported engines:
  - DTLN (default): 16kHz, ONNX Runtime — lightweight, low latency
  - DeepFilterNet: 48kHz, PyTorch — higher quality noise suppression
  - Kim Vocal 2: 44.1kHz, ONNX Runtime — true vocal/music separation (MDX-Net)

Usage:
    python music_remover.py --output "Headphones (Senary Audio)" --latency
    python music_remover.py --engine deepfilter --delay 2.0 --output "Headphones (Senary Audio)" --latency
    python music_remover.py --engine kimvocal --delay 4.5 --output "Headphones (Senary Audio)" --latency
    python music_remover.py --list-devices

Architecture:
    System Audio → Virtual Audio Cable → WASAPI Loopback
    → Ring Buffer → Engine (DTLN/DeepFilterNet/KimVocal2) → [Resample] → Stereo → Output Buffer → Speakers
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
from resampler import resample, mono_to_stereo

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
    RESAMPLE_BATCH = 9600 if model_sr >= 44100 else 2048
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
        audio = resample(audio, model_sr, output_sr)
    if output_channels >= 2:
        audio = mono_to_stereo(audio).flatten()
    output_buffer.push(audio)


def interactive_setup():
    print("\n" + "=" * 60)
    print("  🛠️  Interactive Setup Mode")
    print("=" * 60)

    # 1. Engine selection
    print("\n[1] Pilih Engine Noise/Music Removal:")
    print("    1. DTLN (Cepat, Ringan, 16kHz)")
    print("    2. DeepFilterNet (Speech Enhancement, 48kHz, PyTorch)")
    print("    3. Kim Vocal 2 FP32 (Pemisahan Musik Agresif, 44.1kHz, ONNX)")
    print("    4. Kim Vocal 2 FP16 (Hemat Memori ~32MB, 44.1kHz, ONNX)")
    print("    5. UVR-MDX-NET Voc_FT (Pemisahan Musik Natural, 44.1kHz, ONNX) [Default]")
    engine_choice = input("    Masukkan Pilihan (1/2/3/4/5) [5]: ").strip()
    if engine_choice == "1":
        engine = "dtln"
    elif engine_choice == "2":
        engine = "deepfilter"
    elif engine_choice == "3":
        engine = "kimvocal"
    elif engine_choice == "4":
        engine = "kimvocal_fp16"
    else:
        engine = "vocft"

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

    # 4. Chunk processing delay (for deepfilter / mdx)
    chunk_delay = None
    is_mdx = engine in ("kimvocal", "kimvocal_fp16", "vocft")
    if engine == "deepfilter" or is_mdx:
        default_cd = 4.5 if is_mdx else 2.0
        print("\n[4] Atur Chunk Delay Pemrosesan (detik):")
        print("    Makin lama delay = pemisahan musik lebih stabil & CPU lebih santai.")
        if is_mdx:
            print("    Rekomendasi CPU: 4.5 - 5.5 detik (GPU bisa 1.0 - 2.0 detik)")
        else:
            print("    Rekomendasi: 1.0 - 5.0 detik")
        cd_input = input(f"    Masukkan chunk delay dalam detik [{default_cd}]: ").strip()
        try:
            chunk_delay = float(cd_input) if cd_input else default_cd
            if chunk_delay < 0.5:
                chunk_delay = 0.5
        except ValueError:
            chunk_delay = default_cd

    # 5. Pre-buffer delay
    default_prebuf = (chunk_delay + 4.0) if is_mdx else (3.0 if engine == "deepfilter" else 5.0)
    print(f"\n[{'5' if chunk_delay is not None else '4'}] Pre-buffer Delay (detik):")
    print("    Waktu tunggu awal sebelum audio mulai diputar.")
    print(f"    Rekomendasi: {default_prebuf:.1f} detik")
    delay_input = input(f"    Masukkan pre-buffer delay dalam detik [{default_prebuf:.1f}]: ").strip()
    try:
        prebuffer_delay = float(delay_input) if delay_input else default_prebuf
        if prebuffer_delay < 0:
            prebuffer_delay = default_prebuf
    except ValueError:
        prebuffer_delay = default_prebuf

    print("\n" + "=" * 60)
    print("  🚀 Memulai aplikasi dengan konfigurasi pilihanmu...")
    print("=" * 60 + "\n")

    class Args:
        pass
    
    args = Args()
    args.engine = engine
    args.output = out_device
    args.latency = latency
    args.delay = chunk_delay
    args.prebuffer_delay = prebuffer_delay
    args.input = "loopback"
    args.list_devices = False
    args.quantized = False
    args.model_dir = "models"
    args.threads = 0 if is_mdx else 1
    args.buffer_size = 128
    args.post_filter = True # Default to True in interactive for "bagus"
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
        choices=["dtln", "deepfilter", "kimvocal", "kimvocal_fp16", "vocft"],
        default="dtln",
        help="Engine: 'dtln' (16kHz), 'deepfilter' (48kHz), 'kimvocal', 'kimvocal_fp16', or 'vocft' (44.1kHz ONNX)",
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
    parser.add_argument(
        "--post-filter",
        action="store_true",
        help="Enable DeepFilterNet post-filter for better speech quality (DeepFilterNet only)",
    )
    parser.add_argument(
        "--delay", "--chunk-delay",
        dest="delay",
        type=float,
        default=None,
        help="Processing chunk delay in seconds (deepfilter/kimvocal). Larger = better separation & lower CPU load.",
    )
    parser.add_argument(
        "--prebuffer-delay",
        type=float,
        default=None,
        help="Pre-buffer delay in seconds before starting playback",
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
    use_mdx = args.engine in ("kimvocal", "kimvocal_fp16", "vocft")

    if use_deepfilter:
        engine_label = "DeepFilterNet (48kHz, PyTorch)"
    elif args.engine == "vocft":
        engine_label = "UVR-MDX-NET Voc_FT (44.1kHz, ONNX)"
    elif args.engine == "kimvocal_fp16":
        engine_label = "Kim Vocal 2 FP16 (44.1kHz, ONNX MDX-Net)"
    elif args.engine == "kimvocal":
        engine_label = "Kim Vocal 2 FP32 (44.1kHz, ONNX MDX-Net)"
    else:
        engine_label = "DTLN (16kHz, ONNX Runtime)"

    # Banner
    print(_make_banner(engine_label))

    if use_deepfilter:
        print("[INIT] Loading DeepFilterNet model...")
        try:
            from inference_engine_df import DeepFilterNetEngine
            engine = DeepFilterNetEngine(
                post_filter=args.post_filter,
                chunk_duration_s=args.delay,
            )
            print(f"       Model: DeepFilterNet3")
            print(f"       Chunk delay: {engine.CHUNK_DURATION_S:.2f}s (hop {engine.shift_duration_ms:.1f}ms)")
            print(f"       Sample rate: {engine.sample_rate} Hz")
        except ImportError as e:
            print(f"\nERROR: DeepFilterNet not installed: {e}")
            print("Run: pip install deepfilternet torch torchaudio")
            sys.exit(1)
        except Exception as e:
            print(f"\nERROR: Failed to load DeepFilterNet: {e}")
            sys.exit(1)
    elif use_mdx:
        print(f"[INIT] Loading {engine_label}...")
        try:
            from inference_engine_kim import KimVocalEngine
            chunk_delay = args.delay if args.delay is not None else 4.5
            mdx_threads = args.threads if args.threads != 1 else 0
            try:
                engine = KimVocalEngine(
                    model_dir=args.model_dir,
                    chunk_duration_s=chunk_delay,
                    num_threads=mdx_threads,
                    model_variant=args.engine,
                )
            except FileNotFoundError:
                print(f"\n[!] Model untuk '{args.engine}' tidak ditemukan di '{args.model_dir}/'")
                choice = input("    Download model sekarang (~64 MB)? (y/n) [y]: ").strip().lower()
                if choice == 'n':
                    print(f"Aborting. '{args.engine}' requires its ONNX model to run.")
                    sys.exit(1)
                import download_model
                download_model.download_mdx_model(args.engine, args.model_dir)
                engine = KimVocalEngine(
                    model_dir=args.model_dir,
                    chunk_duration_s=chunk_delay,
                    num_threads=mdx_threads,
                    model_variant=args.engine,
                )
            print(f"       Model: {engine.MODEL_FILENAME} ({engine.provider})")
            print(f"       Chunk delay: {engine.frame_duration_ms/1000:.2f}s (hop {engine.shift_duration_ms/1000:.2f}s)")
            print(f"       Sample rate: {engine.sample_rate} Hz")
        except Exception as e:
            print(f"\nERROR: Failed to load {engine_label}: {e}")
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
        capture_chunk = engine.frame_size  # Matches engine HOP_SIZE
    elif use_mdx:
        capture_chunk = 4096  # ~93ms at 44.1kHz for smooth input streaming
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

    # === SUPER PRE-BUFFER: fill processed audio before playing ===
    print("\n[INIT] Pre-buffering audio...")
    # Use user-specified delay or default
    if args.prebuffer_delay is not None:
        prebuf_delay = args.prebuffer_delay
    elif use_mdx:
        prebuf_delay = (engine.frame_duration_ms / 1000.0) + 4.0
    else:
        prebuf_delay = 3 if use_deepfilter else 5

    if use_mdx:
        prebuf_target = 1  # Each MDX hop is already several seconds of audio
    else:
        prebuf_target = 3 if use_deepfilter else 5  # Minimum chunks needed
    prebuf_start = time.time()
    while output_buffer.size < prebuf_target and (time.time() - prebuf_start) < prebuf_delay:
        time.sleep(0.1)
    print(f"       Pre-buffered {output_buffer.size} chunks (waited {time.time() - prebuf_start:.2f}s)")

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
