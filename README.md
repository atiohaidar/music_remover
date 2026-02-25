# NoMusic — Real-Time Music Remover CLI

Aplikasi CLI untuk Windows yang menghilangkan background music secara real-time dari system audio, dan hanya menyisakan suara manusia (speech/dialog).

## Architecture

```
System Audio (YouTube, VLC, dll)
        ↓
WASAPI Loopback Capture (PyAudioWPatch)
        ↓
Ring Buffer (32ms chunks)
        ↓
DTLN Speech Enhancement (ONNX Runtime)
        ↓
Output Ring Buffer
        ↓
VB-Cable Virtual Audio Device
        ↓
Speaker / Headset
```

## Prerequisites

- **Windows 10/11**
- **Python 3.9+** (tested with 3.10, 3.11)
- **VB-Audio Virtual Cable** (free)

## Setup

### 1. Install VB-Cable

1. Download dari [https://vb-audio.com/Cable/](https://vb-audio.com/Cable/)
2. Extract ZIP file
3. **Run as Administrator**: klik kanan `VBCABLE_Setup_x64.exe` → Run as administrator
4. Klik "Install Driver"
5. **Restart komputer** setelah instalasi

Setelah restart, akan muncul 2 device baru:
- **CABLE Input** (virtual output — untuk menulis audio)
- **CABLE Output** (virtual input — untuk mendengar audio)

### 2. Install Python Dependencies

```bash
cd d:\Project\nomusic
pip install -r requirements.txt
```

### 3. Download AI Model

```bash
python download_model.py
```

Model DTLN (~3.5 MB total) akan didownload ke folder `models/`.

### 4. (Opsional) Quantize Model ke INT8

```bash
python quantize_model.py
```

Menghasilkan model INT8 yang lebih cepat (~30% speedup).

## Usage

### PENTING: Setup Audio Routing

Untuk menghindari echo/gema, routing harus diatur dengan benar:

```
YouTube/VLC/dll → Virtual Audio Cable (tidak terdengar langsung)
        ↓
  WASAPI Loopback capture dari VAC
        ↓
  DTLN Speech Enhancement
        ↓
  Headphones / Speaker (kamu dengar hanya yang diproses)
```

**Langkah Setup:**

1. **Ubah Windows default output** ke **Virtual Audio Cable**:
   - Settings → Sound → Output → pilih "Line 1 (Virtual Audio Cable)"
   - Sekarang semua audio system masuk ke VAC (tidak terdengar langsung)

2. **Jalankan music remover** — output ke speaker/headphone asli kamu:
   ```bash
   python music_remover.py --output "Headphones (Senary Audio)" --latency
   ```

3. **Selesai!** Kamu akan mendengar audio yang sudah dihilangkan musiknya

4. Untuk **berhenti**: tekan `Ctrl+C`, lalu kembalikan Windows output ke speaker biasa

> **⚠️ JANGAN** set output app ke Virtual Audio Cable yang sama — ini akan membuat feedback loop!

### List audio devices
```bash
python music_remover.py --list-devices
```

### Menggunakan model INT8 (lebih cepat)
```bash
python music_remover.py --output "Headphones (Senary Audio)" --quantized --latency
```

## CLI Options

| Flag             | Default       | Description                            |
| ---------------- | ------------- | -------------------------------------- |
| `--input`        | `loopback`    | Input source (WASAPI loopback)         |
| `--output`       | `CABLE Input` | Nama output device                     |
| `--list-devices` | -             | List semua audio device                |
| `--latency`      | -             | Tampilkan diagnostik latency real-time |
| `--quantized`    | -             | Gunakan model INT8                     |
| `--model-dir`    | `models`      | Direktori model ONNX                   |
| `--threads`      | `1`           | Jumlah thread inference                |
| `--buffer-size`  | `64`          | Ukuran ring buffer (chunks)            |
| `--verbose`      | -             | Output verbose                         |

## Performance

| Metric              | Expected                       |
| ------------------- | ------------------------------ |
| Frame size          | 32ms (512 samples @ 16kHz)     |
| Inference per frame | 2-8ms                          |
| End-to-end latency  | ~70ms                          |
| CPU usage           | < 10% (single core)            |
| Model size          | ~3.5 MB (FP32), ~1.8 MB (INT8) |
| Real-time factor    | < 0.25                         |

## Testing Latency

Untuk memastikan latency acceptable:

```bash
python music_remover.py --input loopback --output "CABLE Input" --latency
```

Perhatikan output diagnostik:
- **Latency**: harus < 150ms
- **RTF** (Real-Time Factor): harus < 1.0 (idealnya < 0.3)
- **Overflow**: harus 0 (jika > 0, inference terlalu lambat)

## Troubleshooting

### "No WASAPI loopback device found"
- Pastikan ada audio yang sedang diplay (buka YouTube, dll)
- Cek default output device di Windows Sound Settings

### "Output device 'CABLE Input' not found"
- Pastikan VB-Cable sudah terinstall (restart setelah install)
- Gunakan `--list-devices` untuk melihat nama device yang tepat
- Nama device bisa sedikit berbeda, misal "CABLE Input (VB-Audio Virtual Cable)"

### Audio glitch / stutter
- Naikkan buffer size: `--buffer-size 128`
- Kurangi thread: `--threads 1`
- Gunakan model INT8: `--quantized`

### Latency terlalu tinggi
- Turunkan buffer size: `--buffer-size 32`
- Gunakan model INT8: `--quantized`

## Project Structure

```
nomusic/
├── music_remover.py       # CLI entry point
├── audio_capture.py       # WASAPI loopback capture
├── audio_output.py        # VB-Cable audio output
├── inference_engine.py    # DTLN ONNX inference (2-stage)
├── ring_buffer.py         # Thread-safe ring buffer
├── resampler.py           # Sample rate conversion
├── diagnostics.py         # Latency & performance monitoring
├── download_model.py      # Model downloader
├── quantize_model.py      # INT8 quantization
├── requirements.txt       # Python dependencies
├── README.md              # This file
└── models/
    ├── dtln_1.onnx        # DTLN Stage 1 (STFT path)
    └── dtln_2.onnx        # DTLN Stage 2 (learned basis)
```

## How It Works

1. **Capture**: PyAudioWPatch membuka WASAPI loopback stream yang menangkap semua system audio
2. **Resample**: Audio stereo dikonversi ke mono 16kHz (sesuai model)
3. **Buffer**: Chunks disimpan di ring buffer thread-safe
4. **Inference**: Processing thread mengambil chunks dan menjalankan DTLN model:
   - Stage 1: STFT domain processing (magnitude masking)
   - Stage 2: Learned basis processing (temporal features)
5. **Output**: Audio yang sudah di-enhance di-resample kembali ke native SR dan dikirim ke VB-Cable
6. **Listen**: User mendengar output via CABLE Output device

## Credits

- **DTLN Model**: [breizhn/DTLN](https://github.com/breizhn/DTLN) — Dual-signal Transformation LSTM Network
- **ONNX Runtime**: [Microsoft](https://onnxruntime.ai/)
- **VB-Cable**: [VB-Audio](https://vb-audio.com/Cable/)
