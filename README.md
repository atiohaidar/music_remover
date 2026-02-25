# NoMusic — Real-Time Music Remover CLI

Aplikasi CLI untuk Windows yang menghilangkan background music secara real-time dari system audio, dan hanya menyisakan suara manusia (speech/dialog).

Mulai versi 2.0, aplikasi ini mendukung 2 mesin pemrosesan:
1. **DeepFilterNet (Baru, Default)**: Kualitas pemisahan instrumen/suara jauh lebih jernih, native 48kHz, menggunakan teknik overlap-add (latency ~250ms).
2. **DTLN (Klasik)**: Sangat ringan, menggunakan ONNX, latency sangat rendah (~70ms).

## Architecture (DeepFilterNet Mode)

```text
System Audio (YouTube, VLC, dll) [48kHz]
        ↓
WASAPI Loopback Capture (PyAudioWPatch)
        ↓
Ring Buffer (Accumulate 500ms chunks)
        ↓
DeepFilterNet (PyTorch) + 50% Overlap-Add & Crossfade
        ↓
Output Ring Buffer (250ms hops)
        ↓
VB-Cable Virtual Audio Device / Real Headphones
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

### 3. Jalankan Patch Kompatibilitas (Penting!)

Library `deepfilternet` bawaan PyPI memiliki bug dengan `torchaudio` versi baru (2.10+). Setelah *pip install*, jalankan script ini sekali saja:

```bash
python patch_deepfilter.py
```
*Script ini akan otomatis menambal error kompatibilitas torchaudio pada library deepfilternet di komputermu.*

### 4. Download Model (Otomatis)

- **DeepFilterNet**: Model `DeepFilterNet3` akan diunduh otomatis pada saat pertama kali program dijalankan.
- **DTLN**: Jalankan `python download_model.py` untuk mengunduh model ONNX (~3.5 MB) ke folder `models/`.

## Usage & Controls

### Menjalankan Engine DeepFilterNet (Kualitas Terbaik/Default)
```bash
python music_remover.py --engine deepfilter --output "Headphones (Senary Audio)" --latency
```

### Menjalankan Engine DTLN (Latency Terendah)
```bash
python music_remover.py --engine dtln --output "Headphones (Senary Audio)" --latency
```
*(Atau cukup `python music_remover.py` karena dtln menjaga backward compatibility jika module DF gagal diload).*

### 🎛️ Interactive Controls
Saat program berjalan, kamu bisa mengatur seberapa kuat filternya secara real-time (hanya dengan mengetik di terminal):

- `+` atau `=` : Menambah kekuatan filter (+10%)
- `-` : Mengurangi kekuatan filter (-10%)
- `0` : Bypass (Matikan filter, 0%)
- `9` : Filter maksimal (100%)
- `q` : Keluar aplikasi dengan aman

## CLI Options

| Flag             | Default       | Description                             |
| ---------------- | ------------- | --------------------------------------- |
| `--engine`       | `dtln`        | Pilih engine (`deepfilter` atau `dtln`) |
| `--input`        | `loopback`    | Input source (WASAPI loopback)          |
| `--output`       | `CABLE Input` | Nama output device                      |
| `--list-devices` | -             | List semua audio device                 |
| `--latency`      | -             | Tampilkan diagnostik latency real-time  |
| `--quantized`    | -             | Gunakan model INT8 (hanya untuk DTLN)   |

## Performance Metrics

| Metric             | DeepFilterNet (New)         | DTLN (Classic)              |
| ------------------ | --------------------------- | --------------------------- |
| Sample Rate        | 48kHz (Native, No Resample) | 16kHz (Requires Resampling) |
| Chunk Size / Hop   | Accumulate 500ms, Hop 250ms | Blocks of 32ms, Hop 8ms     |
| End-to-end latency | ~250ms - 350ms              | ~70ms - 150ms               |
| Speech Quality     | ⭐⭐⭐⭐⭐ (Sangat jernih)       | ⭐⭐⭐ (Cukup)                 |
| Background Music   | Terhapus Total              | Tereduksi                   |

## How It Works (DeepFilterNet Mode)

1. **Capture**: loopback stream WASAPI menangkap system audio pada **48kHz native**.
2. **Accumulation**: Audio ditampung hingga mencapai panjang 0.5 detik (500ms).
3. **Inference (Overlap-Add)**: DeepFilterNet memproses sinyal. Outputnya lalu di-crossfade 50% menggunakan Hann Window dengan porsi ekor dari chunk sebelumnya. Ini dilakukan untuk menghindari suara patah-patah/stuttering pada batas frame.
4. **Mix**: Efek filter dicampur menggunakan fitur *native attenuation limit* dalam model, menghasilkan *dry/wet mix* presisi dalam spektrum frekuensi tanpa jeda phasa (menghindari suara comb-filtering/menggema).
5. **Output**: Sinyal (sebesar 0.25 detik hop) dikirim langsung ke output.

## Credits

- **DeepFilterNet**: [Rikorose/DeepFilterNet](https://github.com/Rikorose/DeepFilterNet)
- **DTLN Model**: [breizhn/DTLN](https://github.com/breizhn/DTLN)
- **PyAudioWPatch**: WASAPI loopback extension for Windows


Dibuat oleh antigravity