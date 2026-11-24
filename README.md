# Music Remover — Real-Time Speech Enhancement CLI

Aplikasi CLI untuk Windows yang menghilangkan background music secara real-time dari system audio, dan hanya menyisakan suara manusia (speech/dialog).

Mulai versi 2.0+, project ini mendukung **2 cara penggunaan**:
1. **Python CLI (Windows System Audio)** — mendukung **DTLN** (`16kHz`), **DeepFilterNet3** (`48kHz`), **Kim Vocal 2** (`44.1kHz` MDX-Net), dan **UVR-MDX-NET Voc_FT** (`44.1kHz` MDX-Net) dengan pengaturan `--delay`.
2. **Chrome Extension (100% In-Browser)** — langsung memfilter audio tab Chrome (YouTube, dll.) tanpa Python/VB-Cable menggunakan **GTCRN**, **Kim Vocal 2**, atau **UVR-MDX-NET Voc_FT** via `onnxruntime-web`.

---

## 🌐 Cara Pakai Chrome Extension (100% Browser)

1. Jalankan script setup sekali untuk mengunduh runtime `onnxruntime-web` dan model ONNX ke `chrome_extension/`:
   ```bash
   python setup_extension.py
   ```
2. Buka `chrome://extensions` di Google Chrome dan aktifkan **Developer mode**.
3. Klik **Load unpacked**, lalu pilih folder `chrome_extension/`.
4. Buka tab video/YouTube, klik icon **Music Remover**, pilih model & delay, lalu klik **Start on This Tab**.

---

## 💻 Cara Pakai Python CLI (System Audio)

Bisa langsung double-click `cli/run_python.bat`, atau masuk ke folder `cli/` lalu jalankan manual:

```bash
cd cli
pip install -r requirements.txt
python music_remover.py
```

### 1. **DeepFilterNet / Kim Vocal 2 / Voc_FT**:
   - **DTLN**: Sangat ringan, ONNX `16kHz`, latency sangat rendah (~70ms).
   - **DeepFilterNet**: Speech enhancement `48kHz`, mendukung `--delay` untuk mengatur chunk duration.
   - **Kim Vocal 2 & UVR-MDX-NET Voc_FT**: Pemisahan musik & vokal sejati (MDX-Net ONNX `44.1kHz`), mendukung `--delay`.

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

### 🚀 Cara Menjalankan (Paling Mudah)
Cukup jalankan perintah di bawah ini tanpa argumen untuk masuk ke **Interactive Setup Mode**:
```bash
python music_remover.py
```
Kamu akan diminta memilih:
1. **Engine**: DTLN atau DeepFilterNet.
2. **Device**: List speaker/headset kamu akan muncul otomatis (pilih berdasarkan nomor).
3. **Diagnostics**: Aktifkan atau nonaktifkan tampilan latensi real-time.

---

### Manual Usage (Advanced)
Jika ingin melewati menu interaktif, gunakan perintah manual:

#### Menjalankan Engine DeepFilterNet (Kualitas Terbaik)
```bash
python music_remover.py --engine deepfilter --output "Headphones (Senary Audio)" --latency
```

#### Menjalankan Engine DTLN (Latency Terendah)
```bash
python music_remover.py --engine dtln --output "Headphones (Senary Audio)" --latency
```

### 🎛️ Interactive Controls
Saat program berjalan, kamu bisa mengatur seberapa kuat filternya secara real-time (hanya dengan mengetik di terminal):

- `+` atau `=` : Menambah kekuatan filter (+10%)
- `-` : Mengurangi kekuatan filter (-10%)
- `0` : Bypass (Matikan filter, 0%)
- `9` : Filter maksimal (100%)
- `q` : Keluar aplikasi dengan aman

## CLI Options

| Flag             | Default       | Description                                      |
| ---------------- | ------------- | ------------------------------------------------ |
| `--engine`       | `dtln`        | Pilih engine (`deepfilter` atau `dtln`)          |
| `--input`        | `loopback`    | Input source (WASAPI loopback)                   |
| `--output`       | `CABLE Input` | Nama output device                               |
| `--list-devices` | -             | List semua audio device                          |
| `--latency`      | -             | Tampilkan diagnostik latency real-time           |
| `--quantized`    | -             | Gunakan model INT8 (hanya untuk DTLN)            |
| `--post-filter`  | -             | Aktifkan post-filter (hanya untuk DeepFilterNet) |

## 💡 Tips Kualitas Suara (Troubleshooting)

Jika suara terdengar "patah-patah" atau vocal "muncul-tenggelam" (muncul hilang):

1. **Gunakan DeepFilterNet**: DTLN hanya mendukung 16kHz (kualitas telepon). DeepFilterNet mendukung 48kHz (HD).
2. **Kurangi Kekuatan Filter**: Jika vocal terpotong, berarti filter terlalu agresif. Tekan `-` di keyboard untuk menurunkan kekuatan ke **80% - 90%**. Ini biasanya membuat suara jauh lebih natural tapi musik tetap hilang.
3. **Aktifkan Post-Filter**: Gunakan flag `--post-filter` jika menjalankan lewat CLI untuk kejernihan extra.
4. **Cek Sample Rate Windows**: Pastikan Recording & Playback device kamu di Windows Sound Control Panel diatur ke **48.000 Hz (DVD/Studio Quality)** agar tidak ada distorsi resampling.

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

## Credits & Licenses

Project ini menggunakan model dan library open-source berikut sesuai dengan lisensi aslinya:

- **Kim_Vocal_2.onnx**: Oleh [Kimberley Jensen (KimberleyJSN)](https://github.com/KimberleyJSN) — [Official UVR Model Releases](https://github.com/TRvlvr/model_repo/releases/tag/all_public_uvr_models) (**MIT License**)
- **UVR-MDX-NET-Voc_FT.onnx & Ultimate Vocal Remover**: Oleh Anjok07 & Aufr33 ([Anjok07/ultimatevocalremovergui](https://github.com/Anjok07/ultimatevocalremovergui), [LICENSE](https://github.com/Anjok07/ultimatevocalremovergui/blob/master/LICENSE) — **MIT License**), berbasis arsitektur [Kuielab MDX-Net](https://github.com/kuielab/mdx-net) (**MIT License**)
- **GTCRN (`gtcrn_simple.onnx`)**: Riset & arsitektur oleh Xiaobin Rong et al. ([Xiaobin-Rong/gtcrn](https://github.com/Xiaobin-Rong/gtcrn) — **MIT License**), diekspor ke ONNX oleh [k2-fsa/sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) ([Model Release](https://github.com/k2-fsa/sherpa-onnx/releases/tag/speech-enhancement-models) — **Apache-2.0 License**)
- **DeepFilterNet3**: Oleh Hendrik Schröter ([Rikorose/DeepFilterNet](https://github.com/Rikorose/DeepFilterNet) — **MIT / Apache-2.0 License**)
- **DTLN Model**: Oleh Nils L. Westhausen ([breizhn/DTLN](https://github.com/breizhn/DTLN) — **MIT License**)
- **ONNX Runtime & `onnxruntime-web`**: Oleh Microsoft ([microsoft/onnxruntime](https://github.com/microsoft/onnxruntime) — **MIT License**)
- **PyAudioWPatch**: WASAPI loopback extension oleh s0d3s ([s0d3s/PyAudioWPatch](https://github.com/s0d3s/PyAudioWPatch) — **MIT License**)


Dibuat oleh antigravity