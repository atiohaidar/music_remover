"""
Script otomatis untuk menambal (patch) library DeepFilterNet.

Sistem: DeepFilterNet 0.5.6 memiliki isu kompatibilitas dengan torchaudio 2.10+.
Ia mencoba mengimpor `torchaudio.backend.common.AudioMetaData` yang sudah dihapus
dari torchaudio versi terbaru. 

Script ini akan mencari lokasi instalasi DeepFilterNet di komputer saat ini, 
kemudian otomatis mengganti baris kode yang error tersebut dengan fallback yang aman.
"""

import os
import sys

# Baris target yang bermasalah (ada di df/io.py versi bawaan)
TARGET_IMPORT = "from torchaudio.backend.common import AudioMetaData"

# Kode pengganti untuk kompatibilitas
PATCH_CODE = """try:
    from torchaudio.backend.common import AudioMetaData
except (ImportError, ModuleNotFoundError):
    # Compatibility shim for torchaudio >= 2.10 (AudioMetaData removed)
    from typing import NamedTuple
    class AudioMetaData(NamedTuple):
        sample_rate: int = 0
        num_frames: int = 0
        num_channels: int = 0
        bits_per_sample: int = 0
        encoding: str = ""
"""

def main():
    print("Mencari library DeepFilterNet...")
    try:
        import df.io
    except ModuleNotFoundError:
        print("❌ Error: DeepFilterNet belum terinstall.")
        print("💡 Coba jalankan: pip install deepfilternet")
        sys.exit(1)
    except ImportError as e:
        # Jika import gagal karena *terjadi* error torchaudio, kita masih bisa mencari path-nya secara manual
        print(f"⚠️ Peringatan saat import uji coba: {e}")
        import importlib.util
        spec = importlib.util.find_spec("df.io")
        if spec is None or spec.origin is None:
            print("❌ Error: Tidak dapat menemukan lokasi file df/io.py")
            sys.exit(1)
        file_path = spec.origin
    else:
        file_path = os.path.abspath(df.io.__file__)
    
    print(f"✔️ Ditemukan di: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    if "Compatibility shim for torchaudio" in content:
        print("✔️ Status: Library DeepFilterNet SUDAH di-patch. Tidak butuh tindakan.")
        return

    if TARGET_IMPORT in content:
        print("⚙️ Menerapkan pembaruan kompatibilitas torchaudio...")
        new_content = content.replace(TARGET_IMPORT, PATCH_CODE)
        
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(new_content)
            print("✅ Patch berhasil diterapkan! DeepFilterNet kini siap digunakan.")
        except PermissionError:
            print("❌ Error: Tidak memiliki izin untuk memodifikasi file tersebut.")
            print("💡 Coba jalankan terminal/CMD sebagai Administrator atau cek hak akses file.")
    else:
        print("⚠️ Peringatan: Baris target tidak ditemukan. Mungkin sudah versi terbaru atau bukan struktur DeepFilterNet standar.")

if __name__ == "__main__":
    main()
