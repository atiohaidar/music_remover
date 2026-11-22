@echo off
cd /d "%~dp0"
python -c "import pyaudiowpatch, onnxruntime, numpy, scipy" 2>nul
if errorlevel 1 (
    echo [SETUP] Installing Python dependencies...
    pip install -r requirements.txt
)
python music_remover.py
pause
