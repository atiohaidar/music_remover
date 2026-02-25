"""
Audio capture module using PyAudioWPatch for WASAPI loopback recording.

Captures system audio (all applications) via WASAPI loopback,
converts to mono float32, resamples to 16kHz, and pushes chunks
to a ring buffer for processing.
"""

import threading
import numpy as np

try:
    import pyaudiowpatch as pyaudio
except ImportError:
    raise ImportError(
        "PyAudioWPatch is required. Install with: pip install PyAudioWPatch"
    )

from ring_buffer import RingBuffer
from resampler import resample_to_16k, stereo_to_mono


def list_audio_devices(p: pyaudio.PyAudio = None) -> list[dict]:
    """
    List all available audio devices.
    
    Returns:
        List of device info dicts with keys: index, name, channels, sample_rate, is_loopback.
    """
    own_pa = p is None
    if own_pa:
        p = pyaudio.PyAudio()

    devices = []
    try:
        for i in range(p.get_device_count()):
            info = p.get_device_info_by_index(i)
            devices.append({
                "index": i,
                "name": info.get("name", "Unknown"),
                "channels": int(info.get("maxInputChannels", 0)),
                "sample_rate": int(info.get("defaultSampleRate", 0)),
                "is_loopback": info.get("isLoopbackDevice", False),
                "host_api": int(info.get("hostApi", -1)),
            })
    finally:
        if own_pa:
            p.terminate()

    return devices


def find_loopback_device(p: pyaudio.PyAudio) -> dict:
    """
    Find the default WASAPI loopback device.
    
    Returns:
        Device info dict for the loopback device.
    
    Raises:
        RuntimeError: If no WASAPI loopback device is found.
    """
    try:
        wasapi_info = p.get_host_api_info_by_type(pyaudio.paWASAPI)
    except OSError:
        raise RuntimeError(
            "WASAPI not available on this system. "
            "This application requires Windows with WASAPI support."
        )

    # Get default WASAPI output device
    default_output_idx = wasapi_info.get("defaultOutputDevice", -1)
    if default_output_idx < 0:
        raise RuntimeError("No default WASAPI output device found.")

    default_output = p.get_device_info_by_index(default_output_idx)

    # Find loopback counterpart
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        if (
            info.get("isLoopbackDevice", False)
            and info.get("name", "").startswith(default_output.get("name", "???"))
        ):
            return info

    # Fallback: find any loopback device
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        if info.get("isLoopbackDevice", False):
            return info

    raise RuntimeError(
        "No WASAPI loopback device found. "
        "Make sure audio is playing on the default output device."
    )


class AudioCapture:
    """
    Captures system audio via WASAPI loopback and feeds chunks to a ring buffer.
    
    Args:
        ring_buffer: Target ring buffer for captured audio chunks.
        target_sr: Target sample rate for output chunks (default: 16000 for DTLN).
        chunk_samples: Number of output samples per chunk (at target_sr).
    """

    def __init__(
        self,
        ring_buffer: RingBuffer,
        target_sr: int = 16000,
        chunk_samples: int = 512,
    ):
        self._buffer = ring_buffer
        self._target_sr = target_sr
        self._chunk_samples = chunk_samples
        self._pa: pyaudio.PyAudio | None = None
        self._stream = None
        self._running = False
        self._device_info: dict = {}
        self._native_sr: int = 0
        self._native_channels: int = 0

        # Accumulation buffer for resampled data
        self._accum = np.array([], dtype=np.float32)
        self._accum_lock = threading.Lock()

    def start(self) -> dict:
        """
        Start capturing audio from WASAPI loopback.
        
        Returns:
            Device info dict of the loopback device being used.
        """
        self._pa = pyaudio.PyAudio()
        self._device_info = find_loopback_device(self._pa)
        
        self._native_sr = int(self._device_info["defaultSampleRate"])
        self._native_channels = max(1, int(self._device_info["maxInputChannels"]))

        # Calculate native chunk size for ~32ms
        frames_per_buffer = int(self._native_sr * 0.032)

        self._running = True
        self._stream = self._pa.open(
            format=pyaudio.paFloat32,
            channels=self._native_channels,
            rate=self._native_sr,
            input=True,
            input_device_index=int(self._device_info["index"]),
            frames_per_buffer=frames_per_buffer,
            stream_callback=self._callback,
        )
        self._stream.start_stream()

        return {
            "device_name": self._device_info.get("name", "Unknown"),
            "native_sr": self._native_sr,
            "channels": self._native_channels,
            "frames_per_buffer": frames_per_buffer,
        }

    def _callback(self, in_data, frame_count, time_info, status):
        """PyAudio stream callback — runs in a separate thread."""
        if not self._running:
            return (None, pyaudio.paComplete)

        try:
            # Convert bytes to numpy
            audio = np.frombuffer(in_data, dtype=np.float32)

            # Reshape if stereo/multichannel
            if self._native_channels > 1:
                audio = audio.reshape(-1, self._native_channels)
                audio = stereo_to_mono(audio)

            # Resample to target sample rate
            if self._native_sr != self._target_sr:
                audio = resample_to_16k(audio, self._native_sr)

            # Accumulate and push complete chunks
            with self._accum_lock:
                self._accum = np.concatenate([self._accum, audio])

                while len(self._accum) >= self._chunk_samples:
                    chunk = self._accum[:self._chunk_samples].copy()
                    self._accum = self._accum[self._chunk_samples:]
                    self._buffer.push(chunk)

        except Exception:
            pass  # Never block in callback

        return (None, pyaudio.paContinue)

    def stop(self):
        """Stop capturing audio."""
        self._running = False
        if self._stream:
            try:
                self._stream.stop_stream()
                self._stream.close()
            except Exception:
                pass
            self._stream = None
        if self._pa:
            try:
                self._pa.terminate()
            except Exception:
                pass
            self._pa = None

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def device_name(self) -> str:
        return self._device_info.get("name", "Not started")

    @property
    def native_sample_rate(self) -> int:
        return self._native_sr
