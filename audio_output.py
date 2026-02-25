"""
Audio output module for routing processed audio to a target audio device.

Outputs pre-processed audio (already resampled and formatted) to the target device.
The processing thread is responsible for resampling and stereo conversion —
the output callback just pulls and writes data for minimal latency.
"""

import numpy as np

try:
    import pyaudiowpatch as pyaudio
except ImportError:
    raise ImportError(
        "PyAudioWPatch is required. Install with: pip install PyAudioWPatch"
    )

from ring_buffer import RingBuffer


def find_output_device(p: pyaudio.PyAudio, name: str) -> dict:
    """
    Find an output device by name (partial match, case-insensitive).
    Strongly prefers WASAPI devices (48kHz, lower latency) over MME/DirectSound.
    """
    name_lower = name.lower()

    # Get WASAPI host API index
    wasapi_host_api = -1
    try:
        wasapi_info = p.get_host_api_info_by_type(pyaudio.paWASAPI)
        wasapi_host_api = int(wasapi_info.get("index", -1))
    except OSError:
        pass

    candidates = []
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        max_out = int(info.get("maxOutputChannels", 0))
        dev_name = info.get("name", "")
        is_loopback = info.get("isLoopbackDevice", False)

        if max_out > 0 and name_lower in dev_name.lower() and not is_loopback:
            candidates.append(info)

    if not candidates:
        raise RuntimeError(
            f"Output device '{name}' not found.\n"
            f"Use --list-devices to see available devices."
        )

    # Strongly prefer WASAPI devices (48kHz, lower latency)
    wasapi_candidates = [
        c for c in candidates
        if int(c.get("hostApi", -1)) == wasapi_host_api
    ]
    if wasapi_candidates:
        # Among WASAPI, prefer higher sample rate
        wasapi_candidates.sort(
            key=lambda c: int(c.get("defaultSampleRate", 0)), reverse=True
        )
        return wasapi_candidates[0]

    # Fallback: prefer higher sample rate
    candidates.sort(
        key=lambda c: int(c.get("defaultSampleRate", 0)), reverse=True
    )
    return candidates[0]


class AudioOutput:
    """
    Outputs pre-processed audio to a target audio device.
    
    Data in the ring buffer should already be at the correct sample rate
    and channel count (interleaved float32). The callback simply pulls
    and outputs without any processing.
    """

    def __init__(
        self,
        ring_buffer: RingBuffer,
        device_name: str = "Headphones",
    ):
        self._buffer = ring_buffer
        self._device_name = device_name
        self._pa: pyaudio.PyAudio | None = None
        self._stream = None
        self._running = False
        self._device_info: dict = {}
        self._native_sr: int = 0
        self._native_channels: int = 0

        # Pre-allocated silence + accumulation buffer
        self._accum = np.array([], dtype=np.float32)

    def start(self) -> dict:
        """Start outputting audio to the device."""
        self._pa = pyaudio.PyAudio()
        self._device_info = find_output_device(self._pa, self._device_name)

        self._native_sr = int(self._device_info["defaultSampleRate"])
        self._native_channels = min(2, int(self._device_info["maxOutputChannels"]))

        # ~32ms buffer
        frames_per_buffer = int(self._native_sr * 0.032)

        self._running = True
        self._stream = self._pa.open(
            format=pyaudio.paFloat32,
            channels=self._native_channels,
            rate=self._native_sr,
            output=True,
            output_device_index=int(self._device_info["index"]),
            frames_per_buffer=frames_per_buffer,
            stream_callback=self._callback,
        )
        self._stream.start_stream()

        return {
            "device_name": self._device_info.get("name", "Unknown"),
            "native_sr": self._native_sr,
            "channels": self._native_channels,
            "host_api": int(self._device_info.get("hostApi", -1)),
            "frames_per_buffer": frames_per_buffer,
        }

    def _callback(self, in_data, frame_count, time_info, status):
        """PyAudio output callback — just pulls pre-processed data. No resampling."""
        if not self._running:
            silence = np.zeros(frame_count * self._native_channels, dtype=np.float32)
            return (silence.tobytes(), pyaudio.paComplete)

        needed = frame_count * self._native_channels

        # Pull chunks from ring buffer
        while len(self._accum) < needed:
            chunk = self._buffer.pop(timeout=0)
            if chunk is None:
                break
            self._accum = np.concatenate([self._accum, chunk])

        if len(self._accum) >= needed:
            out_data = self._accum[:needed].astype(np.float32)
            self._accum = self._accum[needed:]
        else:
            # Not enough data — pad with silence (fade to avoid click)
            out_data = np.zeros(needed, dtype=np.float32)
            available = min(len(self._accum), needed)
            if available > 0:
                out_data[:available] = self._accum[:available]
                self._accum = self._accum[available:]

        return (out_data.tobytes(), pyaudio.paContinue)

    def stop(self):
        """Stop audio output."""
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

    @property
    def native_channels(self) -> int:
        return self._native_channels
