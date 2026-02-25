"""
Audio resampler for converting between system audio sample rates and model sample rate.

DTLN model operates at 16 kHz. System audio is typically 44100 or 48000 Hz.
Optimized with fast paths for common integer ratios (48k↔16k = 3:1).
"""

import numpy as np
from math import gcd
from scipy.signal import resample_poly


# Pre-computed resampling ratios for common sample rates
_RATIO_CACHE: dict[tuple[int, int], tuple[int, int]] = {}


def _get_ratio(from_sr: int, to_sr: int) -> tuple[int, int]:
    """Get simplified up/down ratio for resampling."""
    key = (from_sr, to_sr)
    if key not in _RATIO_CACHE:
        g = gcd(from_sr, to_sr)
        _RATIO_CACHE[key] = (to_sr // g, from_sr // g)
    return _RATIO_CACHE[key]


def resample(audio: np.ndarray, from_sr: int, to_sr: int) -> np.ndarray:
    """
    Resample audio from one sample rate to another.
    Uses optimized fast paths for common integer ratios.
    """
    if from_sr == to_sr:
        return audio

    up, down = _get_ratio(from_sr, to_sr)
    resampled = resample_poly(audio, up, down).astype(np.float32)
    return resampled


def resample_to_16k(audio: np.ndarray, orig_sr: int) -> np.ndarray:
    """Downsample audio to 16 kHz for DTLN model input."""
    return resample(audio, orig_sr, 16000)


def resample_from_16k(audio: np.ndarray, target_sr: int) -> np.ndarray:
    """Upsample audio from 16 kHz to target sample rate."""
    return resample(audio, 16000, target_sr)


def stereo_to_mono(audio: np.ndarray) -> np.ndarray:
    """Convert stereo audio to mono by averaging channels."""
    if audio.ndim == 1:
        return audio.astype(np.float32)

    if audio.ndim == 2:
        if audio.shape[1] <= 8:  # (samples, channels)
            return audio.mean(axis=1).astype(np.float32)
        else:  # (channels, samples)
            return audio.mean(axis=0).astype(np.float32)

    return audio.flatten().astype(np.float32)


def mono_to_stereo(audio: np.ndarray) -> np.ndarray:
    """Convert mono audio to stereo by duplicating the channel."""
    mono = audio.astype(np.float32).flatten()
    return np.column_stack([mono, mono])


def calculate_chunk_samples(chunk_ms: float, sample_rate: int) -> int:
    """Calculate number of samples for a given chunk duration."""
    return int(sample_rate * chunk_ms / 1000.0)
