"""
DeepFilterNet Inference Engine for real-time speech enhancement.

Uses the DeepFilterNet3 model via the `df` Python package for significantly
better noise/music suppression than DTLN. Operates natively at 48kHz,
eliminating the need for resampling in the audio pipeline.

Processing strategy:
  - Accumulates incoming audio into large chunks (~0.5s) for quality.
  - Uses 50% overlap between consecutive chunks with Hann crossfade
    to eliminate boundary artifacts (stuttering/clicking).
  - DeepFilterNet gets enough context per chunk for clean output.
"""

import numpy as np
import torch


class DeepFilterNetEngine:
    """
    Real-time DeepFilterNet inference engine with overlap-add.

    Wraps `df.enhance.enhance()` and `df.enhance.init_df()` with the same
    interface as DTLNInferenceEngine for drop-in replacement.

    Uses 50% overlap between consecutive enhanced chunks with Hann window
    crossfade to ensure smooth, artifact-free transitions.

    Args:
        post_filter: Enable post-filter for slightly more aggressive noise removal.
    """

    SAMPLE_RATE = 48000
    
    # === SETTING DELAY (Ubah di sini saja) ===
    # 0.5  = Delay rendah, akurasi ok
    # 1.0  = Delay sedang, akurasi bagus
    # 2.0  = Delay lama, akurasi sangat tinggi (vokal lebih stabil)
    CHUNK_DURATION_S = 5.0
    
    CHUNK_SIZE = int(CHUNK_DURATION_S * SAMPLE_RATE)
    # Gunakan 75% overlap untuk kualitas transisi terbaik
    OVERLAP = int(CHUNK_SIZE * 0.75) 
    HOP_SIZE = CHUNK_SIZE - OVERLAP

    def __init__(self, post_filter: bool = False, chunk_duration_s: float | None = None):
        # Lazy import to avoid loading torch if not needed
        from df.enhance import init_df

        self._post_filter = post_filter
        if chunk_duration_s is not None:
            self.CHUNK_DURATION_S = max(0.5, float(chunk_duration_s))
            self.CHUNK_SIZE = int(self.CHUNK_DURATION_S * self.SAMPLE_RATE)
            self.OVERLAP = int(self.CHUNK_SIZE * 0.75)
            self.HOP_SIZE = self.CHUNK_SIZE - self.OVERLAP

        # Initialize DeepFilterNet model (auto-downloads on first run)
        self._model, self._df_state, _ = init_df(post_filter=self._post_filter)
        self._model.eval()

        # Internal accumulation buffer for streaming
        self._accum = np.array([], dtype=np.float32)

        # Overlap-add state: buffer of size CHUNK_SIZE
        self._ola_buffer = np.zeros(self.CHUNK_SIZE, dtype=np.float32)
        # Window for OLA (Hann window is COLA compliant if hop = 1/4 or 1/2 size)
        self._window = np.hanning(self.CHUNK_SIZE).astype(np.float32)
        
        # Scaling factor for COLA (Total weight at any point should be 1.0)
        # For Hann window with 75% overlap (hop = size/4), sum is 1.5. 
        # For 50% overlap (hop = size/2), sum is 1.0.
        # Scale = 1.0 / (sum of windows at hop intervals)
        self._ola_scale = 0.6666  # Approximated for 75% overlap with Hann

    def reset(self):
        """Reset all internal state (e.g., on audio discontinuity)."""
        self._accum = np.array([], dtype=np.float32)
        self._ola_buffer = np.zeros(self.CHUNK_SIZE, dtype=np.float32)
        # Re-initialize model state
        from df.enhance import init_df
        self._model, self._df_state, _ = init_df(post_filter=self._post_filter)
        self._model.eval()

    def _enhance_chunk(self, audio_np: np.ndarray, strength: float = 1.0) -> np.ndarray:
        from df.enhance import enhance
        
        # --- Pre-processing: Light Compression & Strong HPF ---
        # 1. Soft Compressor (Softer ratio to prevent vocal disappearing)
        def soft_compressor(data, threshold=0.3, ratio=2.0):
            abs_data = np.abs(data)
            mask = abs_data > threshold
            compressed = np.copy(data)
            compressed[mask] = np.sign(data[mask]) * (threshold + (abs_data[mask] - threshold) / ratio)
            return compressed

        # 2. High-Pass Filter (Raised to 120Hz to KILL Bass Guitar)
        from scipy.signal import butter, lfilter
        def butter_highpass(cutoff, fs, order=5):
            nyq = 0.5 * fs
            normal_cutoff = cutoff / nyq
            b, a = butter(order, normal_cutoff, btype='high', analog=False)
            return b, a

        # Apply process: Compress peaks slightly then High-pass
        audio_proc = soft_compressor(audio_np)
        b, a = butter_highpass(120, self.SAMPLE_RATE, order=4)
        audio_passed = lfilter(b, a, audio_proc)
        
        audio_tensor = torch.from_numpy(audio_passed).unsqueeze(0).float()

        # 3. Stronger Attenuation Limit (60dB = Much cleaner music removal)
        import math
        max_suppression = 60.0 
        if strength >= 1.0:
            atten_lim_db = max_suppression
        elif strength <= 0.0:
            atten_lim_db = 0.0
        else:
            atten_lim_db = strength * max_suppression

        enhanced_tensor = enhance(self._model, self._df_state, audio_tensor, pad=True, atten_lim_db=atten_lim_db)
        return enhanced_tensor.squeeze(0).numpy().astype(np.float32)

    def process_chunk(self, chunk: np.ndarray, strength: float = 1.0) -> np.ndarray:
        """
        Process with 75% overlap-add.
        Each CHUNK_SIZE (500ms) output is windowed and added to OLA buffer.
        We then pop HOP_SIZE (125ms) of finalized audio.
        """
        self._accum = np.concatenate([self._accum, chunk])
        output_parts = []

        while len(self._accum) >= self.CHUNK_SIZE:
            process_buf = self._accum[:self.CHUNK_SIZE].copy()
            self._accum = self._accum[self.HOP_SIZE:]

            enhanced = self._enhance_chunk(process_buf, strength)
            
            # Apply Window and Add to OLA buffer
            self._ola_buffer += enhanced * self._window * self._ola_scale

            # Extract finalized hop
            final_hop = self._ola_buffer[:self.HOP_SIZE].copy()
            output_parts.append(final_hop)

            # Shift OLA buffer
            self._ola_buffer[: -self.HOP_SIZE] = self._ola_buffer[self.HOP_SIZE :]
            self._ola_buffer[-self.HOP_SIZE :] = 0

        if output_parts:
            out = np.concatenate(output_parts)
            # --- AGC Logic ---
            if len(out) > 0:
                rms = np.sqrt(np.mean(out**2))
                if not hasattr(self, '_current_gain'): self._current_gain = 1.0
                if not hasattr(self, '_target_rms'): self._target_rms = 0.10
                if rms > 0.005: 
                    target_gain = self._target_rms / (rms + 1e-6)
                    target_gain = np.clip(target_gain, 0.3, 4.0)
                    alpha = 0.05 if target_gain > self._current_gain else 0.01 
                    self._current_gain = (1 - alpha) * self._current_gain + alpha * target_gain
                out *= self._current_gain
                out = np.clip(out, -0.98, 0.98) 
            return out
        return np.array([], dtype=np.float32)

    @property
    def frame_size(self) -> int:
        """Effective output samples per process cycle (HOP_SIZE = 12000 = 250ms)."""
        return self.HOP_SIZE

    @property
    def block_size(self) -> int:
        """Full chunk size for enhancement (24000 = 500ms)."""
        return self.CHUNK_SIZE

    @property
    def sample_rate(self) -> int:
        """Model sample rate (48000)."""
        return self.SAMPLE_RATE

    @property
    def shift_duration_ms(self) -> float:
        """Duration of one output hop in ms (250ms)."""
        return self.HOP_SIZE / self.SAMPLE_RATE * 1000.0

    @property
    def frame_duration_ms(self) -> float:
        """Duration of one full processing chunk in ms (2000ms)."""
        return self.CHUNK_SIZE / self.SAMPLE_RATE * 1000.0
