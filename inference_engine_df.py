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
    # Process 0.5s chunks (24000 samples at 48kHz)
    CHUNK_SIZE = 24000
    # 50% overlap = 0.25s (12000 samples)
    OVERLAP = 12000
    # Hop size = CHUNK_SIZE - OVERLAP
    HOP_SIZE = CHUNK_SIZE - OVERLAP  # 12000

    def __init__(self, post_filter: bool = False):
        # Lazy import to avoid loading torch if not needed
        from df.enhance import init_df

        # Initialize DeepFilterNet model (auto-downloads on first run)
        self._model, self._df_state, _ = init_df()
        self._model.eval()

        self._post_filter = post_filter

        # Internal accumulation buffer for streaming
        self._accum = np.array([], dtype=np.float32)

        # Overlap-add state
        self._prev_tail = None  # Last OVERLAP samples of previous enhanced chunk
        self._first_chunk = True

        # Pre-compute Hann crossfade windows
        self._fade_in = np.sin(np.linspace(0, np.pi / 2, self.OVERLAP, dtype=np.float32)) ** 2
        self._fade_out = np.cos(np.linspace(0, np.pi / 2, self.OVERLAP, dtype=np.float32)) ** 2

    def reset(self):
        """Reset all internal state (e.g., on audio discontinuity)."""
        self._accum = np.array([], dtype=np.float32)
        self._prev_tail = None
        self._first_chunk = True
        # Re-initialize model state
        from df.enhance import init_df
        self._model, self._df_state, _ = init_df()
        self._model.eval()

    def _enhance_chunk(self, audio_np: np.ndarray, strength: float = 1.0) -> np.ndarray:
        """
        Enhance a single chunk of audio using DeepFilterNet.

        Args:
            audio_np: 1D float32 array of mono audio at 48kHz.
            strength: Filter strength from 0.0 (bypass) to 1.0 (full).

        Returns:
            Enhanced 1D float32 array, same length as input.
        """
        from df.enhance import enhance

        # DeepFilterNet expects shape (channels, samples) as torch tensor
        audio_tensor = torch.from_numpy(audio_np).unsqueeze(0).float()

        # Map strength 0.0-1.0 to dB attenuation limit
        atten_lim_db = None
        if strength < 1.0:
            if strength <= 0.0:
                atten_lim_db = 0.0
            else:
                import math
                # lim = 10^(-db/20) => db = -20 * log10(lim)
                # DeepFilterNet uses lim = 1.0 - strength internally
                atten_lim_db = -20.0 * math.log10(1.0 - strength)

        # Enhance
        enhanced_tensor = enhance(
            self._model,
            self._df_state,
            audio_tensor,
            pad=True,
            atten_lim_db=atten_lim_db
        )

        # Convert back to numpy
        return enhanced_tensor.squeeze(0).numpy().astype(np.float32)

    def process_chunk(self, chunk: np.ndarray, strength: float = 1.0) -> np.ndarray:
        """
        Process an arbitrary-length audio chunk through DeepFilterNet.

        Accumulates input audio until we have CHUNK_SIZE samples (0.5s),
        then enhances with 50% overlap-add crossfade for seamless output.

        Args:
            chunk: Audio data, float32, mono, at 48kHz. Any length.
            strength: Filter strength from 0.0 to 1.0.

        Returns:
            Enhanced audio, float32. May be empty if still accumulating.
        """
        # Accumulate incoming audio
        self._accum = np.concatenate([self._accum, chunk])

        # Process when we have enough
        output_parts = []

        while len(self._accum) >= self.CHUNK_SIZE:
            # Extract one full chunk
            process_buf = self._accum[:self.CHUNK_SIZE].copy()
            # Advance by HOP_SIZE (not CHUNK_SIZE) to create overlap
            self._accum = self._accum[self.HOP_SIZE:]

            # Enhance this chunk (apply dry/wet internally)
            enhanced = self._enhance_chunk(process_buf, strength)

            if self._first_chunk:
                # First chunk: output everything except the tail (which overlaps with next)
                output_parts.append(enhanced[:self.HOP_SIZE])
                self._prev_tail = enhanced[self.HOP_SIZE:].copy()
                self._first_chunk = False
            else:
                # Crossfade: blend prev_tail with current chunk's head
                head = enhanced[:self.OVERLAP]
                blended = self._prev_tail * self._fade_out + head * self._fade_in
                # Output: blended region + middle (non-overlapping part)
                output_parts.append(blended)
                # The non-overlapping middle part (if any) - for 50% overlap there's none
                # Save tail for next overlap
                self._prev_tail = enhanced[self.HOP_SIZE:].copy()

        if output_parts:
            return np.concatenate(output_parts)
        else:
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
        """Duration of one full processing chunk in ms (500ms)."""
        return self.CHUNK_SIZE / self.SAMPLE_RATE * 1000.0
