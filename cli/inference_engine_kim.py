
"""
Kim Vocal 2 (MDX-Net) ONNX Inference Engine for real-time music removal.

Uses the Kim_Vocal_2.onnx model from the Ultimate Vocal Remover (UVR) ecosystem
to isolate vocals/speech and remove background music. Operates at 44.1 kHz.

Processing strategy:
  - Maintains a rolling context buffer of 261,120 samples (~5.92s) so the
    MDX-Net model always receives full temporal context (dim_t = 256 frames).
  - Accumulates incoming audio in configurable hops based on `chunk_duration_s`
    and `overlap_ratio` (default 15% raised-cosine crossfade).
  - Extracts the newest `chunk_size` samples from the separated vocal output
    and crossfades seamlessly with the previous chunk.
"""

import os
import numpy as np
import onnxruntime as ort
import torch


class KimVocalEngine:
    """
    Real-time MDX-Net ONNX inference engine (Kim_Vocal_2 / UVR-MDX-NET-Voc_FT).

    Args:
        model_dir: Directory containing the ONNX model file.
        chunk_duration_s: Processing chunk delay in seconds (default: 4.5).
            Larger values reduce CPU load and improve stability; smaller values
            reduce latency (requires fast CPU or GPU).
        overlap_ratio: Crossfade overlap ratio between consecutive chunks (0.0 - 0.4).
        num_threads: ONNX Runtime intra-op threads (0 = auto).
        model_variant: Model key ('kimvocal' or 'vocft').
    """

    MDX_MODELS = {
        "kimvocal": {
            "filename": "Kim_Vocal_2.onnx",
            "compensate": 1.009,
            "label": "Kim Vocal 2 (FP32)",
        },
        "kimvocal_fp16": {
            "filename": "Kim_Vocal_2_fp16.onnx",
            "compensate": 1.009,
            "label": "Kim Vocal 2 (FP16)",
        },
        "vocft": {
            "filename": "UVR-MDX-NET-Voc_FT.onnx",
            "compensate": 1.021,
            "label": "UVR-MDX-NET Voc_FT",
        },
    }

    SAMPLE_RATE = 44100
    N_FFT = 7680
    HOP_LENGTH = 1024
    DIM_F = 3072
    DIM_T = 256

    # Exact samples for DIM_T=256 STFT frames with center=True: (256 - 1) * 1024 = 261120 (~5.92s)
    CONTEXT_SAMPLES = (DIM_T - 1) * HOP_LENGTH

    def __init__(
        self,
        model_dir: str = "models",
        chunk_duration_s: float = 4.5,
        overlap_ratio: float = 0.15,
        num_threads: int = 0,
        model_variant: str = "kimvocal",
    ):
        if model_variant not in self.MDX_MODELS:
            raise ValueError(f"Unknown MDX model variant: {model_variant}")
        spec = self.MDX_MODELS[model_variant]
        self.MODEL_FILENAME = spec["filename"]
        self.COMPENSATE = spec["compensate"]
        self.MODEL_LABEL = spec["label"]

        model_path = os.path.join(model_dir, self.MODEL_FILENAME)
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model not found: {model_path}\n"
                f"Run: python download_model.py"
            )

        self._chunk_duration_s = max(0.5, float(chunk_duration_s))
        self._overlap_ratio = min(0.4, max(0.05, float(overlap_ratio)))

        self.CHUNK_SIZE = max(4096, int(self._chunk_duration_s * self.SAMPLE_RATE))
        self.OVERLAP = int(self.CHUNK_SIZE * self._overlap_ratio)
        self.HOP_SIZE = self.CHUNK_SIZE - self.OVERLAP

        # Configure ONNX Runtime session
        opts = ort.SessionOptions()
        if num_threads > 0:
            opts.intra_op_num_threads = num_threads
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.enable_cpu_mem_arena = True
        opts.enable_mem_pattern = True

        available = ort.get_available_providers()
        providers = [
            p for p in ("CUDAExecutionProvider", "DmlExecutionProvider", "CPUExecutionProvider")
            if p in available
        ]
        if not providers:
            providers = ["CPUExecutionProvider"]

        self._session = ort.InferenceSession(model_path, opts, providers=providers)
        self._input_name = self._session.get_inputs()[0].name
        self._active_provider = self._session.get_providers()[0]

        # Precompute Hann window for torch.stft / torch.istft
        self._stft_window = torch.hann_window(self.N_FFT)

        # Rolling context buffer (holds up to CONTEXT_SAMPLES of recent mono audio)
        self._context = np.zeros(self.CONTEXT_SAMPLES, dtype=np.float32)

        # Incoming accumulation buffer
        self._accum = np.array([], dtype=np.float32)

        # Previous tail for raised-cosine crossfade
        self._prev_tail: np.ndarray | None = None

        # Precompute raised-cosine fade curves (fade_in + fade_out == 1.0)
        if self.OVERLAP > 0:
            t = np.linspace(0.0, np.pi / 2.0, self.OVERLAP, dtype=np.float32)
            self._fade_in = np.sin(t) ** 2
            self._fade_out = np.cos(t) ** 2
        else:
            self._fade_in = np.array([], dtype=np.float32)
            self._fade_out = np.array([], dtype=np.float32)

        # AGC state
        self._current_gain = 1.0
        self._target_rms = 0.10

    def reset(self):
        """Reset all internal buffers and AGC state."""
        self._context[:] = 0.0
        self._accum = np.array([], dtype=np.float32)
        self._prev_tail = None
        self._current_gain = 1.0

    def _separate_context(self, context_mono: np.ndarray) -> np.ndarray:
        """
        Run Kim_Vocal_2.onnx on a CONTEXT_SAMPLES mono buffer.
        Returns separated vocal mono audio of length CONTEXT_SAMPLES.
        """
        wave = torch.from_numpy(context_mono).float()
        # STFT: shape (n_fft//2 + 1, dim_t) = (3841, 256)
        spec = torch.stft(
            wave,
            n_fft=self.N_FFT,
            hop_length=self.HOP_LENGTH,
            window=self._stft_window,
            center=True,
            return_complex=True,
        )
        # Keep only DIM_F bins and exact DIM_T frames
        spec_trim = spec[: self.DIM_F, : self.DIM_T]
        real_np = spec_trim.real.numpy()
        imag_np = spec_trim.imag.numpy()

        # Duplicate mono into L/R stereo planes: [1, 4, DIM_F, DIM_T]
        tensor_in = np.stack([real_np, imag_np, real_np, imag_np], axis=0)[None, ...].astype(np.float32)

        out = self._session.run(None, {self._input_name: tensor_in})[0][0]

        # Average L and R predictions back to mono complex spectrogram
        pred_real = 0.5 * (out[0] + out[2])
        pred_imag = 0.5 * (out[1] + out[3])
        pred_complex = torch.complex(torch.from_numpy(pred_real), torch.from_numpy(pred_imag))

        # Pad frequency bins back to n_fft // 2 + 1
        full_bins = self.N_FFT // 2 + 1
        full_spec = torch.zeros((full_bins, self.DIM_T), dtype=torch.complex64)
        full_spec[: self.DIM_F, :] = pred_complex

        # Inverse STFT back to time domain
        vocal = torch.istft(
            full_spec,
            n_fft=self.N_FFT,
            hop_length=self.HOP_LENGTH,
            window=self._stft_window,
            center=True,
            length=self.CONTEXT_SAMPLES,
        )
        return (vocal.numpy() * self.COMPENSATE).astype(np.float32)

    def process_chunk(self, chunk: np.ndarray, strength: float = 1.0) -> np.ndarray:
        """
        Accumulate incoming mono 44.1kHz audio and run vocal separation
        whenever HOP_SIZE new samples are ready.
        """
        if len(chunk) == 0:
            return np.array([], dtype=np.float32)

        self._accum = np.concatenate([self._accum, chunk.astype(np.float32, copy=False)])
        output_parts = []

        while len(self._accum) >= self.HOP_SIZE:
            new_hop = self._accum[: self.HOP_SIZE]
            self._accum = self._accum[self.HOP_SIZE :]

            # Shift rolling context buffer and append new hop
            if self.HOP_SIZE >= self.CONTEXT_SAMPLES:
                self._context[:] = new_hop[-self.CONTEXT_SAMPLES :]
            else:
                self._context[: -self.HOP_SIZE] = self._context[self.HOP_SIZE :]
                self._context[-self.HOP_SIZE :] = new_hop

            # Separate vocals (unless bypassed)
            if strength <= 0.0:
                vocal_chunk = self._context[-self.CHUNK_SIZE :].copy()
            else:
                sep_full = self._separate_context(self._context)
                vocal_chunk = sep_full[-self.CHUNK_SIZE :]
                if strength < 1.0:
                    orig_chunk = self._context[-self.CHUNK_SIZE :]
                    vocal_chunk = orig_chunk * (1.0 - strength) + vocal_chunk * strength

            # Crossfade head with previous tail, emit HOP_SIZE samples
            if self.OVERLAP > 0:
                if self._prev_tail is not None:
                    head = (
                        self._prev_tail * self._fade_out
                        + vocal_chunk[: self.OVERLAP] * self._fade_in
                    )
                    emitted = np.concatenate([head, vocal_chunk[self.OVERLAP : self.HOP_SIZE]])
                else:
                    emitted = vocal_chunk[: self.HOP_SIZE].copy()
                self._prev_tail = vocal_chunk[self.HOP_SIZE :].copy()
            else:
                emitted = vocal_chunk

            output_parts.append(emitted)

        if not output_parts:
            return np.array([], dtype=np.float32)

        out = np.concatenate(output_parts)

        # Gentle AGC + limiter
        if len(out) > 0:
            rms = float(np.sqrt(np.mean(out ** 2)))
            if rms > 0.005:
                target_gain = np.clip(self._target_rms / (rms + 1e-6), 0.3, 4.0)
                alpha = 0.05 if target_gain > self._current_gain else 0.01
                self._current_gain = (1.0 - alpha) * self._current_gain + alpha * target_gain
            out *= self._current_gain
            out = np.clip(out, -0.98, 0.98)

        return out.astype(np.float32)

    @property
    def frame_size(self) -> int:
        return self.HOP_SIZE

    @property
    def block_size(self) -> int:
        return self.CHUNK_SIZE

    @property
    def sample_rate(self) -> int:
        return self.SAMPLE_RATE

    @property
    def shift_duration_ms(self) -> float:
        return self.HOP_SIZE / self.SAMPLE_RATE * 1000.0

    @property
    def frame_duration_ms(self) -> float:
        return self.CHUNK_SIZE / self.SAMPLE_RATE * 1000.0

    @property
    def provider(self) -> str:
        return self._active_provider
