"""
DTLN ONNX Inference Engine for real-time speech enhancement.

Implementation based on the reference from breizhn/DTLN:
  https://github.com/breizhn/DTLN/blob/master/real_time_processing_onnx.py

The DTLN model consists of two stages:
  - Model 1: Takes FFT magnitude → outputs a mask → masked STFT
  - Model 2: Takes time-domain estimated block → outputs enhanced block

Both models are stateful (hidden states passed as inputs/outputs).

Processing uses overlap-add:
  - block_len = 512 samples (32ms at 16kHz)
  - block_shift = 128 samples (8ms at 16kHz)
  - Each call to process_shift() ingests 128 new samples and outputs 128 enhanced samples
"""

import os
import numpy as np
import onnxruntime as ort


class DTLNInferenceEngine:
    """
    Real-time DTLN inference engine using ONNX Runtime.
    
    Follows the exact reference implementation from breizhn/DTLN.
    Processes audio in shifts of 128 samples (8ms at 16kHz),
    using a sliding window of 512 samples with overlap-add.
    
    Args:
        model_dir: Directory containing model_1.onnx and model_2.onnx.
        use_quantized: If True, use INT8 quantized models (*_int8.onnx).
        num_threads: Number of threads for ONNX Runtime inference.
    """

    BLOCK_LEN = 512       # Samples per FFT frame (32ms at 16kHz)
    BLOCK_SHIFT = 128     # Hop size / shift per step (8ms at 16kHz)
    SAMPLE_RATE = 16000   # Model sample rate

    def __init__(
        self,
        model_dir: str = "models",
        use_quantized: bool = False,
        num_threads: int = 1,
    ):
        suffix = "_int8" if use_quantized else ""
        model1_path = os.path.join(model_dir, f"dtln_1{suffix}.onnx")
        model2_path = os.path.join(model_dir, f"dtln_2{suffix}.onnx")

        for p in [model1_path, model2_path]:
            if not os.path.exists(p):
                raise FileNotFoundError(
                    f"Model not found: {p}\n"
                    f"Run: python download_model.py"
                )

        # Configure session options for low latency
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = num_threads
        opts.inter_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.enable_cpu_mem_arena = True
        opts.enable_mem_pattern = True

        # Load ONNX models
        self._session1 = ort.InferenceSession(model1_path, opts)
        self._session2 = ort.InferenceSession(model2_path, opts)

        # Get input names for both models
        self._input_names_1 = [inp.name for inp in self._session1.get_inputs()]
        self._input_names_2 = [inp.name for inp in self._session2.get_inputs()]

        # Pre-allocate model inputs from model metadata (zeros)
        self._model_inputs_1 = {
            inp.name: np.zeros(
                [dim if isinstance(dim, int) else 1 for dim in inp.shape],
                dtype=np.float32,
            )
            for inp in self._session1.get_inputs()
        }
        self._model_inputs_2 = {
            inp.name: np.zeros(
                [dim if isinstance(dim, int) else 1 for dim in inp.shape],
                dtype=np.float32,
            )
            for inp in self._session2.get_inputs()
        }

        # Sliding buffers for overlap-add
        self._in_buffer = np.zeros(self.BLOCK_LEN, dtype=np.float32)
        self._out_buffer = np.zeros(self.BLOCK_LEN, dtype=np.float32)

    def reset(self):
        """Reset all states and buffers (e.g., on audio discontinuity)."""
        # Re-zero all model input states
        self._model_inputs_1 = {
            inp.name: np.zeros(
                [dim if isinstance(dim, int) else 1 for dim in inp.shape],
                dtype=np.float32,
            )
            for inp in self._session1.get_inputs()
        }
        self._model_inputs_2 = {
            inp.name: np.zeros(
                [dim if isinstance(dim, int) else 1 for dim in inp.shape],
                dtype=np.float32,
            )
            for inp in self._session2.get_inputs()
        }
        self._in_buffer[:] = 0
        self._out_buffer[:] = 0

    def process_shift(self, new_samples: np.ndarray) -> np.ndarray:
        """
        Process one shift of audio (128 samples) through the DTLN pipeline.
        
        This is the core method matching the reference implementation:
        1. Shift input buffer and append new samples
        2. FFT → magnitude + phase
        3. Model 1: magnitude → mask
        4. IFFT with masked magnitude + original phase
        5. Model 2: time-domain → enhanced time-domain
        6. Overlap-add to output buffer
        7. Return the oldest 128 samples
        
        Args:
            new_samples: 128 new audio samples (float32).
        
        Returns:
            128 enhanced audio samples (float32).
        """
        assert len(new_samples) == self.BLOCK_SHIFT, (
            f"Expected {self.BLOCK_SHIFT} samples, got {len(new_samples)}"
        )

        # --- Shift input buffer and append new samples ---
        self._in_buffer[:-self.BLOCK_SHIFT] = self._in_buffer[self.BLOCK_SHIFT:]
        self._in_buffer[-self.BLOCK_SHIFT:] = new_samples

        # --- FFT ---
        in_block_fft = np.fft.rfft(self._in_buffer)
        in_mag = np.abs(in_block_fft).astype(np.float32)
        in_phase = np.angle(in_block_fft)

        # --- Model 1: Magnitude masking ---
        in_mag_reshaped = in_mag.reshape(1, 1, -1)
        self._model_inputs_1[self._input_names_1[0]] = in_mag_reshaped
        
        model_outputs_1 = self._session1.run(None, self._model_inputs_1)
        
        out_mask = model_outputs_1[0]
        # Update states for model 1 (index 1 onwards are states)
        self._model_inputs_1[self._input_names_1[1]] = model_outputs_1[1]

        # --- IFFT with mask ---
        estimated_complex = in_mag * out_mask * np.exp(1j * in_phase)
        estimated_block = np.fft.irfft(estimated_complex)

        # --- Model 2: Time-domain enhancement ---
        estimated_block_reshaped = estimated_block.reshape(1, 1, -1).astype(np.float32)
        self._model_inputs_2[self._input_names_2[0]] = estimated_block_reshaped
        
        model_outputs_2 = self._session2.run(None, self._model_inputs_2)
        
        out_block = model_outputs_2[0]
        # Update states for model 2
        self._model_inputs_2[self._input_names_2[1]] = model_outputs_2[1]

        # --- Overlap-add ---
        self._out_buffer[:-self.BLOCK_SHIFT] = self._out_buffer[self.BLOCK_SHIFT:]
        self._out_buffer[-self.BLOCK_SHIFT:] = 0.0
        self._out_buffer += np.squeeze(out_block)

        # Return the oldest shift
        return self._out_buffer[:self.BLOCK_SHIFT].copy()

    def process_chunk(self, chunk: np.ndarray) -> np.ndarray:
        """
        Process an arbitrary-length audio chunk by splitting into shifts.
        
        Args:
            chunk: Audio data, float32, length should be a multiple of BLOCK_SHIFT (128).
        
        Returns:
            Enhanced audio, same length as input.
        """
        n_samples = len(chunk)
        output = np.zeros(n_samples, dtype=np.float32)

        i = 0
        while i + self.BLOCK_SHIFT <= n_samples:
            shift = chunk[i : i + self.BLOCK_SHIFT]
            enhanced = self.process_shift(shift)
            output[i : i + self.BLOCK_SHIFT] = enhanced
            i += self.BLOCK_SHIFT

        # Handle remaining samples (pad, process, trim)
        if i < n_samples:
            remaining = n_samples - i
            padded = np.zeros(self.BLOCK_SHIFT, dtype=np.float32)
            padded[:remaining] = chunk[i:]
            enhanced = self.process_shift(padded)
            output[i:] = enhanced[:remaining]

        return output

    @property
    def frame_size(self) -> int:
        """Number of samples per processing shift (128)."""
        return self.BLOCK_SHIFT

    @property
    def block_size(self) -> int:
        """Number of samples per FFT block (512)."""
        return self.BLOCK_LEN

    @property
    def sample_rate(self) -> int:
        """Model sample rate (16000)."""
        return self.SAMPLE_RATE

    @property
    def shift_duration_ms(self) -> float:
        """Duration of one processing shift in ms (8ms)."""
        return self.BLOCK_SHIFT / self.SAMPLE_RATE * 1000.0

    @property
    def frame_duration_ms(self) -> float:
        """Duration of one FFT frame in ms (32ms)."""
        return self.BLOCK_LEN / self.SAMPLE_RATE * 1000.0
