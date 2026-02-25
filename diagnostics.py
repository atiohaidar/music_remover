"""
Diagnostics module for real-time latency measurement and performance monitoring.

Tracks per-frame inference time, buffer utilization, and estimated end-to-end latency.
"""

import time
import threading
import numpy as np
from collections import deque

from ring_buffer import RingBuffer


class Diagnostics:
    """
    Performance diagnostics for the audio processing pipeline.
    
    Args:
        input_buffer: Capture ring buffer.
        output_buffer: Output ring buffer.
        model_sr: Model sample rate (16000).
        frame_size: Samples per frame (512).
        print_interval: Seconds between diagnostic prints.
    """

    def __init__(
        self,
        input_buffer: RingBuffer,
        output_buffer: RingBuffer,
        model_sr: int = 16000,
        frame_size: int = 512,
        print_interval: float = 3.0,
    ):
        self._input_buf = input_buffer
        self._output_buf = output_buffer
        self._model_sr = model_sr
        self._frame_size = frame_size
        self._print_interval = print_interval

        # Inference timing
        self._inference_times: deque = deque(maxlen=200)
        self._frame_duration_ms = frame_size / model_sr * 1000.0

        # Thread for periodic printing
        self._running = False
        self._thread: threading.Thread | None = None

    def record_inference_time(self, duration_sec: float):
        """Record the time taken for one inference frame."""
        self._inference_times.append(duration_sec * 1000.0)  # Convert to ms

    def get_stats(self) -> dict:
        """Get current diagnostic statistics."""
        inf_times = list(self._inference_times)
        
        if inf_times:
            avg_inf = np.mean(inf_times)
            max_inf = np.max(inf_times)
            p95_inf = np.percentile(inf_times, 95)
        else:
            avg_inf = max_inf = p95_inf = 0.0

        input_stats = self._input_buf.stats
        output_stats = self._output_buf.stats

        # Estimate total latency
        input_latency = input_stats["current_size"] * self._frame_duration_ms
        output_latency = output_stats["current_size"] * self._frame_duration_ms
        total_latency = input_latency + avg_inf + output_latency

        # Real-time factor (must be < 1.0 to keep up)
        rtf = avg_inf / self._frame_duration_ms if self._frame_duration_ms > 0 else 0

        return {
            "inference_avg_ms": round(avg_inf, 2),
            "inference_max_ms": round(max_inf, 2),
            "inference_p95_ms": round(p95_inf, 2),
            "input_buffer_fill": input_stats["current_size"],
            "output_buffer_fill": output_stats["current_size"],
            "input_overflow": input_stats["overflow_count"],
            "estimated_latency_ms": round(total_latency, 1),
            "real_time_factor": round(rtf, 4),
            "frame_duration_ms": self._frame_duration_ms,
        }

    def _print_loop(self):
        """Background thread that prints diagnostics periodically."""
        while self._running:
            time.sleep(self._print_interval)
            if not self._running:
                break
            
            stats = self.get_stats()
            print(
                f"\r[DIAG] "
                f"Latency: {stats['estimated_latency_ms']:>6.1f}ms | "
                f"Inf: {stats['inference_avg_ms']:>5.2f}ms (p95: {stats['inference_p95_ms']:>5.2f}ms) | "
                f"RTF: {stats['real_time_factor']:.3f} | "
                f"Buf: {stats['input_buffer_fill']:>2d}/{stats['output_buffer_fill']:>2d} | "
                f"Overflow: {stats['input_overflow']}",
                end="", flush=True,
            )

    def start(self):
        """Start periodic diagnostic printing in background."""
        self._running = True
        self._thread = threading.Thread(target=self._print_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop diagnostic printing."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None

    def print_summary(self):
        """Print a final summary of diagnostics."""
        stats = self.get_stats()
        print("\n")
        print("=" * 60)
        print("Performance Summary")
        print("=" * 60)
        print(f"  Avg inference time:   {stats['inference_avg_ms']:.2f} ms")
        print(f"  P95 inference time:   {stats['inference_p95_ms']:.2f} ms")
        print(f"  Max inference time:   {stats['inference_max_ms']:.2f} ms")
        print(f"  Real-time factor:     {stats['real_time_factor']:.4f}")
        print(f"  Frame duration:       {stats['frame_duration_ms']:.1f} ms")
        print(f"  Estimated latency:    {stats['estimated_latency_ms']:.1f} ms")
        print(f"  Input overflows:      {stats['input_overflow']}")
        print(f"  Latency target:       {'PASS ✓' if stats['estimated_latency_ms'] < 150 else 'FAIL ✗'} (< 150ms)")
        print("=" * 60)
