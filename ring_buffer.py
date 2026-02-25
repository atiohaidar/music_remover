"""
Thread-safe ring buffer for real-time audio streaming.

Uses a deque with maxlen to prevent unbounded memory growth.
Designed for single-producer single-consumer (SPSC) pattern:
- Audio capture callback pushes chunks
- Processing thread pops chunks
"""

import threading
import numpy as np
from collections import deque
from typing import Optional


class RingBuffer:
    """
    Thread-safe ring buffer for audio chunks.
    
    Args:
        max_chunks: Maximum number of chunks to store.
                    Older chunks are dropped if buffer is full.
    """

    def __init__(self, max_chunks: int = 64):
        self._buffer: deque = deque(maxlen=max_chunks)
        self._lock = threading.Lock()
        self._event = threading.Event()
        self._max_chunks = max_chunks
        self._overflow_count = 0
        self._total_pushed = 0
        self._total_popped = 0

    def push(self, chunk: np.ndarray) -> None:
        """
        Push an audio chunk into the buffer.
        If buffer is full, the oldest chunk is silently dropped.
        
        Args:
            chunk: Audio data as numpy array (float32).
        """
        with self._lock:
            was_full = len(self._buffer) >= self._max_chunks
            self._buffer.append(chunk.copy())
            self._total_pushed += 1
            if was_full:
                self._overflow_count += 1
        self._event.set()

    def pop(self, timeout: float = 0.01) -> Optional[np.ndarray]:
        """
        Pop the oldest audio chunk from the buffer.
        
        Args:
            timeout: Maximum time to wait for data (seconds).
                     Set to 0 for non-blocking.
        
        Returns:
            Audio chunk as numpy array, or None if buffer is empty.
        """
        if timeout > 0:
            self._event.wait(timeout=timeout)

        with self._lock:
            if len(self._buffer) == 0:
                self._event.clear()
                return None
            chunk = self._buffer.popleft()
            self._total_popped += 1
            if len(self._buffer) == 0:
                self._event.clear()
            return chunk

    def clear(self) -> None:
        """Clear all chunks from the buffer."""
        with self._lock:
            self._buffer.clear()
            self._event.clear()

    @property
    def size(self) -> int:
        """Current number of chunks in the buffer."""
        with self._lock:
            return len(self._buffer)

    @property
    def is_empty(self) -> bool:
        """Check if buffer is empty."""
        with self._lock:
            return len(self._buffer) == 0

    @property
    def overflow_count(self) -> int:
        """Number of times the buffer overflowed (chunks dropped)."""
        return self._overflow_count

    @property
    def stats(self) -> dict:
        """Get buffer statistics."""
        with self._lock:
            return {
                "current_size": len(self._buffer),
                "max_size": self._max_chunks,
                "total_pushed": self._total_pushed,
                "total_popped": self._total_popped,
                "overflow_count": self._overflow_count,
                "utilization": len(self._buffer) / self._max_chunks,
            }
