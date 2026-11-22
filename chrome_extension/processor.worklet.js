// AudioWorkletProcessor running on the real-time Web Audio thread.
// Supports both:
//   - 'gtcrn': 16kHz, 256-sample hops (16ms), biquad anti-aliasing filter + 48ms pre-buffer
//   - 'mdx':   44.1kHz (Kim_Vocal_2 / UVR-MDX-NET-Voc_FT), configurable hop size + 1-hop pre-buffer

class MusicRemoverWorkletProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super();
    const opts = options?.processorOptions || {};
    this.mode = opts.mode || 'mdx'; // 'gtcrn' | 'mdx'
    this.strength = opts.strength ?? 1.0;

    if (this.mode === 'gtcrn') {
      this.nativeSr = sampleRate;
      this.modelSr = 16000;
      this.ratio = this.nativeSr / this.modelSr;
      this.hopSamples = 256; // 16ms at 16kHz
      this.prebufferSamples = Math.round(this.nativeSr * 0.048); // 48ms jitter buffer
      this.outRingSize = 32768;

      // 2nd-order Butterworth Low-Pass at 7.5 kHz to prevent aliasing when downsampling to 16kHz
      const fc = 7500;
      const k = Math.tan((Math.PI * fc) / this.nativeSr);
      const norm = 1.0 / (1.0 + Math.SQRT2 * k + k * k);
      this.b0 = k * k * norm;
      this.b1 = 2.0 * this.b0;
      this.b2 = this.b0;
      this.a1 = 2.0 * (k * k - 1.0) * norm;
      this.a2 = (1.0 - Math.SQRT2 * k + k * k) * norm;
      this.lpX1 = 0; this.lpX2 = 0;
      this.lpY1 = 0; this.lpY2 = 0;

      this.inPhase = 0;
      this.prevSample = 0;
      this.lastOut16k = 0;
    } else {
      // MDX-Net runs natively at 44100 Hz (AudioContext created at 44100 Hz)
      this.hopSamples = opts.hopSamples || 168683;
      this.prebufferSamples = Math.max(4096, Math.floor(this.hopSamples * 0.5));
      this.outRingSize = Math.max(661500, this.hopSamples * 4); // >= 15s capacity at 44.1kHz
    }

    this.inBuf = new Float32Array(this.hopSamples);
    this.inPos = 0;

    this.outRing = new Float32Array(this.outRingSize);
    this.outRead = 0;
    this.outWrite = 0;
    this.outCount = 0;
    this.playbackStarted = false;

    // Aligned dry ring buffer for smooth dry/wet mix
    this.dryRing = new Float32Array(this.outRingSize);
    this.dryWrite = 0;

    this.port.onmessage = (e) => {
      const msg = e.data;
      if (msg.type === 'ENHANCED_HOP') {
        if (this.mode === 'gtcrn') {
          this._pushUpsampled(msg.samples);
        } else {
          this._pushDirect(msg.samples);
        }
        if (!this.playbackStarted && this.outCount >= this.prebufferSamples) {
          this.playbackStarted = true;
        }
      } else if (msg.type === 'SET_STRENGTH') {
        this.strength = msg.strength;
      }
    };
  }

  _pushDirect(samples) {
    const len = samples.length;
    for (let i = 0; i < len; i++) {
      if (this.outCount < this.outRingSize) {
        this.outRing[this.outWrite] = samples[i];
        this.outWrite = (this.outWrite + 1) % this.outRingSize;
        this.outCount++;
      } else {
        // Overwrite oldest if ring overflows
        this.outRing[this.outWrite] = samples[i];
        this.outWrite = (this.outWrite + 1) % this.outRingSize;
        this.outRead = (this.outRead + 1) % this.outRingSize;
      }
    }
  }

  _pushUpsampled(samples16k) {
    const step = this.modelSr / this.nativeSr;
    let phase = 0;
    const len = samples16k.length;
    while (phase < len) {
      const idx = Math.floor(phase);
      const frac = phase - idx;
      const s0 = idx === 0 ? this.lastOut16k : samples16k[idx - 1];
      const s1 = samples16k[idx];
      const val = s0 + (s1 - s0) * frac;

      if (this.outCount < this.outRingSize) {
        this.outRing[this.outWrite] = val;
        this.outWrite = (this.outWrite + 1) % this.outRingSize;
        this.outCount++;
      }
      phase += step;
    }
    this.lastOut16k = samples16k[len - 1];
  }

  process(inputs, outputs) {
    const input = inputs[0];
    const output = outputs[0];
    if (!input || input.length === 0 || !output || output.length === 0) {
      return true;
    }

    const ch0 = input[0];
    const ch1 = input.length > 1 ? input[1] : ch0;
    const n = ch0.length;

    // 1. Capture input into hop buffer
    if (this.mode === 'gtcrn') {
      for (let i = 0; i < n; i++) {
        const mono = 0.5 * (ch0[i] + ch1[i]);
        // Low-pass filter before decimation
        const filtered =
          this.b0 * mono + this.b1 * this.lpX1 + this.b2 * this.lpX2 -
          this.a1 * this.lpY1 - this.a2 * this.lpY2;
        this.lpX2 = this.lpX1; this.lpX1 = mono;
        this.lpY2 = this.lpY1; this.lpY1 = filtered;

        this.inPhase += 1;
        if (this.inPhase >= this.ratio) {
          this.inPhase -= this.ratio;
          this.inBuf[this.inPos++] = filtered;
          if (this.inPos >= this.hopSamples) {
            const copy = this.inBuf.slice();
            this.port.postMessage({ type: 'INPUT_HOP', samples: copy }, [copy.buffer]);
            this.inPos = 0;
          }
        }
      }
    } else {
      for (let i = 0; i < n; i++) {
        const mono = 0.5 * (ch0[i] + ch1[i]);
        this.inBuf[this.inPos++] = mono;
        if (this.inPos >= this.hopSamples) {
          const copy = this.inBuf.slice();
          this.port.postMessage({ type: 'INPUT_HOP', samples: copy }, [copy.buffer]);
          this.inPos = 0;
        }
      }
    }

    // 2. Output playback
    const outL = output[0];
    const outR = output.length > 1 ? output[1] : null;
    const wet = this.strength;
    const dry = 1.0 - wet;

    for (let i = 0; i < n; i++) {
      const orig = 0.5 * (ch0[i] + ch1[i]);
      let sample = 0.0;

      if (wet <= 0.0) {
        sample = orig;
        if (this.playbackStarted && this.outCount > 0) {
          this.outRead = (this.outRead + 1) % this.outRingSize;
          this.outCount--;
        }
      } else if (this.playbackStarted && this.outCount > 0) {
        const enh = this.outRing[this.outRead];
        this.outRead = (this.outRead + 1) % this.outRingSize;
        this.outCount--;
        sample = orig * dry + enh * wet;
      }

      const clamped = Math.max(-0.98, Math.min(0.98, sample));
      outL[i] = clamped;
      if (outR) outR[i] = clamped;
    }

    return true;
  }
}

registerProcessor('music-remover-worklet-processor', MusicRemoverWorkletProcessor);
