// Dedicated Web Worker for 100% in-browser ONNX audio separation
// Supports:
//   - 'kimvocal': Kim_Vocal_2.onnx (44.1kHz Stereo MDX-Net, compensate=1.009)
//   - 'vocft':    UVR-MDX-NET-Voc_FT.onnx (44.1kHz Stereo MDX-Net, compensate=1.021)
//   - 'gtcrn':    gtcrn_simple.onnx (16kHz Mono ultra-fast denoiser)

const ORT_CDN = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/';
importScripts(`${ORT_CDN}ort.all.min.js`);

const MODEL_SPECS = {
  gtcrn: {
    localRelPath: 'models/gtcrn_simple.onnx',
    cacheKey: null, // local same-origin (523 KB)
    sampleRate: 16000,
    channels: 1,
  },
  kimvocal: {
    url: 'https://huggingface.co/seanghay/uvr_models/resolve/main/Kim_Vocal_2.onnx',
    cacheKey: 'https://huggingface.co/seanghay/uvr_models/resolve/main/Kim_Vocal_2.onnx',
    sizeLabel: '~64 MB',
    approxBytes: 66760000,
    compensate: 1.009,
    sampleRate: 44100,
    channels: 2,
  },
  kimvocal_fp16: {
    localRelPath: 'models/Kim_Vocal_2_fp16.onnx',
    cacheKey: 'music-remover-local/Kim_Vocal_2_fp16.onnx',
    sizeLabel: '~32 MB',
    approxBytes: 33400000,
    compensate: 1.009,
    sampleRate: 44100,
    channels: 2,
  },
  vocft: {
    url: 'https://huggingface.co/aaatmy/uvr-mdx-net-onnx/resolve/main/UVR-MDX-NET-Voc_FT.onnx',
    cacheKey: 'https://huggingface.co/aaatmy/uvr-mdx-net-onnx/resolve/main/UVR-MDX-NET-Voc_FT.onnx',
    sizeLabel: '~64 MB',
    approxBytes: 66760000,
    compensate: 1.021,
    sampleRate: 44100,
    channels: 2,
  },
};

const sessions = {};
let cancelRequested = false;

// ===================== 512-point Radix-2 FFT =====================
const N_FFT_512 = 512;
const bitRev512 = new Uint16Array(N_FFT_512);
const cos512 = new Float32Array(N_FFT_512 / 2);
const sin512 = new Float32Array(N_FFT_512 / 2);
{
  const bits = 9;
  for (let i = 0; i < N_FFT_512; i++) {
    let rev = 0;
    for (let j = 0; j < bits; j++) {
      if ((i >> j) & 1) rev |= 1 << (bits - 1 - j);
    }
    bitRev512[i] = rev;
  }
  for (let i = 0; i < N_FFT_512 / 2; i++) {
    const a = (-2.0 * Math.PI * i) / N_FFT_512;
    cos512[i] = Math.cos(a);
    sin512[i] = Math.sin(a);
  }
}

function fft512InPlace(re, im, inverse) {
  for (let size = 2; size <= N_FFT_512; size <<= 1) {
    const half = size >> 1;
    const step = N_FFT_512 / size;
    for (let i = 0; i < N_FFT_512; i += size) {
      for (let j = 0; j < half; j++) {
        const k = j * step;
        const c = cos512[k];
        const s = inverse ? -sin512[k] : sin512[k];
        const idx1 = i + j;
        const idx2 = idx1 + half;
        const tRe = re[idx2] * c - im[idx2] * s;
        const tIm = re[idx2] * s + im[idx2] * c;
        re[idx2] = re[idx1] - tRe;
        im[idx2] = im[idx1] - tIm;
        re[idx1] += tRe;
        im[idx1] += tIm;
      }
    }
  }
  if (inverse) {
    const inv = 1.0 / N_FFT_512;
    for (let i = 0; i < N_FFT_512; i++) {
      re[i] *= inv;
      im[i] *= inv;
    }
  }
}

// ===================== 7680-point Mixed-Radix FFT (15 x 512) =====================
const MDX_N_FFT = 7680;
const MDX_R = 15;
const MDX_M = 512;
const MDX_HOP = 1024;
const MDX_DIM_F = 3072;
const MDX_DIM_T = 256;
const MDX_CHUNK_SAMPLES = (MDX_DIM_T - 1) * MDX_HOP; // 261120 (~5.92s)
const MDX_PAD_LEN = MDX_N_FFT / 2; // 3840

const tw7680Re = new Float32Array(MDX_R * MDX_M);
const tw7680Im = new Float32Array(MDX_R * MDX_M);
const w15Re = new Float32Array(MDX_R * MDX_R);
const w15Im = new Float32Array(MDX_R * MDX_R);
{
  for (let n1 = 0; n1 < MDX_R; n1++) {
    for (let k2 = 0; k2 < MDX_M; k2++) {
      const a = (-2.0 * Math.PI * n1 * k2) / MDX_N_FFT;
      tw7680Re[n1 * MDX_M + k2] = Math.cos(a);
      tw7680Im[n1 * MDX_M + k2] = Math.sin(a);
    }
  }
  for (let k1 = 0; k1 < MDX_R; k1++) {
    for (let n1 = 0; n1 < MDX_R; n1++) {
      const a = (-2.0 * Math.PI * k1 * n1) / MDX_R;
      w15Re[k1 * MDX_R + n1] = Math.cos(a);
      w15Im[k1 * MDX_R + n1] = Math.sin(a);
    }
  }
}

const mdxSubRe = new Float32Array(MDX_R * MDX_M);
const mdxSubIm = new Float32Array(MDX_R * MDX_M);
const mdxTmpRe = new Float32Array(MDX_M);
const mdxTmpIm = new Float32Array(MDX_M);

function fft7680(inRe, inIm, outRe, outIm, inverse) {
  const sgn = inverse ? -1.0 : 1.0;

  for (let n1 = 0; n1 < MDX_R; n1++) {
    for (let n2 = 0; n2 < MDX_M; n2++) {
      const r = bitRev512[n2];
      const srcIdx = n1 + MDX_R * r;
      mdxTmpRe[n2] = inRe[srcIdx];
      mdxTmpIm[n2] = inIm ? inIm[srcIdx] : 0;
    }
    fft512InPlace(mdxTmpRe, mdxTmpIm, inverse);

    const rowOff = n1 * MDX_M;
    for (let k2 = 0; k2 < MDX_M; k2++) {
      const twR = tw7680Re[rowOff + k2];
      const twI = sgn * tw7680Im[rowOff + k2];
      const rVal = mdxTmpRe[k2];
      const iVal = mdxTmpIm[k2];
      mdxSubRe[rowOff + k2] = rVal * twR - iVal * twI;
      mdxSubIm[rowOff + k2] = rVal * twI + iVal * twR;
    }
  }

  const scale = inverse ? 1.0 / MDX_R : 1.0;
  for (let k1 = 0; k1 < MDX_R; k1++) {
    const wOff = k1 * MDX_R;
    for (let k2 = 0; k2 < MDX_M; k2++) {
      let sumRe = 0;
      let sumIm = 0;
      for (let n1 = 0; n1 < MDX_R; n1++) {
        const subIdx = n1 * MDX_M + k2;
        const wR = w15Re[wOff + n1];
        const wI = sgn * w15Im[wOff + n1];
        const xR = mdxSubRe[subIdx];
        const xI = mdxSubIm[subIdx];
        sumRe += xR * wR - xI * wI;
        sumIm += xR * wI + xI * wR;
      }
      const outIdx = k1 * MDX_M + k2;
      outRe[outIdx] = sumRe * scale;
      if (outIm) outIm[outIdx] = sumIm * scale;
    }
  }
}

// ===================== Model Loader with Browser Cache =====================
async function fetchModelBytes(modelKey) {
  const spec = MODEL_SPECS[modelKey];
  if (!spec.cacheKey) {
    const url = new URL(spec.localRelPath, self.location.href).href;
    const res = await fetch(url);
    if (!res.ok) throw new Error(`Gagal memuat model lokal (${res.status})`);
    return new Uint8Array(await res.arrayBuffer());
  }

  const cache = await caches.open('music-remover-web-onnx-v1');
  const cached = await cache.match(spec.cacheKey);
  if (cached) {
    self.postMessage({ type: 'STATUS', text: 'Memuat model dari cache browser...' });
    return new Uint8Array(await cached.arrayBuffer());
  }

  const fetchUrl = spec.localRelPath
    ? new URL(spec.localRelPath, self.location.href).href
    : spec.url;
  const sizeLabel = spec.sizeLabel || '~64 MB';
  const approxBytes = spec.approxBytes || 66760000;

  self.postMessage({ type: 'STATUS', text: `Mengunduh model (${sizeLabel}, hanya sekali)... 0%` });
  const resp = await fetch(fetchUrl);
  if (!resp.ok) {
    throw new Error(`Gagal mengunduh model (${resp.status})`);
  }

  const total = Number(resp.headers.get('content-length')) || approxBytes;
  const reader = resp.body.getReader();
  const chunks = [];
  let received = 0;
  let lastPct = -1;

  while (true) {
    if (cancelRequested) throw new Error('Dibatalkan oleh pengguna.');
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    received += value.length;
    const pct = Math.min(99, Math.round((received / total) * 100));
    if (pct !== lastPct) {
      lastPct = pct;
      self.postMessage({
        type: 'STATUS',
        text: `Mengunduh model (${sizeLabel}, tersimpan di cache browser)... ${pct}%`,
      });
    }
  }

  const buf = new Uint8Array(received);
  let offset = 0;
  for (const c of chunks) {
    buf.set(c, offset);
    offset += c.length;
  }

  try {
    await cache.put(spec.cacheKey, new Response(buf.buffer, {
      headers: { 'Content-Type': 'application/octet-stream' },
    }));
  } catch (_) {
    // Ignore cache quota failures in private browsing
  }

  return buf;
}

function disposeTensor(t) {
  if (t && typeof t.dispose === 'function') {
    try { t.dispose(); } catch (_) { }
  }
}

async function getSession(modelKey, backendMode) {
  const cacheId = `${modelKey}:${backendMode}`;
  if (sessions[cacheId]) return sessions[cacheId];

  ort.env.wasm.wasmPaths = ORT_CDN;
  ort.env.wasm.numThreads = 1;

  const bytes = await fetchModelBytes(modelKey);
  const providers = backendMode === 'webgpu'
    ? ['webgpu', 'wasm']
    : ['wasm'];

  for (const ep of providers) {
    try {
      self.postMessage({ type: 'STATUS', text: `Menginisialisasi runtime (${ep.toUpperCase()})...` });
      const sess = await ort.InferenceSession.create(bytes, {
        executionProviders: [ep],
        graphOptimizationLevel: 'all',
        enableCpuMemArena: true,
        enableMemPattern: true,
      });
      sessions[cacheId] = { sess, provider: ep };
      return sessions[cacheId];
    } catch (_) {
      // Try next provider
    }
  }
  throw new Error(`Gagal menginisialisasi ONNX Runtime untuk ${modelKey}`);
}

// ===================== Offline GTCRN Processor =====================
async function runGtcrnOffline(mono16k, backendMode) {
  const { sess, provider } = await getSession('gtcrn', backendMode);
  const totalLen = mono16k.length;
  const hop = 256;

  const sqrtHann = new Float32Array(N_FFT_512);
  for (let i = 0; i < N_FFT_512; i++) {
    sqrtHann[i] = Math.sqrt(0.5 * (1.0 - Math.cos((2.0 * Math.PI * i) / N_FFT_512)));
  }

  const paddedLen = totalLen + N_FFT_512;
  const padded = new Float32Array(paddedLen);
  padded.set(mono16k, hop); // align 256-sample pre-pad

  const outBuf = new Float32Array(paddedLen);
  const fftRe = new Float32Array(N_FFT_512);
  const fftIm = new Float32Array(N_FFT_512);
  const specRe = new Float32Array(N_FFT_512);
  const specIm = new Float32Array(N_FFT_512);

  let convCache = new ort.Tensor('float32', new Float32Array(2 * 1 * 16 * 16 * 33), [2, 1, 16, 16, 33]);
  let traCache = new ort.Tensor('float32', new Float32Array(2 * 3 * 1 * 1 * 16), [2, 3, 1, 1, 16]);
  let interCache = new ort.Tensor('float32', new Float32Array(2 * 1 * 33 * 16), [2, 1, 33, 16]);

  const totalFrames = Math.ceil(totalLen / hop) + 1;
  let lastProgress = -1;

  for (let frameIdx = 0; frameIdx < totalFrames; frameIdx++) {
    if (cancelRequested) throw new Error('Dibatalkan oleh pengguna.');
    const start = frameIdx * hop;

    for (let i = 0; i < N_FFT_512; i++) {
      const r = bitRev512[i];
      const s = start + r < paddedLen ? padded[start + r] : 0;
      fftRe[i] = s * sqrtHann[r];
      fftIm[i] = 0;
    }
    fft512InPlace(fftRe, fftIm, false);

    const mixData = new Float32Array(257 * 2);
    for (let k = 0; k < 257; k++) {
      mixData[2 * k] = fftRe[k];
      mixData[2 * k + 1] = fftIm[k];
    }

    const mixTensor = new ort.Tensor('float32', mixData, [1, 257, 1, 2]);
    const prevConv = convCache;
    const prevTra = traCache;
    const prevInter = interCache;
    const res = await sess.run({
      mix: mixTensor,
      conv_cache: prevConv,
      tra_cache: prevTra,
      inter_cache: prevInter,
    });
    convCache = res.conv_cache_out;
    traCache = res.tra_cache_out;
    interCache = res.inter_cache_out;

    const enh = res.enh.data;
    for (let k = 0; k < 257; k++) {
      specRe[k] = enh[2 * k];
      specIm[k] = enh[2 * k + 1];
    }
    disposeTensor(mixTensor);
    disposeTensor(prevConv);
    disposeTensor(prevTra);
    disposeTensor(prevInter);
    disposeTensor(res.enh);
    for (let k = 1; k < N_FFT_512 / 2; k++) {
      specRe[N_FFT_512 - k] = specRe[k];
      specIm[N_FFT_512 - k] = -specIm[k];
    }
    for (let i = 0; i < N_FFT_512; i++) {
      const r = bitRev512[i];
      fftRe[i] = specRe[r];
      fftIm[i] = specIm[r];
    }
    fft512InPlace(fftRe, fftIm, true);

    for (let i = 0; i < N_FFT_512 && start + i < paddedLen; i++) {
      outBuf[start + i] += fftRe[i] * sqrtHann[i];
    }

    const pct = Math.round(((frameIdx + 1) / totalFrames) * 100);
    if (pct !== lastProgress) {
      lastProgress = pct;
      self.postMessage({
        type: 'PROGRESS',
        percent: pct,
        text: `Memproses dengan GTCRN (${provider.toUpperCase()})... ${pct}%`,
      });
    }
  }

  const result = outBuf.slice(hop, hop + totalLen);
  return { channels: [result], sampleRate: 16000, provider };
}

// ===================== Offline Stereo MDX-Net Processor =====================
const mdxHann = new Float32Array(MDX_N_FFT);
for (let i = 0; i < MDX_N_FFT; i++) {
  mdxHann[i] = 0.5 * (1.0 - Math.cos((2.0 * Math.PI * i) / MDX_N_FFT));
}

function mdxProcessSingleChunk(chunkL, chunkR, sess, compensate) {
  const paddedL = new Float32Array(MDX_CHUNK_SAMPLES + MDX_N_FFT);
  const paddedR = new Float32Array(MDX_CHUNK_SAMPLES + MDX_N_FFT);
  paddedL.set(chunkL, MDX_PAD_LEN);
  paddedR.set(chunkR, MDX_PAD_LEN);

  const winFrame = new Float32Array(MDX_N_FFT);
  const outRe = new Float32Array(MDX_N_FFT);
  const outIm = new Float32Array(MDX_N_FFT);

  const planeSize = MDX_DIM_F * MDX_DIM_T;
  const tensorData = new Float32Array(4 * planeSize);

  // STFT Left -> planes 0, 1
  for (let f = 0; f < MDX_DIM_T; f++) {
    const start = f * MDX_HOP;
    for (let i = 0; i < MDX_N_FFT; i++) {
      winFrame[i] = paddedL[start + i] * mdxHann[i];
    }
    fft7680(winFrame, null, outRe, outIm, false);
    for (let k = 0; k < MDX_DIM_F; k++) {
      const idx = k * MDX_DIM_T + f;
      tensorData[idx] = outRe[k];
      tensorData[planeSize + idx] = outIm[k];
    }
  }

  // STFT Right -> planes 2, 3
  for (let f = 0; f < MDX_DIM_T; f++) {
    const start = f * MDX_HOP;
    for (let i = 0; i < MDX_N_FFT; i++) {
      winFrame[i] = paddedR[start + i] * mdxHann[i];
    }
    fft7680(winFrame, null, outRe, outIm, false);
    for (let k = 0; k < MDX_DIM_F; k++) {
      const idx = k * MDX_DIM_T + f;
      tensorData[2 * planeSize + idx] = outRe[k];
      tensorData[3 * planeSize + idx] = outIm[k];
    }
  }

  return { tensorData, planeSize, paddedL, paddedR };
}

async function runMdxOffline(left, right, modelKey, backendMode) {
  const { sess, provider } = await getSession(modelKey, backendMode);
  const compensate = MODEL_SPECS[modelKey].compensate;
  const totalLen = left.length;

  // 50% overlap across 261120-sample (~5.92s) chunks
  const step = MDX_CHUNK_SAMPLES / 2; // 130560
  const numChunks = Math.max(1, Math.ceil((totalLen - MDX_CHUNK_SAMPLES) / step) + 1);
  const outL = new Float32Array(totalLen);
  const outR = new Float32Array(totalLen);
  const weightSum = new Float32Array(totalLen);

  // Raised-cosine window for seamless 50% chunk blending
  const chunkWin = new Float32Array(MDX_CHUNK_SAMPLES);
  for (let i = 0; i < MDX_CHUNK_SAMPLES; i++) {
    const t = Math.sin((Math.PI * (i + 0.5)) / MDX_CHUNK_SAMPLES);
    chunkWin[i] = t * t;
  }

  const specInRe = new Float32Array(MDX_N_FFT);
  const specInIm = new Float32Array(MDX_N_FFT);
  const timeRe = new Float32Array(MDX_N_FFT);
  const olaL = new Float32Array(MDX_CHUNK_SAMPLES + MDX_N_FFT);
  const olaR = new Float32Array(MDX_CHUNK_SAMPLES + MDX_N_FFT);
  const winSum = new Float32Array(MDX_CHUNK_SAMPLES + MDX_N_FFT);

  const inputName = sess.inputNames[0];
  const outputName = sess.outputNames[0];

  for (let c = 0; c < numChunks; c++) {
    if (cancelRequested) throw new Error('Dibatalkan oleh pengguna.');
    const chunkStart = c * step;
    const chunkL = new Float32Array(MDX_CHUNK_SAMPLES);
    const chunkR = new Float32Array(MDX_CHUNK_SAMPLES);
    const copyLen = Math.min(MDX_CHUNK_SAMPLES, totalLen - chunkStart);
    if (copyLen > 0) {
      chunkL.set(left.subarray(chunkStart, chunkStart + copyLen));
      chunkR.set(right.subarray(chunkStart, chunkStart + copyLen));
    }

    self.postMessage({
      type: 'PROGRESS',
      percent: Math.round((c / numChunks) * 100),
      text: `Memproses segmen ${c + 1}/${numChunks} (${provider.toUpperCase()})...`,
    });

    const { tensorData, planeSize } = mdxProcessSingleChunk(chunkL, chunkR, sess, compensate);
    const inputTensor = new ort.Tensor('float32', tensorData, [1, 4, MDX_DIM_F, MDX_DIM_T]);
    const results = await sess.run({ [inputName]: inputTensor });
    const outTensor = results[outputName];
    const outData = outTensor.data;

    olaL.fill(0);
    olaR.fill(0);
    winSum.fill(0);

    // iSTFT Left & Right
    for (let f = 0; f < MDX_DIM_T; f++) {
      const start = f * MDX_HOP;

      // Left channel
      specInRe.fill(0);
      specInIm.fill(0);
      for (let k = 0; k < MDX_DIM_F; k++) {
        const idx = k * MDX_DIM_T + f;
        const re = outData[idx];
        const im = outData[planeSize + idx];
        specInRe[k] = re;
        specInIm[k] = im;
        if (k > 0) {
          specInRe[MDX_N_FFT - k] = re;
          specInIm[MDX_N_FFT - k] = -im;
        }
      }
      fft7680(specInRe, specInIm, timeRe, null, true);
      for (let i = 0; i < MDX_N_FFT; i++) {
        const w = mdxHann[i];
        olaL[start + i] += timeRe[i] * w;
        winSum[start + i] += w * w;
      }

      // Right channel
      specInRe.fill(0);
      specInIm.fill(0);
      for (let k = 0; k < MDX_DIM_F; k++) {
        const idx = k * MDX_DIM_T + f;
        const re = outData[2 * planeSize + idx];
        const im = outData[3 * planeSize + idx];
        specInRe[k] = re;
        specInIm[k] = im;
        if (k > 0) {
          specInRe[MDX_N_FFT - k] = re;
          specInIm[MDX_N_FFT - k] = -im;
        }
      }
      fft7680(specInRe, specInIm, timeRe, null, true);
      for (let i = 0; i < MDX_N_FFT; i++) {
        olaR[start + i] += timeRe[i] * mdxHann[i];
      }
    }

    disposeTensor(inputTensor);
    disposeTensor(outTensor);

    // Overlap-add chunk back into full track
    for (let i = 0; i < copyLen; i++) {
      const stftPos = MDX_PAD_LEN + i;
      const denom = winSum[stftPos] > 1e-8 ? winSum[stftPos] : 1.0;
      const vL = (olaL[stftPos] / denom) * compensate;
      const vR = (olaR[stftPos] / denom) * compensate;

      // Keep full weight at very edges of the track so ends don't fade out
      let w = chunkWin[i];
      if (c === 0 && i < step) w = 1.0;
      if (c === numChunks - 1 && i >= step) w = 1.0;

      outL[chunkStart + i] += vL * w;
      outR[chunkStart + i] += vR * w;
      weightSum[chunkStart + i] += w;
    }

    self.postMessage({
      type: 'PROGRESS',
      percent: Math.round(((c + 1) / numChunks) * 100),
      text: `Segmen ${c + 1}/${numChunks} selesai (${provider.toUpperCase()})`,
    });
  }

  for (let i = 0; i < totalLen; i++) {
    const w = weightSum[i] > 1e-6 ? weightSum[i] : 1.0;
    outL[i] /= w;
    outR[i] /= w;
  }

  return { channels: [outL, outR], sampleRate: 44100, provider };
}

self.onmessage = async (e) => {
  const msg = e.data;
  if (msg.type === 'CANCEL') {
    cancelRequested = true;
    return;
  }

  if (msg.type === 'PROCESS') {
    cancelRequested = false;
    try {
      const { modelKey, backendMode, channels } = msg;
      let res;
      if (modelKey === 'gtcrn') {
        res = await runGtcrnOffline(channels[0], backendMode);
      } else {
        res = await runMdxOffline(channels[0], channels[1] || channels[0], modelKey, backendMode);
      }
      const transfers = res.channels.map((ch) => ch.buffer);
      self.postMessage({
        type: 'DONE',
        channels: res.channels,
        sampleRate: res.sampleRate,
        provider: res.provider,
      }, transfers);
    } catch (err) {
      self.postMessage({
        type: 'ERROR',
        error: err.message || String(err),
      });
    }
  }
};
