// Dedicated Web Worker for Chrome Extension ONNX Inference
// Keeps heavy WASM/FFT work completely off the extension main thread so popup.html never freezes.

importScripts('lib/ort.all.min.js');

const MDX_MODELS = {
  kimvocal: {
    localPath: 'models/Kim_Vocal_2.onnx',
    remoteUrl: 'https://huggingface.co/seanghay/uvr_models/resolve/main/Kim_Vocal_2.onnx',
    sizeLabel: '~64 MB',
    approxBytes: 66760000,
    compensate: 1.009,
  },
  kimvocal_fp16: {
    localPath: 'models/Kim_Vocal_2_fp16.onnx',
    remoteUrl: 'https://raw.githubusercontent.com/atiohaidar/music_remover/main/web/models/Kim_Vocal_2_fp16.onnx',
    sizeLabel: '~32 MB',
    approxBytes: 33400000,
    compensate: 1.009,
  },
  vocft: {
    localPath: 'models/UVR-MDX-NET-Voc_FT.onnx',
    remoteUrl: 'https://huggingface.co/aaatmy/uvr-mdx-net-onnx/resolve/main/UVR-MDX-NET-Voc_FT.onnx',
    sizeLabel: '~64 MB',
    approxBytes: 66760000,
    compensate: 1.021,
  },
};

const sessions = {};
let activeEngine = 'kimvocal';
let activeBackendMode = 'wasm';
let activeProvider = 'wasm';
let baseExtensionUrl = '';

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
const MDX_CONTEXT_SAMPLES = (MDX_DIM_T - 1) * MDX_HOP; // 261120 (~5.92s)

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

// ===================== GTCRN State =====================
const GTCRN_HOP = 256;
const GTCRN_BINS = 257;
const gtcrnInFrame = new Float32Array(N_FFT_512);
const gtcrnOla = new Float32Array(N_FFT_512);
const gtcrnSqrtHann = new Float32Array(N_FFT_512);
for (let i = 0; i < N_FFT_512; i++) {
  gtcrnSqrtHann[i] = Math.sqrt(0.5 * (1.0 - Math.cos((2.0 * Math.PI * i) / N_FFT_512)));
}
const gtcrnFftRe = new Float32Array(N_FFT_512);
const gtcrnFftIm = new Float32Array(N_FFT_512);
const gtcrnMixData = new Float32Array(GTCRN_BINS * 2);
const gtcrnSpecRe = new Float32Array(N_FFT_512);
const gtcrnSpecIm = new Float32Array(N_FFT_512);
let convCache = null;
let traCache = null;
let interCache = null;

function disposeTensor(t) {
  if (t && typeof t.dispose === 'function') {
    try { t.dispose(); } catch (_) { }
  }
}

function resetGtcrnCaches() {
  disposeTensor(convCache);
  disposeTensor(traCache);
  disposeTensor(interCache);
  convCache = new ort.Tensor('float32', new Float32Array(2 * 1 * 16 * 16 * 33), [2, 1, 16, 16, 33]);
  traCache = new ort.Tensor('float32', new Float32Array(2 * 3 * 1 * 1 * 16), [2, 3, 1, 1, 16]);
  interCache = new ort.Tensor('float32', new Float32Array(2 * 1 * 33 * 16), [2, 1, 33, 16]);
  gtcrnInFrame.fill(0);
  gtcrnOla.fill(0);
}

// ===================== MDX State =====================
const mdxHann = new Float32Array(MDX_N_FFT);
for (let i = 0; i < MDX_N_FFT; i++) {
  mdxHann[i] = 0.5 * (1.0 - Math.cos((2.0 * Math.PI * i) / MDX_N_FFT));
}
const mdxPadLen = MDX_N_FFT / 2;
const mdxContext = new Float32Array(MDX_CONTEXT_SAMPLES);
const mdxPadded = new Float32Array(MDX_CONTEXT_SAMPLES + MDX_N_FFT);
const mdxWinFrame = new Float32Array(MDX_N_FFT);
const mdxOutRe = new Float32Array(MDX_N_FFT);
const mdxOutIm = new Float32Array(MDX_N_FFT);
const mdxSpecInRe = new Float32Array(MDX_N_FFT);
const mdxSpecInIm = new Float32Array(MDX_N_FFT);
const mdxTimeRe = new Float32Array(MDX_N_FFT);
const mdxOla = new Float32Array(MDX_CONTEXT_SAMPLES + MDX_N_FFT);
const mdxWinSum = new Float32Array(MDX_CONTEXT_SAMPLES + MDX_N_FFT);
const mdxTensorData = new Float32Array(4 * MDX_DIM_F * MDX_DIM_T);

let mdxChunkSize = 198450;
let mdxOverlap = 29767;
let mdxHopSize = 168683;
let mdxFadeIn = null;
let mdxFadeOut = null;
let mdxPrevTail = null;
let mdxGain = 1.0;

function configureMdxDelay(delaySec) {
  const clamped = Math.max(1.0, Math.min(5.9, Number(delaySec) || 4.5));
  mdxChunkSize = Math.max(4096, Math.round(clamped * 44100));
  mdxOverlap = Math.round(mdxChunkSize * 0.15);
  mdxHopSize = mdxChunkSize - mdxOverlap;

  mdxFadeIn = new Float32Array(mdxOverlap);
  mdxFadeOut = new Float32Array(mdxOverlap);
  for (let i = 0; i < mdxOverlap; i++) {
    const t = (i / Math.max(1, mdxOverlap - 1)) * (Math.PI / 2.0);
    const s = Math.sin(t);
    const c = Math.cos(t);
    mdxFadeIn[i] = s * s;
    mdxFadeOut[i] = c * c;
  }
  mdxContext.fill(0);
  mdxPrevTail = null;
  mdxGain = 1.0;
}

// ===================== Model Loader =====================
async function resolveModelSource(engineKey) {
  const localPath = engineKey === 'gtcrn'
    ? 'models/gtcrn_simple.onnx'
    : MDX_MODELS[engineKey].localPath;
  const localUrl = baseExtensionUrl + localPath;

  try {
    const res = await fetch(localUrl, { method: 'HEAD' });
    if (res.ok) return localUrl;
  } catch (_) { }

  const remoteUrl = MDX_MODELS[engineKey]?.remoteUrl;
  if (!remoteUrl) {
    throw new Error(`Model file missing: ${localPath}`);
  }

  const cache = await caches.open('music-remover-onnx-v1');
  const cachedResp = await cache.match(remoteUrl);
  if (cachedResp) {
    return new Uint8Array(await cachedResp.arrayBuffer());
  }

  const sizeLabel = MDX_MODELS[engineKey]?.sizeLabel || '~64 MB';
  const approxBytes = MDX_MODELS[engineKey]?.approxBytes || 66760000;
  self.postMessage({ type: 'STATUS', avgInferenceMs: 0, statusText: `Downloading model (${sizeLabel}) 0%...` });
  const resp = await fetch(remoteUrl);
  if (!resp.ok) {
    throw new Error(`Gagal download model (${resp.status}).`);
  }

  const total = Number(resp.headers.get('content-length')) || approxBytes;
  const reader = resp.body.getReader();
  const chunks = [];
  let received = 0;
  let lastPct = -1;

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    received += value.length;
    const pct = Math.min(99, Math.round((received / total) * 100));
    if (pct !== lastPct) {
      lastPct = pct;
      self.postMessage({
        type: 'STATUS',
        avgInferenceMs: 0,
        statusText: `Downloading model (${sizeLabel}) ${pct}%...`,
      });
    }
  }

  const fullBuf = new Uint8Array(received);
  let offset = 0;
  for (const c of chunks) {
    fullBuf.set(c, offset);
    offset += c.length;
  }

  try {
    await cache.put(remoteUrl, new Response(fullBuf.buffer, {
      headers: { 'Content-Type': 'application/octet-stream' },
    }));
  } catch (_) { }

  return fullBuf;
}

async function getSession(engineKey, backendMode) {
  const cacheKey = `${engineKey}:${backendMode}`;
  if (sessions[cacheKey]) {
    activeProvider = sessions[cacheKey].provider;
    return sessions[cacheKey].sess;
  }

  ort.env.wasm.wasmPaths = baseExtensionUrl + 'lib/';
  const maxThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 4) : 1;
  const modelSource = await resolveModelSource(engineKey);

  // Try requested provider list; for WASM try multi-thread then fall back to 1 thread
  const attempts = backendMode === 'webgpu'
    ? [
      { ep: 'webgpu', threads: 1 },
      { ep: 'wasm', threads: maxThreads },
      { ep: 'wasm', threads: 1 },
    ]
    : [
      { ep: 'wasm', threads: maxThreads },
      { ep: 'wasm', threads: 1 },
    ];

  for (const { ep, threads } of attempts) {
    try {
      ort.env.wasm.numThreads = threads;
      if (ep === 'webgpu' && ort.env.webgpu) {
        ort.env.webgpu.powerPreference = 'high-performance';
      }
      const sess = await ort.InferenceSession.create(modelSource, {
        executionProviders: [ep],
        graphOptimizationLevel: 'all',
        enableCpuMemArena: true,
        enableMemPattern: true,
      });
      const label = ep === 'wasm' && threads > 1 ? `wasm-${threads}t` : ep;
      sessions[cacheKey] = { sess, provider: label };
      activeProvider = label;
      return sess;
    } catch (_) { }
  }
  throw new Error(`Failed to initialize ONNX session for ${engineKey}`);
}

// ===================== Inference =====================
let infSum = 0;
let infCount = 0;

async function processGtcrnHop(hop256) {
  const t0 = performance.now();
  const sess = sessions[`gtcrn:${activeBackendMode}`].sess;

  gtcrnInFrame.copyWithin(0, GTCRN_HOP);
  gtcrnInFrame.set(hop256, GTCRN_HOP);

  for (let i = 0; i < N_FFT_512; i++) {
    const r = bitRev512[i];
    gtcrnFftRe[i] = gtcrnInFrame[r] * gtcrnSqrtHann[r];
    gtcrnFftIm[i] = 0;
  }
  fft512InPlace(gtcrnFftRe, gtcrnFftIm, false);

  for (let k = 0; k < GTCRN_BINS; k++) {
    gtcrnMixData[2 * k] = gtcrnFftRe[k];
    gtcrnMixData[2 * k + 1] = gtcrnFftIm[k];
  }
  const prevConv = convCache;
  const prevTra = traCache;
  const prevInter = interCache;
  const mixTensor = new ort.Tensor('float32', gtcrnMixData, [1, GTCRN_BINS, 1, 2]);
  const results = await sess.run({
    mix: mixTensor,
    conv_cache: prevConv,
    tra_cache: prevTra,
    inter_cache: prevInter,
  });

  convCache = results.conv_cache_out;
  traCache = results.tra_cache_out;
  interCache = results.inter_cache_out;
  const enhData = results.enh.data;

  for (let k = 0; k < GTCRN_BINS; k++) {
    gtcrnSpecRe[k] = enhData[2 * k];
    gtcrnSpecIm[k] = enhData[2 * k + 1];
  }
  for (let k = 1; k < N_FFT_512 / 2; k++) {
    gtcrnSpecRe[N_FFT_512 - k] = gtcrnSpecRe[k];
    gtcrnSpecIm[N_FFT_512 - k] = -gtcrnSpecIm[k];
  }
  disposeTensor(mixTensor);
  disposeTensor(prevConv);
  disposeTensor(prevTra);
  disposeTensor(prevInter);
  disposeTensor(results.enh);

  for (let i = 0; i < N_FFT_512; i++) {
    const r = bitRev512[i];
    gtcrnFftRe[i] = gtcrnSpecRe[r];
    gtcrnFftIm[i] = gtcrnSpecIm[r];
  }
  fft512InPlace(gtcrnFftRe, gtcrnFftIm, true);

  for (let i = 0; i < N_FFT_512; i++) {
    gtcrnOla[i] += gtcrnFftRe[i] * gtcrnSqrtHann[i];
  }
  const outHop = gtcrnOla.slice(0, GTCRN_HOP);
  gtcrnOla.copyWithin(0, GTCRN_HOP);
  gtcrnOla.fill(0, GTCRN_HOP);

  const dt = performance.now() - t0;
  infSum += dt;
  infCount++;
  if (infCount >= 25) {
    self.postMessage({ type: 'STATS', avgInferenceMs: infSum / infCount });
    infSum = 0;
    infCount = 0;
  }
  return outHop;
}

async function processMdxHop(newHop) {
  const t0 = performance.now();
  const sess = sessions[`${activeEngine}:${activeBackendMode}`].sess;
  const compensate = MDX_MODELS[activeEngine].compensate;

  mdxContext.copyWithin(0, mdxHopSize);
  mdxContext.set(newHop, MDX_CONTEXT_SAMPLES - mdxHopSize);

  mdxPadded.fill(0);
  mdxPadded.set(mdxContext, mdxPadLen);

  const planeSize = MDX_DIM_F * MDX_DIM_T;

  for (let f = 0; f < MDX_DIM_T; f++) {
    const start = f * MDX_HOP;
    for (let i = 0; i < MDX_N_FFT; i++) {
      mdxWinFrame[i] = mdxPadded[start + i] * mdxHann[i];
    }
    fft7680(mdxWinFrame, null, mdxOutRe, mdxOutIm, false);

    for (let k = 0; k < MDX_DIM_F; k++) {
      const idx = k * MDX_DIM_T + f;
      const re = mdxOutRe[k];
      const im = mdxOutIm[k];
      mdxTensorData[idx] = re;
      mdxTensorData[planeSize + idx] = im;
      mdxTensorData[2 * planeSize + idx] = re;
      mdxTensorData[3 * planeSize + idx] = im;
    }
  }

  const inputName = sess.inputNames[0];
  const outputName = sess.outputNames[0];
  const inputTensor = new ort.Tensor('float32', mdxTensorData, [1, 4, MDX_DIM_F, MDX_DIM_T]);
  const results = await sess.run({ [inputName]: inputTensor });
  const outTensor = results[outputName];
  const outData = outTensor.data;

  mdxOla.fill(0);
  mdxWinSum.fill(0);

  for (let f = 0; f < MDX_DIM_T; f++) {
    mdxSpecInRe.fill(0);
    mdxSpecInIm.fill(0);

    for (let k = 0; k < MDX_DIM_F; k++) {
      const idx = k * MDX_DIM_T + f;
      const re = 0.5 * (outData[idx] + outData[2 * planeSize + idx]);
      const im = 0.5 * (outData[planeSize + idx] + outData[3 * planeSize + idx]);
      mdxSpecInRe[k] = re;
      mdxSpecInIm[k] = im;
      if (k > 0) {
        mdxSpecInRe[MDX_N_FFT - k] = re;
        mdxSpecInIm[MDX_N_FFT - k] = -im;
      }
    }

    fft7680(mdxSpecInRe, mdxSpecInIm, mdxTimeRe, null, true);

    const start = f * MDX_HOP;
    for (let i = 0; i < MDX_N_FFT; i++) {
      const w = mdxHann[i];
      mdxOla[start + i] += mdxTimeRe[i] * w;
      mdxWinSum[start + i] += w * w;
    }
  }

  disposeTensor(inputTensor);
  disposeTensor(outTensor);

  const vocalChunk = new Float32Array(mdxChunkSize);
  const extractStart = mdxPadLen + MDX_CONTEXT_SAMPLES - mdxChunkSize;
  for (let i = 0; i < mdxChunkSize; i++) {
    const pos = extractStart + i;
    const denom = mdxWinSum[pos] > 1e-8 ? mdxWinSum[pos] : 1.0;
    vocalChunk[i] = (mdxOla[pos] / denom) * compensate;
  }

  const emitted = new Float32Array(mdxHopSize);
  if (mdxPrevTail) {
    for (let i = 0; i < mdxOverlap; i++) {
      emitted[i] = mdxPrevTail[i] * mdxFadeOut[i] + vocalChunk[i] * mdxFadeIn[i];
    }
    emitted.set(vocalChunk.subarray(mdxOverlap, mdxHopSize), mdxOverlap);
  } else {
    emitted.set(vocalChunk.subarray(0, mdxHopSize));
  }
  mdxPrevTail = vocalChunk.slice(mdxHopSize);

  let sumSq = 0;
  for (let i = 0; i < emitted.length; i++) sumSq += emitted[i] * emitted[i];
  const rms = Math.sqrt(sumSq / emitted.length);
  if (rms > 0.005) {
    const targetGain = Math.max(0.3, Math.min(4.0, 0.10 / (rms + 1e-6)));
    const alpha = targetGain > mdxGain ? 0.05 : 0.01;
    mdxGain = (1.0 - alpha) * mdxGain + alpha * targetGain;
  }
  for (let i = 0; i < emitted.length; i++) {
    emitted[i] = Math.max(-0.98, Math.min(0.98, emitted[i] * mdxGain));
  }

  const dt = performance.now() - t0;
  self.postMessage({ type: 'STATS', avgInferenceMs: dt });
  return emitted;
}

self.onmessage = async (e) => {
  const msg = e.data;
  try {
    if (msg.type === 'INIT') {
      baseExtensionUrl = msg.baseExtensionUrl;
      activeEngine = msg.engine;
      activeBackendMode = msg.backendMode;
      infSum = 0;
      infCount = 0;

      await getSession(activeEngine, activeBackendMode);
      if (activeEngine === 'gtcrn') {
        resetGtcrnCaches();
      } else {
        configureMdxDelay(msg.delaySec);
      }

      self.postMessage({
        type: 'INIT_DONE',
        provider: activeProvider,
        hopSamples: activeEngine === 'gtcrn' ? GTCRN_HOP : mdxHopSize,
      });
    } else if (msg.type === 'HOP') {
      const enhanced = activeEngine === 'gtcrn'
        ? await processGtcrnHop(msg.samples)
        : await processMdxHop(msg.samples);
      self.postMessage({ type: 'ENHANCED_HOP', samples: enhanced }, [enhanced.buffer]);
    }
  } catch (err) {
    self.postMessage({ type: 'ERROR', error: err.message || String(err) });
  }
};
