// Offscreen Audio + ONNX Runtime Web Engine
// Supports:
//   - 'kimvocal': Kim_Vocal_2.onnx (44.1kHz MDX-Net, compensate=1.009)
//   - 'vocft':    UVR-MDX-NET-Voc_FT.onnx (44.1kHz MDX-Net, compensate=1.021)
//   - 'gtcrn':    gtcrn_simple.onnx (16kHz ultra-low-latency)

const MDX_MODELS = {
  kimvocal: {
    file: 'models/Kim_Vocal_2.onnx',
    remoteUrl: 'https://huggingface.co/seanghay/uvr_models/resolve/main/Kim_Vocal_2.onnx',
    compensate: 1.009,
  },
  vocft: {
    file: 'models/UVR-MDX-NET-Voc_FT.onnx',
    remoteUrl: 'https://huggingface.co/aaatmy/uvr-mdx-net-onnx/resolve/main/UVR-MDX-NET-Voc_FT.onnx',
    compensate: 1.021,
  },
};

let sessions = {};
let activeEngine = 'kimvocal';
let activeProvider = 'wasm';

let audioCtx = null;
let mediaStream = null;
let workletNode = null;
let isRunning = false;
let processingQueue = Promise.resolve();

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

  // Step 1: 15 sub-FFTs of length 512
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

  // Step 2: 512 DFTs of length 15
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
let convCache = null;
let traCache = null;
let interCache = null;

// ===================== MDX State =====================
const mdxHann = new Float32Array(MDX_N_FFT);
for (let i = 0; i < MDX_N_FFT; i++) {
  mdxHann[i] = 0.5 * (1.0 - Math.cos((2.0 * Math.PI * i) / MDX_N_FFT));
}
const mdxPadLen = MDX_N_FFT / 2; // 3840 (for center=True)
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

// ===================== Session Loader (Local or Auto-Cached Remote) =====================
async function resolveModelSource(engineKey) {
  const localPath = engineKey === 'gtcrn'
    ? 'models/gtcrn_simple.onnx'
    : MDX_MODELS[engineKey].file;
  const localUrl = chrome.runtime.getURL(localPath);

  // 1. Check if bundled locally inside chrome_extension/models/
  try {
    const res = await fetch(localUrl, { method: 'HEAD' });
    if (res.ok) return localUrl;
  } catch (_) {
    // Not bundled locally, proceed to Cache Storage / Remote download
  }

  const remoteUrl = MDX_MODELS[engineKey]?.remoteUrl;
  if (!remoteUrl) {
    throw new Error(`Model file missing: ${localPath}`);
  }

  // 2. Check browser Cache Storage
  const cache = await caches.open('music-remover-onnx-v1');
  const cachedResp = await cache.match(remoteUrl);
  if (cachedResp) {
    const buf = await cachedResp.arrayBuffer();
    return new Uint8Array(buf);
  }

  // 3. Download from Hugging Face with progress reporting to popup
  chrome.runtime.sendMessage({
    type: 'OFFSCREEN_STATS',
    avgInferenceMs: 0,
    statusText: 'Downloading model (~64 MB) 0%...',
  }).catch(() => {});

  const resp = await fetch(remoteUrl);
  if (!resp.ok) {
    throw new Error(`Gagal download model (${resp.status}). Jalankan setup_extension.py.`);
  }

  const total = Number(resp.headers.get('content-length')) || 66760000;
  const reader = resp.body.getReader();
  const chunks = [];
  let received = 0;

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    received += value.length;
    const pct = Math.min(99, Math.round((received / total) * 100));
    chrome.runtime.sendMessage({
      type: 'OFFSCREEN_STATS',
      avgInferenceMs: 0,
      statusText: `Downloading model (~64 MB) ${pct}%...`,
    }).catch(() => {});
  }

  const fullBuf = new Uint8Array(received);
  let offset = 0;
  for (const c of chunks) {
    fullBuf.set(c, offset);
    offset += c.length;
  }

  // Save in Cache Storage for instant load next time
  await cache.put(remoteUrl, new Response(fullBuf.buffer, {
    headers: { 'Content-Type': 'application/octet-stream' },
  }));

  return fullBuf;
}

async function getSession(engineKey) {
  if (sessions[engineKey]) return sessions[engineKey];

  ort.env.wasm.wasmPaths = chrome.runtime.getURL('lib/');
  ort.env.wasm.numThreads = 1;

  const modelSource = await resolveModelSource(engineKey);

  // Try WebGPU first if available, then fall back to WASM
  for (const ep of ['webgpu', 'wasm']) {
    try {
      const sess = await ort.InferenceSession.create(modelSource, {
        executionProviders: [ep],
      });
      sessions[engineKey] = sess;
      activeProvider = ep;
      return sess;
    } catch (_) {
      // Fallback to next provider
    }
  }
  throw new Error(`Failed to initialize ONNX session for ${engineKey}`);
}

function resetGtcrnCaches() {
  convCache = new ort.Tensor('float32', new Float32Array(2 * 1 * 16 * 16 * 33), [2, 1, 16, 16, 33]);
  traCache = new ort.Tensor('float32', new Float32Array(2 * 3 * 1 * 1 * 16), [2, 3, 1, 1, 16]);
  interCache = new ort.Tensor('float32', new Float32Array(2 * 1 * 33 * 16), [2, 1, 33, 16]);
  gtcrnInFrame.fill(0);
  gtcrnOla.fill(0);
}

// ===================== Inference Runners =====================
let infSum = 0;
let infCount = 0;

async function processGtcrnHop(hop256) {
  const t0 = performance.now();
  const sess = sessions.gtcrn;

  gtcrnInFrame.copyWithin(0, GTCRN_HOP);
  gtcrnInFrame.set(hop256, GTCRN_HOP);

  for (let i = 0; i < N_FFT_512; i++) {
    const r = bitRev512[i];
    gtcrnFftRe[i] = gtcrnInFrame[r] * gtcrnSqrtHann[r];
    gtcrnFftIm[i] = 0;
  }
  fft512InPlace(gtcrnFftRe, gtcrnFftIm, false);

  const mixData = new Float32Array(GTCRN_BINS * 2);
  for (let k = 0; k < GTCRN_BINS; k++) {
    mixData[2 * k] = gtcrnFftRe[k];
    mixData[2 * k + 1] = gtcrnFftIm[k];
  }
  const results = await sess.run({
    mix: new ort.Tensor('float32', mixData, [1, GTCRN_BINS, 1, 2]),
    conv_cache: convCache,
    tra_cache: traCache,
    inter_cache: interCache,
  });

  convCache = results.conv_cache_out;
  traCache = results.tra_cache_out;
  interCache = results.inter_cache_out;
  const enhData = results.enh.data;

  const specRe = new Float32Array(N_FFT_512);
  const specIm = new Float32Array(N_FFT_512);
  for (let k = 0; k < GTCRN_BINS; k++) {
    specRe[k] = enhData[2 * k];
    specIm[k] = enhData[2 * k + 1];
  }
  for (let k = 1; k < N_FFT_512 / 2; k++) {
    specRe[N_FFT_512 - k] = specRe[k];
    specIm[N_FFT_512 - k] = -specIm[k];
  }
  for (let i = 0; i < N_FFT_512; i++) {
    const r = bitRev512[i];
    gtcrnFftRe[i] = specRe[r];
    gtcrnFftIm[i] = specIm[r];
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
    chrome.runtime.sendMessage({
      type: 'OFFSCREEN_STATS',
      avgInferenceMs: infSum / infCount,
    }).catch(() => {});
    infSum = 0;
    infCount = 0;
  }
  return outHop;
}

async function processMdxHop(newHop) {
  const t0 = performance.now();
  const sess = sessions[activeEngine];
  const compensate = MDX_MODELS[activeEngine].compensate;

  // Shift rolling context and append newHop
  mdxContext.copyWithin(0, mdxHopSize);
  mdxContext.set(newHop, MDX_CONTEXT_SAMPLES - mdxHopSize);

  // Zero-pad ends for center=True STFT
  mdxPadded.fill(0);
  mdxPadded.set(mdxContext, mdxPadLen);

  // Build [1, 4, 3072, 256] tensor
  const planeSize = MDX_DIM_F * MDX_DIM_T;
  const tensorData = new Float32Array(4 * planeSize);

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
      tensorData[idx] = re;
      tensorData[planeSize + idx] = im;
      tensorData[2 * planeSize + idx] = re;
      tensorData[3 * planeSize + idx] = im;
    }
  }

  const inputName = sess.inputNames[0];
  const outputName = sess.outputNames[0];
  const results = await sess.run({
    [inputName]: new ort.Tensor('float32', tensorData, [1, 4, MDX_DIM_F, MDX_DIM_T]),
  });
  const outData = results[outputName].data;

  // Inverse STFT + Overlap-Add
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

  // Extract newest mdxChunkSize samples (normalized by window squared sum)
  const vocalChunk = new Float32Array(mdxChunkSize);
  const extractStart = mdxPadLen + MDX_CONTEXT_SAMPLES - mdxChunkSize;
  for (let i = 0; i < mdxChunkSize; i++) {
    const pos = extractStart + i;
    const denom = mdxWinSum[pos] > 1e-8 ? mdxWinSum[pos] : 1.0;
    vocalChunk[i] = (mdxOla[pos] / denom) * compensate;
  }

  // Crossfade head with previous tail and emit mdxHopSize samples
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

  // Gentle AGC
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
  chrome.runtime.sendMessage({
    type: 'OFFSCREEN_STATS',
    avgInferenceMs: dt,
  }).catch(() => {});

  return emitted;
}

// ===================== Audio Lifecycle =====================
async function startAudio(streamId, engine, delaySec, strength) {
  await stopAudio();
  activeEngine = engine;

  await getSession(engine);

  if (engine === 'gtcrn') {
    resetGtcrnCaches();
  } else {
    configureMdxDelay(delaySec);
  }

  mediaStream = await navigator.mediaDevices.getUserMedia({
    audio: {
      mandatory: {
        chromeMediaSource: 'tab',
        chromeMediaSourceId: streamId,
      },
    },
  });

  audioCtx = engine === 'gtcrn'
    ? new AudioContext()
    : new AudioContext({ sampleRate: 44100 });

  await audioCtx.audioWorklet.addModule('processor.worklet.js');

  const source = audioCtx.createMediaStreamSource(mediaStream);
  workletNode = new AudioWorkletNode(audioCtx, 'music-remover-worklet-processor', {
    numberOfInputs: 1,
    numberOfOutputs: 1,
    outputChannelCount: [2],
    processorOptions: {
      mode: engine === 'gtcrn' ? 'gtcrn' : 'mdx',
      hopSamples: engine === 'gtcrn' ? GTCRN_HOP : mdxHopSize,
      strength,
    },
  });

  isRunning = true;
  workletNode.port.onmessage = (e) => {
    if (e.data.type === 'INPUT_HOP') {
      const samples = e.data.samples;
      processingQueue = processingQueue.then(async () => {
        if (!isRunning || !workletNode) return;
        const enhanced = activeEngine === 'gtcrn'
          ? await processGtcrnHop(samples)
          : await processMdxHop(samples);
        if (!isRunning || !workletNode) return;
        workletNode.port.postMessage({ type: 'ENHANCED_HOP', samples: enhanced }, [enhanced.buffer]);
      });
    }
  };

  source.connect(workletNode);
  workletNode.connect(audioCtx.destination);
  return activeProvider;
}

async function stopAudio() {
  isRunning = false;
  processingQueue = Promise.resolve();
  if (workletNode) {
    workletNode.disconnect();
    workletNode.port.onmessage = null;
    workletNode = null;
  }
  if (mediaStream) {
    mediaStream.getTracks().forEach((t) => t.stop());
    mediaStream = null;
  }
  if (audioCtx) {
    await audioCtx.close().catch(() => {});
    audioCtx = null;
  }
}

chrome.runtime.onMessage.addListener((msg, _sender, sendResponse) => {
  (async () => {
    try {
      if (msg.type === 'OFFSCREEN_START') {
        const provider = await startAudio(
          msg.streamId,
          msg.engine || 'kimvocal',
          msg.delaySec ?? 4.5,
          msg.strength ?? 1.0
        );
        sendResponse({ ok: true, provider });
      } else if (msg.type === 'OFFSCREEN_STOP') {
        await stopAudio();
        sendResponse({ ok: true });
      } else if (msg.type === 'OFFSCREEN_STRENGTH') {
        if (workletNode) {
          workletNode.port.postMessage({ type: 'SET_STRENGTH', strength: msg.strength });
        }
        sendResponse({ ok: true });
      }
    } catch (err) {
      sendResponse({ ok: false, error: err.message || String(err) });
    }
  })();
  return true;
});
