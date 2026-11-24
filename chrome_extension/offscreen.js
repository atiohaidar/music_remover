// Offscreen Audio Coordinator
// Delegates heavy ONNX inference to dedicated worker.js so popup.html & UI never freeze.

let aiWorker = null;
let audioCtx = null;
let mediaStream = null;
let workletNode = null;
let isRunning = false;

let workerBusy = false;
const pendingHops = [];

function ensureWorker() {
  if (!aiWorker) {
    aiWorker = new Worker(chrome.runtime.getURL('worker.js'));
  }
  return aiWorker;
}

function initWorker(engine, backendMode, delaySec) {
  return new Promise((resolve, reject) => {
    const w = ensureWorker();
    w.onmessage = (e) => {
      const msg = e.data;
      if (msg.type === 'INIT_DONE') {
        resolve(msg);
      } else if (msg.type === 'STATUS') {
        chrome.runtime.sendMessage({
          type: 'OFFSCREEN_STATS',
          avgInferenceMs: msg.avgInferenceMs || 0,
          statusText: msg.statusText,
        }).catch(() => { });
      } else if (msg.type === 'ERROR') {
        reject(new Error(msg.error));
      }
    };
    w.onerror = (err) => reject(new Error(err.message || 'Worker failed to load'));
    w.postMessage({
      type: 'INIT',
      baseExtensionUrl: chrome.runtime.getURL(''),
      engine,
      backendMode,
      delaySec,
    });
  });
}

function dispatchNextHop() {
  if (!isRunning || workerBusy || pendingHops.length === 0 || !aiWorker) return;
  workerBusy = true;
  const next = pendingHops.shift();
  aiWorker.postMessage({ type: 'HOP', samples: next }, [next.buffer]);
}

async function startAudio(streamId, engine, backendMode, delaySec, strength) {
  await stopAudio();

  // Redeem streamId IMMEDIATELY before loading/downloading the ONNX model,
  // because Chrome tabCapture streamIds expire quickly.
  mediaStream = await navigator.mediaDevices.getUserMedia({
    audio: {
      mandatory: {
        chromeMediaSource: 'tab',
        chromeMediaSourceId: streamId,
      },
    },
  });

  let initInfo;
  try {
    initInfo = await initWorker(engine, backendMode, delaySec);
  } catch (err) {
    await stopAudio();
    throw err;
  }

  audioCtx = engine === 'gtcrn'
    ? new AudioContext()
    : new AudioContext({ sampleRate: 44100 });

  if (audioCtx.state === 'suspended') {
    await audioCtx.resume();
  }

  await audioCtx.audioWorklet.addModule('processor.worklet.js');

  const source = audioCtx.createMediaStreamSource(mediaStream);
  workletNode = new AudioWorkletNode(audioCtx, 'music-remover-worklet-processor', {
    numberOfInputs: 1,
    numberOfOutputs: 1,
    outputChannelCount: [2],
    processorOptions: {
      mode: engine === 'gtcrn' ? 'gtcrn' : 'mdx',
      hopSamples: initInfo.hopSamples,
      strength,
    },
  });

  isRunning = true;
  workerBusy = false;
  pendingHops.length = 0;

  aiWorker.onmessage = (e) => {
    const msg = e.data;
    if (msg.type === 'ENHANCED_HOP') {
      workerBusy = false;
      if (isRunning && workletNode) {
        workletNode.port.postMessage({ type: 'ENHANCED_HOP', samples: msg.samples }, [msg.samples.buffer]);
      }
      dispatchNextHop();
    } else if (msg.type === 'STATS') {
      chrome.runtime.sendMessage({
        type: 'OFFSCREEN_STATS',
        avgInferenceMs: msg.avgInferenceMs,
      }).catch(() => { });
    } else if (msg.type === 'ERROR') {
      workerBusy = false;
      chrome.runtime.sendMessage({
        type: 'OFFSCREEN_STATS',
        avgInferenceMs: 0,
        statusText: `Error: ${msg.error}`,
      }).catch(() => { });
      dispatchNextHop();
    }
  };

  workletNode.port.onmessage = (e) => {
    if (!isRunning || e.data.type !== 'INPUT_HOP') return;
    // Cap queued hops so inference backlog never balloons
    const maxQueue = engine === 'gtcrn' ? 8 : 2;
    while (pendingHops.length >= maxQueue) {
      pendingHops.shift();
    }
    pendingHops.push(e.data.samples);
    dispatchNextHop();
  };

  source.connect(workletNode);
  workletNode.connect(audioCtx.destination);
  return initInfo.provider;
}

async function stopAudio() {
  const wasBusy = workerBusy;
  isRunning = false;
  workerBusy = false;
  pendingHops.length = 0;

  if (wasBusy && aiWorker) {
    aiWorker.terminate();
    aiWorker = null;
  }

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
    await audioCtx.close().catch(() => { });
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
          msg.backendMode || 'webgpu',
          msg.delaySec ?? 5.5,
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
