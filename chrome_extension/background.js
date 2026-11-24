
// Background Service Worker (Manifest V3)
// Coordinates tabCapture streamId creation and offscreen audio processing.

const OFFSCREEN_URL = 'offscreen.html';

let state = {
  active: false,
  tabId: null,
  engine: 'kimvocal', // 'kimvocal' | 'kimvocal_fp16' | 'vocft' | 'gtcrn'
  backendMode: 'webgpu', // 'webgpu' | 'wasm'
  delaySec: 5.5,
  strength: 1.0,
  avgInferenceMs: 0,
  provider: '',
  statusText: 'Idle',
  error: null,
};

async function ensureOffscreenDocument() {
  const existing = await chrome.offscreen.hasDocument();
  if (!existing) {
    await chrome.offscreen.createDocument({
      url: OFFSCREEN_URL,
      reasons: ['USER_MEDIA', 'AUDIO_PLAYBACK'],
      justification: 'Capture tab audio and run ONNX speech/vocal separation in real time.',
    });
  }
}

chrome.runtime.onMessage.addListener((msg, _sender, sendResponse) => {
  (async () => {
    try {
      if (msg.type === 'GET_STATUS') {
        sendResponse({ ok: true, state });
        return;
      }

      if (msg.type === 'START_CAPTURE') {
        const {
          tabId,
          engine = 'kimvocal',
          backendMode = 'webgpu',
          delaySec = 5.5,
          strength = 1.0,
        } = msg;

        state.engine = engine;
        state.backendMode = backendMode;
        state.delaySec = delaySec;
        state.strength = strength;
        state.error = null;
        state.avgInferenceMs = 0;
        state.statusText = 'Loading model...';

        if (state.active) {
          await chrome.runtime.sendMessage({ type: 'OFFSCREEN_STOP' }).catch(() => { });
        }

        await ensureOffscreenDocument();
        const streamId = await chrome.tabCapture.getMediaStreamId({ targetTabId: tabId });

        const resp = await chrome.runtime.sendMessage({
          type: 'OFFSCREEN_START',
          streamId,
          engine,
          backendMode,
          delaySec,
          strength,
        });

        if (resp && resp.ok) {
          state.active = true;
          state.tabId = tabId;
          state.provider = resp.provider || 'wasm';
          state.statusText = 'Active';
          sendResponse({ ok: true, state });
        } else {
          state.active = false;
          state.error = resp?.error || 'Failed to start offscreen audio.';
          state.statusText = 'Error';
          sendResponse({ ok: false, error: state.error });
        }
        return;
      }

      if (msg.type === 'STOP_CAPTURE') {
        await chrome.runtime.sendMessage({ type: 'OFFSCREEN_STOP' }).catch(() => { });
        state.active = false;
        state.tabId = null;
        state.avgInferenceMs = 0;
        state.statusText = 'Idle';
        sendResponse({ ok: true, state });
        return;
      }

      if (msg.type === 'SET_CONFIG') {
        if (!state.active) {
          if (msg.engine) state.engine = msg.engine;
          if (msg.backendMode) state.backendMode = msg.backendMode;
          if (msg.delaySec !== undefined) state.delaySec = msg.delaySec;
        }
        sendResponse({ ok: true, state });
        return;
      }

      if (msg.type === 'SET_STRENGTH') {
        state.strength = msg.strength;
        if (state.active) {
          await chrome.runtime.sendMessage({
            type: 'OFFSCREEN_STRENGTH',
            strength: msg.strength,
          }).catch(() => { });
        }
        sendResponse({ ok: true, state });
        return;
      }

      if (msg.type === 'OFFSCREEN_STATS') {
        state.avgInferenceMs = msg.avgInferenceMs;
        if (msg.statusText) state.statusText = msg.statusText;
        return;
      }
    } catch (err) {
      state.active = false;
      state.error = err.message || String(err);
      state.statusText = 'Error';
      sendResponse({ ok: false, error: state.error });
    }
  })();

  return true;
});
