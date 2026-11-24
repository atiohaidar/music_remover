// Background Service Worker (Manifest V3)
// Coordinates tabCapture streamId creation, toolbar badge, persistent config, and offscreen audio processing.

const OFFSCREEN_URL = 'offscreen.html';
const STORAGE_KEY = 'musicRemoverConfig';

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

function updateBadge(isActive, hasError = false) {
  if (isActive) {
    chrome.action.setBadgeText({ text: 'ON' }).catch(() => { });
    chrome.action.setBadgeBackgroundColor({ color: '#15803d' }).catch(() => { });
  } else if (hasError) {
    chrome.action.setBadgeText({ text: '!' }).catch(() => { });
    chrome.action.setBadgeBackgroundColor({ color: '#dc2626' }).catch(() => { });
  } else {
    chrome.action.setBadgeText({ text: '' }).catch(() => { });
  }
}

function savePersistedConfig() {
  chrome.storage.local.set({
    [STORAGE_KEY]: {
      engine: state.engine,
      backendMode: state.backendMode,
      delaySec: state.delaySec,
      strength: state.strength,
    },
  }).catch(() => { });
}

const configReady = (async () => {
  try {
    const stored = await chrome.storage.local.get(STORAGE_KEY);
    const cfg = stored && stored[STORAGE_KEY];
    if (cfg && !state.active) {
      if (cfg.engine) state.engine = cfg.engine;
      if (cfg.backendMode) state.backendMode = cfg.backendMode;
      if (typeof cfg.delaySec === 'number') state.delaySec = cfg.delaySec;
      if (typeof cfg.strength === 'number') state.strength = cfg.strength;
    }
  } catch (_) { }
  updateBadge(state.active, false);
})();

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

async function stopActiveCapture() {
  await chrome.runtime.sendMessage({ type: 'OFFSCREEN_STOP' }).catch(() => { });
  state.active = false;
  state.tabId = null;
  state.avgInferenceMs = 0;
  state.statusText = 'Idle';
  state.error = null;
  updateBadge(false, false);
}

// Auto-cleanup when the captured tab is closed by the user
chrome.tabs.onRemoved.addListener((closedTabId) => {
  if (state.active && state.tabId === closedTabId) {
    stopActiveCapture();
  }
});

chrome.runtime.onMessage.addListener((msg, _sender, sendResponse) => {
  (async () => {
    await configReady;
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
        savePersistedConfig();

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
          updateBadge(true, false);
          sendResponse({ ok: true, state });
        } else {
          state.active = false;
          state.tabId = null;
          state.error = resp?.error || 'Failed to start offscreen audio.';
          state.statusText = 'Error';
          updateBadge(false, true);
          sendResponse({ ok: false, error: state.error });
        }
        return;
      }

      if (msg.type === 'STOP_CAPTURE') {
        await stopActiveCapture();
        sendResponse({ ok: true, state });
        return;
      }

      if (msg.type === 'SET_CONFIG') {
        if (!state.active) {
          if (msg.engine) state.engine = msg.engine;
          if (msg.backendMode) state.backendMode = msg.backendMode;
          if (msg.delaySec !== undefined) state.delaySec = msg.delaySec;
          savePersistedConfig();
        }
        sendResponse({ ok: true, state });
        return;
      }

      if (msg.type === 'SET_STRENGTH') {
        state.strength = msg.strength;
        savePersistedConfig();
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
      state.tabId = null;
      state.error = err.message || String(err);
      state.statusText = 'Error';
      updateBadge(false, true);
      sendResponse({ ok: false, error: state.error });
    }
  })();

  return true;
});
