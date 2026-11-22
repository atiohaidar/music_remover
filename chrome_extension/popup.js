const toggleBtn = document.getElementById('toggleBtn');
const engineSelect = document.getElementById('engineSelect');
const delayControl = document.getElementById('delayControl');
const delaySlider = document.getElementById('delaySlider');
const delayVal = document.getElementById('delayVal');
const strengthSlider = document.getElementById('strengthSlider');
const strengthVal = document.getElementById('strengthVal');
const statusBox = document.getElementById('statusBox');

let currentActive = false;
let initializedFromState = false;

function updateDelayVisibility() {
  const isGtcrn = engineSelect.value === 'gtcrn';
  delayControl.style.display = isGtcrn ? 'none' : 'block';
}

function render(state, syncInputs = false) {
  currentActive = !!state.active;
  if (syncInputs || currentActive || !initializedFromState) {
    if (state.engine) engineSelect.value = state.engine;
    if (state.delaySec) {
      delaySlider.value = state.delaySec;
      delayVal.textContent = `${Number(state.delaySec).toFixed(1)} s`;
    }
    const pct = Math.round((state.strength ?? 1.0) * 100);
    strengthSlider.value = pct;
    strengthVal.textContent = `${pct}%`;
    initializedFromState = true;
  }

  engineSelect.disabled = currentActive;
  delaySlider.disabled = currentActive;
  updateDelayVisibility();

  if (state.error) {
    statusBox.className = 'status error';
    statusBox.textContent = `Error: ${state.error}`;
  } else if (currentActive) {
    statusBox.className = 'status';
    const prov = state.provider ? ` [${state.provider.toUpperCase()}]` : '';
    let timing = ' · Buffering...';
    if (state.avgInferenceMs > 0) {
      timing = state.engine === 'gtcrn'
        ? ` · ${state.avgInferenceMs.toFixed(1)} ms / 16 ms`
        : ` · ${(state.avgInferenceMs / 1000).toFixed(2)} s / chunk`;
    }
    statusBox.textContent = `Active${prov}${timing}`;
  } else {
    statusBox.className = 'status';
    statusBox.textContent = 'Idle — pilih model dan klik Start.';
  }

  toggleBtn.textContent = currentActive ? 'Stop' : 'Start on This Tab';
  toggleBtn.className = currentActive ? 'stop' : 'start';
}

async function refreshStatus(syncInputs = false) {
  const resp = await chrome.runtime.sendMessage({ type: 'GET_STATUS' });
  if (resp?.ok) render(resp.state, syncInputs);
}

engineSelect.addEventListener('change', () => {
  updateDelayVisibility();
  chrome.runtime.sendMessage({ type: 'SET_CONFIG', engine: engineSelect.value }).catch(() => {});
});

delaySlider.addEventListener('input', () => {
  delayVal.textContent = `${Number(delaySlider.value).toFixed(1)} s`;
  chrome.runtime.sendMessage({ type: 'SET_CONFIG', delaySec: Number(delaySlider.value) }).catch(() => {});
});

strengthSlider.addEventListener('input', async () => {
  const pct = Number(strengthSlider.value);
  strengthVal.textContent = `${pct}%`;
  await chrome.runtime.sendMessage({
    type: 'SET_STRENGTH',
    strength: pct / 100,
  });
});

toggleBtn.addEventListener('click', async () => {
  toggleBtn.disabled = true;
  if (!currentActive) {
    statusBox.className = 'status';
    statusBox.textContent = 'Loading model (~64 MB) & starting...';
  }
  try {
    if (currentActive) {
      const resp = await chrome.runtime.sendMessage({ type: 'STOP_CAPTURE' });
      if (resp?.ok) render(resp.state);
    } else {
      const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
      const engine = engineSelect.value;
      const delaySec = Number(delaySlider.value);
      const strength = Number(strengthSlider.value) / 100;

      const resp = await chrome.runtime.sendMessage({
        type: 'START_CAPTURE',
        tabId: tab.id,
        engine,
        delaySec,
        strength,
      });
      if (resp?.ok) {
        render(resp.state);
      } else {
        render({ active: false, engine, delaySec, strength, error: resp?.error || 'Failed to start' });
      }
    }
  } finally {
    toggleBtn.disabled = false;
  }
});

refreshStatus(true);
setInterval(() => {
  if (currentActive) refreshStatus(false);
}, 1000);
