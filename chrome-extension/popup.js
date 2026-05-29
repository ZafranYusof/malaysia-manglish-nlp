document.addEventListener('DOMContentLoaded', init);

let currentTab = 'sentiment';
let apiUrl = 'http://localhost:8000';

async function init() {
  const saved = await chrome.storage.sync.get({ apiUrl: 'http://localhost:8000', theme: 'light' });
  apiUrl = saved.apiUrl;
  document.getElementById('api-url').value = apiUrl;

  if (saved.theme === 'dark') {
    document.documentElement.setAttribute('data-theme', 'dark');
  }

  setupTabs();
  setupAnalyze();
  setupSettings();
  setupTheme();
}

function setupTabs() {
  document.querySelectorAll('.tab').forEach(tab => {
    tab.addEventListener('click', () => {
      document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
      tab.classList.add('active');
      currentTab = tab.dataset.tab;
    });
  });
}

function setupAnalyze() {
  const btn = document.getElementById('analyze-btn');
  const input = document.getElementById('input-text');

  btn.addEventListener('click', () => {
    const text = input.value.trim();
    if (!text) return;
    analyze(text, currentTab);
  });

  input.addEventListener('keydown', e => {
    if (e.key === 'Enter' && e.ctrlKey) {
      btn.click();
    }
  });
}

async function analyze(text, type) {
  const loading = document.getElementById('loading');
  const results = document.getElementById('results');
  const btn = document.getElementById('analyze-btn');

  loading.classList.remove('hidden');
  results.classList.add('hidden');
  btn.disabled = true;

  try {
    let endpoint = '/analyze';
    if (type === 'sentiment') endpoint = '/sentiment';
    else if (type === 'normalize') endpoint = '/normalize';
    else if (type === 'ner') endpoint = '/ner';
    else if (type === 'translate') endpoint = '/translate';

    const res = await fetch(`${apiUrl}${endpoint}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text })
    });

    if (!res.ok) throw new Error(`API error: ${res.status}`);

    const data = await res.json();
    renderResults(data, type);
    results.classList.remove('hidden');
  } catch (err) {
    results.innerHTML = `<div class="error-msg">Error: ${err.message}<br><br>Make sure the Manglish NLP API is running at ${apiUrl}</div>`;
    results.classList.remove('hidden');
  } finally {
    loading.classList.add('hidden');
    btn.disabled = false;
  }
}

function renderResults(data, type) {
  const results = document.getElementById('results');

  if (type === 'sentiment') {
    const label = data.label || data.sentiment || 'unknown';
    const score = data.score ?? data.confidence ?? null;
    const cssClass = `sentiment-${label.toLowerCase()}`;
    results.innerHTML = `
      <div class="result-item">
        <div class="result-label">Sentiment</div>
        <div class="result-value">
          <span class="sentiment-badge ${cssClass}">${label.toUpperCase()}</span>
        </div>
      </div>
      ${score !== null ? `
      <div class="result-item">
        <div class="result-label">Confidence</div>
        <div class="result-value">${(score * 100).toFixed(1)}%</div>
      </div>` : ''}
    `;
  } else if (type === 'normalize') {
    const normalized = data.normalized || data.text || JSON.stringify(data);
    results.innerHTML = `
      <div class="result-item">
        <div class="result-label">Normalized Text</div>
        <div class="result-value normalized-text">${escapeHtml(normalized)}</div>
      </div>
    `;
  } else if (type === 'ner') {
    const entities = data.entities || data.results || [];
    if (entities.length === 0) {
      results.innerHTML = `<div class="result-item"><div class="result-value">No entities found</div></div>`;
    } else {
      results.innerHTML = `
        <div class="result-item">
          <div class="result-label">Entities (${entities.length})</div>
          <div class="result-value">
            ${entities.map(e => `<span class="entity-tag">${escapeHtml(e.text || e.word)}<span class="entity-type">${escapeHtml(e.label || e.entity || '')}</span></span>`).join('')}
          </div>
        </div>
      `;
    }
  } else if (type === 'translate') {
    const translated = data.translated || data.text || JSON.stringify(data);
    results.innerHTML = `
      <div class="result-item">
        <div class="result-label">Translation</div>
        <div class="result-value normalized-text">${escapeHtml(translated)}</div>
      </div>
    `;
  } else {
    results.innerHTML = `<div class="result-item"><pre class="result-value">${escapeHtml(JSON.stringify(data, null, 2))}</pre></div>`;
  }
}

function escapeHtml(str) {
  const div = document.createElement('div');
  div.textContent = str;
  return div.innerHTML;
}

function setupSettings() {
  const panel = document.getElementById('settings-panel');
  const btn = document.getElementById('settings-btn');
  const save = document.getElementById('save-settings');
  const close = document.getElementById('close-settings');

  btn.addEventListener('click', () => panel.classList.remove('hidden'));
  close.addEventListener('click', () => panel.classList.add('hidden'));

  save.addEventListener('click', async () => {
    apiUrl = document.getElementById('api-url').value.trim();
    await chrome.storage.sync.set({ apiUrl });
    panel.classList.add('hidden');
  });
}

function setupTheme() {
  const toggle = document.getElementById('theme-toggle');
  toggle.addEventListener('click', async () => {
    const current = document.documentElement.getAttribute('data-theme');
    const next = current === 'dark' ? 'light' : 'dark';
    document.documentElement.setAttribute('data-theme', next);
    await chrome.storage.sync.set({ theme: next });
  });
}
