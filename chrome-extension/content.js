let tooltip = null;

function removeTooltip() {
  if (tooltip) {
    tooltip.remove();
    tooltip = null;
  }
}

function createTooltip(x, y) {
  removeTooltip();
  tooltip = document.createElement('div');
  tooltip.className = 'manglish-tooltip';
  tooltip.style.left = `${x}px`;
  tooltip.style.top = `${y + 20}px`;
  document.body.appendChild(tooltip);
  return tooltip;
}

function escapeHtml(str) {
  const div = document.createElement('div');
  div.textContent = str;
  return div.innerHTML;
}

function renderSentiment(data) {
  const label = data.label || data.sentiment || 'unknown';
  const score = data.score ?? data.confidence ?? null;
  const cssClass = `manglish-sentiment-${label.toLowerCase()}`;
  return `
    <div class="manglish-result-header">Sentiment Analysis</div>
    <div class="${cssClass}">${label.toUpperCase()}</div>
    ${score !== null ? `<div class="manglish-score">Confidence: ${(score * 100).toFixed(1)}%</div>` : ''}
  `;
}

function renderNormalize(data) {
  const normalized = data.normalized || data.text || JSON.stringify(data);
  return `
    <div class="manglish-result-header">Normalized</div>
    <div class="manglish-normalized">${escapeHtml(normalized)}</div>
  `;
}

function renderNer(data) {
  const entities = data.entities || data.results || [];
  if (entities.length === 0) {
    return `<div class="manglish-result-header">NER</div><div>No entities found</div>`;
  }
  const tags = entities.map(e => {
    const text = e.text || e.word;
    const type = e.label || e.entity || '';
    return `<span class="manglish-entity-tag">${escapeHtml(text)} <small>${escapeHtml(type)}</small></span>`;
  }).join(' ');
  return `
    <div class="manglish-result-header">Entities (${entities.length})</div>
    <div class="manglish-entities">${tags}</div>
  `;
}

function renderTranslate(data) {
  const translated = data.translated || data.text || JSON.stringify(data);
  return `
    <div class="manglish-result-header">Translation</div>
    <div class="manglish-normalized">${escapeHtml(translated)}</div>
  `;
}

chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  if (msg.action === 'show-loading') {
    const t = createTooltip(msg.x, msg.y);
    t.innerHTML = '<div class="manglish-loading"><div class="manglish-spinner"></div> Analyzing...</div>';
  }

  if (msg.action === 'show-result') {
    const t = createTooltip(msg.x, msg.y);
    let html = '<button class="manglish-close">&times;</button>';

    if (msg.type === 'sentiment') html += renderSentiment(msg.data);
    else if (msg.type === 'normalize') html += renderNormalize(msg.data);
    else if (msg.type === 'ner') html += renderNer(msg.data);
    else if (msg.type === 'translate') html += renderTranslate(msg.data);
    else html += `<pre>${escapeHtml(JSON.stringify(msg.data, null, 2))}</pre>`;

    t.innerHTML = html;
    t.querySelector('.manglish-close').addEventListener('click', removeTooltip);

    if (msg.type === 'ner') {
      highlightEntities(msg.data.entities || msg.data.results || []);
    }
  }

  if (msg.action === 'show-error') {
    const t = createTooltip(msg.x, msg.y);
    t.innerHTML = `
      <button class="manglish-close">&times;</button>
      <div class="manglish-error">Error: ${escapeHtml(msg.message)}</div>
    `;
    t.querySelector('.manglish-close').addEventListener('click', removeTooltip);
  }
});

function highlightEntities(entities) {
  entities.forEach(entity => {
    const text = entity.text || entity.word;
    if (!text) return;

    const walker = document.createTreeWalker(
      document.body,
      NodeFilter.SHOW_TEXT,
      null,
      false
    );

    let node;
    while ((node = walker.nextNode())) {
      if (node.parentElement.closest('.manglish-tooltip')) continue;
      if (node.parentElement.classList.contains('manglish-highlight')) continue;

      const idx = node.textContent.toLowerCase().indexOf(text.toLowerCase());
      if (idx === -1) continue;

      const range = document.createRange();
      range.setStart(node, idx);
      range.setEnd(node, idx + text.length);

      const mark = document.createElement('mark');
      mark.className = 'manglish-highlight';
      mark.title = entity.label || entity.entity || '';
      range.surroundContents(mark);
      break;
    }
  });
}

document.addEventListener('click', e => {
  if (tooltip && !tooltip.contains(e.target)) {
    removeTooltip();
  }
});

document.addEventListener('keydown', e => {
  if (e.key === 'Escape') removeTooltip();
});
