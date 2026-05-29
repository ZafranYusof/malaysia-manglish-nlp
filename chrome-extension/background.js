chrome.runtime.onInstalled.addListener(() => {
  chrome.contextMenus.create({
    id: 'manglish-parent',
    title: 'Analyze with Manglish NLP',
    contexts: ['selection']
  });

  chrome.contextMenus.create({
    id: 'manglish-sentiment',
    parentId: 'manglish-parent',
    title: 'Sentiment Analysis',
    contexts: ['selection']
  });

  chrome.contextMenus.create({
    id: 'manglish-normalize',
    parentId: 'manglish-parent',
    title: 'Normalize Text',
    contexts: ['selection']
  });

  chrome.contextMenus.create({
    id: 'manglish-ner',
    parentId: 'manglish-parent',
    title: 'Named Entity Recognition',
    contexts: ['selection']
  });

  chrome.contextMenus.create({
    id: 'manglish-translate',
    parentId: 'manglish-parent',
    title: 'Translate',
    contexts: ['selection']
  });
});

chrome.contextMenus.onClicked.addListener(async (info, tab) => {
  if (!info.menuItemId.startsWith('manglish-') || info.menuItemId === 'manglish-parent') return;

  const type = info.menuItemId.replace('manglish-', '');
  const text = info.selectionText;

  if (!text) return;

  chrome.tabs.sendMessage(tab.id, {
    action: 'show-loading',
    x: info.pageX,
    y: info.pageY
  });

  try {
    const { apiUrl } = await chrome.storage.sync.get({ apiUrl: 'http://localhost:8000' });

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

    chrome.tabs.sendMessage(tab.id, {
      action: 'show-result',
      type,
      data,
      x: info.pageX,
      y: info.pageY
    });
  } catch (err) {
    chrome.tabs.sendMessage(tab.id, {
      action: 'show-error',
      message: err.message,
      x: info.pageX,
      y: info.pageY
    });
  }
});
