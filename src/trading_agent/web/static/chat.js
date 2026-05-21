(() => {
  const els = {
    sidebar: document.getElementById('chat-sidebar'),
    collapse: document.getElementById('chat-collapse'),
    collapseIcon: document.getElementById('chat-collapse-icon'),
    sidebarLabel: document.getElementById('chat-sidebar-label'),
    body: document.getElementById('chat-body'),
    model: document.getElementById('chat-model'),
    modelLimit: document.getElementById('chat-model-limit'),
    reset: document.getElementById('chat-reset'),
    messages: document.getElementById('chat-messages'),
    attachments: document.getElementById('chat-attachments'),
    tokens: document.getElementById('chat-tokens'),
    pct: document.getElementById('chat-context-pct'),
    form: document.getElementById('chat-form'),
    input: document.getElementById('chat-input'),
    send: document.getElementById('chat-send'),
    file: document.getElementById('chat-file'),
    error: document.getElementById('chat-error'),
  };

  let models = [];
  let pendingImages = [];
  let busy = false;

  const STORE_KEY = 'tradingAgent.model';
  const COLLAPSED_KEY = 'tradingAgent.sidebarCollapsed';

  const setCollapsed = (collapsed) => {
    els.sidebar.classList.toggle('w-96', !collapsed);
    els.sidebar.classList.toggle('w-10', collapsed);
    els.body.classList.toggle('hidden', collapsed);
    els.sidebarLabel.classList.toggle('hidden', collapsed);
    els.collapseIcon.textContent = collapsed ? '»' : '«';
    els.collapse.title = collapsed ? 'Expand sidebar' : 'Collapse sidebar';
    localStorage.setItem(COLLAPSED_KEY, collapsed ? '1' : '0');
  };

  els.collapse.addEventListener('click', () => {
    const collapsed = els.sidebar.classList.contains('w-10');
    setCollapsed(!collapsed);
  });

  if (localStorage.getItem(COLLAPSED_KEY) === '1') {
    setCollapsed(true);
  }

  const escapeHtml = (s) => s
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#039;');

  const showError = (msg) => {
    if (!msg) {
      els.error.classList.add('hidden');
      els.error.textContent = '';
      return;
    }
    els.error.textContent = msg;
    els.error.classList.remove('hidden');
  };

  const updateContextBar = (tokens) => {
    const limit = currentModelLimit();
    const pct = limit ? Math.min(100, Math.round((tokens / limit) * 100)) : 0;
    els.tokens.textContent = tokens.toLocaleString();
    els.pct.textContent = pct;
  };

  const currentModelLimit = () => {
    const m = models.find((x) => x.id === els.model.value);
    return m ? m.context_limit : 0;
  };

  const renderModelLimit = () => {
    const m = models.find((x) => x.id === els.model.value);
    els.modelLimit.textContent = m ? `${m.context_limit.toLocaleString()} ctx` : '';
  };

  const renderAttachments = () => {
    els.attachments.innerHTML = '';
    pendingImages.forEach((src, i) => {
      const wrap = document.createElement('div');
      wrap.className = 'relative';
      wrap.innerHTML = `
        <img src="${src}" class="w-12 h-12 object-cover rounded border border-slate-700">
        <button type="button"
                data-i="${i}"
                class="absolute -top-1 -right-1 bg-slate-800 text-rose-400 rounded-full w-4 h-4 text-xs leading-none">
          ×
        </button>`;
      wrap.querySelector('button').addEventListener('click', (e) => {
        const idx = Number(e.currentTarget.dataset.i);
        pendingImages.splice(idx, 1);
        renderAttachments();
      });
      els.attachments.appendChild(wrap);
    });
  };

  const renderMessage = (m) => {
    if (m.role === 'tool') return null;
    if (m.role === 'system') return null;
    const bubble = document.createElement('div');
    const isUser = m.role === 'user';
    bubble.className = isUser
      ? 'ml-6 bg-slate-800 rounded-lg px-3 py-2'
      : 'mr-6 bg-slate-900 border border-slate-800 rounded-lg px-3 py-2';
    const tag = isUser ? 'you' : (m.model || 'assistant').split('/').pop();
    const toolNote = m.tool_calls && m.tool_calls.length
      ? `<p class="text-xs text-slate-500 italic mt-1">called ${m.tool_calls.map(tc => (tc.function && tc.function.name) || 'tool').join(', ')}</p>`
      : '';
    const imageHtml = (m.images || []).map(src =>
      `<img src="${escapeHtml(src)}" class="mt-2 max-w-full rounded border border-slate-700">`
    ).join('');
    const textHtml = m.content ? `<p class="chat-bubble">${escapeHtml(m.content)}</p>` : '';
    bubble.innerHTML = `
      <p class="text-xs text-slate-500 mono mb-1">${escapeHtml(tag)}</p>
      ${textHtml}
      ${imageHtml}
      ${toolNote}`;
    return bubble;
  };

  const renderHistory = (messages) => {
    els.messages.innerHTML = '';
    if (!messages || messages.length === 0) {
      const empty = document.createElement('p');
      empty.className = 'text-slate-500 text-xs italic text-center mt-4';
      empty.textContent = 'Ask about an account, a position, or your recent trades.';
      els.messages.appendChild(empty);
      return;
    }
    messages.forEach(m => {
      const node = renderMessage(m);
      if (node) els.messages.appendChild(node);
    });
    els.messages.scrollTop = els.messages.scrollHeight;
  };

  const setBusy = (b) => {
    busy = b;
    els.send.disabled = b;
    els.send.classList.toggle('opacity-50', b);
    els.send.textContent = b ? '...' : 'send';
  };

  const loadModels = async () => {
    const r = await fetch('/chat/models');
    const data = await r.json();
    models = data.models;
    els.model.innerHTML = '';
    models.forEach(m => {
      const opt = document.createElement('option');
      opt.value = m.id;
      opt.textContent = m.display;
      els.model.appendChild(opt);
    });
    const saved = localStorage.getItem(STORE_KEY);
    els.model.value = saved && models.some(m => m.id === saved) ? saved : data.default;
    renderModelLimit();
  };

  const loadHistory = async () => {
    const r = await fetch('/chat/history');
    const data = await r.json();
    renderHistory(data.messages);
    updateContextBar(data.tokens);
  };

  const send = async () => {
    if (busy) return;
    const text = els.input.value.trim();
    if (!text && pendingImages.length === 0) return;

    showError(null);
    setBusy(true);

    const placeholder = renderMessage({ role: 'user', content: text, images: pendingImages });
    if (placeholder) els.messages.appendChild(placeholder);
    els.messages.scrollTop = els.messages.scrollHeight;

    try {
      const r = await fetch('/chat/send', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          text,
          images: pendingImages,
          model: els.model.value,
        }),
      });
      if (!r.ok) {
        const err = await r.json().catch(() => ({ detail: 'request failed' }));
        throw new Error(err.detail || 'request failed');
      }
      const data = await r.json();
      pendingImages = [];
      renderAttachments();
      els.input.value = '';
      renderHistory(data.messages);
      updateContextBar(data.tokens);
    } catch (e) {
      showError(e.message);
    } finally {
      setBusy(false);
    }
  };

  const reset = async () => {
    if (busy) return;
    if (!confirm('Clear the conversation?')) return;
    await fetch('/chat/reset', { method: 'POST' });
    pendingImages = [];
    renderAttachments();
    renderHistory([]);
    updateContextBar(0);
  };

  const handleImage = (file) => new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });

  els.form.addEventListener('submit', (e) => {
    e.preventDefault();
    send();
  });

  els.input.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      send();
    }
  });

  els.input.addEventListener('paste', async (e) => {
    const items = e.clipboardData ? [...e.clipboardData.items] : [];
    const imageItems = items.filter(i => i.type && i.type.startsWith('image/'));
    if (imageItems.length === 0) return;
    e.preventDefault();
    for (const item of imageItems) {
      const file = item.getAsFile();
      if (file) pendingImages.push(await handleImage(file));
    }
    renderAttachments();
  });

  els.file.addEventListener('change', async (e) => {
    for (const file of e.target.files) {
      pendingImages.push(await handleImage(file));
    }
    e.target.value = '';
    renderAttachments();
  });

  els.reset.addEventListener('click', reset);

  els.model.addEventListener('change', () => {
    localStorage.setItem(STORE_KEY, els.model.value);
    renderModelLimit();
  });

  (async () => {
    try {
      await loadModels();
      await loadHistory();
    } catch (e) {
      showError(`init failed: ${e.message}`);
    }
  })();
})();
