(() => {
  const els = {
    tree: document.getElementById('notes-tree'),
    new: document.getElementById('notes-new'),
    path: document.getElementById('editor-path'),
    dirty: document.getElementById('editor-dirty'),
    textarea: document.getElementById('editor-textarea'),
    preview: document.getElementById('editor-preview'),
    modeEdit: document.getElementById('editor-mode-edit'),
    modePreview: document.getElementById('editor-mode-preview'),
    save: document.getElementById('editor-save'),
    delete: document.getElementById('editor-delete'),

    cstate: document.getElementById('consolidator-state'),
    cindicator: document.getElementById('consolidator-indicator'),
    clast: document.getElementById('consolidator-last'),
    cnext: document.getElementById('consolidator-next'),
    cmodel: document.getElementById('consolidator-model'),
    crun: document.getElementById('consolidator-run'),
    cconfigToggle: document.getElementById('consolidator-config-toggle'),
    cconfigPanel: document.getElementById('consolidator-config'),
    cform: document.getElementById('consolidator-form'),
    cfgEnabled: document.getElementById('cfg-enabled'),
    cfgInterval: document.getElementById('cfg-interval'),
    cfgModel: document.getElementById('cfg-model'),
    cerror: document.getElementById('consolidator-error'),
  };

  let openPath = null;
  let savedContent = '';
  let dirty = false;

  const setDirty = (d) => {
    dirty = d;
    els.dirty.classList.toggle('hidden', !d);
  };

  const showError = (msg) => {
    if (!msg) {
      els.cerror.classList.add('hidden');
      els.cerror.textContent = '';
      return;
    }
    els.cerror.textContent = msg;
    els.cerror.classList.remove('hidden');
  };

  const fetchJSON = async (url, opts = {}) => {
    const r = await fetch(url, opts);
    if (!r.ok) {
      const detail = await r.json().catch(() => ({ detail: r.statusText }));
      throw new Error(detail.detail || `request failed: ${r.status}`);
    }
    return r.json();
  };

  const renderTreeNode = (node, depth) => {
    if (node.type === 'file') {
      const a = document.createElement('a');
      a.href = '#';
      a.textContent = node.name;
      a.dataset.path = node.path;
      a.className = 'tree-file';
      a.style.paddingLeft = `${depth * 14}px`;
      a.addEventListener('click', (e) => {
        e.preventDefault();
        openNote(node.path);
      });
      return a;
    }
    const wrap = document.createElement('div');
    if (depth >= 0) {
      const header = document.createElement('div');
      header.className = 'tree-folder';
      header.style.paddingLeft = `${depth * 14}px`;
      header.textContent = node.name || 'notes';
      wrap.appendChild(header);
    }
    for (const child of node.children) {
      wrap.appendChild(renderTreeNode(child, depth + 1));
    }
    return wrap;
  };

  const loadTree = async () => {
    const root = await fetchJSON('/notes/api/tree');
    els.tree.innerHTML = '';
    if (!root.children || root.children.length === 0) {
      els.tree.innerHTML = '<p class="italic text-ink-40 text-xs">No notes yet.</p>';
      return;
    }
    for (const child of root.children) {
      els.tree.appendChild(renderTreeNode(child, 0));
    }
  };

  const openNote = async (path) => {
    if (dirty && !confirm('Discard unsaved changes?')) return;
    try {
      const data = await fetchJSON(`/notes/api/read?path=${encodeURIComponent(path)}`);
      openPath = data.path;
      savedContent = data.content;
      els.textarea.value = data.content;
      els.path.textContent = data.path;
      setDirty(false);
      switchMode('edit');
    } catch (e) {
      alert(e.message);
    }
  };

  const saveNote = async () => {
    if (!openPath) return;
    try {
      await fetchJSON('/notes/api/write', {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ path: openPath, content: els.textarea.value }),
      });
      savedContent = els.textarea.value;
      setDirty(false);
      await loadTree();
    } catch (e) {
      alert(`save failed: ${e.message}`);
    }
  };

  const deleteNote = async () => {
    if (!openPath) return;
    if (!confirm(`Delete ${openPath}? A backup goes to .history/.`)) return;
    try {
      await fetchJSON('/notes/api/delete', {
        method: 'DELETE',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ path: openPath }),
      });
      openPath = null;
      savedContent = '';
      els.textarea.value = '';
      els.path.textContent = '— no file —';
      setDirty(false);
      await loadTree();
    } catch (e) {
      alert(`delete failed: ${e.message}`);
    }
  };

  const newNote = async () => {
    const folder = prompt(
      'Folder (companies / sectors / macro / general):',
      'companies'
    );
    if (!folder) return;
    const name = prompt('Filename (without .md):', '');
    if (!name) return;
    const path = `${folder}/${name}.md`;
    const today = new Date().toISOString().slice(0, 10);
    const content = `---\ntitle: ${name}\ncreated: ${today}\nupdated: ${today}\n---\n\n`;
    try {
      await fetchJSON('/notes/api/write', {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ path, content }),
      });
      await loadTree();
      await openNote(path);
    } catch (e) {
      alert(`create failed: ${e.message}`);
    }
  };

  const switchMode = (mode) => {
    if (mode === 'edit') {
      els.textarea.classList.remove('hidden');
      els.preview.classList.add('hidden');
      els.modeEdit.classList.add('text-ink-100', 'border-b', 'border-ink-100');
      els.modeEdit.classList.remove('text-ink-60');
      els.modePreview.classList.remove('text-ink-100', 'border-b', 'border-ink-100');
      els.modePreview.classList.add('text-ink-60');
    } else {
      els.preview.innerHTML = window.marked.parse(els.textarea.value);
      els.textarea.classList.add('hidden');
      els.preview.classList.remove('hidden');
      els.modePreview.classList.add('text-ink-100', 'border-b', 'border-ink-100');
      els.modePreview.classList.remove('text-ink-60');
      els.modeEdit.classList.remove('text-ink-100', 'border-b', 'border-ink-100');
      els.modeEdit.classList.add('text-ink-60');
    }
  };

  const formatRelative = (iso) => {
    if (!iso) return '—';
    const dt = new Date(iso);
    const diff = (Date.now() - dt.getTime()) / 1000;
    if (Math.abs(diff) < 60) return diff >= 0 ? 'just now' : 'soon';
    const minutes = Math.round(Math.abs(diff) / 60);
    if (minutes < 60) return diff >= 0 ? `${minutes} min ago` : `in ${minutes} min`;
    const hours = Math.round(minutes / 60);
    return diff >= 0 ? `${hours} hr ago` : `in ${hours} hr`;
  };

  const renderConsolidator = (data) => {
    const config = data.config;
    const status = data.status;

    if (status.running) {
      els.cstate.textContent = 'running';
      els.cindicator.className = 'mark mark-warn animate-pulse';
    } else if (config.enabled) {
      els.cstate.textContent = 'enabled';
      els.cindicator.className = 'mark mark-active';
    } else {
      els.cstate.textContent = 'disabled';
      els.cindicator.className = 'mark mark-paused';
    }

    els.clast.textContent = status.last_run_at
      ? `${formatRelative(status.last_run_at)} (${status.edits_last_run} edits)`
      : 'never';
    els.cnext.textContent = config.enabled
      ? (status.next_run_at ? formatRelative(status.next_run_at) : 'soon')
      : 'disabled';
    els.cmodel.textContent = config.model.split('/').pop();

    if (els.cfgEnabled.dataset.initialized !== '1') {
      els.cfgEnabled.checked = config.enabled;
      els.cfgInterval.value = config.interval_minutes;
      els.cfgEnabled.dataset.initialized = '1';
    }

    if (status.last_error) {
      showError(`last error: ${status.last_error}`);
    }
  };

  const loadConsolidator = async () => {
    try {
      const data = await fetchJSON('/notes/api/consolidator');
      renderConsolidator(data);
      return data.config;
    } catch (e) {
      showError(e.message);
    }
  };

  const loadModels = async () => {
    const data = await fetchJSON('/chat/models');
    els.cfgModel.innerHTML = '';
    for (const m of data.models) {
      const opt = document.createElement('option');
      opt.value = m.id;
      opt.textContent = m.display;
      els.cfgModel.appendChild(opt);
    }
    return data;
  };

  const saveConsolidatorConfig = async (e) => {
    e.preventDefault();
    showError(null);
    try {
      const data = await fetchJSON('/notes/api/consolidator/config', {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          enabled: els.cfgEnabled.checked,
          interval_minutes: Number(els.cfgInterval.value),
          model: els.cfgModel.value,
        }),
      });
      renderConsolidator(data);
    } catch (e) {
      showError(e.message);
    }
  };

  const runConsolidatorNow = async () => {
    els.crun.disabled = true;
    els.crun.textContent = 'running…';
    showError(null);
    try {
      const data = await fetchJSON('/notes/api/consolidator/run', { method: 'POST' });
      renderConsolidator({ config: await loadConsolidator(), status: data.status });
      await loadTree();
      if (data.error) showError(data.error);
    } catch (e) {
      showError(e.message);
    } finally {
      els.crun.disabled = false;
      els.crun.textContent = 'run now';
    }
  };

  els.textarea.addEventListener('input', () => {
    setDirty(els.textarea.value !== savedContent);
  });

  els.textarea.addEventListener('keydown', (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === 's') {
      e.preventDefault();
      saveNote();
    }
  });

  els.save.addEventListener('click', saveNote);
  els.delete.addEventListener('click', deleteNote);
  els.new.addEventListener('click', newNote);
  els.modeEdit.addEventListener('click', () => switchMode('edit'));
  els.modePreview.addEventListener('click', () => switchMode('preview'));
  els.cconfigToggle.addEventListener('click', () => {
    els.cconfigPanel.classList.toggle('hidden');
  });
  els.cform.addEventListener('submit', saveConsolidatorConfig);
  els.crun.addEventListener('click', runConsolidatorNow);

  (async () => {
    await loadTree();
    const models = await loadModels();
    const config = await loadConsolidator();
    if (config) {
      els.cfgModel.value = config.model || models.default;
    }
  })();
})();
