(() => {
  const form = document.getElementById('chat-form');
  if (!form) return;

  const messagesEl = document.getElementById('chat-messages');
  const inputEl = document.getElementById('chat-input');
  const passwordEl = document.getElementById('chat-password');
  const deeperEl = document.getElementById('chat-deeper');
  const sendBtn = document.getElementById('chat-send');
  const costEl = document.getElementById('chat-cost');

  const ticker = form.dataset.ticker;
  const hyphash = form.dataset.hyphash || 'nohyp';

  const history = [];
  let totalCost = 0;
  let totalTokensIn = 0;
  let totalTokensOut = 0;

  const HISTORY_CAP = 20;

  function addBubble(role, text, opts = {}) {
    const div = document.createElement('div');
    div.className = 'chat-bubble ' + role + (opts.loading ? ' loading' : '') + (opts.error ? ' error' : '');
    div.textContent = text;
    if (opts.meta) {
      const m = document.createElement('span');
      m.className = 'meta';
      m.textContent = opts.meta;
      div.appendChild(m);
    }
    messagesEl.appendChild(div);
    messagesEl.scrollTop = messagesEl.scrollHeight;
    return div;
  }

  function updateCost() {
    if (totalCost > 0) {
      costEl.textContent =
        'Session: ' + history.length / 2 + ' Q · ' +
        totalTokensIn.toLocaleString() + ' in / ' +
        totalTokensOut.toLocaleString() + ' out · ~$' +
        totalCost.toFixed(4);
    }
  }

  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    const message = inputEl.value.trim();
    const password = passwordEl.value;
    if (!message || !password) return;

    addBubble('user', message);
    inputEl.value = '';
    sendBtn.disabled = true;
    const loading = addBubble('assistant', 'Thinking…', { loading: true });

    const useSonnet = deeperEl.checked;

    const fd = new FormData();
    fd.append('ticker', ticker);
    fd.append('hypothesis_hash', hyphash);
    fd.append('message', message);
    fd.append('history', JSON.stringify(history));
    fd.append('use_sonnet', useSonnet ? 'true' : 'false');
    fd.append('password', password);

    try {
      const r = await fetch('/chat', { method: 'POST', body: fd });
      loading.remove();
      if (!r.ok) {
        let detail = 'Request failed (' + r.status + ')';
        try {
          const j = await r.json();
          if (j.detail) detail = j.detail;
        } catch (e) { /* swallow */ }
        addBubble('assistant', detail, { error: true });
        sendBtn.disabled = false;
        return;
      }
      const data = await r.json();
      const meta = (data.model || '?') + ' · ' +
                   (data.input_tokens || 0) + ' in / ' +
                   (data.output_tokens || 0) + ' out · ~$' +
                   (data.cost_usd || 0).toFixed(4);
      addBubble('assistant', data.response || '(empty response)', { meta });

      history.push({ role: 'user', content: message });
      history.push({ role: 'assistant', content: data.response || '' });
      while (history.length > HISTORY_CAP) history.shift();

      totalCost += data.cost_usd || 0;
      totalTokensIn += data.input_tokens || 0;
      totalTokensOut += data.output_tokens || 0;
      updateCost();
    } catch (err) {
      loading.remove();
      addBubble('assistant', 'Network error: ' + err.message, { error: true });
    } finally {
      sendBtn.disabled = false;
      inputEl.focus();
    }
  });

  // Cmd/Ctrl+Enter to send
  inputEl.addEventListener('keydown', (e) => {
    if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') {
      e.preventDefault();
      form.requestSubmit();
    }
  });
})();
