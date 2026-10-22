/* ============================================================
   "Add any dataset" modal
   ------------------------------------------------------------
   Lets the visitor drop in any CSV (URL or pasted text), runs
   the browser-side CRISP-DM engine (static/analyze.js) and
   registers the result as a new problem in the top switcher.
   ============================================================ */
(function () {
  'use strict';

  const el = id => document.getElementById(id);
  const MAX_BYTES = 20 * 1024 * 1024;
  const st = { text: null, name: '', source: '', cols: [], rows: [], busy: false };
  let counter = 0;

  /* -------------------------------------------------------- modal -- */
  function open() {
    el('dsModal').hidden = false;
    document.body.classList.add('modal-open');
    setTimeout(() => { const u = el('dsUrl'); if (u) u.focus(); }, 30);
  }
  function close() {
    if (st.busy) return;
    el('dsModal').hidden = true;
    document.body.classList.remove('modal-open');
  }

  function status(msg, kind) {
    const s = el('dsStatus');
    if (!s) return;
    s.textContent = msg;
    s.className = 'hint' + (kind ? ' v-' + kind : '');
  }

  function log(msg, cls) {
    const box = el('dsLog');
    box.hidden = false;
    const line = document.createElement('div');
    line.className = 'ds-log-line' + (cls ? ' ' + cls : '');
    line.textContent = msg;
    box.appendChild(line);
    box.scrollTop = box.scrollHeight;
  }

  /* ------------------------------------------------------ loading -- */
  function fileNameFromUrl(url) {
    try {
      const clean = url.split(/[?#]/)[0];
      return decodeURIComponent(clean.split('/').pop() || 'dataset.csv');
    } catch (e) { return 'dataset.csv'; }
  }

  async function loadUrl(url) {
    status('下載 CSV…');
    let text = null, error = null;
    try {
      const r = await fetch(url, { mode: 'cors', credentials: 'omit' });
      if (!r.ok) throw new Error('HTTP ' + r.status);
      text = await r.text();
    } catch (e) {
      error = e;
      // Static builds have no backend; Flask builds can proxy CORS-blocked hosts.
      try {
        const r2 = await fetch('/api/dataset?url=' + encodeURIComponent(url));
        if (!r2.ok) throw new Error('proxy HTTP ' + r2.status);
        const j = await r2.json();
        text = j.csv;
      } catch (e2) {
        throw new Error(`無法下載（${error.message}）。若該網址未開放 CORS，請改用「貼上 CSV 文字」。`);
      }
    }
    if (text.length > MAX_BYTES) throw new Error('檔案超過 20 MB，請先用試算表裁切後再上傳。');
    useText(text, fileNameFromUrl(url), url);
  }

  function useText(text, name, source) {
    let parsed;
    try {
      parsed = window.CSVEngine.parseCSV(text, window.CSVEngine.detectDelimiter(text));
    } catch (e) {
      throw new Error('CSV 解析失敗：' + e.message);
    }
    if (!parsed.header.length || !parsed.rows.length) throw new Error('檔案裡沒有可用的資料列。');
    if (parsed.header.length < 2) throw new Error('至少需要兩個欄位（特徵 + 目標）。');
    if (parsed.rows.length < 12) throw new Error('資料列太少（至少 12 筆）才能做切分與交叉驗證。');

    st.text = text;
    // Re-read the whole file: parseCSV keeps all rows, but drop obviously empty columns.
    st.name = name;
    st.source = source || '';
    st.cols = parsed.header;
    st.rows = parsed.rows;

    const prof = window.CSVEngine.profileColumns(parsed.header, parsed.rows);
    const numeric = prof.filter(c => c.isNumeric && !c.looksId && !c.looksDate);

    el('dsName').textContent = name;
    const shown = Math.min(parsed.rows.length, 8);
    el('dsShape').textContent =
      `${parsed.rows.length.toLocaleString('en-US')} 列 × ${parsed.header.length} 欄 · 預覽前 ${shown} 列`;

    // preview table (first 12 columns)
    const cols = parsed.header.slice(0, 12);
    el('dsPreviewTable').innerHTML =
      '<thead><tr>' + cols.map(c => `<th>${esc(c)}</th>`).join('') + '</tr></thead><tbody>' +
      parsed.rows.slice(0, shown).map(r =>
        '<tr>' + cols.map((_, j) => `<td class="mono">${esc(r[j] === undefined ? '' : r[j])}</td>`).join('') + '</tr>'
      ).join('') + '</tbody>';

    const tsel = el('dsTarget');
    tsel.innerHTML = numeric.map(c => `<option value="${esc(c.name)}">${esc(c.name)}</option>`).join('');
    // default: the last numeric column (datasets usually put the label last)
    if (numeric.length) tsel.value = numeric[numeric.length - 1].name;

    el('dsPreview').hidden = false;
    status(numeric.length
      ? `偵測到 ${numeric.length} 個數值欄位，請確認目標欄位後開始分析。`
      : '⚠️ 沒有找到數值欄位，無法做迴歸。', numeric.length ? 'emerald' : 'rose');
    if (!numeric.length) el('dsRun').disabled = true;
    else el('dsRun').disabled = false;
  }

  const esc = s => String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');

  /* ------------------------------------------------------ analysis -- */
  async function run() {
    if (st.busy || !st.text) return;
    const target = el('dsTarget').value;
    if (!target) { status('請先選擇目標欄位。', 'rose'); return; }

    st.busy = true;
    el('dsRun').disabled = true;
    el('dsFetch').disabled = true;
    el('dsLog').innerHTML = '';
    el('dsLog').hidden = false;
    log(`▶ 開始分析「${st.name}」→ 目標：${target}`);

    const t0 = performance.now();
    try {
      const res = await window.CSVEngine.analyze({
        text: st.text,
        name: st.name,
        source: st.source,
        target,
        testSize: +el('dsTest').value,
        folds: +el('dsFolds').value,
        kMax: +el('dsK').value,
        id: 'dyn' + (++counter),
        onProgress: m => log('· ' + m)
      });
      const payload = res.payload;
      const cfg = window.CSVEngine.buildDynamicConfig(payload, st.source);

      (window.PROBLEM_DATA = window.PROBLEM_DATA || {})[cfg.id] = payload;
      window.PROBLEMS = window.PROBLEMS || [];
      window.PROBLEMS.push(cfg);

      const sel = el('problemSelect');
      const opt = document.createElement('option');
      opt.value = cfg.id;
      opt.textContent = cfg.label;
      sel.appendChild(opt);

      const secs = ((performance.now() - t0) / 1000).toFixed(1);
      log(`✓ 完成：${payload.meta.dataset.rows.toLocaleString('en-US')} 筆 × ${payload.meta.dataset.features.length} 個編碼特徵，` +
          `${Object.keys(payload.models).length} 種方法 × k = 1–${payload.meta.k_max}（${secs}s）`, 'ok');
      log(`最終模型：${payload.meta.final_decision.algo} · k=${payload.meta.final_decision.k} → ` +
          payload.meta.final_decision.features.join(' + '), 'ok');

      // hand over to the shared engine
      sel.value = cfg.id;
      st.busy = false;
      el('dsRun').disabled = false;
      el('dsFetch').disabled = false;
      close();
      await switchProblem(cfg.id, { force: true });
      window.scrollTo({ top: 0, behavior: 'smooth' });
      toast(`已新增「${st.name}」：${payload.meta.dataset.rows.toLocaleString('en-US')} 筆，最終模型 ${payload.meta.final_decision.algo} · k=${payload.meta.final_decision.k}`);
    } catch (e) {
      log('✗ ' + e.message, 'err');
      status('分析失敗：' + e.message, 'rose');
      st.busy = false;
      el('dsRun').disabled = false;
      el('dsFetch').disabled = false;
    }
  }

  function toast(msg) {
    const d = document.createElement('div');
    d.className = 'toast';
    d.textContent = msg;
    document.body.appendChild(d);
    setTimeout(() => d.classList.add('show'), 20);
    setTimeout(() => { d.classList.remove('show'); setTimeout(() => d.remove(), 400); }, 5200);
  }

  /* -------------------------------------------------------- wiring -- */
  function init() {
    if (!el('dsModal')) return;
    el('dsOpen').addEventListener('click', open);
    el('dsClose').addEventListener('click', close);
    el('dsModal').addEventListener('click', e => { if (e.target === el('dsModal')) close(); });
    document.addEventListener('keydown', e => {
      if (e.key === 'Escape' && !el('dsModal').hidden) close();
    });

    el('dsExample').addEventListener('click', () => {
      el('dsUrl').value = 'https://raw.githubusercontent.com/Avik-Jain/100-Days-Of-ML-Code/master/datasets/50_Startups.csv';
      el('dsUrl').focus();
    });

    el('dsFetch').addEventListener('click', async () => {
      const url = el('dsUrl').value.trim();
      if (!url) { status('請先輸入 CSV 網址。', 'rose'); return; }
      el('dsFetch').disabled = true;
      try {
        await loadUrl(url);
      } catch (e) {
        status(e.message, 'rose');
      } finally {
        el('dsFetch').disabled = false;
      }
    });

    el('dsUrl').addEventListener('keydown', e => {
      if (e.key === 'Enter') { e.preventDefault(); el('dsFetch').click(); }
    });

    el('dsUsePaste').addEventListener('click', () => {
      const text = el('dsPaste').value.trim();
      if (!text) { status('請先貼上 CSV 內容。', 'rose'); return; }
      try {
        useText(text, 'pasted.csv', '');
      } catch (e) { status(e.message, 'rose'); }
    });

    el('dsTest').addEventListener('input', () => {
      el('dsTestOut').textContent = Math.round(+el('dsTest').value * 100) + '%';
    });

    el('dsRun').addEventListener('click', run);
  }

  document.addEventListener('DOMContentLoaded', init);
})();
