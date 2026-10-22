/* ============================================================
   Browser-side CRISP-DM engine for *arbitrary* CSV datasets.
   ------------------------------------------------------------
   Everything runs client side, so the feature also works on the
   static GitHub Pages build (no Python needed).

     parseCSV -> profile -> clean/encode -> hold-out + k-fold
     -> OLS (normal equations) -> CV RMSE -> metrics, t/p, CI, VIF
     -> diagnostics (Jarque-Bera, Breusch-Pagan, DW)
     -> payload {meta, models} + generated narrative config

   The payload mirrors the shape of build_*_data.py so the existing
   renderer (static/app.js) can display it unchanged.
   ============================================================ */
(function () {
  'use strict';

  const MAX_ROWS = 60000;      // hard cap -> keeps the browser responsive
  const MAX_FEATURES = 24;     // hard cap on encoded predictors
  const MAX_LEVELS = 12;       // one-hot beyond this is dropped
  const RANDOM_STATE = 42;

  /* ------------------------------------------------ random / arrays -- */
  function mulberry32(a) {
    return function () {
      a |= 0; a = a + 0x6D2B79F5 | 0;
      let t = Math.imul(a ^ a >>> 15, 1 | a);
      t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
      return ((t ^ t >>> 14) >>> 0) / 4294967296;
    };
  }
  function shuffledIdx(n, seed) {
    const rnd = mulberry32(seed), idx = new Array(n);
    for (let i = 0; i < n; i++) idx[i] = i;
    for (let i = n - 1; i > 0; i--) {
      const j = Math.floor(rnd() * (i + 1));
      const t = idx[i]; idx[i] = idx[j]; idx[j] = t;
    }
    return idx;
  }

  /* --------------------------------------------------------- CSV -- */
  function detectDelimiter(text) {
    const line = text.split(/\r?\n/).find(l => l.trim().length) || '';
    let best = ',', bestCount = -1;
    [',', ';', '\t', '|'].forEach(d => {
      const n = line.split(d).length;
      if (n > bestCount) { bestCount = n; best = d; }
    });
    return best;
  }

  function parseCSV(text, delim) {
    text = text.replace(/^\uFEFF/, '');
    const rows = [];
    let row = [], field = '', inQuotes = false;
    for (let i = 0; i < text.length; i++) {
      const ch = text[i];
      if (inQuotes) {
        if (ch === '"') {
          if (text[i + 1] === '"') { field += '"'; i++; }
          else inQuotes = false;
        } else field += ch;
      } else if (ch === '"') inQuotes = true;
      else if (ch === delim) { row.push(field); field = ''; }
      else if (ch === '\n') { row.push(field); rows.push(row); row = []; field = ''; }
      else if (ch !== '\r') field += ch;
    }
    if (field.length || row.length) { row.push(field); rows.push(row); }
    const header = (rows.shift() || []).map((h, i) => (h || '').trim() || `col_${i + 1}`);
    return { header, rows: rows.filter(r => r.some(v => String(v).trim() !== '')) };
  }

  /* ---------------------------------------------------- numerics -- */
  const toNum = v => {
    if (v === null || v === undefined) return NaN;
    const s = String(v).trim().replace(/,/g, '');
    if (!s) return NaN;
    const n = Number(s);
    return isFinite(n) ? n : NaN;
  };

  function median(arr) {
    if (!arr.length) return 0;
    const a = arr.slice().sort((x, y) => x - y);
    const m = a.length >> 1;
    return a.length % 2 ? a[m] : (a[m - 1] + a[m]) / 2;
  }
  function mean(arr) { return arr.length ? arr.reduce((s, v) => s + v, 0) / arr.length : 0; }
  function std(arr) {
    if (arr.length < 2) return 0;
    const m = mean(arr);
    return Math.sqrt(arr.reduce((s, v) => s + (v - m) * (v - m), 0) / (arr.length - 1));
  }
  function quantileSorted(sorted, q) {
    if (!sorted.length) return 0;
    const pos = (sorted.length - 1) * q;
    const lo = Math.floor(pos), hi = Math.ceil(pos);
    return sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo);
  }

  /* ------------------------------------------- distributions -- */
  function logGamma(x) {
    const c = [76.18009172947146, -86.50532032941677, 24.01409824083091,
      -1.231739572450155, 0.1208650973866179e-2, -0.5395239384953e-5];
    let y = x, tmp = x + 5.5, ser = 1.000000000190015;
    tmp -= (x + 0.5) * Math.log(tmp);
    for (let j = 0; j < 6; j++) ser += c[j] / ++y;
    return -tmp + Math.log(2.5066282746310005 * ser / x);
  }

  function betacf(a, b, x) {
    const MAXIT = 200, EPS = 3e-12, FPMIN = 1e-300;
    let qab = a + b, qap = a + 1, qam = a - 1;
    let c = 1, d = 1 - qab * x / qap;
    if (Math.abs(d) < FPMIN) d = FPMIN;
    d = 1 / d;
    let h = d;
    for (let m = 1; m <= MAXIT; m++) {
      const m2 = 2 * m;
      let aa = m * (b - m) * x / ((qam + m2) * (a + m2));
      d = 1 + aa * d; if (Math.abs(d) < FPMIN) d = FPMIN;
      c = 1 + aa / c; if (Math.abs(c) < FPMIN) c = FPMIN;
      d = 1 / d; h *= d * c;
      aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
      d = 1 + aa * d; if (Math.abs(d) < FPMIN) d = FPMIN;
      c = 1 + aa / c; if (Math.abs(c) < FPMIN) c = FPMIN;
      d = 1 / d;
      const del = d * c; h *= del;
      if (Math.abs(del - 1) < EPS) break;
    }
    return h;
  }

  function betai(a, b, x) {
    if (x <= 0) return 0;
    if (x >= 1) return 1;
    const bt = Math.exp(logGamma(a + b) - logGamma(a) - logGamma(b) + a * Math.log(x) + b * Math.log(1 - x));
    return x < (a + 1) / (a + b + 2) ? bt * betacf(a, b, x) / a : 1 - bt * betacf(b, a, 1 - x) / b;
  }

  function tCdf(t, df) {
    const x = df / (df + t * t);
    const p = 0.5 * betai(df / 2, 0.5, x);
    return t >= 0 ? 1 - p : p;
  }
  const tTwoSidedP = (t, df) => 2 * (1 - tCdf(Math.abs(t), df));

  function tQuantile(p, df) {
    let lo = 0, hi = 40;
    for (let i = 0; i < 80; i++) {
      const mid = (lo + hi) / 2;
      if (tCdf(mid, df) < p) lo = mid; else hi = mid;
    }
    return (lo + hi) / 2;
  }

  function gammp(a, x) {
    if (x <= 0) return 0;
    if (x < a + 1) {                       // series
      let ap = a, sum = 1 / a, del = sum;
      for (let n = 1; n < 300; n++) {
        ap += 1; del *= x / ap; sum += del;
        if (Math.abs(del) < Math.abs(sum) * 1e-14) break;
      }
      return sum * Math.exp(-x + a * Math.log(x) - logGamma(a));
    }
    return 1 - gammq(x, a);
  }
  function gammq(a, x) {                    // continued fraction
    let b = x + 1 - a, c = 1 / 1e-300, d = 1 / b, h = d;
    for (let i = 1; i < 300; i++) {
      const an = -i * (i - a);
      b += 2; d = an * d + b; if (Math.abs(d) < 1e-300) d = 1e-300;
      c = b + an / c; if (Math.abs(c) < 1e-300) c = 1e-300;
      d = 1 / d; const del = d * c; h *= del;
      if (Math.abs(del - 1) < 1e-14) break;
    }
    return Math.exp(-x + a * Math.log(x) - logGamma(a)) * h;
  }
  const chi2Sf = (x, k) => (x <= 0 ? 1 : gammq(k / 2, x / 2));
  const fSf = (f, d1, d2) => (f <= 0 ? 1 : betai(d2 / 2, d1 / 2, d2 / (d2 + d1 * f)));

  /* ------------------------------------------- linear algebra -- */
  function solveLinear(A, b, m) {
    // diagonal preconditioning: normal equations mix units, so normalise every
    // variable by sqrt(A_ii) making the pivot test scale free.
    const sc = new Float64Array(m);
    for (let i = 0; i < m; i++) {
      const d = A[i * m + i];
      if (!(d > 0)) return null;
      sc[i] = Math.sqrt(d);
    }
    const M = new Float64Array(m * m), v = new Float64Array(m);
    for (let i = 0; i < m; i++) {
      v[i] = b[i] / sc[i];
      for (let j = 0; j < m; j++) M[i * m + j] = A[i * m + j] / (sc[i] * sc[j]);
    }
    const tol = 1e-12;
    for (let col = 0; col < m; col++) {
      let piv = col;
      for (let r = col + 1; r < m; r++) if (Math.abs(M[r * m + col]) > Math.abs(M[piv * m + col])) piv = r;
      if (Math.abs(M[piv * m + col]) < tol) return null;
      if (piv !== col) {
        for (let j = 0; j < m; j++) { const t = M[col * m + j]; M[col * m + j] = M[piv * m + j]; M[piv * m + j] = t; }
        const t = v[col]; v[col] = v[piv]; v[piv] = t;
      }
      const d = M[col * m + col];
      for (let r = col + 1; r < m; r++) {
        const f = M[r * m + col] / d;
        if (!f) continue;
        for (let j = col; j < m; j++) M[r * m + j] -= f * M[col * m + j];
        v[r] -= f * v[col];
      }
    }
    const x = new Float64Array(m);
    for (let r = m - 1; r >= 0; r--) {
      let s = v[r];
      for (let j = r + 1; j < m; j++) s -= M[r * m + j] * x[j];
      x[r] = s / M[r * m + r];
    }
    for (let r = 0; r < m; r++) x[r] /= sc[r];
    return x;
  }

  function invertMatrix(A, m) {
    const sc = new Float64Array(m);
    for (let i = 0; i < m; i++) {
      const d = A[i * m + i];
      if (!(d > 0)) return null;
      sc[i] = Math.sqrt(d);
    }
    const M = new Float64Array(m * m), inv = new Float64Array(m * m);
    for (let i = 0; i < m; i++)
      for (let j = 0; j < m; j++) M[i * m + j] = A[i * m + j] / (sc[i] * sc[j]);
    const tol = 1e-12;
    for (let i = 0; i < m; i++) inv[i * m + i] = 1;
    for (let col = 0; col < m; col++) {
      let piv = col;
      for (let r = col + 1; r < m; r++) if (Math.abs(M[r * m + col]) > Math.abs(M[piv * m + col])) piv = r;
      if (Math.abs(M[piv * m + col]) < tol) return null;
      if (piv !== col) {
        for (let j = 0; j < m; j++) {
          let t = M[col * m + j]; M[col * m + j] = M[piv * m + j]; M[piv * m + j] = t;
          t = inv[col * m + j]; inv[col * m + j] = inv[piv * m + j]; inv[piv * m + j] = t;
        }
      }
      const d = M[col * m + col];
      for (let j = 0; j < m; j++) { M[col * m + j] /= d; inv[col * m + j] /= d; }
      for (let r = 0; r < m; r++) {
        if (r === col) continue;
        const f = M[r * m + col];
        if (!f) continue;
        for (let j = 0; j < m; j++) { M[r * m + j] -= f * M[col * m + j]; inv[r * m + j] -= f * inv[col * m + j]; }
      }
    }
    for (let i = 0; i < m; i++)
      for (let j = 0; j < m; j++) inv[i * m + j] /= (sc[i] * sc[j]);
    return inv;
  }

  /* ------------------------------------- sufficient statistics -- */
  function aggregates(rows, y, p) {
    const G = new Float64Array((p + 1) * (p + 1)), c = new Float64Array(p + 1);
    let yy = 0, sy = 0;
    const n = rows.length;
    for (let i = 0; i < n; i++) {
      const r = rows[i], yi = y[i];
      yy += yi * yi; sy += yi;
      for (let a = 0; a <= p; a++) {
        const ra = r[a];
        if (!ra) continue;
        c[a] += ra * yi;
        for (let b = a; b <= p; b++) G[a * (p + 1) + b] += ra * r[b];
      }
    }
    for (let a = 0; a <= p; a++) for (let b = 0; b < a; b++) G[a * (p + 1) + b] = G[b * (p + 1) + a];
    return { G, c, yy, sy, n, p };
  }

  function subsetSolve(agg, idx) {
    const p = agg.p, m = idx.length, G = agg.G, c = agg.c;
    const A = new Float64Array(m * m), b = new Float64Array(m);
    for (let i = 0; i < m; i++) {
      b[i] = c[idx[i]];
      for (let j = 0; j < m; j++) A[i * m + j] = G[idx[i] * (p + 1) + idx[j]];
    }
    const beta = solveLinear(A, b, m);
    if (!beta) return null;
    let rss = agg.yy;
    for (let i = 0; i < m; i++) {
      rss -= 2 * b[i] * beta[i];
      for (let j = 0; j < m; j++) rss += beta[i] * A[i * m + j] * beta[j];
    }
    return { beta, rss: Math.max(rss, 0), A, m };
  }

  /** RMSE of a coefficient vector on another aggregate (used for CV / test). */
  function aggRmse(agg, idx, beta) {
    const p = agg.p, m = idx.length, G = agg.G, c = agg.c;
    let rss = agg.yy;
    for (let i = 0; i < m; i++) {
      rss -= 2 * beta[i] * c[idx[i]];
      for (let j = 0; j < m; j++) rss += beta[i] * G[idx[i] * (p + 1) + idx[j]] * beta[j];
    }
    rss = Math.max(rss, 0);
    return Math.sqrt(rss / agg.n);
  }

  function predictRow(r, idx, beta) {
    let s = 0;
    for (let i = 0; i < idx.length; i++) s += beta[i] * r[idx[i]];
    return s;
  }

  /* ------------------------------------------------- statistics -- */
  function fitStats(agg, idx) {
    const sol = subsetSolve(agg, idx);
    if (!sol) return null;
    const k = idx.length - 1, df = agg.n - k - 1;
    if (df < 1) return null;
    const tss = agg.yy - agg.n * Math.pow(agg.sy / agg.n, 2);
    const r2 = tss > 0 ? 1 - sol.rss / tss : 0;
    const adj = 1 - (1 - r2) * (agg.n - 1) / df;
    const sigma2 = sol.rss / df;
    const inv = invertMatrix(sol.A, sol.m);
    const tcrit = tQuantile(0.975, df);
    const coefs = [];
    for (let i = 0; i < sol.m; i++) {
      const se = inv ? Math.sqrt(Math.max(sigma2 * inv[i * sol.m + i], 0)) : NaN;
      const t = se > 0 ? sol.beta[i] / se : NaN;
      coefs.push({
        coef: sol.beta[i], se, t,
        p: isFinite(t) ? tTwoSidedP(t, df) : NaN,
        lo: sol.beta[i] - tcrit * se, hi: sol.beta[i] + tcrit * se
      });
    }
    const f = k > 0 ? ((tss - sol.rss) / k) / (sol.rss / df) : NaN;
    return {
      beta: sol.beta, coefs, rss: sol.rss, r2, adj, df, k,
      f, fP: k > 0 ? fSf(f, k, df) : NaN,
      aic: agg.n * Math.log(sol.rss / agg.n) + 2 * (k + 1),
      bic: agg.n * Math.log(sol.rss / agg.n) + Math.log(agg.n) * (k + 1)
    };
  }

  function vifFor(agg, idx, j) {
    // regress feature idx[j] (inside the model) on the other model features
    const p = agg.p, target = idx[j];
    const others = idx.filter((_, t) => t !== j);
    if (others.length < 2) return 1;
    const A = new Float64Array(others.length * others.length), b = new Float64Array(others.length);
    for (let i = 0; i < others.length; i++) {
      b[i] = agg.G[others[i] * (p + 1) + target];
      for (let l = 0; l < others.length; l++) A[i * others.length + l] = agg.G[others[i] * (p + 1) + others[l]];
    }
    const beta = solveLinear(A, b, others.length);
    if (!beta) return null;
    let rss = agg.G[target * (p + 1) + target];
    for (let i = 0; i < others.length; i++) rss -= 2 * b[i] * beta[i];
    for (let i = 0; i < others.length; i++)
      for (let l = 0; l < others.length; l++) rss += beta[i] * A[i * others.length + l] * beta[l];
    const meanX = agg.means ? agg.means[target] : 0;
    const tss = agg.G[target * (p + 1) + target] - agg.n * meanX * meanX;
    if (tss <= 0) return null;
    const r2 = 1 - Math.max(rss, 0) / tss;
    return r2 >= 0.999999 ? null : 1 / (1 - r2);
  }

  /* ------------------------------------------------- diagnostics -- */
  function jarqueBera(resid) {
    const n = resid.length;
    if (n < 8) return null;
    const m = mean(resid);
    let m2 = 0, m3 = 0, m4 = 0;
    for (const e of resid) {
      const d = e - m, d2 = d * d;
      m2 += d2; m3 += d2 * d; m4 += d2 * d2;
    }
    m2 /= n; m3 /= n; m4 /= n;
    const S = m3 / Math.pow(m2, 1.5), K = m4 / (m2 * m2);
    const jb = n / 6 * (S * S + (K - 3) * (K - 3) / 4);
    return { stat: jb, p: chi2Sf(jb, 2), skew: S, kurt: K };
  }

  function breuschPagan(resid, rows, p) {
    const n = resid.length;
    if (n < p + 5) return null;
    const e2 = new Float64Array(n);
    for (let i = 0; i < n; i++) e2[i] = resid[i] * resid[i];
    const agg = aggregates(rows, e2, p);
    // regress e2 on all original regressors (X, excluding intercept column 0)
    const idx = [];
    for (let j = 1; j <= p; j++) idx.push(j);
    const sol = fitStats(agg, idx);
    if (!sol) return null;
    const lm = n * sol.r2;
    return { stat: lm, p: chi2Sf(lm, idx.length) };
  }

  function durbinWatson(resid) {
    let num = 0, den = 0;
    for (let i = 0; i < resid.length; i++) {
      den += resid[i] * resid[i];
      if (i) { const d = resid[i] - resid[i - 1]; num += d * d; }
    }
    return den > 0 ? num / den : NaN;
  }

  /* --------------------------------------------------- cleaning -- */
  function profileColumns(header, rows) {
    return header.map((name, j) => {
      const vals = [];
      let nonEmpty = 0, numeric = 0;
      const uniq = new Set();
      const step = Math.max(1, Math.floor(rows.length / 2000));
      for (let i = 0; i < rows.length; i += step) {
        const raw = rows[i][j];
        const s = raw === undefined ? '' : String(raw).trim();
        if (s === '') continue;
        nonEmpty++;
        if (uniq.size < 50) uniq.add(s);
        const n = toNum(s);
        if (isFinite(n)) numeric++;
        else vals.push(s);
      }
      const isNumeric = nonEmpty > 0 && numeric / nonEmpty >= 0.95;
      return {
        name, index: j, isNumeric,
        unique: uniq.size,
        looksId: /^(id|index|unnamed.*|.*_id)$/i.test(name) || /^(unnamed:\s*0|index)$/i.test(name),
        looksDate: !isNumeric && vals.length > 3 &&
          vals.every(v => /^(\d{4}-\d{2}(-\d{2})?|\d{8}T[\d]+|\d{4}\/\d{1,2}\/\d{1,2})/.test(v))
      };
    });
  }

  /* ------------------------------------------------- pipeline -- */
  async function analyze(opts) {
    const { text, name, source, target, testSize = 0.2, folds = 5, kMax = 8, onProgress } = opts;
    const tick = () => new Promise(r => setTimeout(r, 0));
    const say = async msg => { if (onProgress) onProgress(msg); await tick(); };

    await say('解析 CSV…');
    const delim = detectDelimiter(text);
    const parsed = parseCSV(text, delim);
    const header = parsed.header;
    let rows = parsed.rows;
    const rawRows = rows.length, rawCols = header.length;

    if (target === undefined || target === null || !header.includes(target)) {
      throw new Error('找不到目標欄位：' + target);
    }
    if (rows.length > MAX_ROWS) {
      const keep = shuffledIdx(rows.length, RANDOM_STATE).slice(0, MAX_ROWS);
      rows = keep.map(i => rows[i]);
    }

    const profile = profileColumns(header, rows);
    const targetProfile = profile.find(c => c.name === target);
    if (!targetProfile.isNumeric) throw new Error(`目標欄位「${target}」不是數值欄位，無法做迴歸。`);

    /* ---- decide which raw columns survive ---- */
    const dropped = [];
    const keepCols = [];
    for (const c of profile) {
      if (c.name === target) continue;
      if (c.looksId) { dropped.push({ column: c.name, reason: '看起來是流水號 / 索引欄' }); continue; }
      if (c.looksDate) { dropped.push({ column: c.name, reason: '日期欄位（未納入時間趨勢）' }); continue; }
      if (!c.isNumeric && c.unique > MAX_LEVELS) {
        dropped.push({ column: c.name, reason: `類別過多（${c.unique}${c.unique >= 50 ? '+' : ''} 個水準）` });
        continue;
      }
      keepCols.push(c);
    }

    /* ---- clean rows, impute, encode ---- */
    await say('清理缺失值並編碼類別欄位…');
    const colIdx = new Map();
    header.forEach((h, i) => colIdx.set(h, i));

    const numCols = keepCols.filter(c => c.isNumeric);
    const catCols = keepCols.filter(c => !c.isNumeric);
    const numData = new Map();
    const medians = new Map();
    numCols.forEach(c => {
      const arr = [];
      for (const r of rows) { const v = toNum(r[c.index]); if (isFinite(v)) arr.push(v); }
      const med = median(arr);
      medians.set(c.name, med);
      numData.set(c.name, rows.map(r => {
        const v = toNum(r[c.index]);
        return isFinite(v) ? v : med;
      }));
    });
    let imputed = 0;
    numCols.forEach(c => {
      for (const r of rows) if (!isFinite(toNum(r[c.index]))) imputed++;
    });

    const cats = new Map();
    catCols.forEach(c => {
      const counts = new Map();
      for (const r of rows) {
        const v = String(r[c.index] === undefined ? '' : r[c.index]).trim() || '(missing)';
        counts.set(v, (counts.get(v) || 0) + 1);
      }
      const levels = [...counts.entries()].sort((a, b) => b[1] - a[1]).map(e => e[0]);
      cats.set(c.name, levels);
    });

    const y = new Float64Array(rows.length);
    let valid = 0;
    const keepRows = [];
    for (let i = 0; i < rows.length; i++) {
      const v = toNum(rows[i][colIdx.get(target)]);
      if (!isFinite(v)) continue;
      y[valid++] = v;
      keepRows.push(i);
    }
    const yv = y.slice(0, valid);

    // encoded feature names
    const featureNames = [];
    const featureSpec = [];   // {kind:'num', col} | {kind:'cat', col, level}
    numCols.forEach(c => { featureNames.push(c.name); featureSpec.push({ kind: 'num', col: c.name }); });
    catCols.forEach(c => {
      cats.get(c.name).slice(1).forEach(level => {   // first level = reference
        featureNames.push(`${c.name}_${level}`);
        featureSpec.push({ kind: 'cat', col: c.name, level });
      });
    });
    if (!featureNames.length) throw new Error('沒有可用的預測變數。');

    let p = featureNames.length, useSpec = featureSpec, useNames = featureNames;
    const catIdx = new Map();
    cats.forEach((levels, col) => {
      const m = new Map();
      levels.forEach((l, i) => m.set(l, i));
      catIdx.set(col, m);
    });

    // design matrix (n x (p+1)), intercept at column 0
    let design = new Float64Array(valid * (p + 1));
    const buildDesign = (spec, names, out) => {
      const dim = names.length + 1;
      for (let i = 0; i < valid; i++) {
        const base = i * dim;
        out[base] = 1;
        for (let j = 0; j < names.length; j++) {
          const s = spec[j];
          out[base + j + 1] = s.kind === 'num'
            ? numData.get(s.col)[keepRows[i]]
            : (String(rows[keepRows[i]][colIdx.get(s.col)] === undefined ? '' : rows[keepRows[i]][colIdx.get(s.col)]).trim() === s.level ? 1 : 0);
        }
      }
      return out;
    };
    buildDesign(useSpec, useNames, design);

    // correlation with y for feature capping / ranking
    await say('計算特徵與目標的相關性…');
    const corrY = (j) => {
      const n = valid;
      let sx = 0, sy = 0, sxx = 0, syy = 0, sxy = 0;
      for (let i = 0; i < n; i++) {
        const x = design[i * (p + 1) + j + 1], yy = yv[i];
        sx += x; sy += yy; sxx += x * x; syy += yy * yy; sxy += x * yy;
      }
      const cov = sxy - sx * sy / n;
      const vx = sxx - sx * sx / n, vy = syy - sy * sy / n;
      return vx > 0 && vy > 0 ? cov / Math.sqrt(vx * vy) : 0;
    };

    if (p > MAX_FEATURES) {
      const ranked = useNames.map((nm, j) => ({ nm, spec: useSpec[j], r: Math.abs(corrY(j)) }))
        .sort((a, b) => b.r - a.r).slice(0, MAX_FEATURES);
      const keep = new Set(ranked.map(r => r.nm));
      useNames.forEach(nm => {
        if (!keep.has(nm)) dropped.push({ column: nm, reason: `特徵數超過 ${MAX_FEATURES}，依 |r| 排序後移除` });
      });
      useNames = ranked.map(r => r.nm);
      useSpec = ranked.map(r => r.spec);
      p = useNames.length;
      design = new Float64Array(valid * (p + 1));
      buildDesign(useSpec, useNames, design);
    }

    /* ---- remove exactly collinear columns (keeps X'X invertible) ---- */
    await say('檢查完全共線（線性相依）的特徵…');
    const gramMatrix = () => {
      const G = new Float64Array((p + 1) * (p + 1));
      for (let i = 0; i < valid; i++) {
        const b = i * (p + 1);
        for (let a = 0; a <= p; a++) {
          const va = design[b + a];
          if (!va) continue;
          for (let c2 = a; c2 <= p; c2++) G[a * (p + 1) + c2] += va * design[b + c2];
        }
      }
      for (let a = 0; a <= p; a++) for (let b = 0; b < a; b++) G[a * (p + 1) + b] = G[b * (p + 1) + a];
      return G;
    };
    const independent = (G, cols) => {
      const m = cols.length;
      const sc = cols.map(j => Math.sqrt(G[j * (p + 1) + j]));
      if (sc.some(v => !(v > 0))) return false;
      const A = new Float64Array(m * m), b = new Float64Array(m).fill(1);
      for (let i = 0; i < m; i++)
        for (let j = 0; j < m; j++) A[i * m + j] = G[cols[i] * (p + 1) + cols[j]] / (sc[i] * sc[j]);
      return solveLinear(A, b, m) !== null;
    };
    {
      const G0 = gramMatrix();
      const chosen = [0];
      const byCorr = useNames.map((nm, j) => ({ j: j + 1, r: Math.abs(corrY(j)) }))
        .sort((a, b) => b.r - a.r);
      byCorr.forEach(o => {
        if (independent(G0, chosen.concat(o.j))) chosen.push(o.j);
        else dropped.push({ column: useNames[o.j - 1], reason: '與其他特徵完全線性相依（共線），自動移除' });
      });
      if (chosen.length !== p + 1) {
        useNames = chosen.slice(1).map(j => useNames[j - 1]);
        useSpec = chosen.slice(1).map(j => useSpec[j - 1]);
        p = useNames.length;
        design = new Float64Array(valid * (p + 1));
        buildDesign(useSpec, useNames, design);
      }
    }

    /* ---- split ---- */
    await say('切分訓練 / 測試集…');
    const n = valid;
    const order = shuffledIdx(n, RANDOM_STATE);
    const nTest = Math.max(1, Math.round(n * testSize));
    const testPos = new Set(order.slice(0, nTest));
    const trainRows = [], trainY = [], testRows = [], testY = [], testOrigIdx = [];
    for (let i = 0; i < n; i++) {
      const dim = p + 1;
      const r = design.subarray(i * dim, i * dim + dim);
      if (testPos.has(i)) { testRows.push(Float64Array.from(r)); testY.push(yv[i]); testOrigIdx.push(keepRows[i]); }
      else { trainRows.push(Float64Array.from(r)); trainY.push(yv[i]); }
    }
    // column means of the training set (needed for VIF)
    const means = new Float64Array(p + 1);
    for (let j = 0; j <= p; j++) {
      let s = 0; for (const r of trainRows) s += r[j];
      means[j] = s / trainRows.length;
    }
    const trainAgg = aggregates(trainRows, Float64Array.from(trainY), p);
    trainAgg.means = means;
    const testAgg = aggregates(testRows, Float64Array.from(testY), p);
    testAgg.means = means;

    /* ---- k-fold ---- */
    await say(`建立 ${folds}-fold 交叉驗證…`);
    const foldIdx = shuffledIdx(trainRows.length, RANDOM_STATE + 7);
    const foldOf = new Int32Array(trainRows.length);
    foldIdx.forEach((pos, rank) => { foldOf[pos] = rank % folds; });
    const foldAggs = [];
    for (let f = 0; f < folds; f++) {
      const tr = [], trY = [], va = [], vaY = [];
      for (let i = 0; i < trainRows.length; i++) {
        if (foldOf[i] === f) { va.push(trainRows[i]); vaY.push(trainY[i]); }
        else { tr.push(trainRows[i]); trY.push(trainY[i]); }
      }
      const a1 = aggregates(tr, Float64Array.from(trY), p);
      const a2 = aggregates(va, Float64Array.from(vaY), p);
      a1.means = means; a2.means = means;
      foldAggs.push({ train: a1, val: a2 });
    }

    const cvStats = (idx) => {
      const rmses = [];
      for (const f of foldAggs) {
        const sol = subsetSolve(f.train, idx);
        if (!sol) return null;
        rmses.push(aggRmse(f.val, idx, sol.beta));
      }
      const m = mean(rmses);
      return { rmse: m, sd: std(rmses), folds: rmses };
    };

    /* ---- selection helpers ---- */
    const allIdx = () => Array.from({ length: p }, (_, i) => i + 1);

    const corrRanking = () => useNames.map((nm, j) => ({ j: j + 1, r: Math.abs(corrY(j)) }))
      .sort((a, b) => b.r - a.r).map(o => o.j);

    const forwardSfs = async (limit) => {
      const chosen = [];
      const rest = new Set(allIdx());
      for (let step = 0; step < limit; step++) {
        let best = null;
        for (const j of rest) {
          const cv = cvStats([0].concat(chosen, j));
          if (!cv) continue;
          if (!best || cv.rmse < best.rmse) best = { j, rmse: cv.rmse };
        }
        if (!best) break;
        chosen.push(best.j); rest.delete(best.j);
        if (step % 3 === 0) await say(`Forward SFS：已選 ${chosen.length} 個特徵…`);
      }
      return chosen;
    };

    const rfeRanking = () => {
      let idx = allIdx();
      const removed = [];
      while (idx.length > 1) {
        const st = fitStats(trainAgg, [0].concat(idx));
        if (!st) break;
        let worst = 0, worstT = Infinity;
        for (let i = 1; i < st.coefs.length; i++) {
          const t = Math.abs(st.coefs[i].t);
          if (isFinite(t) && t < worstT) { worstT = t; worst = i; }
        }
        if (!worst) return idx.concat(removed.reverse());   // no finite t -> give up gracefully
        removed.push(idx[worst - 1]);
        idx = idx.filter((_, i) => i !== worst - 1);
      }
      return idx.concat(removed.reverse());     // survivors first, then the least important
    };

    const kCap = Math.max(1, Math.min(+kMax || 8, p, trainRows.length - 3));

    await say('特徵排序：相關性 / Forward SFS / RFE…');
    const rankings = {
      'Correlation': corrRanking(),
      'Forward SFS': await forwardSfs(kCap),
      'RFE': rfeRanking()
    };

    /* ---- helpers used while building entries (must precede the loop) ---- */
    const highlight = useNames
      .map((nm, j) => ({ nm, spec: useSpec[j], r: Math.abs(corrY(j)) }))
      .filter(o => o.spec.kind === 'num')
      .sort((a, b) => b.r - a.r).slice(0, 3)
      .map(o => o.nm);
    const num = (v, d = 2) => (isFinite(v) ? Number(v).toLocaleString('en-US', { maximumFractionDigits: d }) : '—');

    const rowValue = (origIdx, nm) => {
      const spec = useSpec[useNames.indexOf(nm)];
      if (!spec) return '—';
      if (spec.kind === 'num') return numData.get(spec.col)[origIdx];
      return String(rows[origIdx][colIdx.get(spec.col)]).trim() === spec.level ? 1 : 0;
    };

    function equationOf(idx, beta) {
      const parts = [`${target} = ${num(beta[0], 2)}`];
      for (let i = 1; i < idx.length; i++) {
        const b = beta[i];
        parts.push(`${b >= 0 ? '+' : '−'} ${num(Math.abs(b), 4)}·${useNames[idx[i] - 1]}`);
      }
      return parts.join(' ');
    }

    /* ---- fit every (method, k) ---- */
    const models = {};
    const methods = Object.keys(rankings);
    for (let mi = 0; mi < methods.length; mi++) {
      const algo = methods[mi];
      const rank = rankings[algo];
      for (let k = 1; k <= kCap; k++) {
        await say(`${algo}：k = ${k}/${kCap}…`);
        const idx = [0].concat(rank.slice(0, k));
        const e = buildEntry(algo, k, idx);
        if (!e) continue;
        (models[algo] = models[algo] || []).push(e);
      }
    }
    Object.keys(models).forEach(a => { if (!models[a] || !models[a].length) delete models[a]; });
    if (!Object.keys(models).length) throw new Error('所有模型都無法擬合（特徵可能完全共線），請改用其他目標欄位或減少類別。');
    const methodList = Object.keys(models);

    function buildEntry(algo, k, idx) {
      const st = fitStats(trainAgg, idx);
      if (!st) return null;
      const beta = st.beta;
      const preds = testRows.map(r => predictRow(r, idx, beta));
      const resid = testY.map((v, i) => v - preds[i]);
      const errs = resid.map(Math.abs);
      const rmse = Math.sqrt(mean(errs.map(e => e * e)));
      const mae = mean(errs);
      const testTss = testY.reduce((s, v) => s + (v - mean(testY)) ** 2, 0);
      const testRss = resid.reduce((s, e) => s + e * e, 0);
      const r2 = testTss > 0 ? 1 - testRss / testTss : 0;
      const cv = cvStats(idx) || { rmse: NaN, sd: NaN };
      const vidx = idx.slice(1);
      const coefs = [];
      for (let i = 0; i < idx.length; i++) {
        const c = st.coefs[i];
        let vif = null;
        if (i > 0) vif = vifFor(trainAgg, idx, i);
        coefs.push({
          feature: i === 0 ? 'Intercept (β₀)' : useNames[vidx[i - 1] - 1],
          coef: c.coef, stderr: isFinite(c.se) ? c.se : null,
          tstat: isFinite(c.t) ? c.t : null,
          pvalue: isFinite(c.p) ? c.p : null,
          ci_lower: isFinite(c.lo) ? c.lo : null,
          ci_upper: isFinite(c.hi) ? c.hi : null,
          vif
        });
      }
      const outlierRows = testRows.map((r, i) => ({
        index: testOrigIdx[i],
        features: highlight.reduce((o, h) => {
          o[h] = rowValue(testOrigIdx[i], h);
          return o;
        }, {}),
        actual: testY[i], predicted: preds[i], residual: resid[i],
        abs_error: Math.abs(resid[i]),
        pct_error: testY[i] ? Math.abs(resid[i]) / Math.abs(testY[i]) * 100 : 0
      })).sort((a, b) => b.abs_error - a.abs_error).slice(0, 10)
        .map((r, i) => Object.assign(r, { rank: i + 1 }));

      return {
        algo, k,
        features: idx.slice(1).map(j => useNames[j - 1]),
        rmse, mae, r2, adj_r2: st.adj, train_r2: st.r2,
        cv_rmse: cv.rmse, cv_rmse_std: cv.sd,
        f_stat: st.f, f_pvalue: st.fP, aic: st.aic, bic: st.bic,
        equation: equationOf(idx, beta),
        coefficients: coefs,
        outliers: outlierRows
      };
    }

    /* ---- baseline + backward elimination ---- */
    await say('baseline 與 backward elimination…');
    const baseIdx = [0].concat(allIdx());
    const baseEntry = buildEntry('Baseline', p, baseIdx);
    if (!baseEntry) throw new Error('全特徵模型無法擬合（完全共線），請改用其他目標欄位。');

    let workIdx = allIdx(), droppedBe = [];
    let beStats = fitStats(trainAgg, [0].concat(workIdx));
    while (workIdx.length > 1 && beStats) {
      let worst = -1, worstP = -1;
      for (let i = 1; i < beStats.coefs.length; i++) {
        const pv = beStats.coefs[i].p;
        if (isFinite(pv) && pv > worstP) { worstP = pv; worst = i; }
      }
      if (worst < 0 || worstP <= 0.05) break;
      droppedBe.push({ feature: useNames[workIdx[worst - 1] - 1], pvalue: worstP });
      workIdx = workIdx.filter((_, i) => i !== worst - 1);
      beStats = fitStats(trainAgg, [0].concat(workIdx));
    }
    const beIdx = [0].concat(workIdx);
    const beEntry = buildEntry('p-value', workIdx.length, beIdx) || baseEntry;
    if (!beStats) beStats = fitStats(trainAgg, beIdx) || { r2: beEntry.r2, adj: beEntry.adj_r2 };

    /* ---- best / noise band / final decision ---- */
    const all = methodList.flatMap(a => models[a].map(e => Object.assign({ algo: a }, e)));
    const cvBest = all.reduce((x, y) => (y.cv_rmse < x.cv_rmse ? y : x));
    const hoBest = all.reduce((x, y) => (y.rmse < x.rmse ? y : x));
    const noise = mean(all.map(e => e.cv_rmse_std).filter(isFinite));
    const cvMin = Math.min(...all.map(e => e.cv_rmse));
    const sfs = models['Forward SFS'] || models[methodList[0]];
    const parsimonious = sfs.find(e => e.cv_rmse <= cvMin * 1.01) || sfs[sfs.length - 1];

    /* ---- diagnostics on the hold-out best model ---- */
    await say('迴歸診斷（常態性 / 異質變異數 / 自相關）…');
    const bestIdx = [0].concat(hoBest.features.map(f => useNames.indexOf(f) + 1));
    const bestSol = subsetSolve(trainAgg, bestIdx);
    const trainPred = bestSol ? trainRows.map(r => predictRow(r, bestIdx, bestSol.beta)) : trainRows.map(() => NaN);
    const trainResid = trainY.map((v, i) => v - trainPred[i]);
    const jb = jarqueBera(trainResid);
    const bp = breuschPagan(trainResid, trainRows, p);
    const dw = durbinWatson(trainResid);
    const vifs = bestIdx.slice(1).map((_, i) => vifFor(trainAgg, bestIdx, i + 1)).filter(v => v !== null);
    const maxVif = vifs.length ? Math.max(...vifs) : 0;

    const assumptions = [
      jb ? {
        test: 'Jarque-Bera（殘差常態）', stat: jb.stat, pvalue: jb.p, criterion: 'p > 0.05',
        status: jb.p > 0.05 ? 'pass' : 'warn',
        note: `偏度 ${num(jb.skew, 2)}、峰度 ${num(jb.kurt, 2)}；樣本數大時容易拒絕常態`
      } : null,
      bp ? {
        test: 'Breusch-Pagan（同變異數）', stat: bp.stat, pvalue: bp.p, criterion: 'p > 0.05',
        status: bp.p > 0.05 ? 'pass' : 'fail',
        note: bp.p > 0.05 ? '未偵測到異質變異數' : '殘差變異數隨特徵改變 → 建議穩健標準誤或對目標取對數'
      } : null,
      {
        test: 'Durbin-Watson（獨立性）', stat: dw, pvalue: null, criterion: '≈ 2',
        status: Math.abs(dw - 2) < 0.5 ? 'pass' : 'warn',
        note: '值接近 2 表示無明顯一階自相關'
      },
      {
        test: 'Max VIF（共線性）', stat: maxVif, pvalue: null, criterion: '< 10',
        status: maxVif < 10 ? 'pass' : 'fail',
        note: maxVif < 10 ? '共線性在可接受範圍內' : '存在嚴重共線性，係數解讀需保守'
      }
    ].filter(Boolean);

    /* ---- describe / correlation / rows sample / inputs ---- */
    await say('整理資料表與圖表資料…');
    const describe = {};
    const correlation = {};
    const fullRows = trainRows.concat(testRows);
    const fullY = trainY.concat(testY);
    const dim = p + 1;
    const colVals = j => fullRows.map(r => r[j]);
    for (let j = 1; j <= p; j++) {
      const vals = colVals(j);
      const s = vals.slice().sort((a, b) => a - b);
      describe[useNames[j - 1]] = {
        count: vals.length, mean: mean(vals), std: std(vals),
        min: s[0], '25%': quantileSorted(s, 0.25), '50%': quantileSorted(s, 0.5),
        '75%': quantileSorted(s, 0.75), max: s[s.length - 1]
      };
    }
    {
      const s = fullY.slice().sort((a, b) => a - b);
      describe[target] = {
        count: fullY.length, mean: mean(fullY), std: std(fullY),
        min: s[0], '25%': quantileSorted(s, 0.25), '50%': quantileSorted(s, 0.5),
        '75%': quantileSorted(s, 0.75), max: s[s.length - 1]
      };
    }
    const corrBetween = (a, b) => {
      const n2 = fullY.length;
      let sa = 0, sb = 0, saa = 0, sbb = 0, sab = 0;
      for (let i = 0; i < n2; i++) {
        const x = a[i], y2 = b[i];
        sa += x; sb += y2; saa += x * x; sbb += y2 * y2; sab += x * y2;
      }
      const cov = sab - sa * sb / n2, va = saa - sa * sa / n2, vb = sbb - sb * sb / n2;
      return va > 0 && vb > 0 ? cov / Math.sqrt(va * vb) : 0;
    };
    const allNames = useNames.concat([target]);
    const allCols = {};
    for (let j = 1; j <= p; j++) allCols[useNames[j - 1]] = colVals(j);
    allCols[target] = fullY;
    allNames.forEach(a => {
      correlation[a] = {};
      allNames.forEach(b => { correlation[a][b] = corrBetween(allCols[a], allCols[b]); });
    });

    const sampleStep = Math.max(1, Math.floor(keepRows.length / 200));
    const sample = [];
    for (let i = 0; i < keepRows.length && sample.length < 200; i += sampleStep) {
      const rec = { index: keepRows[i] };
      header.forEach((h, j) => {
        const raw = rows[keepRows[i]][j];
        const nv = toNum(raw);
        rec[h] = isFinite(nv) && profile[j].isNumeric ? nv : String(raw === undefined ? '' : raw).trim();
      });
      sample.push(rec);
    }

    const predictor_inputs = [];
    numCols.forEach(c => {
      if (!useNames.includes(c.name)) return;
      const d = describe[c.name];
      const span = d.max - d.min;
      const colJ = useNames.indexOf(c.name) + 1;
      const sortedVals = colVals(colJ).slice().sort((a, b) => a - b);
      const binary = d.min === 0 && d.max === 1;
      let lo = quantileSorted(sortedVals, 0.01);
      let hi = quantileSorted(sortedVals, 0.99);
      if (!(hi > lo)) { lo = d.min; hi = d.max; }        // rare binary flags etc.
      if (!(hi > lo)) { hi = lo + 1; }
      const isInt = Number.isInteger(d.min) && Number.isInteger(d.max);
      let step, decimals;
      if (binary) { decimals = 0; step = 1; }
      else if (isInt) { decimals = 0; step = span > 5000 ? 100 : span > 500 ? 10 : 1; }
      else if (span > 1000) { step = 100; decimals = 0; }
      else if (span > 100) { step = 10; decimals = 1; }
      else if (span > 10) { step = 1; decimals = 1; }
      else if (span > 1) { step = 0.1; decimals = 2; }
      else { step = 0.001; decimals = 4; }
      predictor_inputs.push({
        key: c.name, feature: c.name, label: c.name, type: 'range',
        min: Number(lo.toFixed(decimals)), max: Number(hi.toFixed(decimals)),
        step, decimals, default: Number(d['50%'].toFixed(decimals))
      });
    });
    catCols.forEach(c => {
      const levels = cats.get(c.name);
      if (!useNames.some(nm => nm.startsWith(c.name + '_'))) return;
      predictor_inputs.push({
        key: c.name, feature: c.name, label: c.name, type: 'select',
        prefix: c.name + '_', options: levels, reference: levels[0],
        default: levels[0]
      });
    });

    const columns = [];
    header.forEach(h => {
      if (h === target) return;
      const pf = profile.find(x => x.name === h);
      if (!pf) return;
      columns.push({ key: h, label: h, kind: pf.isNumeric ? 'numeric' : 'categorical' });
    });
    columns.push({ key: target, label: target, kind: 'target' });

    const ts = describe[target];
    const meta = {
      id: opts.id, title: opts.title || name,
      k_max: kCap,
      outlier_highlight: highlight,
      top_feature: {
        name: useNames.reduce((best, nm) => (Math.abs(correlation[nm][target]) > Math.abs(correlation[best][target]) ? nm : best), useNames[0]),
        corr: Math.abs(correlation[useNames.reduce((best, nm) => (Math.abs(correlation[nm][target]) > Math.abs(correlation[best][target]) ? nm : best), useNames[0])][target])
      },
      predictor_inputs,
      dataset: {
        name: name, source: source || '', rows: n, raw_rows: rawRows, raw_columns: rawCols,
        features: useNames, numeric_features: numCols.filter(c => useNames.includes(c.name)).map(c => c.name),
        categorical_features: catCols.filter(c => useNames.some(nm => nm.startsWith(c.name + '_'))).map(c => c.name),
        target, target_label: target, unit: '',
        train_rows: trainRows.length, test_rows: testRows.length,
        random_state: RANDOM_STATE, test_size: testSize, cv_folds: folds,
        missing_values: imputed, duplicates: 0, sampled_rows: sample.length,
        dropped,
        columns,
        categorical_values: Object.fromEntries(
          catCols.filter(c => useNames.some(nm => nm.startsWith(c.name + '_')))
            .map(c => [c.name, cats.get(c.name)])
        )
      },
      describe, correlation, rows: sample,
      baseline: baseEntry,
      selected_pvalue: {
        k: workIdx.length, features: workIdx.map(j => useNames[j - 1]),
        equation: beEntry.equation, r2: beEntry.r2, adj_r2: beEntry.adj_r2, rmse: beEntry.rmse
      },
      backward_elimination: {
        retained: workIdx.map(j => useNames[j - 1]),
        dropped: droppedBe,
        coefficients: beEntry.coefficients.map(c => ({
          feature: c.feature.startsWith('Intercept') ? 'const' : c.feature,
          coef: c.coef, stderr: c.stderr, tstat: c.tstat, pvalue: c.pvalue,
          ci_lower: c.ci_lower, ci_upper: c.ci_upper
        })),
        r2: beStats.r2, adj_r2: beStats.adj
      },
      assumptions,
      cv_best: { algo: cvBest.algo, k: cvBest.k, features: cvBest.features, cv_rmse: cvBest.cv_rmse, cv_rmse_std: cvBest.cv_rmse_std, rmse: cvBest.rmse, r2: cvBest.r2 },
      holdout_best: { algo: hoBest.algo, k: hoBest.k, features: hoBest.features, rmse: hoBest.rmse, mae: hoBest.mae, r2: hoBest.r2, cv_rmse: hoBest.cv_rmse, cv_rmse_std: hoBest.cv_rmse_std },
      noise_band: noise,
      final_decision: {
        algo: 'Forward SFS', k: parsimonious.k, model: `Forward SFS · ${parsimonious.k} 特徵 OLS`,
        features: parsimonious.features, rmse: parsimonious.rmse, r2: parsimonious.r2,
        reason: `k=${parsimonious.k} 之後 CV RMSE 的改善已小於 1%（雜訊帶 ±${noise.toLocaleString('en-US', { maximumFractionDigits: 0 })}）`
      },
      tuning: { methods: methodList, test_size: testSize, folds, k_max: kCap, encoded_features: p }
    };

    return {
      payload: { meta, models },
      summary: {
        rows: n, features: p, methods: methodList, kMax: kCap,
        target, top: meta.top_feature, dropped: dropped.length, imputed,
        cvBest: meta.cv_best, hoBest: meta.holdout_best
      }
    };
  }

  /* --------------------------------------- generic narrative config -- */
  function buildDynamicConfig(payload, source) {
    const m = payload.meta;
    const d = m.dataset;
    const label = `自訂 · ${d.name}`;
    return {
      id: m.id,
      dynamic: true,
      dataFile: null,
      label: label.length > 64 ? label.slice(0, 61) + '…' : label,
      short: d.name,
      hint: `${d.rows.toLocaleString('en-US')} 筆 × ${d.features.length} 個編碼特徵 · ${Object.keys(payload.models).length} 種方法 · k = 1–${m.k_max}`,
      navTitle: `${d.name} · Auto CRISP-DM Studio`,
      navSub: 'Imported dataset · Multiple Linear Regression',
      pageTitle: `${d.name} · 自動 CRISP-DM 多元線性迴歸分析`,
      metaDescription: `自動分析 ${d.name}：CRISP-DM 六階段、${Object.keys(payload.models).length} 種特徵選擇法、互動預測`,
      hero: {
        eyebrow: '自動分析 · 任意 CSV · 瀏覽器端計算',
        title: '以<span class="grad">多元線性迴歸</span>即時解析 <span class="grad">{{dataset}}</span>',
        lead: `這份結果由瀏覽器端的自動化 CRISP-DM 流程產生：解析 CSV → 清理與編碼 → hold-out 切分與 {{cv_folds}}-fold CV
          → {{methods}} 種特徵選擇法 × k = 1…{{kmax}} → OLS 推論統計與迴歸診斷。
          使用 {{n}} 筆資料、{{features}} 個編碼後的預測變數，所有計算都在你的瀏覽器完成。`,
        badges: [
          ['cyan', '{{n}} 筆 × {{features}} 個編碼特徵'],
          ['violet', '{{methods}} 種方法 × k = 1–{{kmax}}'],
          ['ok', '{{cv_folds}}-fold CV · 固定隨機種子'],
          ['warn', '缺失值補值 {{imputed}} 個']
        ]
      },
      sections: {
        crispdm: { title: 'CRISP-DM 六階段流程', lead: '自動化流程的每個階段與實際產出；數字全部由本次分析即時計算。' },
        explorer: { title: '模型探索器', lead: '切換方法與 k，觀察測試集指標、CV 誤差、係數推論與模型比較圖。' },
        predict: { title: '互動預測器', lead: '調整特徵值，立即以目前選定的模型（{{selected_tag}}）計算預測值與貢獻拆解。', refSub: '全資料平均' },
        diagnostics: { title: '診斷、比較與結論', lead: '迴歸假設檢定、全模型比較表與原始資料檢視。' }
      },
      phase1: {
        objective: [
          '<b>問題定義：</b>使用者提供了一份自訂資料集，目標是<b>自動</b>找出最能解釋 <span class="mono">{{target}}</span> 的線性模型。',
          '<b>業務決策：</b>哪些欄位值得持續蒐集？哪一個變數是主要槓桿？模型誤差是否足以支撐實務判斷？',
          '<b>分析目標：</b>建立 <span class="mono">{{target}} = β₀ + Σ βᵢ·xᵢ + ε</span>，並在 k 個特徵間尋找準確度與精簡度的平衡。',
          '<b>為何用線性迴歸：</b>係數可直接解讀為「其他條件不變下，該特徵每增加 1 單位，目標變動多少」。'
        ],
        criteria: [
          ['統計', '係數顯著性（p &lt; 0.05）、VIF &lt; 10、殘差診斷'],
          ['模型', '以 hold-out RMSE / MAE / R² 與 {{cv_folds}}-fold CV 比較'],
          ['業務', '模型需可解釋，且能在瀏覽器中即時試算']
        ]
      },
      phase2: {
        conclusions: [
          '原始檔案 {{raw_rows}} 筆 × {{raw_columns}} 欄；本次使用 {{n}} 筆、{{features}} 個編碼特徵。',
          '與 <span class="mono">{{target}}</span> 相關性最高的是 <b>{{top}}</b>（|r| ≈ {{top_corr}}）。',
          '目標值分布：平均 {{mean}}、中位數 {{median}}、範圍 {{min}} – {{max}}。',
          '自動移除 {{dropped_count}} 個欄位（流水號、日期或類別過多），缺失值以中位數／眾數補值 {{imputed}} 個。'
        ]
      },
      phase3: {
        steps: [
          '<b>欄位清理：</b>移除流水號 / 索引、日期與高基數類別欄位（詳見 Phase 2 與原始資料表）。',
          '<b>類別編碼：</b>低基數類別欄位做 One-Hot（第一個水準為參考類別），數值欄位維持原單位。',
          '<b>缺失值：</b>數值以中位數補值、類別以眾數補值（共 {{imputed}} 個）；目標缺失的列直接刪除。',
          '<b>切分：</b>依設定比例 → 訓練 {{train}} 筆、測試 {{test}} 筆（固定隨機種子）。',
          '<b>交叉驗證：</b>{{cv_folds}}-fold，用同一組特徵子集重複擬合，估計樣本外的 RMSE。',
          '<b>推論統計：</b>以正規方程式求解 OLS，並計算 β、SE、t、p、95% CI、VIF、F 檢定。'
        ],
        vifNote: 'VIF &lt; 10 代表無有害共線性；本次最終模型的最大 VIF 為 <b id="max-vif-note">—</b>。'
      },
      phase4: {
        baselineTitle: '📐 基準模型（Baseline OLS，全 {{features}} 特徵）',
        intro: '先把所有編碼特徵放進 OLS 當基準，再讓 {{methods}} 種方法各自排序特徵，取前 k 個重新擬合並以 {{cv_folds}}-fold CV 比較。',
        rankingTitle: '🔁 {{methods}} 種特徵選擇方法 × k = 1…{{kmax}}',
        backwardTitle: '🧭 p-value 逐步淘汰（Backward Elimination）'
      },
      phase5: {
        tensionTitle: '🏁 CV 與 Hold-out 的取捨',
        tension: [
          '<li><b>CV 最佳：</b>{{cv_best_desc}}</li>',
          '<li><b>測試集最佳：</b>{{ho_best_desc}}</li>',
          '<li><b>雜訊帶：</b>各子集 CV RMSE 的標準差平均為 <span class="mono v-amber">{{noise}}</span>。</li>',
          '<li><b>最終決策：</b>採用 <b>{{final_model}}</b>：CV 落在最佳值的 1% 內，但特徵更少、更好解釋。</li>'
        ],
        coefTitle: '🧾 OLS 推論統計（p-value 選出的模型）',
        heatTitle: '📊 {{methods}} 方法 × k 的測試集 RMSE 熱圖',
        decisionNote: `以全 {{features}} 特徵的 baseline（測試 RMSE {{base_rmse}}、R² {{base_r2}}）為上限，
          最終模型 {{final_model}}（{{final_features}}）達到 {{final_rmse}} 的測試 RMSE。
          兩者差距若小於 ± {{noise}} 的 CV 雜訊帶，就沒有理由付出額外的複雜度；
          p-value 逐步淘汰則保留了統計上顯著的欄位（{{sel_count}} 個）。`
      },
      phase6: {
        artifacts: [
          '<span class="mono">瀏覽器端分析</span> — 無需後端；資料不會上傳到伺服器。',
          '<span class="mono">本頁面</span> — 可直接切換 k 與方法、即時預測，重新整理後即消失（僅存於本次工作階段）。',
          '<span class="mono">原始檔案</span> — {{dataset}}（{{n}} 筆）。',
          '<span class="mono">Flask API</span> — 內建資料集仍可使用 <span class="mono">/api/predict</span>；自訂資料集於前端計算。'
        ],
        monitoring: [
          '自訂資料集未經人工檢視，正式使用前請確認欄位定義與單位。',
          '若 R² 偏低，先檢查非線性關係（散佈圖）與離群值，再考慮特徵工程。',
          '若 Breusch-Pagan 未通過，推論請改用穩健標準誤，或對目標取對數。',
          '重新整理頁面後可再次輸入同一網址重新分析（結果具確定性：固定隨機種子）。'
        ]
      },
      diagnostics: {
        conclusion: `<b>自動分析結論：</b><span class="mono">{{target}}</span> 的最佳線性模型為
          <b>{{final_model}}</b>（測試 RMSE {{final_rmse}}、R² {{ho_r2}}）。
          最強的單一訊號是 <span class="mono">{{top}}</span>（|r| ≈ {{top_corr}}）。
          請同時檢視下方的假設檢定：若常態性或同變異數未通過，係數的顯著性推論需保守解讀。`
      },
      datasetSorts: [
        { value: 'index', label: '原始順序' },
        { value: 'target-desc', label: '{{target}} 由高到低' },
        { value: 'target-asc', label: '{{target}} 由低到高' }
      ],
      searchPlaceholder: '搜尋任一欄位值…',
      footer: {
        artifacts: [],
        datasetHtml: `資料集：<a href="${source || '#'}" target="_blank" rel="noopener">${source ? source.split('/').pop() : d.name}</a>
          ｜ 自動分析於瀏覽器完成（{{n}} 筆 × {{features}} 個編碼特徵）`
      }
    };
  }

  window.CSVEngine = { parseCSV, detectDelimiter, profileColumns, toNum, analyze, buildDynamicConfig, MAX_ROWS, MAX_FEATURES };
})();
