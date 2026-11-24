// Transformer Playground: state, inspector panels, token editor, live test score, guided tour and missions.
import { Scene3D } from './scene.js';
import { loadWeights, randomWeights, forward, DEFAULT_OPT, testSet, score, paramCount, topk, norm, softmax } from './model.js';

const $ = s => document.querySelector(s);
function el(tag, props = {}, ...kids) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(props || {})) {
    if (k === 'class') e.className = v; else if (k === 'html') e.innerHTML = v; else if (k === 'style') e.style.cssText = v;
    else if (k.startsWith('on')) e.addEventListener(k.slice(2), v); else if (v !== false && v != null) e.setAttribute(k, v);
  }
  for (const c of kids.flat()) if (c != null && c !== false) e.append(c instanceof Node ? c : document.createTextNode(c));
  return e;
}
const pct = v => (v * 100).toFixed(v > .995 || v < .005 ? 0 : 1) + '%';
const PE_NAME = { none: 'none', learned: 'learned', sin: 'sin/cos', rope: 'RoPE' };

// ---------- state ----------
const S = {
  ix: null, w: null, tr: null, cache: new Map(), set: null, base: null, live: null,
  cfg: { L: 2, h: 2, d: 64, pe: 'rope', ffn: 4 }, mode: 'trained', seed: 1,
  ids: [], focus: 0, sel: null, opt: DEFAULT_OPT(), temp: 1, showVec: true, tour: -1, done: {},
};
const key = c => `L${c.L}H${c.h}D${c.d}-${c.pe}`;
const isTrained = c => c.ffn === 4 && S.ix.models[key(c)];
const words = () => S.ids.map(i => S.ix.vocab[i]);
const STOI = () => Object.fromEntries(S.ix.vocab.map((v, i) => [v, i]));

const scene = new Scene3D($('#stage'), { onPick: ud => select(ud), onHover: hoverTip });

async function getWeights() {
  const c = S.cfg;
  if (S.mode === 'trained' && isTrained(c)) {
    const k = key(c);
    if (!S.cache.has(k)) { $('#busy').hidden = false; S.cache.set(k, loadWeights(await (await fetch(`models/${k}.json`)).json())); $('#busy').hidden = true; }
    return S.cache.get(k);
  }
  const rk = `rand-${key(c)}-f${c.ffn}-s${S.seed}`;
  if (!S.cache.has(rk)) S.cache.set(rk, randomWeights({ ...c, V: S.ix.vocab.length, maxlen: S.ix.maxlen, seed: S.seed }));
  return S.cache.get(rk);
}

async function setModel(patch = {}) {
  Object.assign(S.cfg, patch);
  if (S.cfg.L < S.opt._L) S.opt = DEFAULT_OPT();            // interventions on blocks that no longer exist
  S.w = await getWeights(); S.base = null; S.opt._L = S.cfg.L;
  if (S.sel && S.sel.b !== undefined && S.sel.b !== 'f' && S.sel.b >= S.cfg.L) S.sel = { kind: 'model' };
  if (S.sel?.kind === 'head' && S.sel.j >= S.cfg.h) S.sel = { kind: 'attn', b: S.sel.b };
  refresh(true);
}

let scoreTimer = 0;
function refresh(rebuildScore = false) {
  S.focus = Math.min(S.focus, S.ids.length - 1);
  S.tr = forward(S.w, S.ids, S.opt);
  scene.build({ tr: S.tr, w: S.w, state: S, vocab: S.ix.vocab });
  renderTop(); renderInspector(); renderIO();
  clearTimeout(scoreTimer);
  scoreTimer = setTimeout(() => {
    if (!S.base) S.base = score(S.w, S.ix, S.set, DEFAULT_OPT());
    S.live = activeChanges().length ? score(S.w, S.ix, S.set, S.opt) : S.base;
    renderScore(); checkMissions();
  }, rebuildScore ? 30 : 120);
}

function select(ud) {
  if (ud && ud.kind === 'token') { S.focus = ud.t; }
  if (ud && ud.kind === 'vec') ud = { kind: 'lane', t: ud.t, k: ud.s };
  if (ud && (ud.kind === 'lane' || ud.kind === 'output') && ud.t !== undefined) S.focus = ud.t;
  S.sel = ud || { kind: 'model' };
  refresh(); if (ud && ud.kind !== 'token') scene.flyTo(ud);
  if (innerWidth < 900) $('#insp').scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// ---------- tooltip ----------
const tip = $('#tip');
function hoverTip(ud, e) {
  if (!ud) { tip.hidden = true; return; }
  const n = { token: 'Token', embed: 'Embedding table', pos: 'Position signal', ln: 'LayerNorm', attn: 'Attention', head: `Head ${ud.j + 1}`, add: 'Residual add ⊕', ffn: 'Feed-forward', block: 'Block', unembed: 'Unembedding', output: 'Prediction', lane: 'Residual stream', vec: 'Vector in the stream' }[ud.kind] || ud.kind;
  const where = ud.b === 'f' ? ' · final' : ud.b !== undefined ? ` · block ${ud.b + 1}` : '';
  const tok = ud.t !== undefined ? ` · “${words()[ud.t]}”` : '';
  tip.textContent = `${n}${where}${tok} — click to open`; tip.hidden = false;
  const r = $('#stage').getBoundingClientRect(); tip.style.left = (e.clientX - r.left + 14) + 'px'; tip.style.top = (e.clientY - r.top + 12) + 'px';
}

// ---------- header ----------
function renderTop() {
  const c = S.cfg, tr = S.w.cfg.trained;
  $('#arch').replaceChildren(
    ...[`${c.L} layer${c.L > 1 ? 's' : ''}`, `${c.h} head${c.h > 1 ? 's' : ''}`, `d=${c.d}`, `pos: ${PE_NAME[c.pe]}`, c.ffn !== 4 ? `FFN ×${c.ffn}` : null]
      .filter(Boolean).map(t => el('button', { class: 'chip', onclick: () => select({ kind: 'model' }) }, t)),
    el('span', { class: 'tag ' + (tr ? 'ok' : 'warn') }, tr ? 'trained' : 'untrained · random weights'));
}
function renderScore() {
  const b = S.base, l = S.live; if (!b) return;
  const lab = { capital: 'capital', name: 'name', agree: 'agree' };
  $('#score').replaceChildren(el('span', { class: 'sl' }, 'test'), ...Object.keys(lab).map(k => {
    const d = l[k] - b[k];
    return el('div', { class: 'sc', title: `${k}: ${pct(l[k])} of the test sentences right (no changes: ${pct(b[k])})` },
      el('span', {}, lab[k]), el('i', { style: `--v:${l[k]}` }), el('b', { class: d < -.005 ? 'down' : d > .005 ? 'up' : '' }, pct(l[k])));
  }));
}

// ---------- token editor + next word ----------
const PRESETS = [
  ['capital', 'the capital of france is'], ['capital ←', 'tokyo is the capital of'],
  ['name', 'i am alice . you are bob . who am i ?'], ['name', 'you are emma . i am jack . who are you ?'],
  ['agree', 'the cats near the dog near the bird'], ['agree', 'the fox near the birds near the dogs'],
];
let picker = null;
function openPicker(anchor, onPick) {
  closePicker(); const cats = S.ix.cats;
  picker = el('div', { class: 'picker' }, ...Object.entries(cats).map(([c, ws]) => el('div', { class: 'grp' }, el('span', {}, c), ...ws.map(w => el('button', { onclick: () => { closePicker(); onPick(w); } }, w)))));
  document.body.append(picker);
  const r = anchor.getBoundingClientRect(), pw = Math.min(520, innerWidth - 16);
  picker.style.width = pw + 'px'; picker.style.left = Math.max(8, Math.min(r.left, innerWidth - pw - 8)) + 'px';
  const ph = picker.offsetHeight; picker.style.top = (r.top > ph + 12 ? r.top - ph - 8 : r.bottom + 8) + 'px';
  setTimeout(() => addEventListener('pointerdown', outside), 0);
}
const outside = e => { if (picker && !picker.contains(e.target)) closePicker(); };
function closePicker() { picker?.remove(); picker = null; removeEventListener('pointerdown', outside); }

function setText(text) { const st = STOI(); S.ids = text.split(' ').filter(w => w in st).map(w => st[w]); S.focus = S.ids.length - 1; refresh(); }
function renderIO() {
  const ws = words(), max = S.ix.maxlen;
  $('#tokens').replaceChildren(...ws.map((w, t) => el('span', { class: 'tchip' + (t === S.focus ? ' on' : '') },
    el('button', { class: 'w', title: 'swap this word', onclick: e => openPicker(e.currentTarget, nw => { S.ids[t] = STOI()[nw]; refresh(); }) }, w),
    el('button', { class: 'f', title: 'look from this token', onclick: () => { S.focus = t; refresh(); } }, '◉'),
    ws.length > 1 ? el('button', { class: 'x', title: 'remove', onclick: () => { S.ids.splice(t, 1); refresh(); } }, '×') : null)),
    ws.length < max ? el('button', { class: 'tadd', onclick: e => openPicker(e.currentTarget, nw => { S.ids.push(STOI()[nw]); S.focus = S.ids.length - 1; refresh(); }) }, '+ word') : null);
  $('#presets').replaceChildren(...PRESETS.map(([k, t]) => el('button', { class: 'pre' + (ws.join(' ') === t ? ' on' : ''), onclick: () => setText(t), title: t }, el('i', {}, k), t)));
  const last = S.tr.logits[S.ids.length - 1], p = softmax([...last], S.temp), top = topk(p, 5, S.ix.vocab);
  $('#next').replaceChildren(el('span', { class: 'nl' }, 'next word'), ...top.map(o => el('button', { class: 'nx', disabled: ws.length >= max, onclick: () => { S.ids.push(o.id); S.focus = S.ids.length - 1; refresh(); } },
    el('b', {}, o.w), el('i', { style: `--v:${o.p}` }), el('em', {}, pct(o.p)))),
    el('button', { class: 'nx go', disabled: ws.length >= max, onclick: complete }, 'complete ⏵'));
}
function complete() {
  const dot = STOI()['.'];
  for (let n = 0; n < 8 && S.ids.length < S.ix.maxlen; n++) {
    const tr = forward(S.w, S.ids, S.opt, false), z = tr.logits[S.ids.length - 1]; let best = 0; z.forEach((v, i) => { if (v > z[best]) best = i; });
    S.ids.push(best); if (best === dot) break;
  }
  S.focus = S.ids.length - 1; refresh();
}

// ---------- inspector widgets ----------
function seg(label, opts, cur, on, mark = () => false) {
  return el('div', { class: 'field' }, el('label', {}, label), el('div', { class: 'seg' }, ...opts.map(([v, t]) => el('button', { class: (v === cur ? 'on ' : '') + (mark(v) ? 'dot' : ''), onclick: () => on(v) }, t))));
}
function toggle(label, val, on, hint) {
  return el('label', { class: 'tog' + (val ? ' on' : '') }, el('input', { type: 'checkbox', checked: val || null, onchange: e => on(e.target.checked) }), el('span', { class: 'sw' }), el('span', {}, label, hint ? el('small', {}, hint) : null));
}
function slider(label, min, max, step, val, on, fmt = v => v) {
  const out = el('b', {}, fmt(val));
  return el('div', { class: 'field sl' }, el('label', {}, label, out), el('input', { type: 'range', min, max, step, value: val, oninput: e => { out.textContent = fmt(+e.target.value); on(+e.target.value, false); }, onchange: e => on(+e.target.value, true) }));
}
function heat(M, { rows = [], cols = [], color = '92,225,255', signed = false, cw = 22, ch = 20, hi = -1, hiCol = -1, onCell, maxAbs, rl = 64, cl = 54, vals = false }) {
  const R = M.length, C = M[0].length, dpr = Math.min(devicePixelRatio, 2);
  const W = rl + C * cw, H = (cols.length ? cl : 4) + R * ch, cv = el('canvas', { class: 'heat' });
  cv.width = W * dpr; cv.height = H * dpr; cv.style.width = W + 'px'; cv.style.maxWidth = '100%';
  const g = cv.getContext('2d'); g.scale(dpr, dpr); const top = cols.length ? cl : 4;
  const mx = maxAbs ?? Math.max(...M.flat().map(Math.abs), 1e-9);
  g.font = '11px JBM, monospace'; g.textBaseline = 'middle';
  M.forEach((row, r) => row.forEach((v, c) => {
    const a = Math.min(1, Math.abs(v) / mx);
    g.fillStyle = signed && v < 0 ? `rgba(255,92,138,${a})` : `rgba(${color},${a})`;
    g.fillRect(rl + c * cw + .5, top + r * ch + .5, cw - 1, ch - 1);
    if (vals && cw >= 26 && a > .02) { g.fillStyle = a > .55 ? '#010402' : '#cfe'; g.textAlign = 'center'; g.fillText(v.toFixed(2).replace(/^0/, ''), rl + c * cw + cw / 2, top + r * ch + ch / 2); }
  }));
  g.textAlign = 'right';
  rows.forEach((t, r) => { g.fillStyle = r === hi ? '#E6FFEC' : '#8DFFB0aa'; g.fillText(t, rl - 6, top + r * ch + ch / 2); });
  cols.forEach((t, c) => { g.save(); g.translate(rl + c * cw + cw / 2, top - 6); g.rotate(-Math.PI / 3); g.textAlign = 'left'; g.fillStyle = c === hiCol ? '#E6FFEC' : '#8DFFB0aa'; g.fillText(t, 0, 0); g.restore(); });
  if (hi >= 0) { g.strokeStyle = '#E6FFEC'; g.lineWidth = 1.5; g.strokeRect(rl, top + hi * ch, C * cw, ch); }
  if (onCell) cv.addEventListener('click', e => { const b = cv.getBoundingClientRect(), s = W / b.width; const c = Math.floor(((e.clientX - b.left) * s - rl) / cw), r = Math.floor(((e.clientY - b.top) * s - top) / ch); if (r >= 0 && c >= 0 && r < R && c < C) onCell(r, c); });
  return el('div', { class: 'heatwrap' }, cv);
}
function bars(items, color = 'var(--beam)') {
  return el('div', { class: 'bars' }, ...items.map(o => el('div', { class: 'br' }, el('span', {}, o.w), el('i', { style: `--v:${Math.max(.01, o.p)};--c:${color}` }), el('em', {}, pct(o.p)))));
}
const vecStrip = (v, w = 300) => { const mx = Math.max(...v.map(Math.abs), 1e-9); return el('div', { class: 'vstrip', title: 'green = positive, pink = negative' }, ...[...v].map(x => el('i', { style: `background:${x >= 0 ? `rgba(77,255,136,${Math.abs(x) / mx})` : `rgba(255,92,138,${Math.abs(x) / mx})`}` }))); };
const guide = (html) => el('div', { class: 'guide', html });
const tryit = (...items) => el('div', { class: 'try' }, el('b', {}, 'Try'), el('ul', {}, ...items.map(t => el('li', { html: t }))));
const formula = f => el('div', { class: 'formula', html: f });
const set = (fn) => (...a) => { fn(...a); S.live = null; refresh(); };

// ---------- inspector panels ----------
function renderInspector() {
  const s = S.sel || { kind: 'model' }, P = PANELS[s.kind] || PANELS.model;
  const body = P(s);
  const crumbs = el('div', { class: 'crumbs' }, el('button', { onclick: () => select({ kind: 'model' }) }, 'Model'),
    s.b !== undefined && s.b !== 'f' && s.kind !== 'block' ? el('button', { onclick: () => select({ kind: 'block', b: s.b }) }, `Block ${s.b + 1}`) : null,
    s.kind === 'head' ? el('button', { onclick: () => select({ kind: 'attn', b: s.b }) }, 'Attention') : null);
  $('#insp').replaceChildren(crumbs, ...body.filter(Boolean), changesBox());
}

function activeChanges() {
  const o = S.opt, d = DEFAULT_OPT(), out = [];
  if (!o.causal) out.push(['causal mask off', () => o.causal = true]);
  if (!o.scale) out.push(['no ÷√dₖ', () => o.scale = true]);
  if (o.attnTemp !== 1) out.push([`attention sharpness ×${(1 / o.attnTemp).toFixed(2)}`, () => o.attnTemp = 1]);
  if (o.posScale !== 1) out.push(['position removed', () => o.posScale = 1]);
  for (const k in o.lnOff) if (o.lnOff[k]) out.push([`LayerNorm ${k === 'f' ? 'final' : `B${+k[0] + 1}.${k[2]}`} off`, () => delete o.lnOff[k]]);
  for (const k in o.residOff) if (o.residOff[k]) out.push([`residual B${+k[0] + 1} ${k.slice(2)} off`, () => delete o.residOff[k]]);
  for (const k in o.attnOff) if (o.attnOff[k]) out.push([`attention B${+k + 1} off`, () => delete o.attnOff[k]]);
  for (const k in o.ffnOff) if (o.ffnOff[k]) out.push([`FFN B${+k + 1} off`, () => delete o.ffnOff[k]]);
  for (const k in o.mute) if (o.mute[k]) out.push([`head B${+k.split('.')[0] + 1}.H${+k.split('.')[1] + 1} muted`, () => delete o.mute[k]]);
  for (const k in o.pattern) if (o.pattern[k]) out.push([`head B${+k.split('.')[0] + 1}.H${+k.split('.')[1] + 1} forced: ${o.pattern[k]}`, () => delete o.pattern[k]]);
  for (const k in o.drop) if (o.drop[k]) out.push([`FFN B${+k + 1}: ${Math.round(o.drop[k] * 100)}% neurons off`, () => delete o.drop[k]]);
  for (const k in o.killed) if (o.killed[k]?.length) out.push([`FFN B${+k + 1}: ${o.killed[k].length} neuron(s) killed`, () => delete o.killed[k]]);
  return out;
}
function changesBox() {
  const ch = activeChanges();
  if (!ch.length) return el('div', { class: 'changes none' }, 'No changes yet: the model runs as trained. Every switch you flip is listed here.');
  return el('div', { class: 'changes' }, el('div', { class: 'hd' }, `Your changes (${ch.length})`, el('button', { onclick: set(() => { const L = S.opt._L; S.opt = DEFAULT_OPT(); S.opt._L = L; }) }, 'reset all')),
    ...ch.map(([t, undo]) => el('button', { class: 'cg', onclick: set(undo) }, t, el('i', {}, '×'))));
}

function lensTable(t) {
  const L = S.cfg.L, rows = S.tr.lens.map((p, k) => ({ k, top: topk(p[t], 3, S.ix.vocab) }));
  const fin = rows[L].top[0];
  return el('div', { class: 'lens' }, el('div', { class: 'hd' }, `What “${words()[t]}” would predict if we stopped after each block (logit lens)`),
    ...rows.map(r => {
      const pFin = S.tr.lens[r.k][t][fin.id];
      return el('div', { class: 'lr' }, el('span', {}, r.k === 0 ? 'embedding' : `after block ${r.k}`), el('b', {}, r.top[0].w), el('em', {}, pct(r.top[0].p)),
        el('i', { style: `--v:${pFin}`, title: `P(${fin.w}) = ${pct(pFin)}` }));
    }), el('small', {}, `Bars: probability of the final answer “${fin.w}” at each depth.`));
}

const PANELS = {
  model() {
    const c = S.cfg, w = S.w, pc = paramCount(w), tr = w.cfg.trained, ix = S.ix;
    const parts = [['embedding', pc.emb + pc.pos, 'var(--beam)'], ['attention', pc.attn * c.L, 'var(--cyan)'], ['feed-forward', pc.mlp * c.L, 'var(--amber)'], ['norms', pc.ln * c.L + 2 * c.d, 'var(--hot)']];
    const grid = (rowsK, colsK, rowVals, colVals, fixed) => {
      const tbl = el('div', { class: 'grid', style: `grid-template-columns: 56px repeat(${colVals.length}, 1fr)` }, el('span'), ...colVals.map(v => el('span', { class: 'gh' }, colsK === 'pe' ? PE_NAME[v] : `${colsK === 'h' ? 'H' : 'L'}${v}`)));
      rowVals.forEach(r => { tbl.append(el('span', { class: 'gh' }, `${rowsK === 'L' ? 'L' : 'H'}${r}`)); colVals.forEach(cv => {
        const cfg = { ...c, ...fixed, [rowsK]: r, [colsK]: cv, ffn: 4 }, m = ix.models[key(cfg)], cur = key(cfg) === key(c) && c.ffn === 4 && S.mode === 'trained';
        tbl.append(m ? el('button', { class: 'gc' + (cur ? ' on' : ''), style: `--v:${m.acc.all}`, title: `capital ${pct(m.acc.capital)} · name ${pct(m.acc.name)} · agree ${pct(m.acc.agree)}`, onclick: () => { S.mode = 'trained'; setModel(cfg); } },
          el('b', {}, pct(m.acc.all)), el('small', {}, `${Math.round(m.acc.name * 100)}·${Math.round(m.acc.agree * 100)}`)) : el('span', { class: 'gc na' }, '—'));
      }); });
      return tbl;
    };
    return [
      el('h2', {}, 'The whole model'),
      guide(`A real decoder-only Transformer (the GPT / Qwen kind), small enough to run in your browser. ${tr ? 'It was trained on a tiny language of three sentence types: <b>capitals</b>, <b>names</b> (who said what) and <b>agreement</b> (cat runs / cats run).' : '<b>These weights are random</b>: no trained model exists for this shape, so you see a newborn network. Compare it with a trained one.'} Everything below is computed live. <b>Click any part of the 3D model</b> to open it.`),
      el('div', { class: 'pbar' }, ...parts.map(([n, v, col]) => el('i', { style: `flex:${v};background:${col}`, title: `${n}: ${v.toLocaleString()}` }))),
      el('div', { class: 'plegend' }, ...parts.map(([n, v, col]) => el('span', {}, el('i', { style: `background:${col}` }), `${n} ${Math.round(v / pc.total * 100)}%`)), el('b', {}, `${pc.total.toLocaleString()} weights`)),
      el('h3', {}, 'Architecture'),
      seg('Layers (blocks)', [1, 2, 3, 4, 5, 6].map(v => [v, v]), c.L, v => setModel({ L: v }), v => isTrained({ ...c, L: v })),
      seg('Attention heads', [1, 2, 4, 8].map(v => [v, v]), c.h, v => setModel({ h: v }), v => isTrained({ ...c, h: v })),
      seg('Width d_model', [16, 32, 64, 128].map(v => [v, v]), c.d, v => setModel({ d: v }), v => isTrained({ ...c, d: v })),
      seg('Feed-forward width', [1, 2, 4].map(v => [v, `${v}×d`]), c.ffn, v => setModel({ ffn: v }), v => isTrained({ ...c, ffn: v })),
      seg('Position signal', ['none', 'learned', 'sin', 'rope'].map(v => [v, PE_NAME[v]]), c.pe, v => setModel({ pe: v }), v => isTrained({ ...c, pe: v })),
      el('p', { class: 'note' }, el('span', { class: 'dotk' }), ' = a trained model exists. Any other shape runs with random weights.'),
      seg('Weights', [['trained', 'trained'], ['random', 'random']], S.mode, v => { S.mode = v; setModel(); }),
      S.mode === 'random' || !isTrained(c) ? el('button', { class: 'btn', onclick: () => { S.seed++; setModel(); } }, '🎲 new random seed') : null,
      el('h3', {}, 'Trained models · test accuracy'),
      el('p', { class: 'note' }, `Width d=${c.d}, position ${PE_NAME[c.pe]}. Small print: name · agree. Click a cell to load it.`),
      grid('L', 'h', [1, 2, 3, 4], [1, 2, 4], { d: c.d, pe: c.pe }),
      el('p', { class: 'note' }, `Same, by position signal (${c.h} head${c.h > 1 ? 's' : ''}, d=${c.d}).`),
      grid('L', 'pe', [1, 2, 3, 4], ['none', 'learned', 'sin', 'rope'], { h: c.h, d: c.d }),
      el('h3', {}, 'Everywhere at once'),
      toggle('Causal mask', S.opt.causal, set(v => S.opt.causal = v), 'a token may only look at earlier tokens'),
      toggle('Divide scores by √dₖ', S.opt.scale, set(v => S.opt.scale = v), 'EP01'),
      slider('Attention sharpness', -1.5, 1.5, .1, -Math.log2(S.opt.attnTemp), (v, fin) => { S.opt.attnTemp = 2 ** -v; if (fin) refresh(); else { S.tr = forward(S.w, S.ids, S.opt); scene.build({ tr: S.tr, w: S.w, state: S, vocab: S.ix.vocab }); } }, v => `×${(2 ** v).toFixed(2)}`),
      slider('Output temperature', .2, 3, .1, S.temp, (v, fin) => { S.temp = v; renderIO(); }, v => v.toFixed(1)),
      toggle('Show the numbers in the stream', S.showVec, set(v => S.showVec = v)),
      toggle('Glow', scene.glow, v => { scene.glow = v; renderInspector(); }),
    ];
  },
  token(s) {
    return [el('h2', {}, `Token “${words()[s.t]}”`),
      guide(`The text is cut into tokens and each becomes an id: its row number in a vocabulary of <b>${S.ix.vocab.length}</b> words (Qwen's has 151,936). The model never sees letters, only these ids. The highlighted token is the one whose view you follow (bright lane, arcs, prediction).`),
      el('div', { class: 'kv' }, el('span', {}, 'id'), el('b', {}, S.ids[s.t]), el('span', {}, 'position'), el('b', {}, s.t)),
      tryit('Swap a word in the bar below: <b>france → japan</b>. Does the prediction follow?', 'Press <b>◉</b> on another word to follow its view.')];
  },
  embed(s) {
    const d = S.cfg.d, V = S.ix.vocab.length, E = S.w.E, M = Array.from({ length: V }, (_, v) => [...E.slice(v * d, v * d + d)]);
    const hiRow = S.ids[S.focus];
    return [el('h2', {}, 'Embedding table E'),
      guide(`A table with one row per vocabulary word: <b>${V} × ${d}</b>. Each token id picks its row, and from here on that vector of ${d} numbers <i>is</i> the token. The strips on the lanes show these numbers (green +, pink −). The same table is used again at the very top, backwards.`),
      formula(`x<sub>t</sub> = E[id<sub>t</sub>]`),
      el('div', { class: 'sub' }, `Row ${hiRow} (“${S.ix.vocab[hiRow]}”) picked by the followed token:`), vecStrip(S.tr.emb[S.focus]),
      el('div', { class: 'sub' }, 'The whole table (click a row to see it as a token):'),
      heat(M, { rows: S.ix.vocab, cw: Math.max(2, Math.floor(300 / d)), ch: 9, signed: true, hi: hiRow, rl: 64, onCell: r => { S.ids[S.focus] = r; refresh(); } }),
      tryit(`Words used the same way end up with similar rows. Compare <b>paris</b> and <b>tokyo</b> with <b>cat</b>.`)];
  },
  pos() {
    const pe = S.cfg.pe, text = {
      none: 'This model gets <b>no position signal</b>. Attention by itself is order-blind: “the cat near the dogs” and “the dogs near the cat” look like the same bag of words. Surprisingly, the causal mask leaks some order back (token 5 can see 5 tokens, token 2 only 2), and a 2-layer model can learn to use that.',
      learned: 'A second table with one learned row per position (0…13), added to each token vector before block 1.',
      sin: 'Fixed sine and cosine waves of different frequencies, added to each token vector before block 1 (the original 2017 recipe).',
      rope: '<b>RoPE</b> (used by Qwen, Llama): nothing is added to the stream. Inside every attention head, q and k are <b>rotated</b> by an angle that grows with position, so their dot product depends on the <i>distance</i> between two tokens.',
    }[pe];
    const out = [el('h2', {}, `Position · ${PE_NAME[pe]}`), guide(text)];
    if (S.tr.pos) out.push(el('div', { class: 'sub' }, 'Position vectors for your tokens:'), heat(S.tr.pos.map(r => [...r]), { rows: words().map((w, t) => `${t} ${w}`), cw: Math.max(2, Math.floor(300 / S.cfg.d)), ch: 14, signed: true, hi: S.focus, rl: 80 }));
    if (pe === 'rope') out.push(formula('q′ = R(θ·t)·q,&nbsp; k′ = R(θ·s)·k &nbsp;⇒&nbsp; q′·k′ depends on t − s'));
    if (pe !== 'none') out.push(toggle('Remove the position signal now', S.opt.posScale === 0, set(v => S.opt.posScale = v ? 0 : 1), 'the trained model loses word order'));
    out.push(seg('Switch to a model trained with', ['none', 'learned', 'sin', 'rope'].map(v => [v, PE_NAME[v]]), pe, v => setModel({ pe: v }), v => isTrained({ ...S.cfg, pe: v })),
      tryit('Load <b>1 layer · position none</b> and pick the <b>agree</b> preset: the model has to guess which noun came first.', 'Then switch to RoPE and watch the agree score.'));
    return out;
  },
  block(s) {
    const b = s.b, o = S.opt, B = S.tr.blocks[b], f = S.focus;
    return [el('h2', {}, `Block ${b + 1} of ${S.cfg.L}`),
      guide('Every block does the same two things: <b class=c>attention</b> lets tokens read from each other, then the <b class=a>feed-forward</b> network processes each token alone. Both <b>add</b> their result to the stream instead of replacing it. Stacking blocks lets later ones build on what earlier ones wrote.'),
      formula('x ← x + Attn(LN(x))<br>x ← x + FFN(LN(x))'),
      el('div', { class: 'tiles' },
        el('button', { class: 'tile c', onclick: () => select({ kind: 'attn', b }) }, 'Attention', el('small', {}, `${S.cfg.h} heads`)),
        el('button', { class: 'tile a', onclick: () => select({ kind: 'ffn', b }) }, 'Feed-forward', el('small', {}, `${S.cfg.d}→${S.cfg.ffn * S.cfg.d}→${S.cfg.d}`))),
      el('div', { class: 'sub' }, `Size of “${words()[f]}”'s vector: in ${norm(B.in[f]).toFixed(1)} → after attention ${norm(B.mid[f]).toFixed(1)} → after FFN ${norm(B.out[f]).toFixed(1)}`),
      el('h3', {}, 'Switches'),
      toggle('Attention', !o.attnOff[b], set(v => o.attnOff[b] = !v)),
      toggle('Feed-forward', !o.ffnOff[b], set(v => o.ffnOff[b] = !v)),
      toggle('Residual around attention', !o.residOff[`${b}.attn`], set(v => o.residOff[`${b}.attn`] = !v)),
      toggle('Residual around feed-forward', !o.residOff[`${b}.ffn`], set(v => o.residOff[`${b}.ffn`] = !v)),
      toggle('LayerNorm before attention', !o.lnOff[`${b}.1`], set(v => o.lnOff[`${b}.1`] = !v)),
      toggle('LayerNorm before feed-forward', !o.lnOff[`${b}.2`], set(v => o.lnOff[`${b}.2`] = !v)),
      lensTable(f)];
  },
  attn(s) {
    const b = s.b, B = S.tr.blocks[b], ws = words(), h = S.cfg.h;
    return [el('h2', {}, `Attention · block ${b + 1}`),
      guide(`The only place where tokens exchange information. Each of the <b>${h}</b> head${h > 1 ? 's' : ''} works on its own ${S.cfg.d / h}-number slice: every token sends a <b>query</b> (“what am I looking for?”), every earlier token offers a <b>key</b> (“what do I contain?”), and matching pairs pass along their <b>value</b>. In 3D: bottom edge of a glass panel = keys, top edge = queries, arc brightness = weight.`),
      formula('Attention(Q,K,V) = softmax( QKᵀ / √dₖ ) · V'),
      el('div', { class: 'heads' }, ...B.heads.map((H, j) => el('div', { class: 'hcard' + (H.muted ? ' muted' : '') },
        el('button', { class: 'hh', onclick: () => select({ kind: 'head', b, j }) }, `Head ${j + 1}`, el('small', {}, H.muted ? 'muted' : S.opt.pattern[`${b}.${j}`] || `writes ${H.write.toFixed(2)}`)),
        heat(H.att, { rows: ws, cw: Math.max(8, Math.floor(150 / ws.length)), ch: Math.max(8, Math.floor(150 / ws.length)), rl: 0, hi: S.focus, onCell: r => { S.focus = r; refresh(); } }),
        toggle('on', !H.muted, set(v => S.opt.mute[`${b}.${j}`] = !v))))),
      el('p', { class: 'note' }, 'Rows = the token looking (query), columns = the token looked at (key). Click a row to follow that token.'),
      toggle('All of attention in this block', !S.opt.attnOff[b], set(v => S.opt.attnOff[b] = !v)),
      tryit('Find the head that makes <b>?</b> look at the right name, mute it, and watch the <b>name</b> score.', 'Open one head to force a pattern on it.')];
  },
  head(s) {
    const { b, j } = s, H = S.tr.blocks[b].heads[j], ws = words(), f = S.focus, k = `${b}.${j}`, dh = S.cfg.d / S.cfg.h;
    const row = H.att[f], top = topk(row, Math.min(4, ws.length), ws);
    return [el('h2', {}, `Head ${j + 1} · block ${b + 1}`),
      guide(`This head compares the query of “<b>${ws[f]}</b>” with the key of every token it may see, turns the scores into weights that sum to 1, and takes that mix of their values (${dh} numbers each). ${H.muted ? '<b>It is muted now: it writes nothing.</b>' : ''}`),
      formula(`a<sub>t,s</sub> = softmax<sub>s</sub>( q<sub>t</sub>·k<sub>s</sub> ${S.opt.scale ? `/ √${dh}` : ''} )`),
      heat(H.att, { rows: ws, cols: ws, cw: Math.max(18, Math.min(34, Math.floor(300 / ws.length))), ch: 22, hi: f, vals: true, onCell: r => { S.focus = r; refresh(); } }),
      el('div', { class: 'sub' }, `“${ws[f]}” looks at:`), bars(top, 'var(--cyan)'),
      el('div', { class: 'sub' }, `Raw scores q·k${S.opt.scale ? '/√dₖ' : ''} for “${ws[f]}” (before softmax):`),
      el('div', { class: 'kv' }, ...H.scores[f].flatMap((v, u) => (S.opt.causal && u > f) ? [] : [el('span', {}, ws[u]), el('b', {}, v.toFixed(2))])),
      el('h3', {}, 'Change this head'),
      toggle('Head on', !H.muted, set(v => S.opt.mute[k] = !v)),
      seg('Force a pattern', [['', 'learned'], ['uniform', 'uniform'], ['self', 'self'], ['prev', 'previous'], ['first', 'first']], S.opt.pattern[k] || '', set(v => { if (v) S.opt.pattern[k] = v; else delete S.opt.pattern[k]; })),
      el('p', { class: 'note' }, '“previous” makes every token copy the one before it: the first half of an induction circuit.'),
      tryit('Force <b>uniform</b>: every token averages everything it sees. Which test score falls?')];
  },
  ln(s) {
    const fin = s.b === 'f', k = fin ? 'f' : `${s.b}.${s.n}`, f = S.focus;
    const src = fin ? S.tr.blocks.at(-1)?.out ?? S.tr.x0 : s.n === 1 ? S.tr.blocks[s.b].in : S.tr.blocks[s.b].mid;
    const dst = fin ? S.tr.lnf : s.n === 1 ? S.tr.blocks[s.b].ln1 : S.tr.blocks[s.b].ln2;
    const st = v => { const m = v.reduce((a, b) => a + b, 0) / v.length; return [m, Math.sqrt(v.reduce((a, b) => a + (b - m) ** 2, 0) / v.length)]; };
    const [m0, s0] = st(src[f]), [m1, s1] = st(dst[f]);
    return [el('h2', {}, fin ? 'Final LayerNorm' : `LayerNorm · block ${s.b + 1} · before ${s.n === 1 ? 'attention' : 'feed-forward'}`),
      guide('Rescales each token\'s vector to mean 0 and spread 1, then applies a learned scale and shift. In this <b>pre-norm</b> design only the copy a worker reads is normalised; the stream itself keeps growing.'),
      formula('LN(x) = γ ⊙ (x − mean) / std + β'),
      el('div', { class: 'kv' }, el('span', {}, 'mean in → out'), el('b', {}, `${m0.toFixed(2)} → ${m1.toFixed(2)}`), el('span', {}, 'spread in → out'), el('b', {}, `${s0.toFixed(2)} → ${s1.toFixed(2)}`)),
      el('div', { class: 'sub' }, 'In:'), vecStrip(src[f]), el('div', { class: 'sub' }, 'Out:'), vecStrip(dst[f]),
      toggle('LayerNorm on', !S.opt.lnOff[k], set(v => S.opt.lnOff[k] = !v), 'off = pass the vector through untouched')];
  },
  add(s) {
    const { b, which } = s, k = `${b}.${which}`, B = S.tr.blocks[b], f = S.focus;
    const before = which === 'attn' ? B.in : B.mid, delta = which === 'attn' ? B.attnOut : B.ffnOut, after = which === 'attn' ? B.mid : B.out;
    return [el('h2', {}, `Residual add · after ${which === 'attn' ? 'attention' : 'feed-forward'} · block ${b + 1}`),
      guide('The worker\'s output is <b>added</b> to the stream, not swapped in. The stream is a shared notepad: each layer writes a small edit and everything written earlier stays readable. This is what lets deep models train at all.'),
      formula('x<sub>new</sub> = x + f(x)'),
      el('div', { class: 'sub' }, `stream (size ${norm(before[f]).toFixed(1)})`), vecStrip(before[f]),
      el('div', { class: 'sub' }, `+ edit written by ${which === 'attn' ? 'attention' : 'the feed-forward'} (size ${norm(delta[f]).toFixed(1)})`), vecStrip(delta[f]),
      el('div', { class: 'sub' }, `= new stream (size ${norm(after[f]).toFixed(1)})`), vecStrip(after[f]),
      toggle('Keep the residual connection', !S.opt.residOff[k], set(v => S.opt.residOff[k] = !v), 'off = x_new = f(x): everything before is forgotten'),
      tryit('Switch it off in block 1 and look at the prediction and the test scores.')];
  },
  ffn(s) {
    const b = s.b, B = S.tr.blocks[b], f = S.focus, F = S.cfg.ffn * S.cfg.d, ws = words(), killed = S.opt.killed[b] || [];
    const hid = B.hid.map(r => [...r]), act = hid[f].map((v, i) => [i, v]).sort((a, c) => c[1] - a[1]).slice(0, 6);
    return [el('h2', {}, `Feed-forward · block ${b + 1}`),
      guide(`Each token on its own, no mixing: widen ${S.cfg.d} → <b>${F}</b> numbers, keep the positive part (GELU), squeeze back to ${S.cfg.d}. Each of the ${F} hidden <b>neurons</b> fires for some pattern and writes its own direction into the stream. In real models this is where most weights live (Qwen: 7× attention per block).`),
      formula('FFN(x) = W₂ · GELU( W₁·x + b₁ ) + b₂'),
      el('div', { class: 'sub' }, `Hidden neurons for every token (${F} columns). Click a column to kill that neuron:`),
      heat(hid, { rows: ws, cw: Math.max(1, Math.floor(300 / F)), ch: 14, color: '255,181,71', signed: true, hi: f, rl: 64, onCell: (r, c) => { const L = (S.opt.killed[b] ||= []); const i = L.indexOf(c); i >= 0 ? L.splice(i, 1) : L.push(c); S.live = null; refresh(); } }),
      el('div', { class: 'sub' }, `Strongest neurons for “${ws[f]}”:`),
      el('div', { class: 'kv' }, ...act.flatMap(([i, v]) => [el('span', {}, `#${i}${killed.includes(i) ? ' ✕' : ''}`), el('b', {}, v.toFixed(2))])),
      el('h3', {}, 'Change it'),
      toggle('Feed-forward on', !S.opt.ffnOff[b], set(v => S.opt.ffnOff[b] = !v)),
      slider('Switch off a random share of neurons', 0, 1, .05, S.opt.drop[b] || 0, (v, fin) => { S.opt.drop[b] = v; if (fin) refresh(); }, v => Math.round(v * 100) + '%'),
      tryit('Turn the feed-forward of every block off. Which sentence type survives? (Facts like capitals tend to live here.)')];
  },
  unembed() {
    const f = S.focus, z = [...S.tr.logits[f]], top = z.map((v, i) => [i, v]).sort((a, c) => c[1] - a[1]).slice(0, 8);
    return [el('h2', {}, 'Unembedding (tied)'),
      guide(`The last vector of each token is compared, by dot product, with <b>every row of the same table E</b> used at the bottom. ${S.ix.vocab.length} rows → ${S.ix.vocab.length} scores (logits). Sharing the table is called tying; Qwen2.5-0.5B does the same.`),
      formula('logits = LN(x) · Eᵀ'),
      el('div', { class: 'sub' }, `Highest scores for “${words()[f]}”:`),
      el('div', { class: 'kv' }, ...top.flatMap(([i, v]) => [el('span', {}, S.ix.vocab[i]), el('b', {}, v.toFixed(2))])),
      lensTable(f)];
  },
  output(s) {
    const f = S.focus, p = softmax([...S.tr.logits[f]], S.temp);
    return [el('h2', {}, `Prediction after “${words()[f]}”`),
      guide('Softmax turns the scores into probabilities. During training every position learns to predict the token after it, so each lane shows its own guess. Only the last one is used to continue the text.'),
      formula('p = softmax( logits / temperature )'),
      bars(topk(p, 8, S.ix.vocab), 'var(--hot)'),
      slider('Temperature', .2, 3, .1, S.temp, v => { S.temp = v; renderInspector(); renderIO(); }, v => v.toFixed(1)),
      lensTable(f),
      tryit('Press <b>complete ⏵</b> in the bar below to let the model write the rest.')];
  },
  lane(s) {
    const t = s.t, k = s.k ?? 0, st = [S.tr.x0, ...S.tr.blocks.flatMap(b => [b.mid, b.out])][Math.min(k, 2 * S.cfg.L)];
    const name = k === 0 ? 'after embedding' : `after block ${((k - 1) >> 1) + 1} ${(k - 1) % 2 ? 'feed-forward' : 'attention'}`;
    return [el('h2', {}, `Residual stream · “${words()[t]}”`),
      guide('The lane is the token\'s running vector. It enters as its embedding, and every attention and feed-forward layer adds an edit. Whatever the model knows about this token at a given depth is in these numbers.'),
      el('div', { class: 'sub' }, `${name} · size ${norm(st[t]).toFixed(2)}`), vecStrip(st[t]), lensTable(t)];
  },
};
PANELS.vec = PANELS.lane;

// ---------- missions ----------
const MISSIONS = [
  { id: 'untrained', title: 'Meet a newborn', text: 'Switch the weights to <b>random</b>. Look at the attention arcs and the predictions.', check: () => !S.w.cfg.trained },
  { id: 'name', title: 'Break the name memory', text: 'Pick a <b>name</b> preset. Find the one head whose job is to fetch the name and mute it. Goal: <b>name</b> score below 60%, while <b>capital</b> stays above 90%.', check: () => S.w.cfg.trained && S.live && S.live.name < .6 && S.live.capital > .9 },
  { id: 'order', title: 'Who came first?', text: 'Load a <b>1-layer</b> model with position <b>none</b> (cell in the second table). Goal: see <b>agree</b> fall below 90%. Then find the cheapest fix.', check: () => S.w.cfg.trained && S.cfg.L === 1 && S.cfg.pe === 'none' && S.live && S.live.agree < .9 },
  { id: 'highway', title: 'Cut the highway', text: 'Switch off one <b>residual connection</b> (click a ⊕). Goal: average score below 50%.', check: () => S.live && Object.values(S.opt.residOff).some(Boolean) && (S.live.capital + S.live.name + S.live.agree) / 3 < .5 },
  { id: 'facts', title: 'Where do facts live?', text: 'Switch off the <b>feed-forward</b> in every block, keep attention. Goal: <b>capital</b> below 50%.', check: () => S.live && S.cfg.L && [...Array(S.cfg.L).keys()].every(b => S.opt.ffnOff[b]) && S.live.capital < .5 },
];
function checkMissions() {
  let fresh = null;
  for (const m of MISSIONS) if (!S.done[m.id] && m.check()) { S.done[m.id] = true; fresh = m; }
  if (fresh) toast(`Mission done: ${fresh.title}`);
  renderMissions();
}
function renderMissions() {
  const n = MISSIONS.filter(m => S.done[m.id]).length;
  $('#btnMissions').textContent = `Missions ${n}/${MISSIONS.length}`;
  $('#missions .list').replaceChildren(...MISSIONS.map(m => el('div', { class: 'ms' + (S.done[m.id] ? ' ok' : '') }, el('b', {}, (S.done[m.id] ? '✓ ' : '○ ') + m.title), el('p', { html: m.text }))));
}
function toast(t) { const d = el('div', { class: 'toast' }, t); document.body.append(d); setTimeout(() => d.classList.add('out'), 2600); setTimeout(() => d.remove(), 3200); }

// ---------- guided tour ----------
const TOUR = [
  { sel: null, t: 'A whole Transformer', x: () => `This is a complete language model of the same design as GPT or Qwen, just tiny: ${paramCount(S.w).total.toLocaleString()} weights instead of billions. It reads <b>${words().join(' ')}</b> and guesses the next word. Drag to turn it, scroll or pinch to zoom.` },
  { sel: () => ({ kind: 'token', t: S.focus }), t: '1 · Tokens', x: () => `Each word becomes an id. The bright lane is the token we follow: “${words()[S.focus]}”.` },
  { sel: () => ({ kind: 'embed' }), t: '2 · Embedding', x: () => `Each id picks one row of a table: now every token is a vector of ${S.cfg.d} numbers. Those strips on the lanes are the numbers.` },
  { sel: () => ({ kind: 'pos' }), t: '3 · Position', x: () => S.cfg.pe === 'rope' ? 'Word order is added later, inside attention, by rotating queries and keys (RoPE).' : S.cfg.pe === 'none' ? 'This model gets no word order at all. Remember that for later.' : 'A position vector is added, so the model knows which word came first.' },
  { sel: () => ({ kind: 'block', b: 0 }), t: '4 · A block', x: () => `The green lanes are the residual stream: one running vector per token. Behind them, workers read a copy, compute, and add their result back at ⊕. This model has ${S.cfg.L} identical blocks.` },
  { sel: () => ({ kind: 'attn', b: S.cfg.L - 1 }), t: '5 · Attention', x: () => { const H = S.tr.blocks.at(-1).heads, ws = words(); return `The only place tokens talk. Each glass panel is a head. Arcs go from the token looked at (bottom) to the token looking (top). ${H.map((h, j) => { const r = h.att[S.focus]; let b = 0; r.forEach((v, i) => { if (v > r[b]) b = i; }); return `Head ${j + 1}: “${ws[S.focus]}” → “${ws[b]}” ${r[b].toFixed(2)}`; }).join('. ')}.`; } },
  { sel: () => ({ kind: 'add', b: 0, which: 'attn' }), t: '6 · Add, don\'t replace', x: () => 'The heads\' result is added into the lane. Nothing written before is erased.' },
  { sel: () => ({ kind: 'ffn', b: 0 }), t: '7 · Feed-forward', x: () => `Each token alone, no mixing: ${S.cfg.d} → ${S.cfg.ffn * S.cfg.d} → ${S.cfg.d}. Brightness = how strongly its neurons fire.` },
  { sel: () => ({ kind: 'unembed' }), t: '8 · Back to words', x: () => 'The last vector is scored against every row of the same embedding table.' },
  { sel: () => ({ kind: 'output', t: S.focus }), t: '9 · The guess', x: () => { const p = S.tr.lens[S.cfg.L][S.focus], b = topk(p, 1, S.ix.vocab)[0]; return `Softmax turns scores into probabilities: “${b.w}” ${pct(b.p)}. Now break things: open <b>Missions</b>, or click any part and flip its switches.`; } },
];
function tourStep(i) {
  S.tour = i; const card = $('#tourcard');
  if (i < 0 || i >= TOUR.length) { card.hidden = true; S.tour = -1; return; }
  const st = TOUR[i]; S.sel = st.sel ? st.sel() : { kind: 'model' }; refresh(); scene.flyTo(st.sel ? S.sel : null);
  card.hidden = false;
  card.replaceChildren(el('div', { class: 'tt' }, st.t, el('span', {}, `${i + 1}/${TOUR.length}`)), el('p', { html: st.x() }),
    el('div', { class: 'tb' }, el('button', { onclick: () => tourStep(-1) }, 'close'), el('span'),
      i > 0 ? el('button', { onclick: () => tourStep(i - 1) }, '← back') : null,
      el('button', { class: 'pri', onclick: () => tourStep(i + 1 < TOUR.length ? i + 1 : -1) }, i + 1 < TOUR.length ? 'next →' : 'done')));
}

// ---------- play-through ----------
function narrate(st) {
  const n = $('#narr'); if (!st) { n.classList.remove('on'); $('#btnPlay').textContent = '▶ Run forward pass'; return; }
  const ws = words(), f = S.focus, tr = S.tr, d = S.cfg.d; let t = '';
  const best = r => { let b = 0; r.forEach((v, i) => { if (v > r[b]) b = i; }); return b; };
  if (st.kind === 'token') t = `${ws.length} words → ${ws.length} token ids: ${S.ids.join(', ')}`;
  if (st.kind === 'embed') t = `Each id picks its row of E → ${ws.length} vectors of ${d} numbers`;
  if (st.kind === 'pos') t = `+ position (${PE_NAME[S.cfg.pe]}): now “${ws[0]}” at 0 differs from “${ws[0]}” at 5`;
  if (st.kind === 'ln') t = st.b === 'f' ? 'Final LayerNorm' : 'LayerNorm: a rescaled copy for the worker to read';
  if (st.kind === 'attn') t = `Block ${st.b + 1} attention · ` + tr.blocks[st.b].heads.map((H, j) => H.muted ? `H${j + 1} muted` : `H${j + 1}: “${ws[f]}” → “${ws[best(H.att[f])]}” ${H.att[f][best(H.att[f])].toFixed(2)}`).join(' · ');
  if (st.kind === 'add') { const k = st.b + 1, p = st.which === 'ffn' ? tr.lens[k][f] : null; t = `⊕ added into the stream${p ? ` · if we stopped here: “${S.ix.vocab[best(p)]}” ${pct(p[best(p)])}` : ''}`; }
  if (st.kind === 'ffn') t = `Block ${st.b + 1} feed-forward, each token alone: ${d} → ${S.cfg.ffn * d} → ${d}`;
  if (st.kind === 'unembed') t = `Score “${ws[f]}”'s last vector against all ${S.ix.vocab.length} rows of E`;
  if (st.kind === 'output') { const p = tr.lens[S.cfg.L][f], b = best(p); t = `→ next word: “${S.ix.vocab[b]}” ${pct(p[b])}`; }
  n.textContent = t; n.classList.add('on');
}

// ---------- boot ----------
async function boot() {
  S.ix = await (await fetch('models/index.json')).json();
  S.set = testSet(S.ix);
  const q = new URLSearchParams(location.search), st = Object.fromEntries(S.ix.vocab.map((v, i) => [v, i]));
  if (!S.ix.models[key(S.cfg)]) { const k = Object.keys(S.ix.models)[0]; const m = k.match(/L(\d)H(\d)D(\d+)-(\w+)/); Object.assign(S.cfg, { L: +m[1], h: +m[2], d: +m[3], pe: m[4] }); }
  S.ids = (q.get('text') || 'the capital of france is').split(' ').filter(w => w in st).map(w => st[w]); S.focus = S.ids.length - 1;
  S.sel = { kind: 'model' };
  await setModel();
  scene.flyTo(null, false);
  $('#btnTour').onclick = () => tourStep(0);
  $('#btnPlay').onclick = () => { if (scene.front) { scene.stop(); narrate(null); return; } $('#btnPlay').textContent = '■ Stop'; scene.flyTo(null); scene.play(narrate); };
  $('#btnView').onclick = () => scene.flyTo(null);
  $('#btnMissions').onclick = () => { $('#missions').hidden = !$('#missions').hidden; };
  $('#missions .x').onclick = () => { $('#missions').hidden = true; };
  renderMissions();
  if (!q.has('notour')) setTimeout(() => tourStep(0), 600);
}
boot();
