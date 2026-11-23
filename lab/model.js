// Tiny GPT-style Transformer in plain JS. Mirrors train.py `Model.forward` exactly (pre-LayerNorm, tied embeddings,
// tanh-GELU, rotate-half RoPE) and adds the Playground's runtime interventions. No dependencies: runs in the browser
// and in node (check.mjs compares it with PyTorch logits).

// ---------- weights ----------
const H2F = (() => {                                     // float16 -> float32 lookup table
  const t = new Float32Array(65536);
  for (let h = 0; h < 65536; h++) {
    const s = h & 0x8000 ? -1 : 1, e = (h >> 10) & 31, f = h & 1023;
    t[h] = e === 0 ? s * 2 ** -14 * (f / 1024) : e === 31 ? (f ? NaN : s * Infinity) : s * 2 ** (e - 15) * (1 + f / 1024);
  }
  return t;
})();
const b64bytes = b => typeof atob === 'function'
  ? Uint8Array.from(atob(b), c => c.charCodeAt(0)) : new Uint8Array(Buffer.from(b, 'base64'));
function f16([shape, b64]) {
  const by = b64bytes(b64), u = new Uint16Array(by.buffer, by.byteOffset, by.byteLength / 2), out = new Float32Array(u.length);
  for (let i = 0; i < u.length; i++) out[i] = H2F[u[i]];
  return out;
}

/** Weights from a models/*.json file (trained). */
export function loadWeights(j) {
  const { L, h, d, pe } = j.cfg, W = j.W;
  const blocks = [];
  for (let i = 0; i < L; i++) {
    const g = n => f16(W[`${i}.${n}`]);
    blocks.push({ ln1w: g('ln1.w'), ln1b: g('ln1.b'), qkvW: g('qkv.w'), qkvB: g('qkv.b'), oW: g('o.w'), oB: g('o.b'),
      ln2w: g('ln2.w'), ln2b: g('ln2.b'), f1W: g('f1.w'), f1B: g('f1.b'), f2W: g('f2.w'), f2B: g('f2.b') });
  }
  const E = f16(W.E);
  return { cfg: { L, h, d, pe, ffn: 4, V: E.length / d, trained: true }, E, P: W.P ? f16(W.P) : null,
    lnfw: f16(W['lnf.w']), lnfb: f16(W['lnf.b']), blocks };
}

function rng(seed) {                                       // mulberry32 + Box-Muller
  let a = seed >>> 0;
  const u = () => { a = (a + 0x6D2B79F5) >>> 0; let t = a; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
  return { u, n: () => Math.sqrt(-2 * Math.log(u() + 1e-12)) * Math.cos(2 * Math.PI * u()) };
}

/** Untrained weights for any shape, initialised like train.py (normal 0.02, zero bias, LayerNorm = identity). */
export function randomWeights({ L, h, d, pe, ffn = 4, V, maxlen = 14, seed = 1 }) {
  const r = rng(seed), N = n => Float32Array.from({ length: n }, () => r.n() * .02);
  const Z = n => new Float32Array(n), O = n => new Float32Array(n).fill(1), F = ffn * d;
  const blocks = Array.from({ length: L }, () => ({ ln1w: O(d), ln1b: Z(d), qkvW: N(3 * d * d), qkvB: Z(3 * d), oW: N(d * d), oB: Z(d),
    ln2w: O(d), ln2b: Z(d), f1W: N(F * d), f1B: Z(F), f2W: N(d * F), f2B: Z(d) }));
  return { cfg: { L, h, d, pe, ffn, V, trained: false, seed }, E: N(V * d), P: pe === 'learned' ? N(maxlen * d) : null,
    lnfw: O(d), lnfb: Z(d), blocks };
}

export const paramCount = w => {
  const { L, d, ffn, V } = w.cfg, F = ffn * d;
  const emb = V * d, pos = w.P ? w.P.length : 0;
  const attn = 3 * d * d + 3 * d + d * d + d, mlp = F * d + F + d * F + d, ln = 4 * d;
  return { emb, pos, attn, mlp, ln, block: attn + mlp + ln, total: emb + pos + L * (attn + mlp + ln) + 2 * d };
};

// ---------- maths ----------
function linear(X, W, b, dout) {                           // X: rows of length din; W: (dout x din) row-major like nn.Linear
  const din = X[0].length;
  return X.map(x => {
    const y = new Float32Array(dout);
    for (let o = 0; o < dout; o++) { let s = b ? b[o] : 0; const off = o * din; for (let i = 0; i < din; i++) s += x[i] * W[off + i]; y[o] = s; }
    return y;
  });
}
function layerNorm(X, w, b) {
  return X.map(x => {
    const n = x.length; let m = 0; for (let i = 0; i < n; i++) m += x[i]; m /= n;
    let v = 0; for (let i = 0; i < n; i++) v += (x[i] - m) ** 2; v /= n;
    const s = 1 / Math.sqrt(v + 1e-5), y = new Float32Array(n);
    for (let i = 0; i < n; i++) y[i] = (x[i] - m) * s * w[i] + b[i];
    return y;
  });
}
const gelu = x => .5 * x * (1 + Math.tanh(0.7978845608028654 * (x + .044715 * x * x * x)));
const add = (A, B) => A.map((a, t) => a.map((v, i) => v + B[t][i]));
export const norm = x => Math.sqrt(x.reduce((s, v) => s + v * v, 0));
export function softmax(z, temp = 1) {
  let m = -Infinity; for (const v of z) if (v > m) m = v;
  const e = z.map(v => Math.exp((v - m) / temp)); const s = e.reduce((a, b) => a + b, 0);
  return e.map(v => v / s);
}
function sinusoid(T, d) {
  return Array.from({ length: T }, (_, p) => {
    const r = new Float32Array(d);
    for (let i = 0; i < d; i += 2) { const a = p / 10000 ** (i / d); r[i] = Math.sin(a); if (i + 1 < d) r[i + 1] = Math.cos(a); }
    return r;
  });
}
function rope(x, p) {                                      // one head vector at position p, rotate-half
  const dh = x.length, half = dh >> 1, y = new Float32Array(dh);
  for (let i = 0; i < half; i++) {
    const a = p / 10000 ** (i / half), c = Math.cos(a), s = Math.sin(a);
    y[i] = x[i] * c - x[i + half] * s; y[i + half] = x[i + half] * c + x[i] * s;
  }
  return y;
}
function unembed(X, w) { const { E } = w, d = w.cfg.d, V = w.cfg.V; return X.map(x => { const z = new Float32Array(V); for (let v = 0; v < V; v++) { let s = 0; for (let i = 0; i < d; i++) s += x[i] * E[v * d + i]; z[v] = s; } return z; }); }

function dropMask(n, frac, seed) {                         // deterministic set of neurons switched off
  if (!frac) return null;
  const r = rng(seed * 7919 + n), idx = [...Array(n).keys()];
  for (let i = n - 1; i > 0; i--) { const j = Math.floor(r.u() * (i + 1)); [idx[i], idx[j]] = [idx[j], idx[i]]; }
  const m = new Uint8Array(n); for (const i of idx.slice(0, Math.round(frac * n))) m[i] = 1; return m;
}

export const DEFAULT_OPT = () => ({
  causal: true, scale: true, attnTemp: 1, posScale: 1, ropeOn: true,
  lnOff: {}, residOff: {}, attnOff: {}, ffnOff: {}, mute: {}, pattern: {}, drop: {}, killed: {},
});

/**
 * Full forward pass with a trace of every intermediate the Playground draws.
 * opt keys: causal, scale (divide by sqrt(dh)), attnTemp, posScale (0 removes position), ropeOn,
 *   lnOff['b.1'|'b.2'|'f'], residOff['b.attn'|'b.ffn'], attnOff[b], ffnOff[b], mute['b.j'],
 *   pattern['b.j'] = uniform|self|prev|first, drop[b] = fraction of FFN neurons off, killed['b'] = [neuron ids].
 */
export function forward(w, ids, opt = DEFAULT_OPT(), lens = true) {
  const { L, h, d, pe } = w.cfg, T = ids.length, dh = d / h, F = w.cfg.ffn * d;
  const emb = ids.map(id => w.E.slice(id * d, id * d + d));
  let pos = null;
  if (pe === 'learned') pos = ids.map((_, p) => w.P.slice(p * d, p * d + d).map(v => v * opt.posScale));
  if (pe === 'sin') pos = sinusoid(T, d).map(r => r.map(v => v * opt.posScale));
  let x = pos ? add(emb, pos) : emb.map(r => r.slice());
  const tr = { T, ids, emb, pos, x0: x, blocks: [] };
  const useRope = pe === 'rope' && opt.ropeOn && opt.posScale > 0;

  for (let b = 0; b < L; b++) {
    const B = w.blocks[b], s = { in: x };
    const ln1 = opt.lnOff[`${b}.1`] ? x : layerNorm(x, B.ln1w, B.ln1b);
    const qkv = linear(ln1, B.qkvW, B.qkvB, 3 * d);
    const heads = [];
    const cat = Array.from({ length: T }, () => new Float32Array(d));
    for (let j = 0; j < h; j++) {
      const sl = (r, o) => r.slice(o + j * dh, o + (j + 1) * dh);
      let q = qkv.map(r => sl(r, 0)), k = qkv.map(r => sl(r, d)); const v = qkv.map(r => sl(r, 2 * d));
      if (useRope) { q = q.map((r, p) => rope(r, p)); k = k.map((r, p) => rope(r, p)); }
      const sc = opt.scale ? 1 / Math.sqrt(dh) : 1;
      const scores = q.map((qi, t) => k.map((kk, u) => { let z = 0; for (let i = 0; i < dh; i++) z += qi[i] * kk[i]; return z * sc; }));
      const key = `${b}.${j}`, pat = opt.pattern[key];
      const att = scores.map((row, t) => {
        const ok = u => !opt.causal || u <= t;
        if (pat) {
          const r = row.map(() => 0), lim = opt.causal ? t : T - 1;
          if (pat === 'uniform') for (let u = 0; u <= lim; u++) r[u] = 1 / (lim + 1);
          if (pat === 'self') r[t] = 1;
          if (pat === 'prev') r[Math.max(0, t - 1)] = 1;
          if (pat === 'first') r[0] = 1;
          return r;
        }
        const z = row.map((v, u) => ok(u) ? v / opt.attnTemp : -Infinity);
        return softmax(z);
      });
      const muted = opt.mute[key] || opt.attnOff[b];
      const out = att.map(a => { const o = new Float32Array(dh); a.forEach((p, u) => { if (p) for (let i = 0; i < dh; i++) o[i] += p * v[u][i]; }); return o; });
      if (!muted) out.forEach((o, t) => cat[t].set(o, j * dh));
      heads.push({ q, k, v, scores, att, out, muted: !!muted });
    }
    const attnOut = linear(cat, B.oW, B.oB, d);
    if (opt.attnOff[b]) attnOut.forEach(r => r.fill(0));
    // how much each head writes into the stream: norm of its slice pushed through the output projection
    heads.forEach((hd, j) => { hd.write = hd.muted ? 0 : hd.out.reduce((s, o, t) => { let n = 0; for (let r = 0; r < d; r++) { let z = 0; for (let i = 0; i < dh; i++) z += o[i] * B.oW[r * d + j * dh + i]; n += z * z; } return s + Math.sqrt(n); }, 0) / T; });
    const mid = opt.residOff[`${b}.attn`] ? attnOut : add(x, attnOut);
    const ln2 = opt.lnOff[`${b}.2`] ? mid : layerNorm(mid, B.ln2w, B.ln2b);
    const pre = linear(ln2, B.f1W, B.f1B, F);
    const mask = dropMask(F, opt.drop[b] || 0, b + 1), killed = opt.killed[b] || [];
    const hid = pre.map(r => r.map((v, i) => (mask && mask[i]) || killed.includes(i) ? 0 : gelu(v)));
    const ffnOut = linear(hid, B.f2W, B.f2B, d);
    if (opt.ffnOff[b]) ffnOut.forEach(r => r.fill(0));
    const out = opt.residOff[`${b}.ffn`] ? ffnOut : add(mid, ffnOut);
    Object.assign(s, { ln1, heads, attnOut, mid, ln2, hid, ffnOut, out });
    tr.blocks.push(s); x = out;
  }
  tr.lnf = opt.lnOff.f ? x : layerNorm(x, w.lnfw, w.lnfb);
  tr.logits = unembed(tr.lnf, w);
  if (lens) {                                              // logit lens: read the stream after every block with the final norm + table
    const stages = [tr.x0, ...tr.blocks.map(s => s.out)];
    tr.lens = stages.map((X, k) => (k === L ? tr.logits : unembed(opt.lnOff.f ? X : layerNorm(X, w.lnfw, w.lnfb), w)).map(z => softmax(z)));
  }
  return tr;
}

export const topk = (p, k, vocab) => [...p].map((v, i) => [i, v]).sort((a, b) => b[1] - a[1]).slice(0, k).map(([i, v]) => ({ id: i, w: vocab[i], p: v }));

// ---------- the test set (same templates as train.py) ----------
export function testSet(ix, n = { capital: 24, name: 48, agree: 64 }) {
  const r = rng(42), pick = (a, k) => { const c = a.slice(); for (let i = c.length - 1; i > 0; i--) { const j = Math.floor(r.u() * (i + 1)); [c[i], c[j]] = [c[j], c[i]]; } return c.slice(0, k); };
  const caps = [], names = [], agrees = [];
  const C = ix.cats.country.map((c, i) => [c, ix.cats.city[i]]);
  for (const [c, k] of C) { caps.push({ text: `the capital of ${c} is`, good: [k], task: 'capital' }); caps.push({ text: `${k} is the capital of`, good: [c], task: 'capital' }); }
  const N = ix.cats.name;
  for (const a of N) for (const b of N) if (a !== b) for (const af of [1, 0]) for (const ai of [1, 0]) {
    const p = [`i am ${a} .`, `you are ${b} .`]; const s = (af ? p : p.reverse()).join(' ');
    names.push({ text: `${s} ${ai ? 'who am i ?' : 'who are you ?'}`, good: [ai ? a : b], task: 'name' });
  }
  const forms = ix.cats.noun, verbs = ix.cats.verb;   // verb list: runs run sleeps sleep
  for (const n1 of forms) for (const n2 of forms) for (const n3 of forms) {
    const pl = n1.endsWith('s'); agrees.push({ text: `the ${n1} near the ${n2} near the ${n3}`, good: pl ? [verbs[1], verbs[3]] : [verbs[0], verbs[2]], task: 'agree' });
  }
  return [...pick(caps, n.capital), ...pick(names, n.name), ...pick(agrees, n.agree)];
}

export function score(w, ix, set, opt) {
  const st = { capital: [0, 0], name: [0, 0], agree: [0, 0] }, S = Object.fromEntries(ix.vocab.map((v, i) => [v, i]));
  for (const row of set) {
    const ids = row.text.split(' ').map(t => S[t]);
    const tr = forward(w, ids, opt, false), z = tr.logits[ids.length - 1];
    let best = 0; for (let i = 1; i < z.length; i++) if (z[i] > z[best]) best = i;
    st[row.task][1]++; if (row.good.includes(ix.vocab[best])) st[row.task][0]++;
  }
  return Object.fromEntries(Object.entries(st).map(([k, [a, n]]) => [k, n ? a / n : 0]));
}
