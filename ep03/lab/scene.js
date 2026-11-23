// 3D view of the forward pass. Front: one green lane per token (the residual stream). Behind it, the workers:
// LayerNorm discs, one cyan glass panel per attention head (keys on the bottom edge, queries on the top edge,
// arcs = attention weights), amber feed-forward boxes, and ⊕ where each result is added back into the lane.
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { CSS2DRenderer, CSS2DObject } from 'three/addons/renderers/CSS2DRenderer.js';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';
import { OutputPass } from 'three/addons/postprocessing/OutputPass.js';

export const COL = { green: 0x4DFF88, ui: 0x8DFFB0, cyan: 0x5CE1FF, amber: 0xFFB547, hot: 0xE6FFEC, neg: 0xFF5C8A, grey: 0x3a4a40, red: 0xFF4D5E };
export const LAYOUT = { G: 1.9, Y_EMB: 0.9, Y_POS: 2.1, Y_B0: 3.5, BH: 7.4, Z_BR: -1.5, Z_HEAD: .95 };
const o = { ln1: .55, br: .3, attB: 1.35, attT: 3.15, add1: 3.7, ln2: 4.25, ffn: 5.35, ffnH: 1.3, add2: 6.75 };

export class Scene3D {
  constructor(host, { onPick = () => {}, onHover = () => {}, manual = false } = {}) {
    this.host = host; this.onPick = onPick; this.onHover = onHover; this.manual = manual;   // manual: no animation loop, frame(t) draws one frame (film capture)
    let r;
    try { r = this.renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: 'high-performance' }); }
    catch (e) { host.insertAdjacentHTML('beforeend', '<p style="padding:24px;color:#FFB547">This browser could not start WebGL, so the 3D view is off. The panels still work. Try enabling hardware acceleration.</p>'); throw e; }
    r.setPixelRatio(Math.min(devicePixelRatio, 1.75)); r.setClearColor(0x010402); host.appendChild(r.domElement);
    this.css = new CSS2DRenderer(); this.css.domElement.className = 'labels'; host.appendChild(this.css.domElement);
    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(42, 1, .1, 400);
    this.controls = new OrbitControls(this.camera, this.css.domElement);
    Object.assign(this.controls, { enableDamping: true, dampingFactor: .08, screenSpacePanning: true, minDistance: 3, maxDistance: 160 });
    this.composer = new EffectComposer(r); this.composer.addPass(new RenderPass(this.scene, this.camera));
    this.bloom = new UnrealBloomPass(new THREE.Vector2(256, 256), .5, .05, .55); this.composer.addPass(this.bloom);
    this.composer.addPass(new OutputPass());
    this._glow = false; this.root = new THREE.Group(); this.scene.add(this.root);
    this.ray = new THREE.Raycaster(); this.ray.params.Line.threshold = .08; this.ptr = new THREE.Vector2();
    this.pickables = []; this.parts = []; this.front = null; this.fly = null; this.t0 = performance.now();
    this.css.domElement.addEventListener('pointermove', e => this.hover(e));
    let down = null;
    this.css.domElement.addEventListener('pointerdown', e => { down = [e.clientX, e.clientY]; });
    this.css.domElement.addEventListener('pointerup', e => {
      if (!down || Math.hypot(e.clientX - down[0], e.clientY - down[1]) > 6) return;
      const hit = this.pick(e); this.onPick(hit ? hit.userData : null);
    });
    new ResizeObserver(() => this.resize()).observe(host); this.resize();
    if (!manual) { const loop = () => { requestAnimationFrame(loop); this.tick(); }; loop(); }
  }

  /** Film mode: draw the frame at time t (s) with the camera at cam = { pos: [x,y,z], tgt: [x,y,z] }. Deterministic. */
  frame(t, cam) {
    if (cam) { this.camera.position.set(...cam.pos); this.controls.target.set(...cam.tgt); this.camera.lookAt(this.controls.target); }
    this.tick(t);
  }

  get glow() { return this._glow; }
  set glow(v) {   // plain render = exact sRGB colours; bloom chain encodes twice unless the output is linear (measured)
    this._glow = v; this.renderer.outputColorSpace = v ? THREE.LinearSRGBColorSpace : THREE.SRGBColorSpace;
  }

  resize() {
    const w = this.host.clientWidth, h = this.host.clientHeight; if (!w || !h) return;
    this.renderer.setSize(w, h); this.css.setSize(w, h); this.composer.setSize(w, h);
    this.camera.aspect = w / h; this.camera.updateProjectionMatrix();
  }

  // ---------- helpers ----------
  mat(color, opacity = 1, add = true) {
    return new THREE.MeshBasicMaterial({ color, transparent: opacity < 1 || add, opacity, blending: add ? THREE.AdditiveBlending : THREE.NormalBlending, depthWrite: !add });
  }
  mesh(geo, mat, pos, ud) {
    const m = new THREE.Mesh(geo, mat); m.position.set(...pos); this.root.add(m);
    if (ud) { m.userData = ud; this.pickables.push(m); }
    return m;
  }
  edges(geo, color, opacity, pos) {
    const l = new THREE.LineSegments(new THREE.EdgesGeometry(geo), new THREE.LineBasicMaterial({ color, transparent: true, opacity, blending: THREE.AdditiveBlending, depthWrite: false }));
    l.position.set(...pos); this.root.add(l); return l;
  }
  line(pts, color, opacity = .5) {
    const g = new THREE.BufferGeometry().setFromPoints(pts.map(p => new THREE.Vector3(...p)));
    const l = new THREE.Line(g, new THREE.LineBasicMaterial({ color, transparent: true, opacity, blending: THREE.AdditiveBlending, depthWrite: false }));
    this.root.add(l); return l;
  }
  label(html, pos, cls = '', ud) {
    const d = document.createElement('div'); d.className = 'lbl ' + cls; d.innerHTML = cls.includes('side') ? `<div class=in>${html}</div>` : html;
    if (ud) { d.classList.add('clk'); d.addEventListener('pointerup', e => { e.stopPropagation(); this.onPick(ud); }); d.addEventListener('pointerdown', e => e.stopPropagation()); }
    const ob = new CSS2DObject(d); ob.position.set(...pos); this.root.add(ob); return ob;
  }
  sel(ud) {                                                   // is this object part of the current selection?
    const s = this.state.sel; if (!s || !ud || s.kind !== ud.kind) return false;
    return ['b', 'j', 'n', 'which', 't'].every(k => s[k] === undefined || ud[k] === undefined || s[k] === ud[k]);
  }

  clear() {
    this.root.traverse(ob => { if (ob.geometry) ob.geometry.dispose(); if (ob.material) ob.material.dispose(); if (ob.element) ob.element.remove(); });
    this.scene.remove(this.root); this.root = new THREE.Group(); this.scene.add(this.root); this.pickables = []; this.parts = []; this.glowers = []; this.flows = []; this.marks = [];
    let a = 12345; this.rnd = () => { a = (a + 0x6D2B79F5) >>> 0; let x = a; x = Math.imul(x ^ (x >>> 15), x | 1); x ^= x + Math.imul(x ^ (x >>> 7), x | 61); return ((x ^ (x >>> 14)) >>> 0) / 4294967296; };   // seeded: same dots on every render
  }

  // ---------- build the whole model from a forward trace ----------
  build({ tr, w, state, vocab }) {
    this.clear(); this.state = state; this.tr = tr; this.w = w;
    const { L, h, d, pe } = w.cfg, T = tr.T, { G, Y_EMB, Y_POS, Y_B0, BH, Z_BR, Z_HEAD } = LAYOUT, opt = state.opt;
    const X = t => (t - (T - 1) / 2) * G, W = T * G, xl = X(0) - G * .5 - .4, focus = state.focus;
    const YF = Y_B0 + L * BH; this.dims = { T, L, h, W, YF, X };
    const words = tr.ids.map(i => vocab[i]);
    const stages = this.stages = [];                          // what the play-through narrates, bottom to top
    const glow = (m, y, base) => { this.glowers.push({ m, y, base }); return m; };

    // tokens
    words.forEach((wd, t) => this.label(`<b>${wd}</b><i>${tr.ids[t]}</i>`, [X(t), -.25, .4], 'tok' + (t === focus ? ' on' : ''), { kind: 'token', t }));
    stages.push({ y: -.2, kind: 'token' });

    // embedding table
    const embGeo = new THREE.BoxGeometry(W + .8, .08, 2.4);
    glow(this.mesh(embGeo, this.mat(COL.green, this.sel({ kind: 'embed' }) ? .42 : .2), [0, Y_EMB - .35, -.3], { kind: 'embed' }), Y_EMB, .2);
    this.edges(embGeo, COL.green, .5, [0, Y_EMB - .35, -.3]);
    this.label(`Embedding <span>table E · ${vocab.length}×${d}</span>`, [xl, Y_EMB - .35, .9], 'side', { kind: 'embed' });
    stages.push({ y: Y_EMB, kind: 'embed' });

    // positional encoding
    const yRes0 = pe === 'learned' || pe === 'sin' ? Y_POS + .45 : Y_EMB + .25;
    if (pe === 'learned' || pe === 'sin') {
      const pg = new THREE.BoxGeometry(W + .8, .06, 1.2);
      glow(this.mesh(pg, this.mat(COL.ui, this.sel({ kind: 'pos' }) ? .38 : .15), [0, Y_POS - .25, -1.2], { kind: 'pos' }), Y_POS, .15);
      this.edges(pg, COL.ui, opt.posScale ? .45 : .15, [0, Y_POS - .25, -1.2]);
      const wave = []; for (let i = 0; i <= 120; i++) { const x = -W / 2 + W * i / 120; wave.push([x, Y_POS - .1 + .18 * Math.sin(i / 120 * Math.PI * 2 * T / 2), -1.2]); }
      this.line(wave, COL.ui, opt.posScale ? .7 : .15);
      tr.ids.forEach((_, t) => { this.line([[X(t), Y_POS - .2, -1.2], [X(t), Y_POS + .45, 0]], COL.ui, .25); this.addRing(X(t), yRes0, opt.posScale ? COL.ui : COL.grey, { kind: 'pos' }); });
      this.label(`+ Position <span>${pe === 'sin' ? 'sin / cos waves' : 'learned table'}${opt.posScale ? '' : ' · OFF'}</span>`, [xl, Y_POS - .25, .9], 'side', { kind: 'pos' });
      stages.push({ y: Y_POS, kind: 'pos' });
    } else {
      this.label(pe === 'rope' ? `Position <span>RoPE: rotates q and k inside attention${opt.posScale ? '' : ' · OFF'}</span>` : `Position <span>none: the model gets no word order</span>`,
        [xl, Y_POS - .25, .9], 'side dim', { kind: 'pos' });
    }

    // residual lanes, split into segments so a switched-off residual shows as a break
    const resY = [yRes0];
    for (let b = 0; b < L; b++) resY.push(Y_B0 + b * BH + o.add1, Y_B0 + b * BH + o.add2);
    resY.push(YF + .4);
    const segOff = k => { if (k === 0) return false; const b = (k - 1) >> 1; return !!opt.residOff[`${b}.${(k - 1) % 2 ? 'ffn' : 'attn'}`]; };
    tr.ids.forEach((_, t) => {
      this.line([[X(t), -.05, 0], [X(t), yRes0, 0]], COL.green, .35);
      for (let k = 0; k < resY.length - 1; k++) {
        const off = segOff(k + 1), y0 = resY[k], y1 = resY[k + 1];
        const r = off && state.loud ? .09 : t === focus ? .06 : .04, g = new THREE.CylinderGeometry(r, r, y1 - y0 - .3, 6);
        this.mesh(g, this.mat(off ? COL.red : COL.green, off ? (state.loud ? .95 : .2) : (t === focus ? 1 : .68)), [X(t), (y0 + y1) / 2, 0], { kind: 'lane', t, k });
      }
    });

    // residual vectors as strips of d cells (what is in the lane at each stage)
    if (state.showVec) this.vectorStrips(tr, resY, X, d, focus);

    // blocks
    for (let b = 0; b < L; b++) {
      const y0 = Y_B0 + b * BH, S = tr.blocks[b];
      const frame = new THREE.BoxGeometry(W + 1.4, BH - .3, 2.6 + h * Z_HEAD);
      this.edges(frame, COL.green, this.state.sel?.kind === 'block' && this.state.sel.b === b ? .7 : .22, [0, y0 + BH / 2 - .15, Z_BR / 2 - h * Z_HEAD / 2 + .2]);
      this.label(`Block ${b + 1}`, [xl - .2, y0 + BH - .7, .9], 'blk', { kind: 'block', b });
      // branch to attention: LayerNorm discs, fan out to heads, back to ⊕
      const lnOff1 = opt.lnOff[`${b}.1`], attnOff = opt.attnOff[b];
      tr.ids.forEach((_, t) => {
        const x = X(t);
        this.line([[x, y0 + o.br, 0], [x, y0 + o.br, Z_BR], [x, y0 + o.attB - .25, Z_BR]], COL.green, .38);
        this.disc(x, y0 + o.ln1, Z_BR, lnOff1, { kind: 'ln', b, n: 1 });
        for (let j = 0; j < h; j++) {
          const z = Z_BR - .3 - j * Z_HEAD;
          this.line([[x, y0 + o.attB - .25, Z_BR], [x, y0 + o.attB, z]], COL.cyan, .32);
          this.line([[x, y0 + o.attT, z], [x, y0 + o.attT + .25, Z_BR]], COL.cyan, .32);
        }
        this.line([[x, y0 + o.attT + .25, Z_BR], [x, y0 + o.add1, Z_BR], [x, y0 + o.add1, 0]], attnOff ? COL.grey : COL.cyan, .55);
        this.addRing(x, y0 + o.add1, opt.residOff[`${b}.attn`] ? COL.red : COL.green, { kind: 'add', b, which: 'attn' });
        if (!attnOff) {
          const z0 = Z_BR - .3;
          this.flow([[x, y0 + o.br, 0], [x, y0 + o.br, Z_BR], [x, y0 + o.attB - .25, Z_BR], [x, y0 + o.attB, z0]], COL.ui);
          this.flow([[x, y0 + o.attT, z0], [x, y0 + o.attT + .25, Z_BR], [x, y0 + o.add1, Z_BR], [x, y0 + o.add1, 0]], COL.cyan);
        }
      });
      this.label(`LayerNorm${lnOff1 ? ' <span>OFF</span>' : ''}`, [xl, y0 + o.ln1, Z_BR], 'side small', { kind: 'ln', b, n: 1 });
      // heads
      for (let j = 0; j < h; j++) this.headPanel(b, j, S.heads[j], y0, X, W, words, focus);
      this.label(`Attention <span>${h} head${h > 1 ? 's' : ''} × ${d / h} dims${attnOff ? ' · OFF' : ''}</span>`, [xl, y0 + (o.attB + o.attT) / 2, Z_BR - .3], 'side', { kind: 'attn', b });
      this.label('⊕ add', [xl, y0 + o.add1, 0], 'side small', { kind: 'add', b, which: 'attn' });
      stages.push({ y: y0 + o.ln1, kind: 'ln', b, n: 1 }, { y: y0 + (o.attB + o.attT) / 2, kind: 'attn', b }, { y: y0 + o.add1, kind: 'add', b, which: 'attn' });

      // feed-forward
      const lnOff2 = opt.lnOff[`${b}.2`], ffnOff = opt.ffnOff[b];
      const act = S.hid.map(r => r.reduce((s, v) => s + Math.abs(v), 0) / r.length), amax = Math.max(...act, 1e-6);
      tr.ids.forEach((_, t) => {
        const x = X(t), sel = this.sel({ kind: 'ffn', b }) || this.sel({ kind: 'ffn', b, t });
        this.line([[x, y0 + o.add1 + .25, 0], [x, y0 + o.add1 + .25, Z_BR], [x, y0 + o.ffn - o.ffnH / 2, Z_BR]], COL.green, .38);
        this.disc(x, y0 + o.ln2, Z_BR, lnOff2, { kind: 'ln', b, n: 2 });
        const g = new THREE.BoxGeometry(.95, o.ffnH, 1.1), lvl = ffnOff ? .06 : .16 + .5 * act[t] / amax;
        const offC = state.loud ? COL.red : COL.grey;
        glow(this.mesh(g, this.mat(ffnOff ? offC : COL.amber, sel ? Math.min(1, lvl + .15) : lvl), [x, y0 + o.ffn, Z_BR - .2], { kind: 'ffn', b, t }), y0 + o.ffn, lvl);
        this.edges(g, ffnOff ? offC : COL.amber, ffnOff && state.loud ? .95 : sel ? 1 : .75, [x, y0 + o.ffn, Z_BR - .2]);
        this.line([[x, y0 + o.ffn + o.ffnH / 2, Z_BR], [x, y0 + o.add2, Z_BR], [x, y0 + o.add2, 0]], ffnOff ? COL.grey : COL.amber, .55);
        this.addRing(x, y0 + o.add2, opt.residOff[`${b}.ffn`] ? COL.red : COL.green, { kind: 'add', b, which: 'ffn' });
        if (!ffnOff) {
          this.flow([[x, y0 + o.add1 + .25, 0], [x, y0 + o.add1 + .25, Z_BR], [x, y0 + o.ffn - o.ffnH / 2, Z_BR]], COL.ui);
          this.flow([[x, y0 + o.ffn - o.ffnH / 2, Z_BR - .2], [x, y0 + o.ffn + o.ffnH / 2, Z_BR - .2]], COL.amber, .45);
          this.flow([[x, y0 + o.ffn + o.ffnH / 2, Z_BR], [x, y0 + o.add2, Z_BR], [x, y0 + o.add2, 0]], COL.amber);
        }
      });
      this.label(`LayerNorm${lnOff2 ? ' <span>OFF</span>' : ''}`, [xl, y0 + o.ln2, Z_BR], 'side small', { kind: 'ln', b, n: 2 });
      this.label(`Feed-forward <span>${d} → ${w.cfg.ffn * d} → ${d}${ffnOff ? ' · OFF' : ''}</span>`, [xl, y0 + o.ffn, Z_BR - .2], 'side amber', { kind: 'ffn', b });
      this.label('⊕ add', [xl, y0 + o.add2, 0], 'side small', { kind: 'add', b, which: 'ffn' });
      stages.push({ y: y0 + o.ln2, kind: 'ln', b, n: 2 }, { y: y0 + o.ffn, kind: 'ffn', b }, { y: y0 + o.add2, kind: 'add', b, which: 'ffn' });
    }

    // final norm, the same table again, output
    tr.ids.forEach((_, t) => this.disc(X(t), YF + .55, 0, opt.lnOff.f, { kind: 'ln', b: 'f' }));
    this.label(`Final LayerNorm${opt.lnOff.f ? ' <span>OFF</span>' : ''}`, [xl, YF + .55, .9], 'side small', { kind: 'ln', b: 'f' });
    const ug = new THREE.BoxGeometry(W + .8, .08, 2.4);
    glow(this.mesh(ug, this.mat(COL.hot, this.sel({ kind: 'unembed' }) ? .4 : .18), [0, YF + 1.3, -.3], { kind: 'unembed' }), YF + 1.3, .18);
    this.edges(ug, COL.hot, .45, [0, YF + 1.3, -.3]);
    this.label(`Unembedding <span>the same table E, used backwards</span>`, [xl, YF + 1.3, .9], 'side', { kind: 'unembed' });
    tr.ids.forEach((_, t) => {
      const p = tr.lens[L][t]; let best = 0; p.forEach((v, i) => { if (v > p[best]) best = i; });
      this.line([[X(t), YF + .7, 0], [X(t), YF + 1.9, 0]], COL.hot, .3);
      if (t !== focus && T <= 8) this.label(`${vocab[best]}<i>${Math.round(p[best] * 100)}%</i>`, [X(t), YF + 2.25, 0], 'pred', { kind: 'output', t });
    });
    const top = [...tr.lens[L][focus]].map((p, i) => [i, p]).sort((a, b) => b[1] - a[1]).slice(0, 5);
    this.label(`<div class=hd>next after “${words[focus]}”</div>` + top.map(([i, p]) => `<div class=row><span>${vocab[i]}</span><b style="width:${Math.max(2, p * 100)}%"></b><em>${(p * 100).toFixed(1)}%</em></div>`).join(''),
      [X(focus), YF + 3.2, 0], 'card', { kind: 'output', t: focus });
    stages.push({ y: YF + .55, kind: 'ln', b: 'f' }, { y: YF + 1.3, kind: 'unembed' }, { y: YF + 2.6, kind: 'output' });

    this.particles(T, X, resY, YF);
    this.flowPoints();
    if (state.mark) this.drawMark(state.mark);
    this.resetCamera(false);
  }

  lit(m, y) { this.glowers.push({ m, y, base: m.material.opacity }); return m; }

  /** Film/teaching overlay: a box around one part with a big tag. mark = { target, tone: 'warn' | 'off', text }.
   *  tone 'info' (cyan, steady) points at something that still works.
 *  target: { kind: 'head', b, j } | { kind: 'ffn' } (every block) | { kind: 'resid', b } | { kind: 'attnAll' } */
  drawMark({ target, tone, text }) {
    const { W, L, h, T, X } = this.dims, { Y_B0, BH, Z_BR, Z_HEAD } = LAYOUT, col = tone === 'off' ? COL.red : tone === 'info' ? COL.cyan : COL.amber, boxes = [];
    const yb = b => Y_B0 + b * BH, zh = j => Z_BR - .3 - j * Z_HEAD;
    if (target.kind === 'head') boxes.push([[0, yb(target.b) + (o.attB + o.attT) / 2, zh(target.j)], [W + 1.1, o.attT - o.attB + .9, .35]]);
    if (target.kind === 'attnAll') for (let b = 0; b < L; b++) boxes.push([[0, yb(b) + (o.attB + o.attT) / 2, Z_BR - .3 - (h - 1) * Z_HEAD / 2], [W + 1.1, o.attT - o.attB + .9, (h - 1) * Z_HEAD + .5]]);
    if (target.kind === 'ffn') for (let b = 0; b < L; b++) boxes.push([[0, yb(b) + o.ffn, Z_BR - .2], [W + 1.1, o.ffnH + .6, 1.6]]);
    if (target.kind === 'resid') boxes.push([[0, yb(target.b) + (o.add1 + o.add2) / 2, 0], [W + 1.1, o.add2 - o.add1 + .7, .7]]);
    this.marks = boxes.map(([c, sz], i) => {
      const g = new THREE.BoxGeometry(...sz), fill = this.mesh(g, this.mat(col, .12), c), edge = this.edges(g, col, 1, c);
      const tag = this.label(text, [c[0], c[1] + sz[1] / 2 + .55, c[2]], 'mtag ' + tone);
      return { fill, edge, tag, tone };
    });
  }
  addRing(x, y, color, ud) {
    const g = new THREE.TorusGeometry(.17, .035, 6, 20);
    this.lit(this.mesh(g, this.mat(color, this.sel(ud) ? 1 : .75), [x, y, 0], ud), y);
  }
  disc(x, y, z, off, ud) {
    const g = new THREE.CylinderGeometry(.3, .3, .05, 20);
    this.lit(this.mesh(g, this.mat(off ? COL.grey : COL.hot, this.sel(ud) ? .75 : (off ? .12 : .3)), [x, y, z], ud), y);
  }

  headPanel(b, j, H, y0, X, W, words, focus) {
    const { Z_BR, Z_HEAD } = LAYOUT, z = Z_BR - .3 - j * Z_HEAD, opt = this.state.opt, key = `${b}.${j}`;
    const sel = this.sel({ kind: 'head', b, j }) || this.sel({ kind: 'attn', b }), muted = H.muted, pat = opt.pattern[key];
    const g = new THREE.PlaneGeometry(W + .5, o.attT - o.attB + .4), loud = muted && this.state.loud;
    if (!loud) this.lit(this.mesh(g, new THREE.MeshBasicMaterial({ color: muted ? COL.grey : COL.cyan, transparent: true, opacity: sel ? .11 : .05, side: THREE.DoubleSide, depthWrite: false, blending: THREE.AdditiveBlending }),
      [0, y0 + (o.attB + o.attT) / 2, z], { kind: 'head', b, j }), y0 + (o.attB + o.attT) / 2);
    this.edges(g, loud ? COL.red : muted ? COL.grey : (pat ? COL.amber : COL.cyan), loud ? .95 : sel ? 1 : .6, [0, y0 + (o.attB + o.attT) / 2, z]);
    this.label(`H${j + 1}${muted ? ' · muted' : pat ? ' · ' + pat : ''}`, [X(words.length - 1) + 1.15, y0 + o.attT, z], 'head', { kind: 'head', b, j });
    if (muted) return;
    // arcs: key u (bottom edge) -> query t (top edge); the focus token's arcs bright, others faint
    const yb = y0 + o.attB, yt = y0 + o.attT;
    H.att.forEach((row, t) => row.forEach((p, u) => {
      if (p < .04) return;
      const on = t === focus, a = new THREE.Vector3(X(u), yb, z), c = new THREE.Vector3(X(t), yt, z);
      const m = new THREE.Vector3((X(u) + X(t)) / 2, (yb + yt) / 2, z + .15 + .05 * Math.abs(t - u));
      const curve = new THREE.QuadraticBezierCurve3(a, m, c);
      if (on) { const k = 1 + Math.round(p * 3); for (let n = 0; n < k; n++) this.flows.push({ curve, col: COL.hot, speed: .55, off: n / k, size: 1 }); }
      const tube = new THREE.TubeGeometry(curve, 14, .012 + (on ? .055 : .025) * p, 5, false);
      const mesh = this.mesh(tube, this.mat(on ? COL.hot : COL.cyan, on ? .4 + .6 * p : .08 + .3 * p), [0, 0, 0], { kind: 'head', b, j });
      this.glowers.push({ m: mesh, y: (yb + yt) / 2, base: mesh.material.opacity });
    }));
    // query/key dots on the edges
    words.forEach((_, t) => {
      const dg = new THREE.SphereGeometry(t === focus ? .07 : .045, 8, 6);
      this.mesh(dg, this.mat(COL.cyan, .8), [X(t), yb, z]); this.mesh(dg, this.mat(t === focus ? COL.hot : COL.cyan, .8), [X(t), yt, z]);
    });
    const r = H.att[focus]; let best = 0; r.forEach((v, i) => { if (v > r[best]) best = i; });
    if (j === 0 || this.sel({ kind: 'head', b, j }) || this.sel({ kind: 'attn', b })) {
      // "looks at", not "→": this is where the token reads from, not the answer. A repeated word gets its context.
      const dup = words.filter(x => x === words[best]).length > 1;
      const ctx = dup ? ` <span class=ctx>(in “${words.slice(Math.max(0, best - 1), best + 2).join(' ')}”)</span>` : '';
      this.label(`H${j + 1} · “${words[focus]}” looks at “${words[best]}”${ctx} <i>${r[best].toFixed(2)}</i>`, [Math.max(X(0) + 2.5, X(focus) - 4.4), yt + .34 + j * .48, z], 'att');
    }
  }

  vectorStrips(tr, resY, X, d, focus) {
    const vecs = [tr.x0, ...tr.blocks.flatMap(s => [s.mid, s.out])];
    const n = vecs.length * tr.T * d, cellW = 1.3 / d;
    const g = new THREE.BoxGeometry(cellW * .82, .13, .05);
    const m = new THREE.MeshBasicMaterial({ transparent: true, blending: THREE.AdditiveBlending, depthWrite: false });
    const inst = new THREE.InstancedMesh(g, m, n), M = new THREE.Matrix4(), c = new THREE.Color(), cp = new THREE.Color(COL.green), cn = new THREE.Color(COL.neg);
    const map = []; let k = 0;
    vecs.forEach((V, s) => {
      const mx = Math.max(...V.flatMap(r => [...r].map(Math.abs)), 1e-6);
      V.forEach((row, t) => {
        for (let i = 0; i < d; i++) {
          const v = row[i] / mx, a = Math.min(1, .08 + Math.abs(v)) ** .6 * (t === focus ? 1 : .78);
          M.makeTranslation(X(t) - .65 + (i + .5) * cellW, resY[s] + .32, .12); inst.setMatrixAt(k, M);
          c.copy(v >= 0 ? cp : cn).multiplyScalar(a); inst.setColorAt(k, c); map[k] = { kind: 'vec', t, s }; k++;
        }
      });
    });
    inst.instanceMatrix.needsUpdate = true; inst.instanceColor.needsUpdate = true;
    inst.userData = { kind: 'vec', map }; this.root.add(inst); this.pickables.push(inst);
  }

  flow(pts, col, speed = .7) {                              // a dot that keeps travelling along a polyline
    const curve = new THREE.CurvePath(), V = pts.map(p => new THREE.Vector3(...p));
    for (let i = 0; i < V.length - 1; i++) curve.add(new THREE.LineCurve3(V[i], V[i + 1]));
    this.flows.push({ curve, col, speed: speed / Math.max(.5, curve.getLength() / 2), off: this.rnd(), size: .8 });
  }
  flowPoints() {
    const n = this.flows.length; if (!n) { this.fpts = null; return; }
    const g = new THREE.BufferGeometry(), c = new Float32Array(n * 3), col = new THREE.Color();
    this.flows.forEach((f, i) => { col.set(f.col); c.set([col.r, col.g, col.b], i * 3); });
    g.setAttribute('position', new THREE.BufferAttribute(new Float32Array(n * 3), 3)); g.setAttribute('color', new THREE.BufferAttribute(c, 3));
    const m = new THREE.PointsMaterial({ size: .17, vertexColors: true, transparent: true, opacity: 1, blending: THREE.AdditiveBlending, depthWrite: false });
    this.fpts = new THREE.Points(g, m); this.root.add(this.fpts);
  }

  particles(T, X, resY, YF) {
    const per = 7, n = T * per, g = new THREE.BufferGeometry(), pos = new Float32Array(n * 3);
    g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    const m = new THREE.PointsMaterial({ color: COL.hot, size: .12, transparent: true, opacity: .9, blending: THREE.AdditiveBlending, depthWrite: false });
    const pts = new THREE.Points(g, m); this.root.add(pts);
    this.parts = { pts, pos, n, T, X, y0: resY[0], y1: YF + 1.9, seed: Array.from({ length: n }, () => this.rnd()) };
  }

  // ---------- picking ----------
  pick(e) {
    const r = this.css.domElement.getBoundingClientRect();
    this.ptr.set((e.clientX - r.left) / r.width * 2 - 1, -(e.clientY - r.top) / r.height * 2 + 1);
    this.ray.setFromCamera(this.ptr, this.camera);
    const hits = this.ray.intersectObjects(this.pickables, false);
    for (const h of hits) {
      if (h.object.isInstancedMesh) return { userData: h.object.userData.map[h.instanceId] };
      return h.object;
    }
    return null;
  }
  hover(e) {
    if (e.buttons) return;
    const hit = this.pick(e); this.css.domElement.style.cursor = hit ? 'pointer' : 'grab';
    this.onHover(hit ? hit.userData : null, e);
  }

  // ---------- camera ----------
  frameFor(ud) {
    const { W, YF, L, h } = this.dims, { Y_B0, BH, Z_BR, Z_HEAD } = LAYOUT, fit = Math.max(W, 9) * .9;
    if (!ud) { const H = YF + 6; return { tgt: [0, H / 2 - .8, -1.8], cam: [W * .6 + 5, H / 2 + 4, Math.max(H * 1.05, W * .95) + 9] }; }
    const yb = b => Y_B0 + b * BH;
    let y = YF / 2, z = -1.5, span = fit;
    if (ud.kind === 'token') y = 0;
    else if (ud.kind === 'embed' || ud.kind === 'pos') y = 1.5;
    else if (ud.kind === 'block') { y = yb(ud.b) + BH / 2; span = Math.max(fit, BH * .9); }
    else if (ud.kind === 'attn' || ud.kind === 'head') { y = yb(ud.b) + 2.25; z = Z_BR - h * Z_HEAD / 2; }
    else if (ud.kind === 'ffn') { y = yb(ud.b) + o.ffn; z = Z_BR; }
    else if (ud.kind === 'ln') y = ud.b === 'f' ? YF + .5 : yb(ud.b) + (ud.n === 1 ? o.ln1 : o.ln2);
    else if (ud.kind === 'add') y = yb(ud.b) + (ud.which === 'attn' ? o.add1 : o.add2);
    else if (ud.kind === 'unembed' || ud.kind === 'output') y = YF + 2;
    else if (ud.kind === 'lane' || ud.kind === 'vec') y = YF / 2;
    if (ud.kind === 'head') return { tgt: [0, y, z], cam: [span * .35, y + span * .45 + 1, z + span * 1.05 + 4] };
    return { tgt: [0, y, z], cam: [span * .7 + 2, y + span * .35 + 1, z + span * 1.1 + 4] };
  }
  flyTo(ud, animate = true) {
    const f = this.frameFor(ud);
    if (!ud) { const dist = Math.hypot(f.cam[0] - f.tgt[0], f.cam[1] - f.tgt[1], f.cam[2] - f.tgt[2]); this.controls.maxDistance = dist * 1.8; }
    if (!animate) { this.controls.target.set(...f.tgt); this.camera.position.set(...f.cam); return; }
    this.fly = { t0: performance.now(), dur: 900, a: [this.controls.target.clone(), this.camera.position.clone()], b: [new THREE.Vector3(...f.tgt), new THREE.Vector3(...f.cam)] };
  }
  resetCamera(animate = true) { if (!this.framed || animate) { this.flyTo(null, animate); this.framed = true; } }

  // ---------- play-through: a bright front climbs the stack ----------
  play(onStage, dur = 14000) { this.front = { t0: performance.now(), dur, onStage, i: -1 }; }
  stop() { this.front = null; this.glowers?.forEach(g => g.m.material && (g.m.material.opacity = g.base)); }

  tick(tFilm) {
    const now = performance.now(), t = tFilm ?? (now - this.t0) / 1000;
    if (this.fly) {
      const k = Math.min(1, (now - this.fly.t0) / this.fly.dur), e = k < .5 ? 2 * k * k : 1 - (-2 * k + 2) ** 2 / 2;
      this.controls.target.lerpVectors(this.fly.a[0], this.fly.b[0], e); this.camera.position.lerpVectors(this.fly.a[1], this.fly.b[1], e);
      if (k >= 1) this.fly = null;
    }
    let fy = null;
    if (this.front && this.stages) {
      const k = (now - this.front.t0) / this.front.dur, y0 = -.5, y1 = this.dims.YF + 3; fy = y0 + (y1 - y0) * k;
      const i = this.stages.findLastIndex(s => s.y <= fy);
      if (i !== this.front.i) { this.front.i = i; if (i >= 0) this.front.onStage(this.stages[i], i); }
      this.controls.target.y += (Math.min(fy, y1 - 2) - this.controls.target.y) * .04;
      if (k >= 1) { this.front.onStage(null); this.stop(); fy = null; }
    }
    if (this.glowers?.length && this.dims) {                  // the wave: what is computing now glows; during a run the rest dims
      const play = fy !== null, wy = play ? fy : -1 + ((t / 7) % 1) * (this.dims.YF + 4);
      for (const g of this.glowers) {
        if (!g.m.material) continue;
        const near = Math.max(0, 1 - Math.abs(wy - g.y) / (play ? 1.4 : 1.8));
        g.m.material.opacity = Math.min(1, play ? g.base * (.35 + .65 * near) + near * .75 : g.base + near * .22);
      }
    }
    for (const mk of this.marks || []) {                      // warn: blink; off: steady
      const k = mk.tone === 'warn' ? .35 + .65 * Math.abs(Math.sin(t * 5)) : 1;
      if (mk.fill.material) { mk.fill.material.opacity = .14 * k; mk.edge.material.opacity = k; }
      if (mk.tag.element) mk.tag.element.style.opacity = mk.tone === 'warn' ? .55 + .45 * k : 1;
    }
    if (this.fpts) {
      const pos = this.fpts.geometry.attributes.position, v = new THREE.Vector3();
      this.flows.forEach((f, i) => { f.curve.getPointAt((f.off + t * f.speed) % 1, v); pos.setXYZ(i, v.x, v.y, v.z); });
      pos.needsUpdate = true;
    }
    const P = this.parts;
    if (P && P.pts) {
      const per = P.n / P.T, span = P.y1 - P.y0;
      for (let i = 0; i < P.n; i++) {
        const lane = Math.floor(i / per), ph = (P.seed[i] + t * .08) % 1;
        let y = P.y0 + ph * span; if (fy !== null && y > fy) y = P.y0 + (ph * (fy - P.y0 > 0 ? fy - P.y0 : 0));
        P.pos[i * 3] = P.X(lane); P.pos[i * 3 + 1] = y; P.pos[i * 3 + 2] = .02;
      }
      P.pts.geometry.attributes.position.needsUpdate = true;
    }
    if (tFilm === undefined) this.controls.update();
    if (this.glow) this.composer.render(); else this.renderer.render(this.scene, this.camera);
    this.css.render(this.scene, this.camera);
  }
}
