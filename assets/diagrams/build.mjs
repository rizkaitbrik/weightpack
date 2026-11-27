// Generates the README diagrams as light and dark SVGs in the Weightpack
// brand palette. Run with `node assets/diagrams/build.mjs` after editing.
import { writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const OUT = dirname(fileURLToPath(import.meta.url));

const THEMES = {
  light: {
    bg: '#F4F1EA', card: '#FFFFFF', ink: '#171917', muted: '#6E6C64',
    border: '#CFC9BA', group: '#ECE7DB', groupBorder: '#D3CCBB',
    ember: '#E65032', emberTint: '#FBE2DA', line: '#3B3D39',
    inv: '#171917', invText: '#F4F1EA',
  },
  dark: {
    bg: '#171917', card: '#20231F', ink: '#F4F1EA', muted: '#9C9A91',
    border: '#3B3F39', group: '#1C1F1B', groupBorder: '#363A33',
    ember: '#E65032', emberTint: '#3A231C', line: '#C9C5BA',
    inv: '#F4F1EA', invText: '#171917',
  },
};

const SANS = "Inter, 'Segoe UI', -apple-system, Helvetica, Arial, sans-serif";
const MONO = "'IBM Plex Mono', ui-monospace, SFMono-Regular, Menlo, Consolas, monospace";

const esc = (s) => String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
const lines = (v) => (v == null ? [] : Array.isArray(v) ? v : [v]);
const textWidth = (s, size = 12, mono = false) => s.length * size * (mono ? 0.62 : 0.56);

// ---------- primitives ----------

function text(x, y, s, cls = 's', anchor = 'middle') {
  return `<text x="${x}" y="${y}" class="${cls}" text-anchor="${anchor}">${esc(s)}</text>`;
}

function stack(cx, cy, title, sub, { titleCls = 't', subCls = 's' } = {}) {
  const subs = lines(sub);
  const rows = [title ? { s: title, cls: titleCls, h: 18 } : null, ...subs.map((s) => ({ s, cls: s.startsWith('`') ? (subCls.includes('inv') ? 'm inv' : 'm') : subCls, h: 16 }))].filter(Boolean);
  const total = rows.reduce((a, r) => a + r.h, 0);
  let y = cy - total / 2;
  return rows.map((r) => { y += r.h; return text(cx, y - 4, r.s.replace(/^`|`$/g, ''), r.cls); }).join('');
}

const FILL = {
  card: ['card', 'border', ''], ext: ['bg', 'border', 'stroke-dasharray="5 4"'],
  accent: ['emberTint', 'ember', ''], inv: ['inv', 'inv', ''], plain: ['group', 'groupBorder', ''],
};

function box(x, y, w, h, title, sub, { kind = 'card', tag, rx = 10 } = {}) {
  const [f, s, extra] = FILL[kind];
  const inv = kind === 'inv';
  let out = `<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="${rx}" class="f-${f} k-${s}" ${extra}/>`;
  if (tag) out += text(x + w - 10, y + 15, tag, inv ? 'tag inv' : 'tag', 'end');
  out += stack(x + w / 2, y + h / 2 + (tag ? 3 : 0), title, sub, inv ? { titleCls: 't inv', subCls: 's inv' } : {});
  return out;
}

function pill(x, y, w, h, title, sub, opts = {}) {
  return box(x, y, w, h, title, sub, { ...opts, rx: h / 2 });
}

function cyl(x, y, w, h, title, sub, { kind = 'card' } = {}) {
  const [f, s, extra] = FILL[kind];
  const ry = 8;
  const body = `M${x},${y + ry} v${h - 2 * ry} a${w / 2},${ry} 0 0 0 ${w},0 v${-(h - 2 * ry)}`;
  return `<path d="${body}" class="f-${f} k-${s}" ${extra}/>` +
    `<ellipse cx="${x + w / 2}" cy="${y + ry}" rx="${w / 2}" ry="${ry}" class="f-${f} k-${s}" ${extra}/>` +
    stack(x + w / 2, y + h / 2 + 5, title, sub);
}

function group(x, y, w, h, label, { tag } = {}) {
  let out = `<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="14" class="f-group k-groupBorder"/>`;
  out += text(x + 16, y + 22, label.toUpperCase(), 'cap', 'start');
  if (tag) out += text(x + w - 14, y + 22, tag, 'tag', 'end');
  return out;
}

function roundedPath(pts, r = 8) {
  let d = `M${pts[0][0]},${pts[0][1]}`;
  for (let i = 1; i < pts.length - 1; i++) {
    const [px, py] = pts[i - 1], [cx, cy] = pts[i], [nx, ny] = pts[i + 1];
    const l1 = Math.hypot(cx - px, cy - py), l2 = Math.hypot(nx - cx, ny - cy);
    const rr = Math.min(r, l1 / 2, l2 / 2);
    const ax = cx - ((cx - px) / l1) * rr, ay = cy - ((cy - py) / l1) * rr;
    const bx = cx + ((nx - cx) / l2) * rr, by = cy + ((ny - cy) / l2) * rr;
    d += ` L${ax},${ay} Q${cx},${cy} ${bx},${by}`;
  }
  const last = pts[pts.length - 1];
  return d + ` L${last[0]},${last[1]}`;
}

function label(x, y, s, { anchor = 'middle', cls = 'lbl' } = {}) {
  const w = textWidth(s, 11.5, cls.includes('m')) + 14;
  const rx = anchor === 'middle' ? x - w / 2 : anchor === 'start' ? x - 4 : x - w + 4;
  return `<rect x="${rx}" y="${y - 11}" width="${w}" height="18" rx="9" class="f-bg"/>` +
    text(anchor === 'start' ? x + 3 : anchor === 'end' ? x - 3 : x, y + 2, s, cls, anchor);
}

function arrow(pts, { lbl, at, anchor, dashed, both, ember, end = true } = {}) {
  const c = ember ? 'ember' : 'line';
  let out = `<path d="${roundedPath(pts)}" class="edge k-${c}" ${dashed ? 'stroke-dasharray="5 5"' : ''} ` +
    `${end ? `marker-end="url(#a-${c})"` : ''} ${both ? `marker-start="url(#a-${c})"` : ''}/>`;
  if (lbl) {
    const [lx, ly] = at ?? [(pts[0][0] + pts[1][0]) / 2, (pts[0][1] + pts[1][1]) / 2];
    out += label(lx, ly, lbl, { anchor });
  }
  return out;
}

function legend(x, y, items) {
  let cx = x, out = '';
  for (const it of items) {
    if (it.kind === 'edge' || it.kind === 'dashed') {
      out += `<path d="M${cx},${y} h26" class="edge k-${it.ember ? 'ember' : 'line'}" ${it.kind === 'dashed' ? 'stroke-dasharray="5 5"' : ''} marker-end="url(#a-${it.ember ? 'ember' : 'line'})"/>`;
      cx += 36;
    } else {
      const [f, s, extra] = FILL[it.kind];
      out += `<rect x="${cx}" y="${y - 7}" width="20" height="14" rx="4" class="f-${f} k-${s}" ${extra}/>`;
      cx += 28;
    }
    out += text(cx, y + 4, it.label, 'lg', 'start');
    cx += textWidth(it.label, 12) + 26;
  }
  return out;
}

// A sequence diagram: participants across, steps down.
function sequence({ width, parts, steps, top = 0 }) {
  const X = Object.fromEntries(parts.map((p) => [p.id, p.x]));
  let out = '', y = top + 92;
  const body = [];
  const frames = [];
  const walk = (list, depth) => {
    for (const st of list) {
      if (st.frame) {
        const ids = collect(st.steps);
        const xs = ids.map((i) => X[i]);
        const f = { x1: Math.min(...xs) - 86 + depth * 6, x2: Math.max(...xs) + 86 - depth * 6, y1: y, label: st.frame, cond: st.cond };
        y += 34;
        walk(st.steps, depth + 1);
        f.y2 = y - 6;
        y += 26;
        frames.push(f);
      } else if (st.section) {
        body.push(`<rect x="16" y="${y}" width="${width - 32}" height="26" rx="8" class="f-group"/>` + text(30, y + 17, st.section.toUpperCase(), 'cap', 'start'));
        y += 56;
      } else if (st.self) {
        const x = X[st.self];
        body.push(`<path d="${roundedPath([[x, y - 4], [x + 34, y - 4], [x + 34, y + 14], [x + 4, y + 14]], 6)}" class="edge k-${st.ember ? 'ember' : 'line'}" marker-end="url(#a-${st.ember ? 'ember' : 'line'})"/>`);
        body.push(label(x + 44, y + 5, st.text, { anchor: 'start' }));
        y += 48;
      } else {
        const x1 = X[st.from], x2 = X[st.to];
        const dir = Math.sign(x2 - x1);
        body.push(`<path d="M${x1 + dir * 4},${y} L${x2 - dir * 6},${y}" class="edge k-${st.ember ? 'ember' : 'line'}" ${st.reply ? 'stroke-dasharray="5 5"' : ''} marker-end="url(#a-${st.ember ? 'ember' : 'line'})"/>`);
        body.push(label((x1 + x2) / 2, y - 12, st.text));
        y += 44;
      }
    }
  };
  const collect = (list) => list.flatMap((s) => (s.frame ? collect(s.steps) : s.self ? [s.self] : s.section ? [] : [s.from, s.to]));
  walk(steps, 0);
  const end = y;
  for (const p of parts) {
    out += `<path d="M${p.x},${top + 52} V${end}" class="life"/>`;
  }
  for (const f of frames) {
    const tw = textWidth(f.label, 11, true) + 18;
    out += `<rect x="${f.x1}" y="${f.y1}" width="${f.x2 - f.x1}" height="${f.y2 - f.y1}" rx="10" class="f-none k-ember" stroke-dasharray="4 4"/>` +
      `<path d="M${f.x1},${f.y1 + 20} V${f.y1 + 10} a10,10 0 0 1 10,-10 H${f.x1 + tw} V${f.y1 + 12} a8,8 0 0 1 -8,8 Z" class="f-ember"/>` +
      text(f.x1 + 10, f.y1 + 14, f.label, 'tag inv-e', 'start') +
      text(f.x1 + tw + 10, f.y1 + 16, f.cond, 'lbl', 'start');
  }
  out += body.join('');
  for (const p of parts) {
    out += box(p.x - 80, top, 160, 50, p.label, p.sub, { kind: p.kind ?? 'card', tag: p.tag });
  }
  return { svg: out, height: end + 10 };
}

// ---------- document ----------

function doc(name, fig, title, w, h, content) {
  for (const [mode, c] of Object.entries(THEMES)) {
    const css = `
      .t{font:600 14px ${SANS};fill:${c.ink}}
      .s{font:400 12px ${SANS};fill:${c.muted}}
      .m{font:400 11.5px ${MONO};fill:${c.muted}}
      .tag{font:500 10px ${MONO};fill:${c.muted};letter-spacing:.06em}
      .cap{font:600 10.5px ${MONO};fill:${c.muted};letter-spacing:.12em}
      .lbl{font:500 11.5px ${SANS};fill:${c.ink}}
      .lg{font:400 12px ${SANS};fill:${c.muted}}
      .fig{font:600 11px ${MONO};fill:${c.muted};letter-spacing:.14em}
      .edge{fill:none;stroke-width:1.6}
      .life{fill:none;stroke:${c.border};stroke-width:1.2;stroke-dasharray:2 5}
      .f-none{fill:none}
      ${Object.entries(c).map(([k, v]) => `.f-${k}{fill:${v}} .k-${k}{stroke:${v};stroke-width:1.4}`).join('\n      ')}
      .k-line{stroke-width:1.6} .k-ember{stroke-width:1.6}
      .inv{fill:${c.invText}} .inv-e{fill:#FFFFFF;font-weight:600}`;
    const marker = (id, col) => `<marker id="a-${id}" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0.8 L9.5,5 L0,9.2 z" fill="${col}"/></marker>`;
    const H = h + 56;
    const svg = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${w} ${H}" width="${w}" height="${H}" role="img" aria-label="${esc(title)}">
<title>${esc(title)}</title>
<defs><style>${css}</style>${marker('line', c.line)}${marker('ember', c.ember)}</defs>
<rect width="${w}" height="${H}" rx="18" fill="${c.bg}"/>
<path d="M22,30 l3.2,-8 h11 l-3.2,8 z" fill="${c.ember}"/>
${text(44, 30, `FIG ${fig} · ${title.toUpperCase()}`, 'fig', 'start')}
<g transform="translate(0 48)">${content}</g>
</svg>
`;
    writeFileSync(join(OUT, `${name}-${mode}.svg`), svg);
  }
}

// ---------- 1. architecture ----------

doc('architecture', 1, 'Architecture', 960, 500, [
  box(20, 14, 180, 58, 'wpk', ['command-line client', '`wpkd.sock`'], { tag: 'GO' }),
  cyl(20, 170, 180, 66, 'OCI registry', 'GHCR · Harbor · ECR', { kind: 'ext' }),
  cyl(20, 262, 180, 66, 'Hugging Face Hub', 'hf:// sources', { kind: 'ext' }),
  box(20, 360, 180, 60, 'OpenAI client', 'any SDK or curl', { kind: 'ext' }),

  group(250, 20, 390, 440, 'wpkd · one daemon per machine', { tag: 'GO' }),
  box(274, 64, 160, 66, 'Transfer', ['push · pull · hf://', 'signatures · trust']),
  box(456, 64, 160, 66, 'Builder', ['Weightfile · chain IDs', 'fingerprint · cache']),
  cyl(274, 188, 342, 84, 'Store', ['`blobs/sha256/<ab>/<digest>`', 'tags · build cache · decoded cache']),
  box(274, 340, 342, 66, 'Runtime', ['serving bases · engine processes', 'OpenAI gateway on :8090']),

  box(724, 64, 216, 66, 'weightpack-worker', ['squash · xor · bit1 · sparse', 'evals · router training'], { kind: 'accent', tag: 'PY' }),
  box(724, 340, 216, 66, 'Engine', ['vLLM + weightpack-vllm', 'or weightpack-engine'], { kind: 'accent', tag: 'PY' }),
  text(832, 200, 'Only wpkd writes the store.', 's'),
  text(832, 218, 'Workers read blobs read-only', 's'),
  text(832, 236, 'and hand back staged files.', 's'),

  arrow([[200, 43], [250, 43]]),
  arrow([[274, 88], [230, 88], [230, 203], [200, 203]], { both: true }),
  arrow([[274, 110], [240, 110], [240, 295], [200, 295]]),
  arrow([[200, 390], [274, 390]]),
  arrow([[354, 130], [354, 188]]),
  arrow([[536, 130], [536, 188]]),
  arrow([[445, 340], [445, 272]]),
  arrow([[616, 97], [724, 97]], { ember: true, lbl: 'gRPC', at: [670, 84] }),
  arrow([[616, 373], [724, 373]], { ember: true, lbl: 'spawns', at: [670, 360] }),
  label(670, 388, 'proxies'),

  legend(24, 490, [
    { kind: 'card', label: 'Go, in wpkd' }, { kind: 'accent', label: 'Python, touches tensors' },
    { kind: 'ext', label: 'External' },
  ]),
].join(''));

// ---------- 2. image format ----------

const chip = (x, y, mt, sub, lossy) => box(x, y, 160, 58, mt, sub, { kind: lossy ? 'accent' : 'card' });
doc('image-format', 2, 'Image format', 960, 380, [
  box(20, 150, 180, 70, 'OCI manifest', ['one entry per blob', '`wpk.layer · wpk.tensor`'], { kind: 'inv' }),
  box(260, 20, 300, 110, 'Config blob', ['`weightpack.config.v1`', 'architecture fingerprint · tokenizer', 'layer graph · chain IDs', 'fidelity records']),
  box(600, 20, 340, 110, 'Canonical blob layout', ['tensors sorted by name, packed data', 'compact header, sorted keys', 'same tensor → same digest, anywhere'], { kind: 'plain' }),
  group(260, 160, 680, 190, 'layer blobs · one tensor per blob · vnd.weightpack.*'),
  chip(278, 200, 'tensor.v1', 'one base tensor', false),
  chip(444, 200, 'lora.v1', 'A and B of one module', false),
  chip(610, 200, 'delta.xor.v1+zstd', 'bits XORed with parent', false),
  chip(776, 200, 'router.v1', 'a trained MIX router', false),
  chip(278, 272, 'meta.v1', 'config · tokenizer · template', false),
  chip(444, 272, 'delta.bit1.v1', 'sign bits + one scale', true),
  chip(610, 272, 'delta.sparse.v1+zstd', 'bitmap + kept values', true),
  arrow([[200, 170], [230, 170], [230, 75], [260, 75]]),
  arrow([[200, 200], [260, 200]]),
  legend(24, 376, [{ kind: 'card', label: 'Lossless' }, { kind: 'accent', label: 'Lossy, carries a fidelity record' }]),
].join(''));

// ---------- 3. chain IDs ----------

doc('chain-ids', 3, 'Chain IDs', 960, 510, [
  box(330, 20, 300, 58, 'MIX tasks', ['router: trained · top_k=1']),
  box(170, 130, 170, 58, 'python', 'lora · expert', { kind: 'accent' }),
  box(395, 130, 170, 58, 'sql', 'lora · expert', { kind: 'accent' }),
  box(620, 130, 170, 58, 'math', 'lora · expert', { kind: 'accent' }),
  box(330, 240, 300, 58, 'LAYER support', ['`chain = H(chain(base), digest)`'], { kind: 'accent' }),
  box(330, 322, 300, 58, 'base', ['`chain = digest(base)`'], { kind: 'inv' }),
  arrow([[255, 188], [255, 214], [440, 214], [440, 240]]),
  arrow([[480, 188], [480, 240]]),
  arrow([[705, 188], [705, 214], [520, 214], [520, 240]]),
  arrow([[480, 322], [480, 298]], { end: true }),
  arrow([[255, 130], [255, 49], [330, 49]], { dashed: true, ember: true }),
  arrow([[480, 130], [480, 78]], { dashed: true, ember: true, lbl: 'by reference', at: [480, 104] }),
  arrow([[705, 130], [705, 49], [630, 49]], { dashed: true, ember: true }),
  box(20, 410, 450, 62, 'Chain ID', ['`chain(L) = sha256(chain(parent) + " " + digest(L))`'], { kind: 'plain' }),
  box(490, 410, 450, 62, 'A layer is valid only on its chain', 'pull and build refuse it anywhere else', { kind: 'plain' }),
  legend(24, 500, [{ kind: 'edge', label: 'sits on (parent)' }, { kind: 'dashed', ember: true, label: 'mounted as an expert' }]),
].join(''));

// ---------- 4. build ----------

{
  const s = sequence({
    width: 960,
    parts: [
      { id: 'u', label: 'wpk', sub: 'CLI', x: 100 },
      { id: 'd', label: 'wpkd', sub: 'daemon', x: 300, kind: 'inv' },
      { id: 's', label: 'Store', sub: 'CAS', x: 490 },
      { id: 'h', label: 'Sources', sub: 'HF Hub · registry', x: 670, kind: 'ext' },
      { id: 'w', label: 'Worker', sub: 'Python', x: 860, kind: 'accent' },
    ],
    steps: [
      { from: 'u', to: 'd', text: 'build -t name:tag (Weightfile)' },
      { self: 'd', text: 'parse, validate, resolve the cache' },
      { from: 'd', to: 'h', text: 'fetch missing files · hash-checked · trust policy' },
      { from: 'd', to: 's', text: 'split into tensor blobs, no decoding' },
      { self: 'd', text: 'check each layer against its parent: names, shapes, dtypes' },
      { frame: 'opt', cond: 'codec = xor · bit1 · sparse', steps: [
        { from: 'd', to: 'w', text: 'Import(parent blobs, fine-tune, staging dir)', ember: true },
        { from: 'w', to: 'd', text: 'one file per changed tensor + eval scores', reply: true, ember: true },
        { self: 'd', text: 'canonicalise · zstd · gate on max_drop' },
      ] },
      { frame: 'opt', cond: 'ROUTER train', steps: [
        { from: 'd', to: 'w', text: 'TrainRouter(serving base, experts, data) · streams', ember: true },
        { from: 'w', to: 'd', text: 'router.safetensors', reply: true, ember: true },
      ] },
      { from: 'd', to: 's', text: 'write config + manifest · tag' },
      { from: 'd', to: 'u', text: 'image digest', reply: true },
    ],
  });
  doc('build', 4, 'Build', 960, s.height, s.svg);
}

// ---------- 5. import ----------

doc('import', 5, 'Importing full fine-tunes', 960, 600, [
  box(40, 20, 220, 58, 'Full fine-tune', 'safetensors · hf:// · path', { kind: 'ext' }),
  pill(340, 20, 280, 58, 'Same tensors, tokenizer', 'and config as the parent?', { kind: 'plain' }),
  box(700, 20, 240, 58, 'Build error', 'nothing is stored', { kind: 'inv' }),
  box(340, 120, 280, 58, 'Delta vs parent', 'per tensor the fine-tune changed'),
  box(40, 220, 260, 66, 'xor', ['bits XORed with the parent', 'lossless · bit-exact']),
  box(350, 220, 260, 66, 'bit1', ['sign bits + one scale per matrix', 'scales calibrated by KL distillation'], { kind: 'accent' }),
  box(660, 220, 240, 66, 'sparse', ['top-density entries', 'bitmap + kept values'], { kind: 'accent' }),
  box(40, 340, 260, 58, 'Lossless layer', 'no fidelity record needed'),
  box(350, 340, 550, 58, 'Eval: the fine-tune vs the decoded layer', 'your suite (eval=) or the self suite, served as the image will be'),
  pill(500, 440, 240, 52, 'drop ≤ max_drop ?', null, { kind: 'plain' }),
  box(500, 530, 240, 58, 'Lossy layer', '+ fidelity record', { kind: 'accent' }),
  arrow([[260, 49], [340, 49]]),
  arrow([[620, 49], [700, 49]], { lbl: 'no', at: [660, 36] }),
  arrow([[480, 78], [480, 120]], { lbl: 'yes', at: [500, 99], anchor: 'start' }),
  arrow([[480, 178], [480, 198], [170, 198], [170, 220]]),
  arrow([[480, 178], [480, 220]]),
  arrow([[480, 178], [480, 198], [780, 198], [780, 220]]),
  arrow([[170, 286], [170, 340]]),
  arrow([[480, 286], [480, 340]], { ember: true }),
  arrow([[780, 286], [780, 340]], { ember: true }),
  arrow([[620, 398], [620, 440]]),
  arrow([[620, 492], [620, 530]], { lbl: 'yes', at: [640, 511], anchor: 'start' }),
  arrow([[740, 466], [925, 466], [925, 78]], { lbl: 'no', at: [835, 453] }),
].join(''));

// ---------- 6. distribution ----------

{
  const s = sequence({
    width: 960,
    parts: [
      { id: 'u', label: 'wpk', sub: 'holds keys', x: 110 },
      { id: 'd', label: 'wpkd', sub: 'holds credentials', x: 360, kind: 'inv' },
      { id: 'r', label: 'OCI registry', sub: 'blobs · referrers', x: 610, kind: 'ext' },
      { id: 'f', label: 'Sigstore', sub: 'Fulcio · Rekor', x: 850, kind: 'ext' },
    ],
    steps: [
      { section: 'Publish' },
      { from: 'u', to: 'd', text: 'push name:tag' },
      { from: 'd', to: 'r', text: 'HEAD each blob · upload only missing · manifest' },
      { from: 'u', to: 'f', text: 'sign: key, or keyless with OIDC', ember: true },
      { from: 'u', to: 'd', text: 'Sigstore bundle (no private key)', ember: true },
      { from: 'd', to: 'r', text: 'OCI 1.1 referrer, subject = image', ember: true },
      { section: 'Consume' },
      { from: 'u', to: 'd', text: 'pull [--exact] name:tag' },
      { from: 'd', to: 'r', text: 'resolve the reference' },
      { self: 'd', text: 'trust policy: longest scope wins' },
      { from: 'd', to: 'r', text: 'fetch referrers · verify bundles', ember: true },
      { from: 'd', to: 'r', text: 'manifest + config · recompute digests and chains' },
      { from: 'd', to: 'r', text: 'fetch only the missing tensor blobs' },
    ],
  });
  doc('distribution', 6, 'Distribution and trust', 960, s.height, s.svg);
}

// ---------- 7. serving ----------

const slab = (x, y, w, name, model, mix) => box(x, y, w, 52, name, `\`model: ${model}\``, { kind: 'accent', tag: mix ? 'MIX' : undefined });
doc('serving', 7, 'Serving', 960, 470, [
  box(20, 190, 160, 60, 'OpenAI client', 'POST /v1/chat/…', { kind: 'ext' }),
  box(230, 160, 200, 120, 'wpkd gateway', [':8090', 'routes by model name', 'adds the image template', 'no tensor work'], { kind: 'inv' }),
  group(480, 0, 460, 250, 'Engine A · one process', { tag: 'vLLM' }),
  slab(500, 40, 130, 'support:v1', 'support:v1'),
  slab(642, 40, 130, 'french:v1', 'fr'),
  slab(784, 40, 140, 'qwen-tasks', 'qwen-tasks', true),
  box(500, 104, 424, 30, null, 'adapters, loaded at runtime · /v1/load_lora_adapter', { kind: 'plain' }),
  box(500, 146, 424, 84, 'Serving base · chain(qwen base)', ['dense weights, loaded once', 'shared by every image on this base'], { kind: 'inv' }),
  group(480, 270, 460, 200, 'Engine B · one process', { tag: 'REFERENCE' }),
  slab(500, 310, 130, 'legal:v2', 'legal:v2'),
  box(500, 376, 424, 76, 'Serving base · chain(llama + xor)', ['the xor layer is squashed into the base', 'loaded once'], { kind: 'inv' }),
  arrow([[180, 220], [230, 220]]),
  arrow([[430, 200], [455, 200], [455, 125], [480, 125]]),
  arrow([[430, 240], [455, 240], [455, 370], [480, 370]]),
].join(''));

// ---------- 8. token routing ----------

doc('token-routing', 8, 'Token routing in vLLM', 960, 420, [
  pill(20, 170, 90, 52, 'x', 'hidden state'),
  box(170, 30, 240, 60, 'W x', 'base product, once per token', { kind: 'inv' }),
  box(170, 160, 240, 72, 'Router g(x)', ['`logit_i = |x · w_i| + b_i`', 'top-k · softmax'], { kind: 'accent' }),
  box(170, 290, 240, 66, 'Shrink  z = x Aᵀ', '`A = [A₁; A₂; A₃]`', { kind: 'accent' }),
  box(470, 250, 210, 106, 'Gate', ['scale each expert’s', 'block of z by its', 'routing coefficient'], { kind: 'accent' }),
  box(730, 290, 210, 66, 'Expand  z Bᵀ', '`B = [s₁B₁, s₂B₂, s₃B₃]`', { kind: 'accent' }),
  `<circle cx="835" cy="60" r="20" class="f-card k-line"/>`, text(835, 66, '+', 't'),
  pill(880, 40, 60, 40, 'y', null),
  arrow([[110, 196], [138, 196], [138, 60], [170, 60]]),
  arrow([[110, 196], [170, 196]]),
  arrow([[110, 196], [138, 196], [138, 323], [170, 323]]),
  arrow([[410, 60], [815, 60]]),
  arrow([[410, 196], [440, 196], [440, 280], [470, 280]], { ember: true }),
  arrow([[410, 323], [470, 323]], { ember: true }),
  arrow([[680, 323], [730, 323]], { ember: true }),
  arrow([[835, 290], [835, 80]], { ember: true, lbl: 'Σ experts', at: [835, 190] }),
  arrow([[855, 60], [880, 60]]),
  box(20, 380, 920, 34, null, 'One adapter slot holds the whole MIX. Mix tokens, single-adapter tokens and base tokens share one batch.', { kind: 'plain' }),
].join(''));

// ---------- 9. kubernetes ----------

const node = (x, label, base, models) => [
  group(x, 200, 430, 210, label),
  box(x + 20, 240, 390, 52, 'wpkd agent', 'store on node disk · trust policy · gateway'),
  box(x + 20, 306, 390, 84, null, null, { kind: 'inv' }),
  text(x + 36, 330, `engine · ${base}`, 't inv', 'start'),
  text(x + 394, 330, 'one copy of the base', 's inv', 'end'),
  ...models.map((m, i) => box(x + 36 + i * 128, 342, 118, 34, null, `\`${m}\``, { kind: 'accent', rx: 8 })),
];
doc('kubernetes', 9, 'Kubernetes', 960, 560, [
  box(20, 30, 180, 60, 'OpenAI client', 'one cluster endpoint', { kind: 'ext' }),
  box(260, 14, 440, 92, 'wpk-control', ['Deployment + Service · stateless', 'gateway · placement · catalogue · /metrics'], { kind: 'inv' }),
  cyl(760, 26, 180, 68, 'ConfigMap', 'catalogue of models', { kind: 'ext' }),
  ...node(20, 'GPU node A · DaemonSet pod', 'base X', ['model-1', 'model-2']),
  ...node(510, 'GPU node B · DaemonSet pod', 'base Y', ['model-3']),
  cyl(330, 460, 300, 66, 'OCI registry', 'missing blobs only · signatures checked', { kind: 'ext' }),
  arrow([[200, 60], [260, 60]]),
  arrow([[760, 60], [700, 60]]),
  arrow([[440, 106], [440, 150], [200, 150], [200, 200]]),
  arrow([[460, 106], [460, 170], [300, 170], [300, 200]], { dashed: true, ember: true }),
  arrow([[520, 106], [520, 150], [760, 150], [760, 200]]),
  arrow([[500, 106], [500, 170], [660, 170], [660, 200]], { dashed: true, ember: true }),
  arrow([[420, 460], [420, 435], [235, 435], [235, 410]]),
  arrow([[540, 460], [540, 435], [725, 435], [725, 410]]),
  legend(660, 500, [{ kind: 'edge', label: 'requests' }, { kind: 'dashed', ember: true, label: 'placement' }]),
].join(''));

console.log('wrote diagrams to', OUT);
