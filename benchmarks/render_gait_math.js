#!/usr/bin/env node
/**
 * Render the gait-explorer math section to a static HTML fragment.
 *
 *   cd benchmarks && npm ci && npm run render
 *   -> benchmarks/gait_math.generated.html   (committed; injected by build_gait_explorer.py)
 *
 * `npm ci` installs the exact tree in package-lock.json, so the fragment is
 * byte-reproducible: KaTeX is pinned to an exact version because its emitted
 * markup and stylesheet are what get committed here, and a minor bump would
 * silently rewrite them.
 *
 * Why pre-render: the page is published as an artifact, where a strict CSP blocks
 * every external request -- no CDN for katex.js, its stylesheet, or the ~20 font
 * files the stylesheet references. Shipping KaTeX to the browser would mean
 * inlining the 272 KB parser plus 296 KB of fonts, and flashing raw TeX on load.
 * The formulas never change with the data, so they are rendered once here and the
 * result is committed. Only the fonts the output actually uses are inlined
 * (detected below, not guessed), and no KaTeX JS reaches the page.
 *
 * Node is a render-time dependency only. Building the page from the committed
 * fragment (benchmarks/build_gait_explorer.py) needs nothing but Python.
 */
const fs = require("fs");
const path = require("path");
const katex = require("katex");

const KATEX_DIR = path.dirname(require.resolve("katex/package.json"));
const DIST = path.join(KATEX_DIR, "dist");
const OUT = path.join(__dirname, "gait_math.generated.html");

/* ------------------------------------------------------------------ *
 * Content. `tex` lines render as display math; prose is declarative -- no
 * rhetorical questions. `ranks` marks the one number in a group that
 * separates the four policies.
 * ------------------------------------------------------------------ */
const GROUPS = [
  {
    head: "Energy — what the gait cost",
    entries: [
      {
        name: "Actuator work",
        tex: [String.raw`\tau_j = \mathrm{gear}_j\, u_j \qquad W_j = \tau_j\, \Delta q_j`],
        prose: `The six leg motors are fixed-gain torque actuators, so <span class="v">τ</span> is exact and holds
                constant across a control step under the zero-order hold. The step's work is therefore torque times
                angle change, with no sub-step integration.`,
      },
      {
        name: "Gross positive work",
        tex: [String.raw`E^{+} = \sum_{\mathrm{steps}} \sum_{j} \max(W_j,\, 0)`],
        prose: `The energy the actuators delivered. These motors recover nothing from braking, so negative work earns
                no credit.`,
      },
      {
        name: "Cost of transport",
        ranks: true,
        tex: [String.raw`\mathrm{CoT} = \frac{E^{+}}{M g d} \qquad M = 14\ \mathrm{kg}`],
        prose: `Energy per unit weight per unit distance, with <span class="v">d</span> the forward travel —
                dimensionless, and blind to how far a faster policy gets on the same clock. Absolute values sit far
                above an animal's (raw positive work, 2-D, no regeneration), so it reads across policies rather than
                against biology.`,
      },
      {
        name: "Return per joule",
        ranks: true,
        tex: [String.raw`R \,/\, E^{+}`],
        prose: `Reward bought per joule. Cost of transport prices distance and this prices reward; the two diverge
                because return saturates at 10 m/s while the energy bill does not.`,
      },
    ],
  },
  {
    head: "Spectrum — whole body, all six joints and both feet",
    entries: [
      {
        name: "Detrend and estimate",
        tex: [
          String.raw`\tilde x(t) = x(t) - (a t + b)`,
          String.raw`P(f) = \mathrm{Welch}\{\tilde x(t)\}, \quad f_s = 1/\mathrm{median}(\Delta t)`,
        ],
        prose: `Irregular control intervals are first resampled onto a uniform grid. The linear fit removes slow
                posture creep, which would otherwise swamp the spectrum and every phase measure downstream. Welch
                averages eight Hann-tapered, half-overlapping FFTs: one FFT of a real signal stays noisy at any
                recording length, since extra data buys frequency bins rather than a cleaner estimate in each.`,
      },
      {
        name: "Stride band",
        tex: [String.raw`\mathrm{band} = [\,0.5,\ 8\,]\ \mathrm{Hz}, \quad f_{\max} \le 0.45\, f_s`],
        prose: `Every spectral quantity is read inside this band. A motionless limb's power sits entirely at DC
                (<span class="v">f = 0</span>), the constant offset fixing where the limb rests, which falls outside
                it — so a standing policy cannot register as periodic.`,
      },
      {
        name: "Stride-band power fraction",
        tex: [String.raw`\beta = \frac{\sum_{f \in \mathrm{band}} P(f)}{\sum_{f > 0} P(f)}`],
        prose: `The share of a limb's motion at stride rate, DC excluded from the denominator for the reason above.
                This is the gate's evidence that a gait exists at all. <span class="sp">0.98 across all four,
                spread 0.006.</span>`,
      },
      {
        name: "Stride frequency",
        tex: [String.raw`f_0 = \operatorname*{argmax}_{f \in \mathrm{band}} P(f), \quad
                         \mathrm{stride\ freq} = \operatorname{median}_{\,8\ \mathrm{signals}} f_0`],
        prose: `The stride rate, taken as the median across the eight signals so one odd joint cannot drag it.
                Descriptive. <span class="sp">3.00–3.68 Hz.</span>`,
      },
      {
        name: "Fundamental power fraction",
        ranks: true,
        tex: [
          String.raw`\delta = \max(0.15\, f_0,\ 1.5\, \Delta f)`,
          String.raw`\rho_{\mathrm{peak}} = \frac{\sum_{|f - f_0| \le \delta} P(f)}{\sum_{f \in \mathrm{band}} P(f)}`,
        ],
        prose: `In-band power concentrated in the fundamental's main lobe; near 1 the motion is a single clean sine.
                The <span class="v">1.5Δf</span> floor keeps the Hann lobe from being clipped, which would score even a
                perfect tone as impure. Harmonics are excluded — a 3 Hz fundamental's would fill the band and saturate
                the measure for every policy. It also picks the reference joint. <span class="sp">0.73–0.86,
                spread 0.129.</span>`,
      },
      {
        name: "Spectral entropy",
        tex: [String.raw`p_k = P(f_k) \Big/ \textstyle\sum P, \qquad
                         H = -\frac{\sum_k p_k \log p_k}{\log N}`],
        prose: `Entropy of the in-band spectrum over its <span class="v">N</span> bins: 0 a pure tone, 1 an even smear.
                Kept as an honest null — a long tail of small-but-nonzero bins pins it mid-range however clean the
                fundamental is. <span class="sp">0.696–0.751, spread 0.055.</span>`,
      },
    ],
  },
  {
    head: "Reference joint — moves with the joint selector",
    entries: [
      {
        name: "Periodicity",
        ranks: true,
        tex: [String.raw`\rho(\tau) = \frac{\sum_t \tilde\theta(t)\, \tilde\theta(t + \tau)}{\sum_t \tilde\theta(t)^2},
                         \qquad \max_{\tau \,\in\, [1/f_{\max},\, 1/f_{\min}]} \rho(\tau)`],
        prose: `The strength of the best repeat period, reaching 1 for an exactly repeating signal. This is the phase
                portrait's band tightness as a number.`,
      },
      {
        name: "Stride period and CV",
        ranks: true,
        tex: [
          String.raw`\varphi(t) = \operatorname{unwrap} \arg\!\big(\tilde\theta(t) + i\, \mathcal{H}\{\tilde\theta\}(t)\big)`,
          String.raw`\varphi(t_k) = 2\pi k, \quad T_k = t_{k+1} - t_k, \quad
                     \mathrm{CV} = \frac{\operatorname{std}(T_k)}{\operatorname{mean}(T_k)}`,
        ],
        prose: `One stride is one <span class="v">2π</span> of the analytic signal's instantaneous phase, so
                segmentation is indifferent to waveform shape where a threshold counter would double-count a
                multi-lobed stride. CV vanishes when every stride takes the same time.`,
      },
      {
        name: "Poincaré dispersion",
        ranks: true,
        tex: [
          String.raw`\Sigma:\ \tilde\theta = 0,\ \dot{\tilde\theta} > 0, \qquad
                     s_k = (\theta_{1..6},\, \dot\theta_{1..6}) \in \mathbb{R}^{12}`,
          String.raw`d_P = \frac{1}{K} \sum_{k=1}^{K} \left\lVert \frac{s_k - \bar s}{\sigma} \right\rVert_2`,
        ],
        prose: `The full twelve-dimensional leg state at each upward mean-crossing, standardised per coordinate and
                measured from the crossings' centroid. A stable cycle pierces the section at one point, so
                <span class="v">d_P → 0</span>; standardising stops fast joints drowning slow ones. This is the return
                map's scatter in twelve dimensions rather than two.`,
      },
      {
        name: "Front/back phase-locking",
        ranks: true,
        tex: [String.raw`\mathrm{PLV} = \left| \frac{1}{T} \sum_t e^{\,i(\varphi_b(t) - \varphi_f(t))} \right|`],
        prose: `Reaches 1 when the back–front thigh phase difference is constant at any offset, as in a bound or
                gallop, and falls toward 0 when the limbs wander independently. Invariance to <em>which</em> phase is
                held is what lets it score coordination without presupposing a gait.`,
      },
    ],
  },
  {
    head: "Gate — when the reference-joint numbers are allowed to exist",
    entries: [
      {
        name: "Gait detection",
        tex: [String.raw`\mathrm{gait} = \big(n_{\mathrm{strides}} \ge 3\big) \,\wedge\, \big(\bar\beta > 0.2\big)`],
        prose: `A near-stationary policy's jitter still has an analytic phase, so Hilbert segmentation would
                manufacture strides from noise. Failing the gate returns the four reference-joint measures as NaN
                (shown as —), leaving <span class="v">β</span> and the stride count reported, so an absent gait stays
                visible rather than scoring as a bad one.`,
      },
    ],
  },
];

/* ------------------------------------------------------------------ *
 * Render
 * ------------------------------------------------------------------ */
const render = (tex) =>
  katex.renderToString(tex.replace(/\s+/g, " ").trim(), {
    displayMode: true,
    throwOnError: true,
    output: "html",          // drop the MathML twin: it doubles size and the
                             // page ships a plain-text equivalent already
    strict: "error",
  });

let body = "";
for (const g of GROUPS) {
  body += `\n    <div class="grp-head">${g.head}</div>\n    <div class="mathlist">\n`;
  for (const e of g.entries) {
    const tags = e.ranks ? ` <span class="chip-rank">ranks</span>` : "";
    const eqs = e.tex.map((t) => `<div class="mtex">${render(t)}</div>`).join("\n          ");
    body += `      <div class="mrow">
        <div class="mname">${e.name}${tags}</div>
        <div class="mbody">
          ${eqs}
          <p>${e.prose.replace(/\s+/g, " ").trim()}</p>
        </div>
      </div>\n`;
  }
  body += "    </div>\n";
}

/* ------------------------------------------------------------------ *
 * Font subsetting: keep only the faces the rendered output actually uses.
 * KaTeX picks a face via CSS class, so the classes present in the output
 * determine the set -- detected here rather than guessed.
 * ------------------------------------------------------------------ */
const CLASS_TO_FILES = {
  mathnormal: ["KaTeX_Math-Italic"],
  mathit: ["KaTeX_Main-Italic"],
  mathbf: ["KaTeX_Main-Bold"],
  textbf: ["KaTeX_Main-Bold"],
  mathbb: ["KaTeX_AMS-Regular"],
  amsrm: ["KaTeX_AMS-Regular"],
  mathcal: ["KaTeX_Caligraphic-Regular"],
  mathscr: ["KaTeX_Script-Regular"],
  mathfrak: ["KaTeX_Fraktur-Regular"],
  mathsf: ["KaTeX_SansSerif-Regular"],
  mathtt: ["KaTeX_Typewriter-Regular"],
  size1: ["KaTeX_Size1-Regular"],
  size2: ["KaTeX_Size2-Regular"],
  size3: ["KaTeX_Size3-Regular"],
  size4: ["KaTeX_Size4-Regular"],
  "op-symbol": ["KaTeX_Size1-Regular", "KaTeX_Size2-Regular"],
};
const needed = new Set(["KaTeX_Main-Regular"]);   // upright roman is always used
for (const [cls, files] of Object.entries(CLASS_TO_FILES)) {
  if (new RegExp(`\\b${cls}\\b`).test(body)) files.forEach((f) => needed.add(f));
}

let css = fs.readFileSync(path.join(DIST, "katex.min.css"), "utf8");
const kept = new Set();
css = css.replace(/@font-face\{[^}]*\}/g, (block) => {
  const m = block.match(/fonts\/(KaTeX_[\w-]+)\.woff2/);
  if (!m || !needed.has(m[1])) return "";
  const b64 = fs.readFileSync(path.join(DIST, "fonts", `${m[1]}.woff2`)).toString("base64");
  kept.add(m[1]);
  // `src` is the last declaration in the block, so it terminates at `}` rather
  // than `;` -- match up to either, and drop the woff/ttf fallbacks with it.
  const inlined = block.replace(
    /src:[^;}]+/,
    `src:url(data:font/woff2;base64,${b64}) format("woff2")`
  );
  if (inlined === block) throw new Error(`src rewrite failed for ${m[1]}`);
  return inlined;
});

const missing = [...needed].filter((f) => !kept.has(f));
if (missing.length) throw new Error(`no @font-face matched: ${missing.join(", ")}`);
// The page is published under a CSP that blocks every external fetch: a single
// surviving relative url() means silently missing glyphs.
if (/url\((?!data:)/.test(css)) {
  throw new Error("a non-data: url() survived in the CSS - fonts would not load under CSP");
}

const bytes = [...kept].reduce(
  (n, f) => n + fs.statSync(path.join(DIST, "fonts", `${f}.woff2`)).size, 0);

const out = `<!-- generated by benchmarks/render_gait_math.js - do not edit by hand -->
<style>
/* KaTeX ${katex.version}, font faces inlined as data URIs (strict CSP: no external fetches).
   Faces: ${[...kept].sort().join(", ")} */
${css}
</style>
${body}`;
fs.writeFileSync(OUT, out);

console.log(`fonts inlined (${kept.size}/20): ${[...kept].sort().join(", ")}`);
console.log(`  raw woff2 ${(bytes / 1024).toFixed(0)} KB -> base64 ~${((bytes * 4) / 3 / 1024).toFixed(0)} KB`);
console.log(`wrote ${path.relative(process.cwd(), OUT)} (${(out.length / 1024).toFixed(0)} KB)`);
