# Independent Audit V3 — Follow-up on commit `23546eae`

**Repository root:** `/Users/zengweiguo/Documents/sequence_model_explainer` (confirmed via `git rev-parse --show-toplevel`)
**HEAD:** `23546eae35a0c832cbf724e8a206c4ff1b5dae71` — *Resolve independent audit release blockers*
**Branch:** `codex/dilation-tensor-evolution-demo`
**Working tree:** clean (`git status --short` empty, before and after)
**Baseline audited:** `3d56b978` in `AUDIT-INDEPENDENT-V3.md`

**Scope.** Focused re-test of the V3 findings plus a regression sweep of commit `23546eae`. Not a full
re-audit. Nothing in the application was modified; the only file written is this report.

---

# VERDICT: READY WITH CAVEATS

All three V3 release blockers are **fixed and independently verified**:

- **B-1** (red, non-deterministic release gate) — `npm test` and the browser suite passed **4/4 runs**,
  including one cold run and one under deliberate 4-way CPU load. No failure of any kind was observed.
- **B-2** (main-page residual example degenerate in 5 of 8 blocks) — all **8 K562 blocks × 3 selection
  modes = 24 combinations** now render, `balanced` shows both paths > 0 in every block, and
  `correction + shortcut = output` holds to < 5 × 10⁻⁴ in all 24.
- **B-3** (softmax caption contradicting its chart) — both captions now match the plotted quantity, in
  both locations.

**No scientific or numerical error was found at this commit.** Every value I re-derived reproduced
exactly. The remaining caveats are teaching-clarity, accessibility, performance and packaging issues.

**One new medium finding** (F-1) is the same defect class the commit fixed on `/dilation-trace`,
reappearing on `/`: the main page's residual prose does not update when the reader drags the position
slider, and lands on a `0.0000 + 0.0000` cell while still asserting the balanced selection rule. In a
live demo or screen recording, one drag reaches it. **This is the one item I would fix before the
video.**

Two other new items are packaging side-effects of the bundle fix (F-3 empty server-rendered HTML,
F-4 duplicated data with no sync path) that matter for a published artifact but not for correctness.

---

## 1. Commands executed

| # | Command | Result |
|---|---|---|
| 1 | `git rev-parse --show-toplevel` | `/Users/zengweiguo/Documents/sequence_model_explainer` ✅ |
| 2 | `git log -1 --oneline` | `23546eae Resolve independent audit release blockers` ✅ |
| 3 | `git rev-parse --abbrev-ref HEAD` | `codex/dilation-tensor-evolution-demo` ✅ |
| 4 | `git status --short` | empty ✅ |
| 5 | **`npm test`** (cold: dev server killed, `.vinext/deps` cleared) | **EXIT 0** ✅ — lint ✅ · typecheck ✅ · build ✅ · **19/19 unit** ✅ · **7/7 browser** (39.4 s) ✅ |
| 6 | **`npm run verify:python`** | **EXIT 0** ✅ — `model-analysis invariants: PASS`, `Basset adapter verification passed` |
| 7 | `npx playwright test` — **run 2** (warm) | **EXIT 0** ✅ — 7/7 passed (23.1 s) |
| 8 | `npx playwright test` — **run 3** (warm) | **EXIT 0** ✅ — 7/7 passed (20.1 s) |
| 9 | `npx playwright test` — **run 4** (with 4 busy-loop CPU hogs running) | **EXIT 0** ✅ — 7/7 passed (20.6 s) |

**Every run is reported above as executed; none was re-run to reach green.** Four consecutive passes,
zero failures. No failure occurred, so the "loading timeout vs. scientific mismatch" distinction did not
arise in practice — but the suite is now structurally able to make it: the timeout that previously
failed is explicitly widened (`tests/browser/explainer.spec.ts:20`, `{ timeout: 30_000 }`), and the new
test at `:48` gates on a text sentinel (`waitForFunction(() => !…includes("Loading verified checkpoint
data"))`) before asserting numeric values, so a data-loading stall and a numeric regression now fail on
different lines with different messages.

---

## 2. Required checks — results

### 2.1 Main residual lesson ✅ PASS

**Default K562 Block 1** (browser, `#residual .merge-lane`):

```
transformed value 0.3757  +  preserved shortcut 0.3024  =  0.6781
```

Correction > 0 ✅ · shortcut > 0 ✅ · `0.3757 + 0.3024 = 0.6781` ✅ (|Δ| < 5 × 10⁻⁵).

**All eight K562 blocks × three modes** (browser-read merge lane; `sum-ok` = |corr + short − out| < 5 × 10⁻⁴):

| Block | Balanced merge | Strongest learned correction | Strongest final activation |
|---|---|---|---|
| 1 | 0.3757 + 0.3024 = 0.6781 ✅ | 0.7806 + 0.0000 = 0.7806 | 0.1149 + 0.9770 = 1.0919 |
| 2 | 0.3915 + 0.6891 = 1.0807 ✅ | 1.1568 + 0.0000 = 1.1568 | 1.1568 + 0.0000 = 1.1568 |
| 3 | 0.4045 + 0.4998 = 0.9043 ✅ | 1.0482 + 0.0000 = 1.0482 | 0.0000 + 1.1568 = 1.1568 |
| 4 | 0.7267 + 0.6157 = 1.3424 ✅ | 1.4182 + 0.0000 = 1.4182 | 1.4182 + 0.0000 = 1.4182 |
| 5 | 1.9610 + 1.0393 = 3.0003 ✅ | 1.9610 + 1.0393 = 3.0003 | 1.9610 + 1.0393 = 3.0003 |
| 6 | 0.8859 + 0.6327 = 1.5186 ✅ | 1.7845 + 0.0000 = 1.7845 | 0.0000 + 3.0003 = 3.0003 |
| 7 | 1.0337 + 1.0547 = 2.0883 ✅ | 2.5585 + 0.0000 = 2.5585 | 0.0000 + 3.0003 = 3.0003 |
| 8 | 4.6421 + 3.0003 = 7.6425 ✅ | 8.2779 + 0.0000 = 8.2779 | 6.5752 + 1.9298 = 8.5050 |

**`balanced` yields two active paths in 8 of 8 blocks. Sum identity holds in 24 of 24.** The zeros in
the `correction` and `output` columns are correct behaviour — those rules explicitly maximize one path.

**Independent numerical confirmation.** I recomputed all 24 from the raw `.f32.gz` tensors
(`conv = Σ_taps input[:, p + t·d] · W[:, t]`, `corr = max(0, conv + bias)`, `shortcut = input[ch, p+d]`)
and cross-checked against the *actual block-output tensor*:
`all 24 K562 example_modes reconcile with raw tensors AND the block-output tensor: True (0 mismatches)`.

**Labels accurately describe the rules** ✅. The prose at `app/page.tsx:574` prints the exporter's own
`selection_rule` string verbatim, so the label cannot drift from the rule:

- balanced → *"maximize min(ReLU correction, shortcut) inside input-aligned coordinates 558–1557. Channel 118, input-aligned coordinate 948."*
- correction → *"largest ReLU correction inside input-aligned coordinates 558–1557. Channel 471, input-aligned coordinate 971."*
- output → *"largest block output inside input-aligned coordinates 558–1557. Channel 334, input-aligned coordinate 976."*

These match `scripts/run_chrombpnet_checkpoint.py:62-100` exactly.

**GM21515 / synthetic do not falsely claim a globally searched balanced example** ✅.
Confirmed: `example_modes` is present in 8/8 K562 blocks and **0/8** blocks of both other presets. The
selector correctly falls back:

| preset | selector value | disabled | prose |
|---|---|---|---|
| k562 | `balanced` | no | the rule string above |
| gm21515 | `legacy` → "Stored checkpoint example" | **yes** | *"This preset predates the multi-example export. The displayed cell is the exact stored checkpoint example; switch to K562 for the balanced/correction/output teaching selector."* |
| synthetic | `legacy` → "Stored checkpoint example" | **yes** | same |

That is honest and sufficient. Two residues are noted as **F-6** and **F-7** below.

### 2.2 Softmax ✅ PASS

**Both captions now match the plotted quantity.**

| location | caption at `23546eae` | plotted array | match |
|---|---|---|---|
| `app/page.tsx:651` | *"1. Relative logits for an upward bar chart: **logit − minimum logit** — the smallest becomes 0. Subtracting one shared constant changes no softmax probability, and the tallest bar remains the largest logit."* | `relativeLogitHeights = (logit − max) − min(logit − max) = logit − min(logit)` | ✅ |
| `app/page.tsx:775` | *"**logit − minimum logit**; the smallest becomes zero and ordering is unchanged"* | same array | ✅ |
| `app/page.tsx:653` (stage 2) | *"exp(logit − max); the same position is still highest"* | `exponentials = shiftedLogits.map(Math.exp)` where `shiftedLogits = logit − max` | ✅ |

**Browser evidence.** Across the four `#outputs` stages the **tallest bar is index 68 and the shortest
is index 157 in all four** — i.e. the chart's polarity now matches the caption ("the tallest bar remains
the largest logit"), which is precisely what was contradicted at `3d56b978` (where the caption claimed
the largest became 0 while the tallest bar sat at the peak).

**Argmax preservation, all three presets** (independent NumPy):

| preset | logit | logit − min | exp(logit − max) | probability | expected counts | |
|---|---:|---:|---:|---:|---:|---|
| k562 | 426 | 426 | 426 | 426 | 426 | ALL EQUAL ✅ |
| gm21515 | 341 | 341 | 341 | 341 | 341 | ALL EQUAL ✅ |
| synthetic | 244 | 244 | 244 | 244 | 244 | ALL EQUAL ✅ |

**Does the visualization still imply softmax created the peak?** ✅ **No.** Chart 1 now shows the peak
as the tallest bar in the same place as charts 2–4, so the reader sees the peak *already present* before
softmax; the note *"Softmax sharpens relative differences; it does not invent a new peak"* is now
consistent with the picture rather than contradicted by it.

### 2.3 Loading and release-gate determinism ✅ PASS (with one packaging caveat)

**The main route no longer statically bundles the activation JSONs or the audit artifact.** ✅
`app/page.tsx:1-21` — the four `import … from "./data/*.json"` statements are gone, replaced by
`PRESET_URLS` + `useJsonAsset` runtime fetches.

**Production main-route JavaScript payload** (from `dist/client/_next/static/chunks/`, route identified
by grepping each chunk for its unique heading string):

| route | chunk | raw | gzip |
|---|---|---:|---:|
| **`/`** | `page-DCyFt4mX.js` (contains `ONE CALCULATION`) | **72 KB** | **≈ 21 KB** |
| `/basset` | `page-CnenHlQ0.js` | 216 KB | 0.06 MB |
| `/model-audit` | `page-DeyRNd-9.js` + shared `model-audit-summary-*.js` | 28 KB + 980 KB | |
| `/dilation-trace` | `page-BFiEmK8j.js` (contains `DILATION EVOLUTION LAB`) + shared audit chunk | **6.6 MB** | **2.47 MB** |

**`/` went from 13.4 MB raw / 4.6 MB gzip at `3d56b978` to 72 KB raw / ≈ 21 KB gzip — a ~186× reduction
in route-chunk size.** Browser timings on the dev server improved correspondingly:

```
navigation → load event        197 ms
navigation → JSON ready        1,098 ms   (was: tensor fetches not even starting until ~3,700 ms)
navigation → stem tensor drawn 1,999 ms
```

**Caveat (F-2):** the dev server returns `/data/demos/k562.json` with
`Content-Length: 6,924,468` and **no `Content-Encoding`** — 6.9 MB uncompressed on the wire, plus
980 KB for the audit JSON (7.56 MB of JSON total observed for `/`). The `.f32.gz` tensors are
pre-compressed; these new JSON assets are not. Production compression must be confirmed or the files
pre-gzipped.

**Browser suite determinism:** 4 runs, 4 passes (§1). Fixed.

### 2.4 Dilation lab ✅ PASS — fully fixed

Round-trip verified in the browser at `app/dilation-trace/page.tsx:257`:

| step | coordinate | scope prose | flow |
|---|---|---|---|
| 1. default | 948 | *"Channel 118 and input-aligned coordinate 948 were selected by the **maximize min(ReLU correction, shortcut) inside input-aligned coordinates 558–1557** rule. This selected example teaches the stated criterion…"* | corr 0.3757 + short 0.3024 = 0.6781 |
| 2. slider → 1400 | 1400 | *"**You moved** from the selected **balanced** example at coordinate 948 to coordinate 1400. **This live cell was not selected by that rule, so either path may now be zero.** Choose 'Show selected example' to return."* | corr 0.0000 + short 0.0000 = 0.0000 |
| 3. "Show selected example" | **948** | back to the rule text ✅ | 0.3757 + 0.3024 = 0.6781 ✅ |
| 4. mode → correction | **971** | *"…selected by the **largest ReLU correction**… rule"* ✅ | 0.7806 + 0.0000 = 0.7806 |
| 5. slider → 600 | 600 | *"You moved from the selected **correction** example at coordinate 971 to coordinate 600…"* ✅ | 0.0000 + 0.0000 = 0.0000 |

The prose **explicitly** states the live cell was not chosen by the displayed rule ✅; the return button
restores both the coordinate and the explanation ✅; the mode name and coordinate in the warning track
the current selection ✅. **No stale coordinate or selection-rule text.**

### 2.5 Profile zoom and audit page ✅ PASS (one mouse-affordance regression)

**Profile zooms include the predicted peak on every main-page preset** ✅ — browser-observed window
matches the independently computed peak, and the peak bar is at index 30 of 61 (dead centre) in all
three:

| preset | computed peak (profile pos → input) | browser axis | peak bar |
|---|---|---|---|
| k562 | 426 → **984** | `input 954 … peak at input 984 … input 1014` | 30/61 ✅ |
| gm21515 | 341 → **899** | `input 869 … peak at input 899 … input 929` | 30/61 ✅ |
| synthetic | 244 → **802** | `input 772 … peak at input 802 … input 832` | 30/61 ✅ |

`ProfileLayerView` (`app/page.tsx:728`) likewise now initialises `profilePosition` to the peak.

**CKA prose accurately states the sampling** ✅ (`app/model-audit/page.tsx:170`):

> *"This pilot CKA compares **128 aligned positions** and each layer's **64 highest-variance channels**,
> as stated in the method label—not every tensor entry."*

Matches the artifact's `method` string exactly and matches `scripts/build_model_audit.py:82-102`.

**Layer rows — mouse / Enter / Space:**

| interaction | result |
|---|---|
| Mouse click on the row **button** (first cell) | ✅ `res8` selected |
| **Enter** on a focused row button | ✅ `res1` selected |
| **Space** on a focused row button | ✅ `res5` selected |
| **Mouse click on a non-button cell** of a different row (`512 × 2,066`, 140 × 32 px, `cursor: pointer`) | ❌ **no effect** — selection stayed on `res8` |

**Zero-box focusable layer rows: none remain** ✅ — the zero-box sweep on `/model-audit` returns `[]`,
and the row buttons measure 90 × 32 px. The non-button cell click is logged as **F-5**.

### 2.6 Regression sweep — 4 routes × 3 viewports

| viewport | route | console errors | page/hydration errors | failed requests | horizontal overflow | zero-box focusables |
|---|---|---|---|---|---|---|
| 1440 | `/` | 0 | 0 | 0 | none | none |
| 1440 | `/dilation-trace` | 0 | 0 | 0 | none | none |
| 1440 | `/model-audit` | 0 | 0 | 0 | none | none |
| 1440 | `/basset` | 0 | 0 | 0 | none | none |
| 900 | `/` | 0 | 0 | 0 | none (one `LABEL` 264→250) | none tabbable |
| 900 | `/dilation-trace` | 0 | 0 | 0 | none | none |
| 900 | `/model-audit` | 0 | 0 | 0 | none | none |
| 900 | `/basset` | 0 | 0 | 0 | none | none |
| 390 | `/` | 0 | 0 | 0 | **416 vs 390** (`.topbar`, `.output-branches`) | none tabbable |
| 390 | `/dilation-trace` | 0 | 0 | 0 | none (two inner `reachLane` 313→304) | none |
| 390 | `/model-audit` | 0 | 0 | 0 | **560 vs 390** (CKA grid, 512-column `signedStrip`) | none |
| 390 | `/basset` | 0 | 0 | 0 | none | none |

**Zero console errors, zero page errors, zero hydration errors and zero failed requests across all
twelve combinations.** No new console/network regression from `23546eae`.

Inaccessible controls at ≤ 1000 px are unchanged from V3 (**I-4**, still present): the main page's
`nav` and its "Channel order" `<label>` both compute to `display:none` at 900 px and 390 px.

---

## 3. Findings at `23546eae`

Each finding is tagged with its category: **[SCI]** scientific/numerical correctness · **[TEACH]**
teaching clarity · **[A11Y]** accessibility · **[PERF]** performance · **[TEST]** test reliability ·
**[PKG]** packaging/provenance.

### F-1 · Main-page residual prose does not update when the position slider moves — **NEW**
**Severity: medium · [TEACH] · Blocks video/live demo: YES · Blocks blog/slides/docs: no**

`app/page.tsx:574` (prose) vs `app/page.tsx:581` (the "Slide this dilated kernel" range input).

The `<p>` in `.residual-example-controls` is rendered purely from `selectedExample` and is never
compared with the live `position` state. Browser evidence, K562 Block 1, default `balanced` mode:

| | coordinate readout | merge lane | prose |
|---|---|---|---|
| before | `Shared input coordinate: 948` | `0.3757 + 0.3024` | *"maximize min(ReLU correction, shortcut)… Channel 118, input-aligned coordinate **948**."* |
| after dragging the slider to output 300 | `Shared input coordinate: **313**` | **`0.0000 + 0.0000`** | *"maximize min(ReLU correction, shortcut)… Channel 118, input-aligned coordinate **948**."* — **unchanged** |

This is the same defect class as V3's **I-1**, which `23546eae` fixed on `/dilation-trace` — the fix was
applied to one of the two pages that needed it, exactly as **B-2** was at the previous commit. The
degenerate state is one drag away from the default view of the section titled *"A residual block
combines features across channels and distance."*

**Fix.** Mirror `app/dilation-trace/page.tsx:257`:

```tsx
<p>{position === selectedExample.output_position_zero_based
  ? `${selectedExample.selection_rule}. Channel ${…}, input-aligned coordinate ${…}.`
  : `You moved from the selected ${exampleMode} example at output ${selectedExample.output_position_zero_based + 1}
     to output ${position + 1}. This live cell was not selected by that rule, so either path may now be zero.`}</p>
```

and add a "Return to the selected example" button next to the selector.

### F-2 · New runtime JSON assets are served uncompressed — **NEW**
**Severity: medium · [PERF] · Blocks anything: no**

`app/page.tsx:21` + `public/data/demos/*.json`. Measured on the dev server:

```
GET /data/demos/k562.json        Content-Length: 6,924,468   Content-Encoding: (none)
GET /data/model-audit-summary.json  Content-Length: 1,001,752   Content-Encoding: (none)
```

Browser-observed totals for `/`: **7.56 MB of JSON, 0 bytes of it compressed**. The equivalent bundled
chunks at `3d56b978` were served gzipped (~2.5 MB + 0.3 MB). Every `.f32.gz` tensor in the same
directory tree *is* pre-compressed, so the project already has the pattern.

The critical-path win is real and unaffected (JSON-ready at 1,098 ms vs. fetches not starting until
~3,700 ms before), but total transfer is currently worse unless the production host compresses.

**Fix.** Ship `k562.json.gz` / `gm21515.json.gz` / `synthetic.json.gz` / `model-audit-summary.json.gz`
and reuse the existing gzip-aware path in `app/tensor-loader.ts`, or verify and document that the
deployment target applies gzip/brotli to `application/json`.

### F-3 · The main route now server-renders an empty loading screen — **NEW**
**Severity: medium · [PERF]/[PKG] · Blocks blog/SEO/social preview: YES · Blocks video/slides: no**

`app/page.tsx:923` returns `<main className="route-loading">` before any content when `demo` or
`auditCheckpoint` is null, which is always true during SSR. `curl http://localhost:4321/` now yields, as
the entire visible text of a 24 KB document:

```
Sequence CNN Explainer   Loading verified checkpoint data…
```

At `3d56b978` the full narrative was server-rendered. Consequences for a published artifact: no content
for search indexing, no social/link preview text, nothing for readers with JS disabled or blocked, and
nothing in a print/PDF capture that runs before hydration. `/dilation-trace`, `/model-audit` and
`/basset` are unaffected (they still import their data statically).

**Fix.** Render the static narrative shell (headings, prose, architecture map, the cross-model frame)
server-side and gate only the data-dependent panels on the fetch, instead of gating the whole `<main>`.

### F-4 · The activation JSONs are now duplicated with no synchronisation path — **NEW**
**Severity: medium · [PKG]/[SCI-risk] · Blocks anything today: no · Silent-divergence hazard**

`23546eae` added `public/data/demos/{k562,gm21515,synthetic}.json` and
`public/data/model-audit-summary.json`. I verified with `cmp` that all four are **byte-identical** to
their `app/data/` counterparts today. But:

- `/` fetches the **`public/`** copies (`app/page.tsx:21`).
- `/dilation-trace` still `import`s `../data/k562-peak-activations.json`, and `/dilation-trace` and
  `/model-audit` still `import` `../data/model-audit-summary.json`.
- **No script writes the `public/` copies.** `scripts/build_model_audit.py:266` defaults its output to
  `app/data/model-audit-summary.json`; `:32` reads `app/data/{preset}-activations.json`;
  `scripts/export_basset_demo.py:290` likewise targets `app/data/`. The README's rebuild commands
  therefore update only `app/data/`.

A future re-export silently leaves `public/data/demos/` stale, at which point `/` and `/dilation-trace`
would display **different numbers for the same checkpoint** with no error. `public/data` also grew
**110 MB → 131 MB** purely from the duplication.

**Fix.** Make `public/data/demos/*` build-generated (a `prebuild` copy step or a symlink) and add a unit
test asserting `cmp`-equality of each pair, or migrate `/dilation-trace` and `/model-audit` to the same
runtime fetch and delete the `app/data` copies from the client graph.

### F-5 · Model-audit layer rows advertise a click target that no longer works — **REGRESSION**
**Severity: medium-low · [A11Y]/[TEACH] · Blocks anything: no**

`app/model-audit/page.tsx:209` moved the click handler from the row `<div>` onto a `<button>` in the
first cell — which correctly fixed V3's **I-3** (keyboard) — but
`app/model-audit/page.module.css:8` still carries `.layerTable>div{…cursor:pointer}` and
`.layerTable>div:hover>*{background:#f8efcf}` for the **whole row**.

Browser evidence: clicking the `512 × 2,066` span in the `res3` row (140 × 32 px, computed
`cursor: pointer`, row highlights on hover) produced **no change** — the selection stayed on `res8`.
Before `23546eae` that click worked. So the row is now keyboard-correct and mouse-dead outside its
90 px first cell, while still signalling "click anywhere" via cursor and hover.

**Fix.** Either restore an `onClick` on the row `<div>` alongside the button (harmless duplication), or
drop `cursor:pointer` and the row-wide hover so only the button reads as interactive.

### F-6 · The residual selection caveat no longer describes the selected example
**Severity: low · [TEACH] · Blocks anything: no**

`app/page.tsx:587` still reads *"This is a useful **high-activation** example, not a claim that this
channel is the most biologically important."* That is accurate for the `output` mode only. Under
`balanced` the cell is chosen to maximize `min(correction, shortcut)`; under `correction` it maximizes
the ReLU correction. The sentence now sits directly under prose that states the actual rule, and
contradicts it in two of three modes.

**Fix.** Change to *"This cell was chosen by the stated rule for visibility, not as a claim of
biological importance."*

### F-7 · Non-K562 presets can still open on a degenerate merge, unwarned
**Severity: low · [TEACH] · Blocks anything: no**

Because `example_modes` exists only for K562 (verified: 8/8 vs 0/8 and 0/8), gm21515 and synthetic fall
back to the stored `trace`. Browser, Block 1: gm21515 → `0.3795 + 0.7102` (fine); **synthetic →
`0.0000 + 1.0697`** (zero learned correction). The fallback prose is honest about *which* example is
shown but does not warn that it may show a zero path.

**Fix.** Re-export `example_modes` for both presets (the exporter already supports it —
`scripts/run_chrombpnet_checkpoint.py:62-100` is preset-agnostic), or add one clause to the fallback
prose.

### F-8 · Switching presets replaces the whole page with a loading screen
**Severity: low · [TEACH]/[PERF] · Blocks video: minor**

`app/page.tsx:923`. Browser-observed: selecting GM21515 or synthetic sets `demo` to `null` (the hook
returns `null` while the URL and the loaded URL disagree), so the entire `<main>` unmounts to
*"Loading verified checkpoint data…"*, discarding scroll position and every panel's local state. At
`3d56b978` the switch was instantaneous. On screen this is a full-page white flash mid-narrative.

**Fix.** Keep the previously loaded demo rendered while the next one is in flight (retain the last
non-null value and show an inline "switching checkpoint…" badge).

### Carried over from V3, retested and unchanged

| V3 ref | Finding | Category | Evidence at `23546eae` |
|---|---|---|---|
| **I-4** | Below 1000 px the main page hides its `nav` and its "Channel order" control, leaving `/dilation-trace` with no inbound link from `/` | [A11Y] | `app/globals.css:21, 29` unchanged; at 900 px and 390 px both compute to `display:none` |
| **I-6** | Basset K562 rank 13 shown but the 0.54–0.97 / median 0.80 distribution across all 164 targets is not | [TEACH] | `app/basset/page.tsx` untouched by this commit |
| **I-7** | `Producing Channel 118` beside `immutable ID 117` (zero-based where every other surface is one-based) | [TEACH] | `app/page.tsx:586` unchanged; browser-confirmed live text |
| **I-8** | Mobile horizontal overflow: `/` 416 vs 390, `/model-audit` 560 vs 390 | [A11Y] | reproduced identically at 390 px |
| **I-9** | `MiniBars` renders 160 bars under captions reading "512 pooled features" | [TEACH] | `app/page.tsx:610` `count = 160` unchanged; 160 `<i>` counted in the DOM |
| **I-10** | Profile probabilities quantized to 1 × 10⁻⁶ (590/1000 distinct); `Σ = 0.999985` beside "sum to 1" | [SCI] | data files unchanged |
| **I-11** | Four node tests are pure source-regex; `app/model-analysis.ts` still has no direct test; `verify_basset_adapter.py` still asserts a self-reported error field | [TEST] | partly improved — one real browser test added at `:48`; the four regex tests remain and two gained assertions |
| **I-12** | `scripts/build_model_audit.py:27` hard-codes `/private/tmp/model.chrombpnet_nobias.fold_0.ENCSR960KGO.h5` | [PKG] | unchanged |
| **P-3** | `public/data` size | [PERF] | **worse**: 110 MB → 131 MB (see F-4) |
| **P-5** | Basset `k562Rank` derived from a 15-entry slice; would render "rank 0" if K562 fell outside | [PKG] | `app/basset/page.tsx:332` unchanged |
| **P-7** | Synthetic preset's planted `TGACTCA` (AP-1) site undisclosed; README never mentions the preset | [SCI-risk] | unchanged — `grep -c synthetic README.md` = **0** |
| **P-8** | `VectorCanvas` per-strip normalization; coral = base A *and* positive; gain clipping undisclosed | [TEACH] | unchanged |
| **P-1** (partial) | `/dilation-trace` still bundles 6.6 MB raw / 2.47 MB gzip; `/model-audit` still bundles the 980 KB audit chunk | [PERF] | measured from `dist/client` |

---

## 4. Disposition of every V3 release-blocking and medium finding

| V3 ref | Finding | Category | Disposition | Evidence |
|---|---|---|---|---|
| **B-1** | `npm test` red and non-deterministic | [TEST] | **FIXED** | 4/4 runs pass (1 cold, 1 under CPU load); `explainer.spec.ts:20` timeout 30 s; new sentinel-gated test at `:48` |
| **B-2** | Main-page residual default degenerate in 5/8 blocks | [SCI]/[TEACH] | **FIXED** | 8/8 blocks both paths > 0 under `balanced`; 24/24 sum identities hold; all 24 reconcile with raw tensors |
| **B-3** | Softmax step-1 caption contradicts its chart | [TEACH] | **FIXED** | captions at `:651` and `:775` now say `logit − minimum logit`; tallest bar index 68 in all four stages |
| **I-1** | `/dilation-trace` scope note goes stale | [TEACH] | **FIXED** | 5-step round trip verified; explicit "was not selected by that rule" warning; Show-button restores |
| **I-2** | Profile zoom hard-coded to 470–530, missing the peak | [TEACH] | **FIXED** | peak-centred on all three presets; browser windows match computed peaks exactly |
| **I-3** | `/model-audit` rows focusable but zero-box and keyboard-dead | [A11Y] | **PARTLY FIXED** | zero-box sweep now `[]`; Enter ✅ Space ✅ button-click ✅; **non-button cell click regressed** → **F-5** |
| **I-4** | Channel-order control and nav hidden below 1000 px | [A11Y] | **STILL PRESENT** | `app/globals.css:21, 29` unchanged; confirmed at 900 px and 390 px |
| **I-5** | CKA prose said "whole representations" | [SCI] | **FIXED** | `app/model-audit/page.tsx:170` now states 128 positions / 64 highest-variance channels |
| **I-6** | Basset K562 rank shown without distribution context | [TEACH] | **STILL PRESENT** | `app/basset/page.tsx` untouched |
| **I-7** | Zero-based "immutable ID" beside one-based "Channel" | [TEACH] | **STILL PRESENT** | `app/page.tsx:586`; live text `display rank 118 · immutable ID 117` |
| **I-8** | Mobile horizontal overflow on `/` and `/model-audit` | [A11Y] | **STILL PRESENT** | 416 vs 390 and 560 vs 390 reproduced |
| **I-9** | 160 bars labelled "512 pooled features" | [TEACH] | **STILL PRESENT** | `app/page.tsx:610`; 160 bars counted |
| **I-10** | Profile probabilities quantized to 1e-6 | [SCI] | **STILL PRESENT** | data unchanged |
| **I-11** | Test-quality gaps (source-regex tests, untested `model-analysis.ts`, self-reported Basset gate) | [TEST] | **PARTLY FIXED** | +1 genuine browser test; 4 regex tests remain; `model-analysis.ts` still untested |
| **I-12** | `build_model_audit.py` hard-coded `/private/tmp` checkpoint path | [PKG] | **STILL PRESENT** | `scripts/build_model_audit.py:27` |
| **P-1** | ~7.4 MB gzip of JS on `/` | [PERF] | **FIXED for `/`; STILL PRESENT for `/dilation-trace`** | `/` chunk 72 KB raw / ≈21 KB gzip; `/dilation-trace` 6.6 MB raw / 2.47 MB gzip |
| **P-3** | 110 MB in `public/data` | [PERF] | **REGRESSED** | now 131 MB from byte-identical duplication (F-4) |
| — | *(new)* Main-page residual prose stale on slider move | [TEACH] | **NEW — F-1** | prose fixed at coordinate 948 while merge shows `0.0000 + 0.0000` at 313 |
| — | *(new)* Runtime JSON served uncompressed | [PERF] | **NEW — F-2** | 7.56 MB JSON, no `Content-Encoding` |
| — | *(new)* `/` server-renders only a loading screen | [PERF]/[PKG] | **NEW — F-3** | `curl` returns 24 KB whose entire visible text is the loading line |
| — | *(new)* Duplicated data with no sync path | [PKG]/[SCI-risk] | **NEW — F-4** | `cmp`-identical today; no script writes the `public/` copies |
| — | *(new)* Selection caveat contradicts the stated rule | [TEACH] | **NEW — F-6** | `app/page.tsx:587` |
| — | *(new)* gm21515/synthetic can open degenerate, unwarned | [TEACH] | **NEW — F-7** | synthetic Block 1 = `0.0000 + 1.0697` |
| — | *(new)* Preset switch blanks the page | [TEACH]/[PERF] | **NEW — F-8** | `showedLoadingScreen = true` on both switches |

**Not retested** (unchanged files, outside the focused scope, and no reason to expect drift): V3's
long-tail P-4 (`ActivationLogo` height ceiling, still latent — no motif artifact exists), P-6 (schema
version type mismatch), and the full §7 numerical re-derivation of the Basset chain and audit artifact.
I did re-verify the three ChromBPNet items most likely to be affected by this commit (residual
identities, softmax chain, profile peaks) and all reproduced exactly.

---

## 5. Category summary

**Scientific / numerical correctness — clean.** No error found at this commit. All 24 K562 residual
example modes reconcile with the raw tensors and with the block-output tensor (0 mismatches); the
softmax chain preserves argmax across all five displayed quantities on all three presets; the CKA prose
now matches the computation. The two open [SCI] items are **I-10** (1e-6 probability quantization, which
makes three on-page identity statements false at the sixth decimal) and **P-7** (undisclosed planted
motif in the synthetic preset) — neither changes a displayed number materially.

**Teaching clarity — good, one item to fix before recording.** B-2, B-3, I-1 and I-2 are all resolved
and the residual lesson now works as intended. **F-1** is the one reachable contradiction left: one drag
of the main page's most prominent slider produces `0.0000 + 0.0000` under prose asserting a balanced
merge. F-6, F-7, I-6, I-7 and I-9 are smaller wording/labelling residues.

**Accessibility — net improvement, one regression.** Zero-box focusables are gone from `/model-audit`
(I-3), and Enter/Space both work. But the row is now mouse-dead outside its first cell while still
showing `cursor: pointer` and a row-wide hover (**F-5**), and **I-4** (controls hidden below 1000 px)
and **I-8** (mobile overflow) are unchanged.

**Performance — large win on the critical path, two new packaging costs.** `/` dropped from 13.4 MB to
72 KB of route JavaScript and reaches interactive data in ~1.1 s. Against that: the fetched JSON is
uncompressed (**F-2**), `/` no longer server-renders content (**F-3**), `public/data` grew 110 → 131 MB
from duplication (**F-4**), and `/dilation-trace` still ships 2.47 MB gzip.

**Test reliability — fixed.** Four consecutive green runs including a cold run and one under CPU load;
the previously failing assertion has an explicit 30 s budget and the new test separates loading from
numeric assertions. The deeper coverage gaps from V3's I-11 (source-regex tests, no direct test of
`app/model-analysis.ts`, a self-reported Basset verification field) are unchanged.

---

## 6. Recommended order of work before publication

1. **F-1** — mirror the dilation lab's live-state prose on `app/page.tsx:574`, and add a "return to the
   selected example" control. *(the only item I would block a recording on)*
2. **F-4** — make `public/data/demos/*` build-generated and add a `cmp`-equality test, so the two copies
   cannot silently diverge.
3. **F-3** — server-render the narrative shell; gate only the data panels.
4. **F-5** — restore the row-wide click on `/model-audit`, or remove the pointer/hover affordance.
5. **F-2** — pre-gzip the four JSON assets or confirm host compression.
6. **F-6 / F-7 / I-7** — three one-line wording fixes (selection caveat, degenerate-fallback warning,
   one-based immutable ID).
7. **I-4 / I-8** — restore the channel-order control below 1000 px and wrap the two overflowing
   `/model-audit` panels in `overflow-x: auto`.
8. **I-6, I-9, I-10, I-11, I-12, P-3, P-5, P-7** — as scheduled in `AUDIT-INDEPENDENT-V3.md` §9.
