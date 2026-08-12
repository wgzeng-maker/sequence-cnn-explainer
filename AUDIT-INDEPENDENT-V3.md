# Independent Audit V3 — Sequence CNN Explainer

**Repository root:** `/Users/zengweiguo/Documents/sequence_model_explainer` (confirmed with `git rev-parse --show-toplevel`)
**Commit:** `3d56b978` *Fix audited explainer teaching and verification gaps*
**Branch:** `codex/dilation-tensor-evolution-demo`
**Working tree:** clean (`git status --short` produced no output, before and after this audit)

**Method.** Source read of all four route components, both shared modules, all six Python/Node scripts,
both docs, the unit suite and the Playwright suite; independent NumPy re-derivation of the ChromBPNet
stem, all eight residual blocks, both output heads, the complete audit artifact, and the complete
Basset forward chain from the committed binaries; direct execution of `app/model-analysis.ts`; and live
browser interaction with all four routes at three viewport widths using a headless Chromium driven
outside the project's own test suite.

**Order of work.** All findings in §§2–7 were recorded before `AUDIT-INDEPENDENT-V2.md` was opened.
V2 was read only afterwards, to build §8.

**Nothing in the application was modified.** The only file this audit writes is this report.

Legend — **[VERIFIED]** reproduced numerically or in a browser · **[JUDGMENT]** argued design/usability
position · **[LATENT]** real defect not currently reachable by a reader · **[PERF]** performance.

---

## 1. Commands executed and results

| Command | Result |
|---|---|
| `git rev-parse --show-toplevel` | `/Users/zengweiguo/Documents/sequence_model_explainer` ✅ |
| `git status --short` | empty — clean tree ✅ |
| `git log -1 --oneline` | `3d56b978 Fix audited explainer teaching and verification gaps` ✅ |
| `git rev-parse --abbrev-ref HEAD` | `codex/dilation-tensor-evolution-demo` |
| `npm test` — **run 1** | **EXIT 1** ❌ · lint ✅ · typecheck ✅ · build ✅ · unit 19/19 ✅ · **Playwright 5/6, 1 failed** |
| `npx playwright test -g "dilation lab opens" --repeat-each=2` | **2 of 2 failed** ❌ |
| `npx playwright test` (after clearing `.vinext/deps`, warm machine) | 6/6 passed ✅ |
| `npm test` — **run 2** | EXIT 0 ✅ · 19 unit + 6 browser passed |
| `npm run verify:python` | EXIT 0 ✅ — `model-analysis invariants: PASS`, `Basset adapter verification passed` |
| `npx eslint . --ignore-pattern dist --ignore-pattern .next` | 0 problems ✅ |
| Browser: `/`, `/dilation-trace`, `/model-audit`, `/basset` at 1440 / 820 / 390 px | 0 console errors, 0 page errors, 0 hydration errors, 0 failed requests ✅ |

The failing test is `tests/browser/explainer.spec.ts:20` — the one test that guards audit question B1.
Failure cause and reproduction analysis are in **B-1** below.

### Release recommendation

> ## NOT READY — for the public narrative (blog / slides / video)
> ## READY WITH CAVEATS — for internal and expert review

This is **not** a verdict on the science. Every architectural and numerical claim I was able to check
independently reproduced exactly, often to float32 precision, across three ChromBPNet presets and the
whole Basset chain (§7). Three specific defects block public release, and all three are hours of work:

1. `npm test`, the documented one-command release gate, is **red and non-deterministic** (**B-1**).
2. On the **main explainer** — the page that will drive the narrative — the default residual example
   shows the learned transform path contributing **exactly 0.0000** in 5 of 8 blocks of the default
   checkpoint (**B-2**). The section is titled *"A residual block combines features across channels and
   distance."*
3. The softmax panel's step-1 caption states a formula that is **not** the one plotted (**B-3**), and
   the mismatch is visible at a glance.

---

## 2. Release-blocking findings

### B-1 · `npm test` is a red, non-deterministic release gate
**Severity: high · Classification: verified defect (+ performance root cause) · [VERIFIED] [PERF]**

`package.json:11` · `tests/browser/explainer.spec.ts:17-27` · `playwright.config.ts`

My first invocation of `npm test` at this commit exited **1**. A targeted re-run of the same test with
`--repeat-each=2` failed **2 of 2**. A later run on a warmer machine passed 6/6, and a second full
`npm test` exited 0. The gate therefore passes or fails depending on machine load, not on code.

**Evidence.** Playwright's saved failure snapshot
(`test-results/explainer-dilation-lab-ope-4d268-.../error-context.md`) shows the page rendered but the
data never arrived: `"loading full tensor…"`, `"Loading real input-feature vectors…"`,
`"calculated logit …"`, `"0 of 732 shown cells receive a positive correction"`. The assertion at
`explainer.spec.ts:20` uses Playwright's default **5 s** `toBeVisible` timeout on
`[data-testid="residual-numeric-flow"]`, which only mounts after `useRawTensors`
(`app/dilation-trace/page.tsx:37-51`) resolves nine tensor fetches.

**Root cause is a real payload problem, not a test-config nit.** I measured the production build:

| `dist/client/_next/static/chunks/…` | raw | gzip |
|---|---:|---:|
| `page-DRAWMZl0.js` (the `/` route chunk) | **13.4 MB** | **4.6 MB** |
| `k562-peak-activations-kc3iDmhw.js` | 6.87 MB | 2.5 MB |
| `model-analysis-l00gkAR_.js` (the audit JSON) | 1.00 MB | 0.3 MB |

Loading `/` transfers **≈ 7.4 MB gzipped / ≈ 21 MB uncompressed of JavaScript** before a single pixel
of tensor is fetched, because `app/page.tsx:4-7` statically imports all three activation JSONs plus the
audit artifact into a `"use client"` module. In the browser I measured tensor fetches on
`/dilation-trace` not *starting* until **t ≈ 3.7 s** after navigation; the fetches themselves take
35–75 ms each. Time-to-`residual-numeric-flow` on a warm dev server measured **2.40 s / 2.84 s / 2.54 s**
over three runs — comfortably inside 5 s, and comfortably outside it under load.

**Likely reader misunderstanding.** None directly — but a maintainer will read a green `npm test` as
evidence that B1 (balanced default merge) holds, and will read a red one as flake and re-run it. Both
readings are wrong for different reasons.

**Recommended correction.** (a) Raise this assertion's timeout explicitly
(`await expect(flow).toBeVisible({ timeout: 30_000 })`) and add a `waitForFunction` on tensor arrival so
the failure mode is legible; (b) move the three activation JSONs and the audit artifact out of the
client bundle and fetch them like the `.f32.gz` tensors — the loader in `app/tensor-loader.ts` already
does exactly this correctly for binaries; (c) README currently says "five Playwright browser tests";
there are six.

---

### B-2 · The main explainer's default residual example has a zero learned correction in 5 of 8 blocks
**Severity: high · Classification: verified defect + scientific-risk judgment · [VERIFIED]**

`app/page.tsx:519-528` (`ResidualStory` reads `demo.residual_kernel_demos[block-1].trace`, not
`.example_modes`) · rendered at `app/page.tsx:580-582`

The `/dilation-trace` route was fixed to open on a balanced merge. The **main page was not**. It still
selects its example from the legacy `trace` field, whose stated rule is *"output channel with the
largest peak activation in this block; position of that channel's peak"* — a rule that
systematically finds the cell the shortcut carried forward untouched.

I recomputed every block's default from the raw tensors:

| Block (K562 default preset) | ReLU correction | shortcut | shown as |
|---|---:|---:|---|
| **1** | **0.00000** | 1.18282 | `0.0000 + 1.1828 = 1.1828` |
| **2** | **0.00000** | 1.18282 | `0.0000 + 1.1828 = 1.1828` |
| **3** | **0.00000** | 1.18282 | `0.0000 + 1.1828 = 1.1828` |
| **4** | 1.41818 | **0.00000** | `1.4182 + 0.0000 = 1.4182` |
| 5 | 1.96102 | 1.03930 | ✅ both paths active |
| **6** | **0.00000** | 3.00032 | `0.0000 + 3.0003 = 3.0003` |
| **7** | **0.00000** | 3.00032 | `0.0000 + 3.0003 = 3.0003` |
| 8 | 6.57520 | 1.92982 | ✅ both paths active |

Synthetic preset: **7 of 8** degenerate. GM21515: 3 of 8 degenerate.

**Browser evidence.** Screenshot of `#residual` at default state, Block 1:

```
TRANSFORM PATH   3 tap sums 0.098 + -0.229 + 0.002 → conv sum -0.1289
                 + bias -0.1295 → before ReLU -0.2584 → after ReLU 0.0000
SHORTCUT PATH    1.1828
                 transformed value 0.0000 + preserved shortcut 1.1828 = 1.1828
```

directly beneath the heading **"A residual block combines features across channels and distance"** and
above the takeaway *"A residual kernel can give different weights to motif-like features in different
channels … allowing learned cooperation or competition."*

**Likely reader misunderstanding.** A reader stepping 1 → 2 → 3 sees the dilated convolution produce
literally nothing three times in a row, then sees it produce something only when the shortcut is zero
(block 4). The natural conclusion is that dilated residual blocks are pass-throughs and the whole
mechanism the page is teaching is decorative. This is the same class of defect the project already
solved on `/dilation-trace`; the fix simply was not carried across to the primary narrative.

**Recommended correction.** Point `ResidualStory` at `kernel.example_modes.balanced` (which exists and
is correct for all 8 blocks of all 3 presets — I verified all 24 against the raw tensors, §7), and
surface the same three-way `balanced / correction / output` selector the dilation lab has, with its
declared selection rules. The exporter (`scripts/run_chrombpnet_checkpoint.py:62-100`) already computes
everything needed.

---

### B-3 · The softmax panel's step-1 caption states the opposite of what the chart plots
**Severity: medium-high · Classification: verified defect · [VERIFIED]**

`app/page.tsx:607-609` computes:

```ts
const shiftedLogits = logits.map(v => v - maximumLogit);
const minimumShiftedLogit = Math.min(...shiftedLogits);
const relativeLogitHeights = shiftedLogits.map(v => v - minimumShiftedLogit);   // == logit − min(logit)
```

`app/page.tsx:628` then labels the chart of `relativeLogitHeights`:

> **1. Relative logits: logit − maximum logit** — *the largest becomes 0; subtracting one constant
> changes no softmax probability. Bars preserve the original ordering.*

The plotted quantity is `logit − min(logit)`, not `logit − max(logit)`. Measured in the live DOM:

| quantity | claimed | measured on page |
|---|---:|---:|
| bar value at `argmax(logit)` | 0 | **5.9223** (tallest bar, 75 px, index 68 of 160) |
| bar value at `argmin(logit)` | −5.9223 | **0.0000** |

The same mislabel appears a second time at `app/page.tsx:707-709` / `:730`, where `ProfileLayerView`
renders the identical array under *"AFTER CONVOLUTION · SHIFTED FOR DISPLAY … logit − maximum logit."*

**Likely reader misunderstanding.** The caption tells the reader the tallest bar is the *lowest* logit
and that the peak has collapsed to zero. The picture shows the opposite. A reader who trusts the
caption reads the chart's polarity backwards; a reader who trusts the picture concludes the caption's
arithmetic is unreliable and discounts the (correct and valuable) shift-invariance explanation beside
it.

**Recommended correction.** Either change the label to `logit − minimum logit` (with "only differences
matter; softmax ignores any constant shift"), or plot `shiftedLogits` on a descending axis and keep the
current label. Do not leave the two disagreeing. Note that the *substantive* claim on the same panel —
*"raw logits, shifted logits, exponentials, probabilities, and expected counts all have the same peak
position"* — **is true**: I confirmed `argmax = 426` in all four arrays.

---

## 3. Improvements recommended before the blog / slides / video

### I-1 · `/dilation-trace` scope note goes stale and can assert a balanced merge over a degenerate one
**Severity: medium · Classification: verified defect · [VERIFIED]**

`app/dilation-trace/page.tsx:257` prints the *example's* coordinate and the sentence *"Balanced merge
maximizes the weaker of correction and shortcut, so the default visibly teaches their addition."*
`:236-253` computes the displayed flow at the **current** `globalCenter`, which the reader can move
freely with the "Tracked input coordinate" slider.

Browser evidence — stage 8, mode `balanced`, slider moved to 1400:

```
scope note : "Channel 160 and input-aligned coordinate 948 were selected by a declared rule …
              Balanced merge maximizes the weaker of correction and shortcut …"
flow shown : three tap sums 3.0768 + bias 1.1023 → ReLU correction 4.1790 + shortcut 0.0000 = 4.1790
slider     : 1400
```

The prose names coordinate 948 while the arithmetic is at 1400, and asserts a balanced merge while the
shortcut is exactly zero.

**Correction.** Make the note reflect live state: show the example coordinate only while
`globalCenter === example.input_aligned_coordinate_one_based`, otherwise switch to "you have moved off
the selected example — this cell was not chosen by any rule."

### I-2 · The profile zoom on `/` is hard-coded to a window that excludes the predicted peak
**Severity: medium · Classification: usability + scientific-risk judgment · [VERIFIED]**

`app/page.tsx:617` fixes the zoom to profile positions **470–530** (input 1,028–1,088). The K562
profile peak is at profile position **426** (input 984) — 44 bases outside the window. In the captured
screenshot, panel 4 above shows a sharp peak while the "61-position zoom" directly beneath it renders
an almost flat blue block labelled *"expected cuts per base."* `ProfileLayerView`'s default
`profilePosition = 500` (`app/page.tsx:704`) has the same problem.

**Correction.** Default both to `argmax(profile_signal.position_max)` for the active preset, or add a
"jump to predicted peak" control.

### I-3 · `/model-audit` layer-table rows are keyboard-focusable but have no box and do not activate
**Severity: medium · Classification: verified defect (accessibility) · [VERIFIED]**

`app/model-audit/page.tsx:209` renders each layer row as `<div role="button" tabIndex={0} onKeyDown={…}>`,
but `app/model-audit/page.module.css:8` sets `.layerTable>div{display:contents}`. A `display:contents`
element generates no box.

Measured: all **9** rows return `getBoundingClientRect() = 0 × 0`. Focusing row index 8 (`res8`) and
pressing <kbd>Enter</kbd> left the selection on `stem` — the default. It is the only zero-box focusable
element on any of the four routes (`/`, `/dilation-trace`, `/basset` all return 0).

**Correction.** Move `role`/`tabIndex`/`onKeyDown` onto the first cell of each row, or drop
`display:contents` and use `grid-template-columns: subgrid` / an actual `<table>` with
`<button>` cells.

### I-4 · Below 1000 px the main page silently removes its navigation and its channel-order control
**Severity: medium · Classification: usability judgment · [VERIFIED]**

`app/globals.css:21` — `@media(max-width:1000px){.topbar nav{display:none} …}`
`app/globals.css:29` — `@media(max-width:1000px){… .topbar-controls label:last-child{display:none}}`

Measured at 900 px: `nav` → `display:none`; the "Channel order" `<label>` → `display:none`. No
hamburger, no in-body substitute. Consequences:

- The **global channel-order** control — one of the release's headline features, and the mechanism
  behind the "immutable channel registry" story — is unreachable on any laptop below 1000 px.
- **`/dilation-trace` has exactly one inbound link in the whole app from `/`** (the hidden nav). Below
  1000 px the dilation lab is reachable only via `/model-audit`, which is itself reachable only via a
  link inside the `ActivationMotifState` empty state — i.e. only if the reader happens to switch the
  kernel view to "Activation motif."

**Correction.** Keep the channel-order select in the body (near the `channel-order-note` banner, which
is already always visible), and add body-level links to all three sibling routes.

### I-5 · The CKA panel's prose overstates what was computed
**Severity: medium · Classification: scientific-risk judgment · [VERIFIED]**

`app/model-audit/page.tsx:170` states: *"CKA compares **whole representations** after aligning
positions."* The artifact's own `method` string, shown in small type beside the heading, says:
*"linear CKA on **128 shared positions and 64 highest-variance channels per layer**"*
(`scripts/build_model_audit.py:82-102`, seeded `default_rng(1729)`).

This is not a whole-representation comparison: it uses 12 % of positions and a **different**
top-variance channel subset for each layer, which is feature selection inside the similarity measure.
I recomputed linear CKA on all 1,074 aligned positions × all 512 channels; the pattern is qualitatively
identical (monotone decay away from the diagonal) but individual entries differ by up to **0.222**:

| pair | artifact (shown to 2 d.p.) | full 1074 × 512 |
|---|---:|---:|
| stem × res3 | 0.66 | **0.75** |
| res1 × res7 | 0.35 | **0.21** |
| res4 × res6 | 0.71 | **0.60** |

**Correction.** Change "whole representations" to "a 128-position subsample of each layer's 64 most
variable channels", and either widen the sample or print one decimal place.

### I-6 · Basset K562 is correctly ranked but not distributionally contextualized
**Severity: medium · Classification: scientific-risk judgment · [VERIFIED]**

`app/basset/page.tsx:416`. The page shows the top 12 predictions plus K562 pinned with "· rank 13", and
that rank is **exactly right** — I reconstructed all 164 logits from `outputs.dense2_activations`,
`output_biases` and `output_weights.f32.gz`, and K562's true global rank among 164 is **13**
(prob 0.9060933, matching the stored value to 3 × 10⁻¹⁰).

What is missing is the shape of the output distribution on this sequence:

| statistic across all 164 targets | value |
|---|---:|
| max (H1-hESC) | 0.9734 |
| **median** | **0.8010** |
| **min** | **0.5416** |
| K562 | 0.9061 |

Every one of 164 cell types gets ≥ 0.54. A reader sees a bar chart of thirteen bars between 0.906 and
0.973 and can easily read K562's 0.906 as strong cell-type-specific evidence, when it is roughly one
decile above the median of an output layer that is high everywhere for this sequence. The page's
existing caveat (*"model outputs for one sequence, not experimental measurements"*) is necessary but
does not carry this.

**Correction.** Add the median/min of all 164 beside the chart, or plot the full 164-value distribution
with K562 marked.

### I-7 · Channel identity is printed zero-based in one place and one-based everywhere else
**Severity: low-medium · Classification: verified defect · [VERIFIED]**

`app/page.tsx:565` renders `display rank {displayRank(...)} · immutable ID {outputChannel}` — the
**zero-based** index — immediately beside `Producing Channel {outputChannel + 1}`. The rendered result
is literally:

> **Producing Channel 205** — display rank 205 · immutable ID 204

Everywhere else, "immutable ID" is one-based: `app/page.tsx:330` (`Channel {filterNumber + 1}`),
`app/dilation-trace/page.tsx:159` (`immutable IDs: … channel + 1`), `app/model-audit/page.tsx:217`
(`Channel {row.id_zero_based + 1}`). A reader comparing panels will conclude the registry is off by one.

**Correction.** Print `Channel {outputChannel + 1}` under the "immutable ID" label, or state the
zero-based convention explicitly at that one site.

### I-8 · Mobile horizontal overflow on `/` and `/model-audit`
**Severity: low · Classification: usability judgment · [VERIFIED]**

At 390 px viewport:

| route | `documentElement.scrollWidth` vs `clientWidth` | worst offenders |
|---|---|---|
| `/` | **416** vs 390 | `.topbar` (399), `.output-branches` (387 in 332) |
| `/model-audit` | **560** vs 390 | `_similarityGrid` (398 in 332), `_signedStrip` (512 in 294) ×2 |
| `/dilation-trace` | 390 vs 390 ✅ | — |
| `/basset` | 390 vs 390 ✅ | — |

The 512-column `SignedStrip` elements have no `overflow-x` container, so they push the whole page.

### I-9 · `MiniBars` labels 160 bars as "512 pooled features"
**Severity: low · Classification: usability judgment · [VERIFIED]**

`app/page.tsx:589-601, 640-642`. The count-head panels bin 512 channels into 160 bars (per-bin max)
under captions reading "512 pooled features" and "multiply by matching dense weights · same immutable
IDs". Counted in the DOM: 160 `<i>` elements. Each bar is the extremum of ~3 channels, so bar *k* does
not correspond to channel *k* — which quietly undercuts the "same immutable IDs" claim in the same
sentence.

### I-10 · Shipped profile probabilities are quantized to 1 × 10⁻⁶
**Severity: low · Classification: verified defect (data fidelity) · [VERIFIED]**

`tensors.profile_probabilities.position_max` holds only **590 distinct values out of 1,000** for K562
(726/1000 for GM21515, 677/1000 for synthetic) — the array is rounded to six decimals. Consequences:

- **415 of 1,000** positions have a different rank under the stored probabilities than under the stored
  logits. The *peak* claim on `app/page.tsx:635` still holds exactly (argmax = 426 in logits,
  probabilities and expected counts); the broader "preserves ordering" reading does not hold of the
  shipped numbers.
- `Σ probabilities = 0.9999850`, beside a caption reading *"all 1,000 values sum to 1"*
  (`app/page.tsx:632`).
- `probability × predicted_total_count` differs from the stored `profile_signal` by up to 1.5 × 10⁻³,
  beside the "Final identity" panel (`app/page.tsx:649`) that asserts they are equal.

README claims raw tensors are stored "so weak nonzero activations are not lost to display quantization"
— true of the `.f32.gz` binaries, not of these JSON arrays.

### I-11 · Test suite: what is actually being tested
**Severity: medium · Classification: verified defect (test quality) · [VERIFIED]**

**Claims tested only by source-string matching (F1).** Four of the 19 node tests are pure
`assert.match(sourceText, /…/)` over `.tsx` files and never render anything:

- `tests/rendered-html.test.mjs:129-161` — 30 regexes over `app/page.tsx`, covering the Conv1D
  cross-correlation convention, "no ReLU after this addition", "38,400 products + bias",
  "1,074 − 75 + 1 = 1,000", the softmax vocabulary, and the signed-colour legend.
- `:163-178` (dilation-trace), `:222-244` (model-audit), `:257-275` (basset) — same pattern.
- `:149` asserts on an **exact code expression**
  (`nextStage === "stem" ? (computation === "output" ? "relu" : computation) : layer === "stem" ? "output"`),
  which tests source formatting, not behaviour, and will break on any reformat.

**`app/model-analysis.ts` has no direct test at all.** Its only coverage is
`tests/rendered-html.test.mjs:157` — `assert.match(page, /centeredStemWeights/)`, i.e. a check that the
*name appears in page.tsx*. `scripts/verify_model_analysis.py` never imports the TypeScript; it
re-implements the same maths in NumPy and asserts against itself. I therefore executed the module
directly; it is **correct** (see §7), but nothing in the repo would catch a regression.

**`verify_basset_adapter.py` cannot verify what its banner claims.** Line 33 asserts
`data["verification"]["maximum_absolute_error"] < 1e-9` — a number the exporter writes about itself.
The UI renders this as *"NUMERICAL GATE PASSED · NumPy ↔ TensorFlow · 9.95e-14"*
(`app/basset/page.tsx:342`). The dual-implementation comparison is genuinely strong, but the committed
check is a self-report, not a reproduction. (The rest of that script does real independent work.)

**Failures the current tests would miss (F2).** B-2 (main-page degenerate residual defaults), B-3 (the
softmax caption/chart mismatch), I-1 (stale scope note after slider movement), I-3 (dead keyboard
activation), I-4 (controls hidden below 1000 px), I-6 (missing distribution context), I-7 (zero-based
ID), I-8 (mobile overflow), I-10 (probability quantization).

**What the tests do well (F4).** The data tests genuinely recompute rather than compare copies:
`:46-52` re-derives residual tap sums, ReLU and shortcut from the raw `.f32.gz`; `:70-79` reconstructs
three profile logits from `res8.f32.gz` × the 512 × 75 kernel; `:82-126` re-derives the `_bias` and
`_relu` display transforms and checks the shortcut identity at 9 cells per layer; `:180-191` checks all
eight balanced examples are non-degenerate and in range. That is the right instinct — it is just
applied to a sparse sample and stops short of the stem and the heads.

### I-12 · Reproducibility of the audit artifact
**Severity: low-medium · Classification: verified defect · [VERIFIED]**

`scripts/build_model_audit.py:27` hard-codes
`"gm21515": Path("/private/tmp/model.chrombpnet_nobias.fold_0.ENCSR960KGO.h5")`, and `main()` raises
`FileNotFoundError` if a checkpoint is missing. The README's documented rebuild command
(`models/.extract-env/bin/python scripts/build_model_audit.py`) therefore cannot succeed on any machine
that does not happen to have that file in `/private/tmp`. Make it a CLI argument.

---

## 4. Longer-term research / performance improvements

- **P-1 [PERF]** — Move the four static JSON imports in `app/page.tsx:4-7` (and the equivalents in the
  other routes) behind runtime fetches. This is the single highest-leverage change in the repo: it cuts
  ~7.4 MB gzip from the critical path, fixes B-1's root cause, and makes each route's cost proportional
  to what the reader actually opens.
- **P-2 [PERF]** — `app/dilation-trace/page.tsx:37-51` fetches all nine backbone tensors on mount
  (~2.6 MB compressed, ~33 MB in the JS heap) regardless of which stage is selected. Fetch the selected
  stage plus its input; prefetch the rest idle.
- **P-3** — `public/data` is **110 MB** in git (98 MB tensors, 13 MB Basset). Every byte is referenced,
  but clone and deploy both carry it. Consider Git LFS or fetch-on-demand before publication.
- **P-4 [LATENT]** — `informationContent` (`app/model-analysis.ts:63-70`) correctly generalizes to
  relative entropy, but `ActivationLogo` (`app/page.tsx:225`) scales letter height as
  `value / 2 * 145` px inside a 250 px `overflow:hidden` box, i.e. it assumes IC ≤ 2 bits. I measured
  IC = **4.32 bits** for a pure-A column against background `[.05,.45,.45,.05]`, and IC can also be
  **negative** (the code floors it at a 2 px stub with no sign cue). Unreachable today
  (`activation_motifs.motifs` is `[]`), but it will silently clip the first real motif with a
  non-uniform background.
- **P-5 [LATENT]** — `app/basset/page.tsx:332` derives `k562Rank` as the index inside the 15-entry
  `top_predictions` array. If a re-export ever placed K562 outside the top 15, `findIndex` returns −1
  and the page renders "rank 0" and drops the K562 bar. Compute the rank in the exporter.
- **P-6** — `basset-demo.json` uses `schema_version: 1` (number); `model-audit-summary.json` uses
  `"1.0.0"` (string). Unify before either is consumed elsewhere. *(Carried over from V2 §10.6.)*
- **P-7** — The synthetic preset's sequence has a **`TGACTCA` (AP-1) site planted at input 76–82**
  (`scripts/run_chrombpnet_checkpoint.py:22-27`). Neither the UI nor the README discloses this; the
  README does not mention the synthetic preset at all. A reader exploring that preset and finding a
  strong stem response there could read a planted control as a discovered motif.
- **P-8 [JUDGMENT]** — Two carried-over encoding issues remain unaddressed and are worth revisiting:
  **(a)** `VectorCanvas` (`app/page.tsx:499-517`) normalizes each of the `features × weights = products`
  strips independently, so multiplying the left two images does not yield the right one, and learned
  weights use the same colormap as computed contributions (`app/page.tsx:570-572`); **(b)** the palette
  reuses coral for base **A** and for **positive** weight, and blue for base **C** and for **negative**,
  with the base-coloured `.sequence-cells` row sitting directly above the sign-coloured
  `.selected-weights` row (`app/page.tsx:342-343`); **(c)** `transformValue`
  (`app/page.tsx:47-51`) saturates at `ratio ≥ (1/gain)²` — 30.9 % of layer max at the sqrt × 1.8
  default — and the clipping is still undisclosed.

---

## 5. Strong sections that should be preserved

| # | What | Where | Why it survives independent checking |
|---|---|---|---|
| **S1** | **Exactness of every displayed tensor** | all routes | I re-derived the stem from one-hot × kernel bank (max err 5.4 × 10⁻⁷), all 1,000 profile logits from `res8` × the 512 × 75 kernel (3.6 × 10⁻⁶), the count head (`pooled = res8.mean(axis=1)`, err 1.3 × 10⁻⁶), and all 24 residual `example_modes` against both the raw input tensors and the actual output tensors. Nothing disagreed. |
| **S2** | **The audit artifact is fully reproducible from the committed binaries** | `app/data/model-audit-summary.json` | Every statistic I could recompute matched exactly on both checkpoints: 18 `exact_zero_fraction` values, 18 `median_active_channels_per_position`, `stem_occupancy` and `final_rms` (err **0.0**), `count_contribution` (4.4 × 10⁻⁸), `profile_position_energy` (1.4 × 10⁻⁷), the count–profile correlation (−0.08114 / −0.33043 to 5 d.p.), and all 5 `channel_orders` confirmed as true permutations correctly sorted by their named metric. |
| **S3** | **Dynamic, value-derived statistical prose** | `app/model-audit/page.tsx:155-161` | `CorrelationRuler` derives band and direction from the number and prints r² plus a replication caveat. Live: K562 → "−0.081 … very weak negative … 0.7 %"; GM21515 → "−0.330 … moderate negative … 10.9 %". Guarded by a browser test. This is the correct pattern and should be copied wherever a number sits next to a sentence. |
| **S4** | **Cross-channel mixing is demonstrated, not asserted** | `app/model-audit/page.tsx:222-225`, `app/page.tsx:551, 567-576` | `diagonal_energy_fraction` = 0.24–0.40 % (chance ≈ 0.195 %) and median `effective_input_channels` = **329–379 of 512** in every block are exactly the right evidence that dilation is not per-channel. Paired with three visible 512-element `features × weights = products` columns. **Answers B4 and B6 affirmatively.** |
| **S5** | **The evidence ladder and single-locus scope banner** | `app/model-audit/page.tsx:195-205` | Descriptive / mechanism / biology with definitions, plus `"1 exact locus"` and an explicit interpretation limit on every panel. Rare in public explainers. |
| **S6** | **Refusing to fabricate activation motifs** | `app/page.tsx:231-238`, `app/model-audit/page.tsx:239-242` | Labeled empty state with planned corpus, reservoir size and selection rule; enforced by a test asserting `status === "not_generated"`. |
| **S7** | **The Basset chain is exact end to end** | `/basset`, `docs/basset-checkpoint-adapter.md` | I reconstructed all 164 output probabilities from the committed weights and confirmed K562's true global rank = 13; verified channel-major flatten (err 1.1 × 10⁻⁷ vs 3.24 for position-major); verified all three max-pool identities at **err = 0.0**; verified the full RF/spacing table (19/21/51/60/132/168 bp; 1/3/3/12/12/48 bp) and that `168 + 9 × 48 = 600` exactly. **Answers E1 and E2 affirmatively.** |
| **S8** | **Declared, honest example-selection rules** | `scripts/run_chrombpnet_checkpoint.py:62-100` | `balanced` = argmax `min(correction, shortcut)`; `correction` = argmax correction; `output` = argmax output — all restricted to input-aligned 558–1557. **Each rule does exactly what its label claims (B2 ✅)**, and all 24 selected coordinates are valid for the relevant layer *and* the profile head (B3 ✅). |
| **S9** | **Coordinate consistency across all four routes** | throughout | `offsetForLength` / `inputCoordinate` reconcile at every point I checked: stem output *p* ↔ input centre *p*+10; profile output *p* ↔ input *p*+558 with a 1,115 bp footprint (`globalCenter ± 557`); block-8 taps 690 / 946 / 1202 around shared coordinate 946; the `SequenceOverview` marker box is exactly `21/2114` wide at `start/2114`. **Answers A4 affirmatively.** |
| **S10** | **The four-state computation control** | `app/page.tsx:809, 817` | `conv → +bias → ReLU → +shortcut` as four real inspectable tensors, signed rendering auto-enabled for pre-ReLU stages, correct captions at each point. **Answers A6 affirmatively for ChromBPNet**; Basset's `conv bias → batch norm → ReLU` ordering is likewise correct and now shows the intermediate. |
| **S11** | **`display_transform` storage design** | `app/page.tsx:95-112`, `app/tensor-loader.ts` | 35 tensor states served from 18 files by deriving `_bias`/`_relu` from `_conv` + `channel_bias` in the browser. I re-derived the transform independently and it reconciles. |
| **S12** | **The cross-model frame on the main page** | `app/page.tsx:904-908` | *"The output task changes which spatial compromises are acceptable"* + a shared-vocabulary table (Channel · Receptive field · Crossing distance · Output). **Answers E4 and E5 affirmatively** — this is what converts two demos into one framework. |
| **S13** | **Attribution and reach boundaries** | `app/page.tsx:931`, `app/dilation-trace/page.tsx:194, 368`, `app/basset/page.tsx:424` | Geometric reach, observed activation change, learned weights, and mechanistic contribution are kept verbally distinct and never collapsed. **Answers B5 affirmatively.** |

---

## 6. Audit questions — direct answers

### A. Scientific and numerical correctness

**A1 — shapes.** ✅ All correct. ChromBPNet: 2114 → 2094 (k=21 valid) → 2090/2082/2066/2034/1970/1842/1586/1074
(each block losing `2 × dilation`) → 1000 (k=75 valid); `LENGTHS`/`RECEPTIVE_FIELDS`
(`app/page.tsx:27-28`) re-derived independently and exact. Basset: 600 → 582 → 194 → 184 → 46 → 40 → 10
→ 2000 → 1000 → 1000 → 164, all confirmed against the committed binaries.

**A2 — operations.** ✅ Correct throughout. Cross-correlation orientation is stated explicitly with the
index formula (`app/page.tsx:346`) and proved by an asymmetric toy kernel in
`verify_model_analysis.py::test_orientation`. Dilation, symmetric cropping (`d` per side), the absence
of a post-addition activation, both heads, `expm1(logcount)` (verified: 3081.5254 = expm1(8.0335)), the
Basset BN convention (legacy Torch7 `running_std` = inverse std, documented and applied correctly), and
the 164 sigmoid readers are all right.

**A3 — do the example calculations reproduce the tensors?** ✅ Yes, at every point I checked.
Stem: displayed `dot 1.265 + bias −0.082 → 1.183` for Filter 205 at input 1635–1655 reproduces exactly
(recomputed 1.26497 / −0.08215 / 1.18282). Weight-logo compensating bias: displayed **−1.16131** for
Filter 118 reproduces exactly. Residual: all 24 example modes reconcile with the raw tensors *and* with
the actual block-output tensor. Count head: displayed `7.896 + 0.137 = 8.034` reproduces.

**A4 — coordinate systems.** ✅ Consistent (see S9).

**A5 — the 75-position kernel's 1,115 bp footprint.** ⚠️ **Partly.** `/dilation-trace` does this well:
*"tensor centers C−37 – C+37; combined input footprint C−557 – C+557 (1,115 bp)"* — arithmetically
exact. On `/` the number appears only as a terminal chip in the receptive-field meter
(`app/page.tsx:584`: `→ profile head: 1,115 bp`) with no derivation, while the surrounding copy talks
about "75 positions" and "38,400 products". A reader on the main page alone is not shown that
`1041 + 74 = 1115`.

**A6 — bias / ReLU / shortcut / BN / dense placement.** ✅ Correct (see S10).

**A7 — colour consistency.** ⚠️ **Mostly, with two carried-over problems.** Zero is consistently paper
(`rgb(247 244 236)`) in `heatColor`, `signedColor` and `MiniBars`, and exact zero now draws a 1 px bar
versus ≥ 3 px for nonzero — a real improvement. But coral means both base **A** and **positive weight**,
blue means both base **C** and **negative weight**, on adjacent rows; and the four pages use three
slightly different coral/blue pairs (`[225,91,69]/[44,129,158]` on `/` and `/dilation-trace`,
`[232,100,75]/[50,136,163]` on `/model-audit`, `[231,95,72]/[45,136,166]` on `/basset`). See P-8.

**A8 — information content.** ✅ **Correct for both uniform and nonuniform backgrounds.** I executed
`app/model-analysis.ts` directly: `informationContent([[1,0,0,0],[.25,.25,.25,.25],[.7,.1,.1,.1]])` →
`[2, 0, 0.6432]` (classic 2 − H); with background `[.1,.4,.4,.1]` → `[3.3219, 0.3219, 1.5651]` (correct
relative entropy). The background argument is normalized and validated. The *renderer* does not yet
handle IC > 2 bits or negative IC (P-4).

### B. Residual and dilation explanation

**B1 — does the default residual example show both a positive correction and a nonzero shortcut?**
**Split.** ✅ on `/dilation-trace` (0.3757 + 0.3024, verified in browser and matching
`example_modes.balanced`). ❌ on `/` — see **B-2**: 5 of 8 blocks show correction 0.0000 and one shows
shortcut 0.0000.

**B2 — do the three example-selection rules do what their labels claim?** ✅ Yes (S8), verified against
the exporter and numerically for all 24 (mode × block) combinations.

**B3 — are the selected coordinates valid for the relevant layers and the profile head?** ✅ Yes. All
24 lie in input-aligned [558, 1557] by construction, which is exactly the profile head's output domain
(profile index 0–999 ↔ input 558–1557), and I confirmed each tap index stays inside its input tensor.

**B4 — does the visualizer demonstrate all-512-channel mixing?** ✅ Yes, at two levels: the visual
`features × weights = products` triple over 512 elements per tap, and the kernel diagnostics
(diagonal energy 0.24–0.40 %; median effective input channels 329–379 of 512).

**B5 — are reach / observed change / weights / mechanism separated?** ✅ Yes, and explicitly
(`/dilation-trace` §2 caption, §3 heading, §4 contributions, closing "Interpretation boundary").

**B6 — could a reader conclude dilation is per-channel?** ✅ **No** — this is one of the release's
strongest points. Three independent surfaces contradict that reading.

### C. Visual encoding and interaction

**C1 — softmax ordering / "does softmax create the peak?"** ⚠️ The *substance* is right (peak
preserved at 426 across all four arrays; explicit "does not invent a new peak" note; four-stage chain
with the shift explained). The *label* is wrong — **B-3**.

**C2 — does binning hide peaks?** ✅ **No, this is fixed.** `MiniBars` now uses per-bin **max**
(`app/page.tsx:594`). Verified: the rendered 160-bar max equals the true `profile_signal` peak
**52.89105** exactly. (The 160-vs-512 labelling issue is separate — I-9.)

**C3 — does the magnitude control behave as labeled?** ✅ Yes on both surfaces. `/`'s Inspector toggle
switches between per-layer and a shared ±8.51 and says so; I confirmed the shared maximum (8.505, from
`res8`) exceeds every conv/bias-stage magnitude (max 8.278), so nothing clips.
`/dilation-trace`'s "Color scale: shared / normalized per stage" changes the rendered canvas for the
stem, res4 and res8 panels (pixel-diffed).

**C4 — can readers distinguish pattern normalization from absolute growth?** ✅ Yes, via C3's control
plus the disclosure text *"Color is normalized within this layer; compare patterns, not magnitude."*

**C5 — alignment of markers, zoom windows, axes, labels, coordinates.** ✅ Precise. The 21-base window
marker is `width: calc(21 / 2114 * 100%)` at `left: start/2114` — exact grid alignment; the residual
zoom's base letters are offset by +10 (window centre); the profile zoom's base letters by +557; the
inspector's by `offsetForLength(length)`. All verified.

**C6 — are whole-tensor widths proportional?** ✅ Yes. Measured in the DOM: `res8` frame = **51.2894 %**
(= 1074/2094) and the profile output stack = **47.7555 %** (= 1000/2094).

**C7 — misleading, unreachable, jammed or broken controls?** Three issues: **I-4** (channel order and
all nav hidden < 1000 px), **I-3** (layer-table rows focusable but dead), **I-1** (scope note asserts a
condition the numbers contradict after slider movement).

**C8 — console / hydration / fetches / overflow.** ✅ Zero console errors, zero page errors, zero
hydration warnings, zero failed or ≥ 400 requests on all four routes at 1440 / 820 / 390 px. Overflow:
clean at 1440 and 820; **I-8** at 390.

**C9 — keyboard and a narrower viewport.** Keyboard: `/`, `/dilation-trace` and `/basset` expose 62 / 26 /
22 focusable controls with **zero** zero-box elements; `/dilation-trace`'s magnifier responds correctly
to arrow keys (coordinate 1400 → 1401, channel band shifted by one). `/model-audit` has 9 zero-box
focusables (**I-3**). Narrower viewports: 820 px clean apart from **I-4**; 390 px per **I-8**.

### D. Checkpoint and evidence integrity

**D1 — are K562, GM21515 and synthetic clearly distinguished?** ✅ Mostly. Distinct `provenance`,
`locus_label`, `preset_id`; the synthetic entry is labeled *"Synthetic teaching sequence · 2,114 bp"*
and its empirical channel orders are correctly disabled. Gap: the planted AP-1 site is undisclosed and
the README omits the preset entirely (P-7).

**D2 — is single-locus evidence labeled descriptively?** ✅ Yes, consistently — `"1 exact locus"`,
`"Single-locus descriptive audit; not a population or biological conclusion"`, and per-panel caveats.

**D3 — are checkpoint statistics interpreted dynamically?** ✅ Yes (S3). This was V2's most serious
finding and it is thoroughly fixed.

**D4 — are activations / weights / contributions / ablations / attributions / biology kept separate?**
✅ Yes, and stated as an explicit three-rung ladder with an ordered protocol to climb it.

**D5 — does "no-bias" clearly mean enzyme-bias-corrected?** ✅ Yes. `app/page.tsx:901`:
*"'no-bias' in the source filename means the separate enzyme-sequence-bias model was factored out;
ordinary learned layer biases remain."*

**D6 — channel identity and global ordering without changing predictions or breaking the shortcut?**
✅ **Yes, functionally.** All five orders are true permutations of 0–511 correctly sorted by their named
metric; reordering is applied only at render time (`channelOrder[displayRow]`), never inside a
computation; the shortcut always uses the immutable `outputChannel`; and
`verify_model_analysis.py::test_global_permutation_invariance` proves the producer/consumer axes must
move together. The one labelling wart is **I-7**.

### E. Basset and cross-model explanation

**E1** ✅ Faithful to the published architecture and to the checkpoint's stored graph, including the
`padW = 0` finding. **E2** ✅ All stages connected into one flow, with the intermediate `before batch
norm` now shown. **E3** ⚠️ K562 is visible and its rank (13, independently confirmed) is honest, but
the output distribution is not contextualized (**I-6**). **E4** ✅ Yes — *"the output task changes which
spatial compromises are acceptable."* **E5** ✅ Yes — the main page now carries the shared-vocabulary
frame, so the two demos are no longer disconnected.

### F. Test quality

Answered in **I-11** (F1, F2, F4), **B-1** (F3), and the proposals below (F5).

---

## 7. Independent numerical evidence (summary of my own re-derivations)

```
STEM   conv recompute (15 filters × 10 positions, one-hot × kernel bank)   max err 5.4e-07
STEM   relu(conv + bias) vs stem.f32.gz                                    max err 4.1e-08
STEM   peak_positions_zero_based vs argmax(stem)                           512 / 512 match
STEM   maximum_activations vs max(stem)                                    max err 5.4e-07
PROFILE  all 1,000 logits from res8 × (512×75 kernel) + bias               max err 3.6e-06
PROFILE  softmax(logits) vs stored probabilities                           max err 5.0e-07
PROFILE  argmax logits / probabilities / expected counts                   426 / 426 / 426
COUNT  count_pooled_features vs res8.mean(axis=1)                          max err 1.3e-06
COUNT  pooled·w + bias  = 8.03351 (stored 8.03350); expm1 = 3081.5254      = stored total
RESID  24 example_modes (8 blocks × 3 rules) vs raw tensors AND outputs    all OK, |err| < 2e-5
AUDIT  18 exact_zero_fraction, 18 median_active, occupancy, RMS, contrib   all exact
AUDIT  count–profile correlation  k562 -0.08114 / gm21515 -0.33043         = stored, 5 d.p.
BASSET all three max-pool identities                                       err 0.0
BASSET dense1 channel-major flatten (vs position-major 3.24)               err 1.1e-07
BASSET all 164 probabilities from dense2_activations × output weights      K562 rank 13/164
MODEL-ANALYSIS.TS  centering identity over all 16 one-hot combinations     holds to 1e-12
MODEL-ANALYSIS.TS  informationContent, uniform and nonuniform background   correct
```

---

## 8. Disposition against `AUDIT-INDEPENDENT-V2.md`

*(V2 audited commit `b515ac6d`; this section was written only after §§2–7 were complete.)*

### Fixed

| V2 | Finding | Evidence at `3d56b978` |
|---|---|---|
| **N1** | Hardcoded "very close to zero" correlation sentence, false on GM21515 | **FIXED.** `CorrelationRuler` (`model-audit/page.tsx:155-161`) derives band + direction from the value and prints r². Browser: K562 → "very weak negative … 0.7 %"; GM21515 → "moderate negative … 10.9 %". Guarded by a browser test. |
| **N2** | "Jump to strongest example" silently clamped to 1557 for blocks 1–3 | **FIXED by replacement.** The button is now "Show selected example" and targets `example_modes[mode].input_aligned_coordinate_one_based`; the exporter constrains every candidate to [558, 1557], so no clamping can occur. I verified all 24 coordinates in range. |
| **N4** | Reach band used `(RF − 1)/2` — 6× overstatement at Block 1 | **FIXED.** `dilation-trace/page.tsx:180` is now `(RECEPTIVE_FIELDS[stage] - 21) / 2`, labeled "±N stem centers". Browser: stage 8 → "±510 centers … 1041-base receptive field". |
| **N5** | "no-bias" undefined | **FIXED.** See D5. |
| **N7** | `MiniBars` bin-**averaging** understated the peak by 25 % | **FIXED.** Now per-bin max (`page.tsx:594`); rendered peak = true peak = 52.89105 exactly. Signed series use max-magnitude, so ± no longer cancel inside a bin. |
| **N8** | 7.2× layer magnitude growth invisible; no shared-scale toggle in the Inspector | **FIXED.** "Magnitude scale · Per layer / Shared" added with disclosure text ("All backbone layers share ±8.51"). Guarded by a browser test. I confirmed no clipping under the shared scale. |
| **N10** | Basset top-12 chart excluded K562 (rank 13); highlight branch was dead code | **FIXED.** `displayedPredictions` appends K562 with "· rank 13" plus an aside. Rank independently confirmed correct. |
| **N11** | Basset stem/Conv2 equations skipped `after_raw_bias` | **FIXED.** Browser: `sum products 3.178 + conv bias 2.619 = before batch norm 5.797 → after batch norm 1.817 → ReLU 1.817`. Same for the Conv2 mixing equation. |
| **N12c** | Zero rendered as visible tint in `MiniBars` | **FIXED.** `heatColor(value/maximum)` maps 0 → paper; exact zero draws 1 px vs ≥ 3 px for nonzero. |
| **N13** | Synthetic preset borrowed K562's empirical channel orders | **FIXED.** `empiricalOrderingAvailable = preset !== "synthetic"` disables the selector and forces the identity order, with an explanatory tooltip. |
| **N15** | `npm test` did not run ESLint; 13 lint errors | **FIXED.** `npm test` now runs lint → typecheck → build → unit → e2e. ESLint reports **0** problems. |
| **N16** | `npm test` unrunnable from a clean clone (hardcoded `models/.extract-env/bin/python`) | **FIXED.** `npm test` no longer touches Python; `verify:python` uses `python3` and is separately documented. |
| **N17** | Rules-of-Hooks violation in `ExactCommunication` | **FIXED.** `useMemo` at `:236` now precedes the early return at `:254`; lint clean. |
| **N18** | `informationContent` silently ignored its `background` argument | **FIXED.** Verified by direct execution — correct relative entropy, normalized and validated background. |
| **T5** | Profile footprint understated as "75 positions" | **FIXED.** `dilation-trace/page.tsx:295` now prints the 1,115 bp combined footprint. |
| **§8.1–8.3** | Two-model contrast one-directional; no shared vocabulary; "task forces architecture" unstated | **FIXED.** `page.tsx:904-908` adds the cross-model frame with all three. |
| **V4** | Filmstrip `sharedScale` defaulted to the misleading setting | **FIXED.** `dilation-trace/page.tsx:310` now `useState(true)`. |

### Partly fixed

| V2 | Finding | Status |
|---|---|---|
| **N6** | Logits→softmax panel teaches the opposite of what softmax does | **PARTLY FIXED, AND REPLACED BY A NEW PROBLEM.** The zero-baseline defect is genuinely fixed: the panel now subtracts a constant, uses the full axis, adds a four-stage chain and an explicit "does not invent a new peak" note (peak preserved at 426 — I confirmed). But the implemented shift is `logit − min` while the label says `logit − maximum logit / the largest becomes 0`. **See B-3.** |
| **N19** | `/dilation-trace` fetches all nine tensors on mount | **STILL PRESENT, AND NOW COMPOUNDED.** `useRawTensors` is unchanged, and the client bundle has grown to ~7.4 MB gzip on `/`. Together they are the mechanical cause of the red gate (**B-1**). |
| **N12d / V9** | Overview marker ~1 % skew | **EFFECTIVELY FIXED.** The marker box is exactly `21/2114` wide at `start/2114`, with a code comment explaining the choice; residual click-mapping error < 0.05 %. |
| **Numbering 1,2,3,—,5** | Missing section 4 on `/dilation-trace` | **PARTLY FIXED.** A "4 ·" now exists, but as an eyebrow `<small>` rather than the numbered `<span>` badge the other four sections use — the badge sequence still reads 1, 2, 3, —, 5. |
| **N14** | No browser or interaction tests | **PARTLY FIXED.** Six Playwright tests now exist and cover five of V2's defects. But three of the six assert only on page *text*, none would catch B-2, B-3, I-1, I-3 or I-4, and the suite is flaky (**B-1**). |
| **T1** | `4 × 21` vs `512 × 3` side-by-side kernel panel | **STILL ABSENT.** The 512 × 21 bank and the 512-element tap vectors partly compensate. |

### Still present

| V2 | Finding | Current location |
|---|---|---|
| **N9** | Weights and contributions share a colormap; the three strips are independently normalized, so "×" is not multiplicative | `page.tsx:499-517, 570-572` — unchanged (P-8a) |
| **N12a** | Palette collision: coral = base A **and** positive; blue = base C **and** negative, on adjacent rows | `page.tsx:22, 70, 342-343` — unchanged (P-8b) |
| **N12b** | Gain clipping undisclosed (sqrt × 1.8 saturates above 30.9 % of layer max) | `page.tsx:47-51` — unchanged (P-8c) |
| **T2** | `start` means two things | `page.tsx:341` "Input {start+1}–{start+21}" vs `:347` "output position {start+1}" — unchanged |
| **T3** | Two separate colormap functions | `dilation-trace/page.tsx:55-63` `color()` vs `page.tsx:53-74` `heatColor`/`signedColor` — unchanged |
| **§10.5** | `top_predictions` holds only 15 entries | Unchanged; rank is now exposed, but derived from the slice (P-5) |
| **§10.6** | Schema version type mismatch (`1` vs `"1.0.0"`) | Unchanged (P-6) |
| **§10.7** | 110 MB in `public/data` | Unchanged (P-3) |

### Replaced by a new problem

| V2 | What changed |
|---|---|
| **N3** — "flagship cell calculation shows 0.0000 + 0.0000 on first load" on `/dilation-trace` | **The fix was applied to `/dilation-trace` only.** That page now opens on a genuine balanced merge (0.3757 + 0.3024) with declared rules and a browser test. **The identical defect now sits on the main explainer**, which still reads `trace` (`page.tsx:527`) and shows a zero learned correction in 5 of 8 blocks of the default preset and 7 of 8 of the synthetic one — **B-2**. V2 called the selection rule "the most valuable single change"; the change was made in one of the two places that needed it. |
| **N6** — logits chart baseline | The baseline problem was fixed and a **caption/chart contradiction** was introduced in its place — **B-3**. |
| **N1** — hardcoded interpretation on `/model-audit` | Fixed for the correlation panel, but the *same failure mode* survives one panel away: `model-audit/page.tsx:170` asserts "CKA compares **whole representations**" while the artifact's method string says 128 positions × 64 per-layer channels — **I-5**. |

---

## 9. The ten highest-value next actions, in priority order

1. **Point `ResidualStory` at `example_modes.balanced`** (`app/page.tsx:527-528`) and expose the same
   three-rule selector `/dilation-trace` has. One import change converts the main page's weakest panel
   into its intended centerpiece. *(B-2)*
2. **Fix the softmax step-1 caption** in both places (`app/page.tsx:628`, `:730`) so the stated formula
   matches the plotted array. *(B-3)*
3. **Make `npm test` deterministic**: raise the `toBeVisible` timeout at
   `tests/browser/explainer.spec.ts:20` and gate on tensor arrival rather than wall-clock. *(B-1)*
4. **Move the four JSON imports out of the client bundle** (`app/page.tsx:4-7` and siblings) and fetch
   them at runtime. Removes ~7.4 MB gzip from `/`, and is the root fix for #3. *(P-1, B-1)*
5. **Make `/dilation-trace`'s scope note live-state-aware** so it never claims a balanced merge while
   displaying a zero shortcut. *(I-1)*
6. **Stop hiding the channel-order control and the route nav below 1000 px**; add body-level links to
   all three sibling routes so `/dilation-trace` is reachable. *(I-4)*
7. **Correct the CKA prose** to describe the 128-position × 64-channel subsample, and print one decimal
   place. *(I-5)*
8. **Default the profile zooms to the predicted peak** instead of the hard-coded 470–530 window.
   *(I-2)*
9. **Add the five high-value tests below.** *(F5)*
10. **Fix `/model-audit`'s `display:contents` rows** (I-3), the zero-based "immutable ID" (I-7), the
    Basset output-distribution context (I-6), and the mobile overflow (I-8) — a single cleanup pass.

### F5 · Five proposed tests, with exact assertions

```js
// 1. tests/rendered-html.test.mjs — the main page's default residual example must be non-degenerate.
//    Would have caught B-2. Currently no test looks at `trace` at all.
for (const preset of ["k562-peak", "gm21515", "synthetic"]) {
  const demo = await loadDemo(preset);
  for (const block of demo.residual_kernel_demos) {
    const t = block.trace;                       // ← whatever app/page.tsx:527 actually reads
    assert.ok(t.transformed_after_relu > 0, `${preset} block ${block.block}: zero correction`);
    assert.ok(t.shortcut_value       > 0, `${preset} block ${block.block}: zero shortcut`);
  }
}

// 2. tests/rendered-html.test.mjs — recompute the stem forward pass the main page displays.
//    Nothing currently verifies the single most prominent calculation on the site.
const bank   = new Float32Array(gunzipSync(await readFile(kernelUrl)).buffer);   // 512×4×21
const stem   = new Float32Array(gunzipSync(await readFile(stemUrl)).buffer);     // 512×2094
const oneHot = (p, b) => demo.input.sequence[p] === "ACGT"[b] ? 1 : 0;
for (const f of [0, 117, 204, 511]) {
  for (const p of [0, 1046, 2093]) {
    let dot = 0;
    for (let k = 0; k < 21; k++) for (let b = 0; b < 4; b++) dot += oneHot(p + k, b) * bank[f*84 + b*21 + k];
    assert.ok(Math.abs(Math.max(0, dot + demo.stem_kernel_bank.biases[f]) - stem[f*2094 + p]) < 2e-5);
  }
}

// 3. tests/rendered-html.test.mjs — the softmax chain identities the page asserts in prose.
const L = demo.tensors.profile_logits.position_max;
const P = demo.tensors.profile_probabilities.position_max;
const S = demo.tensors.profile_signal.position_max;
const m = Math.max(...L), e = L.map(v => Math.exp(v - m)), z = e.reduce((a, b) => a + b, 0);
assert.equal(L.indexOf(m), P.indexOf(Math.max(...P)));           // "same peak position"
assert.equal(L.indexOf(m), S.indexOf(Math.max(...S)));
for (let i = 0; i < 1000; i++) {
  assert.ok(Math.abs(e[i] / z - P[i]) < 2e-6, `softmax mismatch at ${i}`);
  assert.ok(Math.abs(P[i] * demo.outputs.predicted_total_count - S[i]) < 2e-3, `identity at ${i}`);
}

// 4. tests/browser/explainer.spec.ts — the step-1 caption must match the plotted array.
//    Would have caught B-3.
await page.goto("/");
const bars = await page.locator("#outputs .head-stage").first()
  .locator(".mini-bars i").evaluateAll(es => es.map(e => parseFloat(e.style.height)));
const label = await page.locator("#outputs .head-stage").first().locator("b").textContent();
if (/maximum logit/.test(label)) {
  // "logit − max ⇒ largest becomes 0" implies the tallest bar is the SMALLEST logit.
  expect(bars.indexOf(Math.max(...bars))).toBeGreaterThan(150);   // fails today: index 68
}

// 5. tests/browser/explainer.spec.ts — every interactive control has a hit box and is reachable.
//    Would have caught I-3 and I-4.
for (const route of ["/", "/dilation-trace", "/model-audit", "/basset"]) {
  await page.goto(route);
  const bad = await page.evaluate(() => [...document.querySelectorAll(
    'button,select,input,summary,[role=button],[tabindex]:not([tabindex="-1"])')]
    .filter(e => { const r = e.getBoundingClientRect(); return r.width < 2 || r.height < 2; })
    .map(e => e.tagName + ":" + (e.getAttribute("aria-label") || e.textContent || "").slice(0, 40)));
  expect(bad, `${route} has zero-box focusables`).toEqual([]);    // fails today on /model-audit (9)
}
await page.setViewportSize({ width: 900, height: 900 });
await page.goto("/");
await expect(page.getByLabel("Channel order")).toBeVisible();     // fails today (display:none)
```

---

## 10. Unresolved questions requiring the project owner's judgment

1. **Which residual selection rule should the main page teach by default?** `balanced` is the obvious
   fix for B-2, but "the shortcut carries strong features through untouched" is itself a real and
   interesting lesson. If that is the intended message, it needs to be *stated as the lesson* and paired
   with a counterexample — not left as a silent default beside a heading that promises the opposite.
2. **Should the profile probabilities be re-exported at full float32?** The 1 × 10⁻⁶ quantization
   (I-10) is invisible in the charts but makes three on-page identity claims literally false at the
   sixth decimal. Re-exporting costs a few hundred kB; leaving it costs a footnote. Your call which is
   cheaper.
3. **How much of `/model-audit` should survive into the public narrative?** It is the strongest
   scientific surface in the repo and the least suited to a video. Does the blog carry it in full, link
   to it, or distill the evidence ladder into one slide?
4. **Is the GM21515 count–profile correlation (r = −0.330, r² = 10.9 %) worth promoting from a display
   statistic to a stated observation?** It suggests the two heads lean on partly different channels.
   The weight half is checkpoint-wide; the activation half is n = 1 locus. V2 asked this; the panel now
   reports it honestly but still frames it as a display metric.
5. **Should the CKA be recomputed at full resolution before publication?** My full-resolution
   recomputation preserves the qualitative story but moves individual cells by up to 0.22. If any
   number from that matrix will be quoted, the subsample should be widened first.
6. **What is the intended reading of the Basset output layer?** Every one of 164 targets scores ≥ 0.54
   on the HOXA-boundary sequence. Is that a property of this sequence, of this checkpoint's calibration,
   or of the training targets? The answer changes how I-6 should be worded.
7. **Should the synthetic preset stay?** It carries an undisclosed planted AP-1 site, is absent from the
   README, and is the preset with the most degenerate residual examples (7 of 8). Either document it as
   a positive control or drop it from the public build.
8. **Which route leads the narrative?** `/` is a guided story, `/dilation-trace` an exploratory lab,
   `/model-audit` a technical workspace, `/basset` a second story. A blog, a slide deck and a video want
   different subsets and different entry points; the current single-nav structure serves none of them
   optimally, and below 1000 px it serves the lab not at all (I-4).
