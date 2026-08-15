# Independent Audit V2 — Sequence CNN Explainer

**Repo:** `<repository-root>`
**Commit:** `b515ac6d` *Complete signed kernel and Basset dense readout views* · working tree **clean**
**Method:** full read of all 4 route components, both shared modules, both docs, the test suite and the
Python verifiers; numeric re-derivation of all three ChromBPNet presets, the complete Basset forward
chain, and the audit artifact in Node; ESLint; `node --test`; both Python verifiers; **and live
browser interaction against `http://localhost:3000`** (dev server already running from this repo).

**Nothing in the application was modified.** `git status` is clean at the end of this audit.

Legend: **[VERIFIED]** reproduced numerically, by tooling, or in the browser · **[JUDGMENT]** design
preference, argued not proven · **[LATENT]** real defect not currently reachable by a user.

---

## 0. Verification gate

| Check | Result |
|---|---|
| `pwd` | `<repository-root>` ✅ |
| `git rev-parse --show-toplevel` | same path (repo root) ✅ |
| `git log -1 --oneline` | `b515ac6d Complete signed kernel and Basset dense readout views` ✅ |
| working tree | clean, 0 modified/untracked ✅ |
| `app/page.tsx` | **898** lines |
| `app/dilation-trace/page.tsx` | **365** lines |
| `app/model-audit/page.tsx` | **254** lines — exists ✅ |
| `app/basset/page.tsx` | **423** lines — exists ✅ |
| `docs/basset-checkpoint-adapter.md` | exists (73 lines) ✅ |
| Scripts | 8 present, incl. `verify_model_analysis.py`, `verify_basset_adapter.py`, `export_basset_demo.py`, `run_chrombpnet_checkpoint.py` ✅ |
| Tests | 18 pass · both Python verifiers pass · ESLint **13 errors** |

My earlier audit examined `~/Downloads/Learn_AI/sequencemodel_visualizer`, a much older uncommitted
snapshot (548-line main page, one supporting route, no scripts or docs). **That report should be
discarded.** This one supersedes it.

---

## 1. Disposition of every prior finding

Re-tested against the current files, not assumed.

### Prior findings that are now FIXED

| Prior | Status | Evidence |
|---|---|---|
| Missing `/model-audit`, `/basset`, `docs/`, scripts | **INAPPLICABLE** — wrong repo | All present |
| README is starter boilerplate | **FIXED** | `README.md` is a real, accurate 69-line project description |
| 83 MB of unreferenced `.bin`/`.f32` artifacts | **FIXED** | Orphan scan: **68 files on disk, 68 referenced, 0 orphans** [VERIFIED] |
| Repo has zero commits | **FIXED** | Real history; clean tree |
| Convolution orientation never stated | **FIXED — now exemplary** | `page.tsx:339` states the TF Conv1D cross-correlation convention with the explicit index formula and a doc link; `verify_model_analysis.py::test_orientation` proves it with a decisive asymmetric toy kernel |
| "Why dilation instead of pooling?" never asked | **FIXED** | `basset/page.tsx:341` thesis line + §7 `compareGrid` make this the organizing contrast |
| Evidence ledger absent / n=1 not stated | **FIXED — now a highlight** | `/model-audit` `evidence_levels` ladder + `evidence_scope.activation_sample_count: 1` rendered as "1 exact locus" |
| No corpus/attribution roadmap | **FIXED** | `activation_motifs.status = "not_generated"` with a stated plan, plus the TF-MoDISco → Tomtom → perturbation protocol |
| Residual block internals text-only | **FIXED** | Inspector "Computation state" (`conv` → `+bias` → `ReLU` → `+shortcut`) exposes each stage as a real tensor |
| No way to see all 512 filters | **FIXED** | `KernelBankCanvas` 512 × 21 bank + free filter-number entry |

### Prior findings CONFIRMED unchanged

| Prior | Status | Current location |
|---|---|---|
| **C2** receptive-field unit mismatch | **CONFIRMED, unchanged** | `dilation-trace/page.tsx:179` — still `(RECEPTIVE_FIELDS[stage] - 1) / 2` |
| **C3** residual example is a no-op | **CONFIRMED and WORSE** | see §2 N2/N3 |
| **C4** "no-bias" undefined | **CONFIRMED, unchanged** | `page.tsx:869` "published no-bias fold-0 checkpoint"; browser-confirmed as the only occurrence |
| **V1** logits→softmax encoding | **CONFIRMED, unchanged + second instance** | `page.tsx:611`; now also `page.tsx:706` |
| **V2** `MiniBars` bin-averaging | **CONFIRMED, unchanged** | `page.tsx:582-589`, byte-identical logic |
| **V3** per-layer max normalization | **CONFIRMED, unchanged** | `page.tsx:414` `tensor.max`; stem 1.183 → res8 8.505 (7.2×) still invisible |
| **V4** filmstrip default per-stage scale | **CONFIRMED, unchanged** | `dilation-trace:306` `useState(false)` |
| **V5** palette collision | **CONFIRMED, roles swapped** | `signedColor` positive is now **coral** `[225,91,69]`, negative **blue**. Base A is `#e96b54` (coral), base C `#4f97b2` (blue). The collision is unchanged; A now collides with *positive* instead of negative |
| **V6** weights vs contributions identical encoding | **CONFIRMED, unchanged** | `page.tsx:563-565` — all three strips still independently normalized; weights and products both `signed` |
| **V7** gain clipping undisclosed | **CONFIRMED, unchanged** | `page.tsx:48` `clamp(mapped * gain, 0, 1)`, default sqrt × 1.8 → saturation at ratio ≥ 0.309 |
| **V8** zero renders as signal in `MiniBars` | **CONFIRMED, unchanged** | `page.tsx:589` `heatColor(.25 + .75 * value / maximum)` |
| **V9** overview marker ~1% skew | **CONFIRMED, unchanged** | `page.tsx:155` |
| **T2** `start` means two things | **CONFIRMED, unchanged** | `page.tsx:334` "Input {start+1}–{start+21}" vs `:340` "output position {start+1}" |
| **T5** profile footprint understated | **CONFIRMED, unchanged** | `dilation-trace:291` — "75 final-tensor positions" beside "input coordinates C−37–C+37"; true footprint 1,115 bp |
| Section numbering 1,2,3,—,5 | **CONFIRMED** | Browser DOM query returned exactly `["1","2","3","5"]` [VERIFIED] |

### Prior findings CHANGED

| Prior | New status |
|---|---|
| **C1** hooks violation → "page crashes" | **DOWNGRADED. My prior severity claim was wrong.** The Rules-of-Hooks violation is real (`dilation-trace:224` early return before `useMemo` at `:233`; ESLint `react-hooks/rules-of-hooks`), but it **does not crash**. I drove both transitions in the browser: Block 1 → Stem and Stem → Block 1 both render correctly with an empty console. **Reason:** the guarded region contains exactly **one** hook. Going 1→0 hooks leaves `currentHook === null`, so React's `didRenderTooFewHooks` check never fires; going 0→1, React sees `current.memoizedState === null` and selects the *mount* dispatcher, so `useMemo` mounts fresh. It is a latent trap, not a live crash — see §2 N7. |
| **T1** 4→512 channel jump never shown | **PARTIALLY IMPROVED.** The 512 × 21 kernel bank makes "512 filters" concrete. The specific missing beat — `4 × 21` and `512 × 3` side by side — is still absent. |
| **T3** `/dilation-trace` is a separate island | **PARTIALLY IMPROVED.** Now shares the channel-order registry and audit artifact. Still has its own `color()` function (`:54`) distinct from `heatColor` (`page.tsx:51`), its own coordinate control, and the 1,2,3,—,5 numbering. |

---

## 2. Critical scientific / numerical errors

### N1 — The count–profile correlation panel prints a false interpretation on the GM21515 checkpoint [VERIFIED]

`model-audit/page.tsx:160-162`, `CorrelationRuler`:

```jsx
<p><strong>{value.toFixed(3)}</strong> is very close to zero: across channels, large count-head
weight magnitudes are almost unrelated to large profile-kernel energies.</p>
```

The sentence is **hardcoded**. The value is not.

| Checkpoint | r | t (df 510) | p | r² | 95% CI |
|---|---|---|---|---|---|
| K562 · DNase | −0.081 | −1.84 | 0.066 | 0.7% | [−0.167, **+0.006**] |
| **GM21515 · ATAC** | **−0.330** | **−7.91** | **2.7 × 10⁻¹⁵** | **10.9%** | [−0.405, **−0.251**] |

For K562 the sentence is defensible. For GM21515 it is **false**: r = −0.330 across 512 channels is a
highly significant, moderate negative association whose confidence interval excludes zero by a wide
margin. I confirmed the live rendering: with GM21515 selected the page shows **"−0.330"** immediately
above **"is very close to zero … almost unrelated."**

This is the most serious defect in the current build, because it is on the page whose entire purpose is
calibrating claim strength — and it *understates* a real finding. The negative sign is itself
interesting (channels the count head weights heavily tend to carry *less* profile-kernel energy, i.e.
the two heads lean on partly different channels) and is being discarded as noise.

**Fix:** derive the wording from the value (thresholded bands, or state r, CI and p directly). Never
hardcode a statistical interpretation next to a variable number.

### N2 — "Jump to this channel's strongest example" cannot reach the peak for blocks 1–3, and silently lands on the slider's edge [VERIFIED]

`dilation-trace/page.tsx:319-323`:

```js
setGlobalCenter(clamp(localToGlobal(block.trace.output_position_zero_based, LENGTHS[stage]), 558, 1557));
```

The tracked-coordinate slider is restricted to **[558, 1557]** — the *profile head's* output domain
(profile index 0–999 ↔ input 558–1557). The backbone tensors are valid over wider input ranges, so
three blocks' peaks fall outside it:

| Block | trace output pos | true input coord | after clamp | reachable? |
|---|---|---|---|---|
| 1 | 1632 | **1645** | 1557 | **no — clamped** |
| 2 | 1628 | **1645** | 1557 | **no — clamped** |
| 3 | 1620 | **1645** | 1557 | **no — clamped** |
| 4–8 | — | 989, 948, 948, 948, 946 | unchanged | yes |

I clicked the button in the browser on Block 1: the coordinate readout jumped to **1557**, not 1645,
and the cell calculation showed `ReLU correction 0.0000 + shortcut 0.0000 = 0.0000`. The button's
label promises the strongest example and delivers an arbitrary edge position with no indication that
clamping occurred.

Note the coincidence that makes this worse: blocks 1–3 are exactly the blocks whose stored traces have
a zero learned correction (§N3).

### N3 — The flagship cell calculation shows `0.0000 + 0.0000 = 0.0000` on first load [VERIFIED in browser]

On page load (`stage = 1`, `globalCenter = 1058`), the "ONE EXACT CELL CALCULATION" panel renders:

```
three tap sums  −0.3510   +   bias  −0.1295   →   ReLU correction  0.0000
                                              +   shortcut  0.0000   =   0.0000
```

Both lanes are zero. The centerpiece of the page — the panel titled *"How three feature neighborhoods
communicate in Block 1"* — demonstrates a block computing nothing, at the default coordinate, before
any user interaction. Clicking "Jump to this channel's strongest example" (N2) also yields all zeros.

The underlying stored traces confirm the pattern is structural, not incidental:

| Preset | blocks with **exactly zero** ReLU correction at the traced cell |
|---|---|
| **k562-peak** (default) | **1, 2, 3, 6, 7** — 5 of 8 |
| synthetic | **1, 2, 3, 4, 5** — 5 of 8 |
| gm21515 | 5, 6 — 2 of 8 |

Cause is unchanged from my prior audit: selecting the **argmax** cell systematically selects the cell
the shortcut carried forward untouched (for stem→res3 on K562, the block max *equals* the stem max,
1.1828, exactly). Choosing "the strongest cell" preferentially finds where the block did the least.

**Note the accidental good news:** the new **GM21515 preset is a far better teaching example** (only
2 of 8 degenerate, and blocks 1–4 all show genuine mixing). Making GM21515 the default, or selecting
the demo cell by `max(ReLU(conv))` rather than `max(output)`, would fix the pedagogy at a stroke.

### N4 — Receptive-field reach band mixes two coordinate units [VERIFIED — unchanged from prior audit]

`dilation-trace/page.tsx:179`: `const half = (RECEPTIVE_FIELDS[stage] - 1) / 2;`

`RECEPTIVE_FIELDS` is in **input bases**; the plotted dots (`:168-177`) are **stem-feature peak
positions**, i.e. centers of 21-base windows. The correct half-width for stem-feature centers is the
cumulative dilation sum `(RF − 21)/2`, not `(RF − 1)/2`.

| Stage | code `half` | true reach | overstatement |
|---|---|---|---|
| Stem | 10 | **0** | band drawn where reach is a single point |
| Block 1 | 12 | **2** | **6×** |
| Block 2 | 16 | 6 | 2.7× |
| Block 3 | 24 | 14 | 1.7× |
| Block 8 | 520 | 510 | 1.02× |

The prose caption ("This map shows possible reach through the stacked kernels… does not claim that
every reachable feature affected the prediction") is exactly the right hedge — the arithmetic just
needs to catch up to it.

### N5 — "no-bias" still collides with the bias terms shown throughout [VERIFIED]

`page.tsx:869` — *"published no-bias fold-0 checkpoint"*. Browser-confirmed as the sole occurrence;
nothing defines it anywhere in the app or README. `chrombpnet_nobias` means the Tn5/DNase **enzymatic
sequence-bias model has been factored out**, not that the network lacks bias parameters. The same app
prominently displays `bias −0.082` (stem, `:343`), residual `bias` (`:573`), `count_dense_bias`
(`:624`), and `profile bias 7.80105` (`:677`). One clause fixes it.

---

## 3. Misleading visual encodings

### N6 — The logits→softmax panel still teaches the opposite of what softmax does [VERIFIED with measured pixels]

`page.tsx:611-613`. I measured the rendered bar heights in the live DOM:

| Panel | rendered min px | rendered max px | min/max |
|---|---|---|---|
| **1. Raw logits** | **33.3** | 75.0 | **0.444** |
| 2. Profile probabilities | 3.3 | 75.0 | 0.044 |
| 3. Expected counts | 3.3 | 75.0 | 0.044 |

K562 logits are **all positive** (3.71 – 9.63) and softmax is **shift-invariant**, so only differences
between logits carry meaning; their zero point is arbitrary. Anchoring them at zero compresses the
entire structure into the top 56% of the axis, and the probability panel directly below then looks
dramatically spiky. The reader's takeaway is *"softmax created the peak."* It did not — softmax is
monotone, and I verified `argmax(logits) === argmax(probabilities)` on this data.

Second instance, same defect: `page.tsx:706` renders the same logits through `SignalCanvas` with
`signed`, where `min = Math.min(...visible, 0)` evaluates to 0 for all-positive logits, again forcing
a zero baseline.

Worth noting that the GM21515 preset accidentally masks this (its logits reach 0.044, so the panel
looks spiky too) — the flaw is only visible on the default preset.

**Fix:** plot `logit − max(logit)` on its own axis, labeled "only differences matter — softmax ignores
any constant shift," and drop `signed` (sign is meaningless for a shift-invariant quantity).

### N7 — `MiniBars` bin-averaging understates the headline peak by 25% [VERIFIED]

`page.tsx:582-589` averages ~6.25 positions per bar. True `profile_signal` peak **52.89** expected
cuts renders as **39.65**. For `signed` series, positive and negative values cancel *inside* a bin
before `Math.abs` is taken for the height, so a bin containing a large + and a large − draws as a
zero-height bar. Use max-magnitude (or a min/max envelope) per bin for peak-carrying series.

### N8 — Layer magnitude growth of 7.2× is invisible [VERIFIED]

`page.tsx:414` normalizes each layer by its own `tensor.max`:

| stem | res1–res3 | res4 | res5–res7 | res8 |
|---|---|---|---|---|
| 1.183 | 1.183 | 1.418 | 3.000 | **8.505** |

Stepping the Inspector's Layer dropdown from Stem to Residual 8 shows sparsity changing correctly
(well labeled) but implies magnitudes are stable. They grow 7.2×. There is no shared-scale toggle in
the Inspector (the filmstrip has one; it defaults to the misleading setting). This discards a genuinely
interesting real result: a sum-based skip connection *should* accumulate magnitude.

### N9 — Weights and contributions remain visually identical, and the "×" is not multiplicative [VERIFIED]

`page.tsx:562-566`. The `features × weights = products` row still renders all three strips with
**independent** per-strip normalization (`VectorCanvas:500`), and both the weights and the products
strips use `signed` — the same colormap. A reader cannot distinguish a learned parameter from a
computed contribution, and multiplying the left two images does not produce the right one. (The strips
are now consistently permuted by `channelOrder`, which is a real improvement in a different dimension.)

### N10 — Basset's top-12 prediction chart excludes the very cell type the section walks through [VERIFIED]

`basset/page.tsx:412` renders `top_predictions.slice(0, 12)` and applies `styles.k562` when
`item.label === "K562"`. **K562 ranks 13th** (0.9061), one place below the cut:

```
1. H1-hESC 0.973  2. H9ES 0.950  3. ESC.H1 0.948 … 12. KID.FET 0.911  |  13. K562 0.906
```

So the highlight branch is dead code, and a reader who has just followed the entire Dense-1 → Dense-2 →
K562-reader walkthrough sees a bar chart of twelve *other* cell types. The aside separately reports
K562 = 0.906, which reads as a contradiction. The model genuinely thinks this HOXA-boundary sequence is
most accessible in ESC/iPS lines — that is worth *saying*, not hiding behind an off-by-one slice.

### N11 — Basset's stem equation skips the one step that makes the arithmetic checkable [VERIFIED]

`basset/page.tsx:355` renders: `sum products 3.178` **+** `conv bias 2.619` **→** `batch norm 1.817`
**→** `ReLU`. A reader computes 3.178 + 2.619 = 5.797 and sees 1.817. The intermediate
`after_raw_bias` is present in the data and is **displayed in the Dense-2 panel** (`:405`, "before
batch norm"), so the app is internally inconsistent about it. Same omission in the Conv2 mixing
equation (`:380`).

### N12 — Minor encoding issues, all unchanged

- **Palette collision** (`page.tsx:20` vs `:66`): coral = base A **and** positive weight **and** mid-high
  activation heat; blue = base C **and** negative weight. In `StemStory` the base-colored
  `.sequence-cells` row sits directly on top of the sign-colored `.selected-weights` row.
- **Gain clipping** (`:48`): default sqrt × 1.8 saturates everything above 31% of layer max.
- **Zero renders as signal** (`:589`): `heatColor(.25 + .75·v/max)` gives v = 0 a visible tint.
- **Marker skew** (`:155`): `start / (length − 21)` in a full-width ribbon, ~1% off at the right edge.

---

## 4. Architecture fidelity

### ChromBPNet — correct [VERIFIED]

All nine tensor lengths, all eight dilations, the crop-per-edge rule, the absence of a post-add
activation, both heads, and the cumulative receptive fields re-derived independently and all correct.
`→ profile head: 1,115 bp` (`page.tsx:577`) = 1041 + 74 ✓. Head arithmetic reproduces stored outputs on
all three presets: `Σ profile_signal` = `predicted_total_count` to 3 d.p., `Σ probabilities` = 1.000,
`expm1(logcount)` = stored total exactly.

### Basset — correct, and unusually well evidenced [VERIFIED]

Every dimension in `docs/basset-checkpoint-adapter.md` and on the page matches a real Torch7 Basset
checkpoint, and I re-derived the full receptive-field/stride chain independently:

| Stage | shape | RF | spacing | my check |
|---|---|---|---|---|
| conv1 (k=19, valid) | 300 × 582 | 19 | 1 | ✓ 600−19+1 |
| pool1 (3) | 300 × 194 | 21 | 3 | ✓ 19+(3−1)·1 |
| conv2 (k=11) | 200 × 184 | 51 | 3 | ✓ 21+(11−1)·3 |
| pool2 (4) | 200 × 46 | 60 | 12 | ✓ 51+(4−1)·3 |
| conv3 (k=7) | 200 × 40 | 132 | 12 | ✓ 60+(7−1)·12 |
| pool3 (4) | 200 × 10 | 168 | 48 | ✓ 132+(4−1)·12 |
| flatten | 2,000 | — | — | ✓ 168+9·48 = 600 exactly |

**Dense-layer calculations — all exact:**

| Check | Result |
|---|---|
| Stem: page's `(dot + bias − μ)·σ⁻¹·γ + β`, ReLU vs stored track, **all 582 positions × 8 filters** | max err **1.2 × 10⁻⁷** ✓ |
| Max-pool: `pooled_track[i] == max(track[3i..3i+2])`, all 194 × 8 | **0 mismatches** ✓ |
| Conv2: Σ 3,300 contributions vs `sum_products` | exact ✓ |
| Dense 1: Σ 2,000 contributions vs `sum_products` | 48.357034 vs 48.357033 ✓ |
| **Dense 1 flatten order** | `pool3[channel·10 + position]` (**channel-major**) → err 3.7 × 10⁻⁷ ✓; the position-major alternative errs by **2.37** — the choice is consequential and correct |
| Dense 1 JSON weights vs `dense1_weights.f32.gz` row 147 at index `c·10+p` | 5.0 × 10⁻⁸ ✓ |
| Dense 2: contributions = w × act; Σ vs `sum_products`; BN recomputed | all ✓ |
| Output: K562 logit = Σ + bias; `sigmoid(2.26684)` = 0.906093 = stored | ✓ |

The `verification` block reports a NumPy↔TensorFlow max absolute disagreement of **9.95 × 10⁻¹⁴**
across 21 named intermediate states — an independent-implementation cross-check, which is a
substantially stronger guarantee than a snapshot test.

---

## 5. Model-audit metrics — verified, with one caveat

All internally consistent [VERIFIED]: `tap_energy_fraction` sums to 1.000000000 in all 16 blocks;
`Σ count_contribution` = `logcount − bias` to 6 d.p. on both checkpoints; registry `count_weight`
matches `heads.count_weights` (6 × 10⁻⁸); CKA matrices are symmetric with unit diagonal and lie in
[0, 1]; `value_quantiles` are monotone; `channel_orders` are genuine permutations and differ correctly
between checkpoints.

The **"nearly equal thirds" tap-energy claim is well supported** — max deviation from exactly 1/3 is
2.24 pp (K562) and 1.80 pp (GM21515) across all 16 blocks. The accompanying note ("Every bar uses the
full 0–100% width with no magnification") is exactly the right disclosure, and there is a regression
test asserting the absence of the old magnified scale. This is model behavior for how to present a
near-null result.

**Caveat — N13 [VERIFIED]:** `page.tsx:862` maps the preset to an audit checkpoint as
`preset === "gm21515" ? "gm21515" : "k562-peak"`. The **synthetic preset therefore borrows K562-peak's
channel orders**. Those orders (`stem_occupancy`, `final_rms`, `profile_influence`,
`count_influence`) are *locus-dependent empirical* metrics. On the synthetic sequence the dropdown
label says "Stem activation occupancy" while the ordering is occupancy measured on a **different
sequence**. Either compute a synthetic entry or disable the empirical orders for that preset.

---

## 6. Testing gaps

**N14 — No browser or interaction tests exist [VERIFIED].** The 18 Node tests are data-integrity plus
regex matches against source text; the two Python verifiers check invariants and the artifact. None of
them render a component. Every defect in §2 and §3 is invisible to the suite:

- N1 (false correlation sentence) — needs a render with `preset = gm21515`
- N2 (clamped jump button) — needs a click
- N3 (all-zero default cell) — needs a render
- N6 (bar-height ratios) — needs computed styles
- The hooks violation — needs a state transition

**N15 — `npm test` does not run ESLint**, and ESLint reports **13 errors** including the
`rules-of-hooks` violation. Adding `npm run lint` to the test script is a one-line change that would
have surfaced it.

**N16 — `npm test` is not reproducible from a clean clone.** It hardcodes
`models/.extract-env/bin/python`, and `models/` is `.gitignore`d (line 4). A fresh clone's `npm test`
fails at step 2. Guard it, or document the bootstrap.

**N17 — The Rules-of-Hooks violation is a latent trap [LATENT].** `dilation-trace:224/233`. Benign
today only because the guarded region holds exactly one stateless `useMemo`. Adding a second hook, or
converting it to `useState`/`useEffect`, turns it into a hard crash on the Stem↔Block transition. Fix
it now while it is a one-line move.

**N18 — `informationContent` silently ignores its `background` argument [LATENT].** `model-analysis.ts`
takes `background` then executes `void background` and returns `2 − H`, i.e. it assumes a uniform
background. `ActivationLogo` (`page.tsx:213`) passes `motif.background_frequencies`. Unreachable today
(`activation_motifs.motifs` is `[]` and status is `not_generated`), but it will silently produce wrong
letter heights the moment real motifs land with a non-uniform background.

**N19 — Payload [VERIFIED].** `/dilation-trace` fetches **all nine** tensors unconditionally on mount:
**2.5 MB** compressed → **32.9 MB** decompressed in the JS heap, before any interaction. `public/data`
totals **110 MB** in git (all of it referenced — no waste, but heavy for clone and deploy).

---

## 7. Strong sections — preserve these

| # | What | Where |
|---|---|---|
| **S1** | **The evidence ladder.** Three named claim strengths (`descriptive` / `mechanism` / `biology`) with definitions, plus a per-panel scope banner stating checkpoint, sample size ("1 exact locus"), and an explicit interpretation limit. I have not seen this done in a public model explainer. | `model-audit:203-206, 196-200` |
| **S2** | **Refusing to fabricate activation motifs.** The activation-logo slot renders a labeled empty state with the planned corpus, reservoir size, and selection rule rather than inventing a logo. Enforced by a test (`status === "not_generated"`). | `page.tsx:222-229`, `model-audit:240-243` |
| **S3** | **The Basset numerical gate.** Two independent implementations (NumPy + TensorFlow) compared across 21 named states at 9.95 × 10⁻¹⁴, with checkpoint SHA-256 published in both the doc and the UI. | `docs/basset-checkpoint-adapter.md`, `basset:338` |
| **S4** | **The orientation convention block.** States cross-correlation explicitly, gives the index formula, notes the base axis is a feature channel and is never rotated, defines reverse complement, and links the TF docs — *and* `verify_model_analysis.py::test_orientation` proves it with an asymmetric toy kernel. | `page.tsx:339` |
| **S5** | **Exact reparameterized weight logo.** Per-position mean removal with the compensating bias printed (`adjustedBias`), so the logo is an exact reparameterization rather than a cosmetic recentering — with a numerical test. | `page.tsx:329`, `model-analysis.ts`, `verify_model_analysis.py::test_exact_centering` |
| **S6** | **The immutable channel registry.** Display rank is separable from identity; permutation is applied consistently to producer and consumer axes; "equal IDs across independently trained checkpoints do not imply correspondence" is stated outright; and `test_global_permutation_invariance` proves the permutation is applied to *both* axes. | `model-audit:216-219`, `model-analysis.ts` |
| **S7** | **The `display_transform` storage design.** 35 tensor states served from 18 files by deriving `_bias` and `_relu` from `_conv` + `channel_bias` in the browser. Zero orphans on disk. Elegant and honest. | `page.tsx:92-105`, `tensor-loader.ts` |
| **S8** | **The Computation-state control.** `conv → +bias → ReLU → +shortcut` as four real inspectable tensors, with signed rendering auto-enabled for the pre-ReLU stages. This is the single best pedagogical addition since the prior version. | `page.tsx:779, 729` |
| **S9** | **"Where is dropout?"** Explicitly answers a question every careful reader has, and explains why it is the identity here. | `basset:411` |
| **S10** | **The `padW = 0` finding.** Notes that current repo code *can* build padded models but the published checkpoint stores valid convolutions, and treats the stored graph as authoritative. Exactly the right instinct. | `basset:346`, adapter doc |
| **S11** | **The two-model thesis line.** "preserve a long spatial grid and communicate with dilation *versus* progressively compress positions, then communicate globally through dense layers." This is the reusable mental model, stated in one sentence. | `basset:341`, §7 compare grid |
| **S12** | **Attribution boundary,** retained everywhere and never softened. | `page.tsx:894`, `basset:420`, `dilation-trace:363` |

---

## 8. Does the two-model comparison produce a reusable mental model? (question 9)

**Largely yes — this is the biggest gain over the prior version.** The contrast is correctly framed
(resolution-preserving dilation vs. resolution-destroying pooling + dense), correctly dimensioned on
both sides, and stated in one memorable sentence. Three things still block it from landing fully:

1. **It is one-directional.** `/basset` links to and contrasts with ChromBPNet; the ChromBPNet pages
   never state the contrast. A reader who never clicks "Basset model ↗" gets no comparison at all.
   Put the thesis line on the main page too.
2. **No shared vocabulary panel.** The two pages use different words for the same idea — Basset says
   "receptive field / center spacing"; ChromBPNet says "receptive field / dilation". A small shared
   table (*what is a channel · what is a receptive field · how does distance get crossed*) applied to
   both models would convert two explanations into one framework.
3. **The output tasks differ** (base-resolution profile + count vs. 164 binary labels), which is a
   confound in the comparison. The pages never say that the *task* is what forces the architecture.
   That single sentence is the payoff of the whole comparison.

---

## 9. Recommended revisions, ordered by impact

**Tier 0 — correctness (before any external use)**

1. **N1** Derive the correlation sentence from the value. It is currently false on GM21515 — on the
   page about claim calibration.
2. **N4** `half = (RECEPTIVE_FIELDS[stage] − 21) / 2`, relabel the band "reach in stem-feature centers."
3. **N2** Widen the tracked-coordinate range per stage, or disable/annotate the jump button when the
   target is out of range. It currently misreports.
4. **N5** Define "no-bias" in one clause.
5. **N15/N17** Add `npm run lint` to `npm test`; move the `stage === 0` guard below the `useMemo`.

**Tier 1 — the biggest teaching wins**

6. **N3** Select the residual demo cell by largest *learned correction*, not largest output — or make
   **GM21515 the default preset**, which alone takes the degenerate blocks from 5/8 to 2/8.
7. **N6** Re-baseline the logits charts (`logit − max`) in both locations; add one sentence on
   shift-invariance.
8. **N10** Show 13+ predictions, or pin K562 into the chart with a "rank 13" annotation and say plainly
   that the model favors ESC/iPS for this sequence.
9. **N11** Show `after_raw_bias` in the Basset stem and Conv2 equations, matching the Dense-2 panel.
10. **§8** Put the two-model thesis line on the main page; add the shared-vocabulary panel and the
    "the task forces the architecture" sentence.

**Tier 2 — encoding integrity**

11. **N9** Share one normalization across each tap's three strips; distinguish weights from contributions.
12. **N7** Bin by max-magnitude for peak-carrying series. **N8** Add a shared-scale toggle to the
    Inspector and surface the 1.18 → 8.51 growth as a finding. **V4** Flip `sharedScale` to default true.
13. **N12** Split the palette (diverging coral↔blue for signed values only; a distinct qualitative
    scheme for A/C/G/T); disclose gain clipping; fix the zero tint and the marker skew.

**Tier 3 — structure, tests, docs**

14. **N14** Add browser interaction tests. The five defects above are all one assertion each.
15. **N13** Fix the synthetic preset's borrowed channel orders. **N18** Fix `informationContent`
    before real motifs land. **N16** Make `npm test` work from a clean clone.
16. **T1** Add the `4 × 21` vs `512 × 3` side-by-side kernel panel — still the missing keystone.
17. **T2/T5** Fix the dual-meaning `start` label and the 75-vs-1,115 bp footprint caption.
18. **T3** Unify the two colormap functions; fix the 1,2,3,—,5 numbering.
19. **N19** Lazy-load `/dilation-trace` tensors per stage rather than all nine on mount.

---

## 10. Questions requiring author judgment

1. **Should GM21515 become the default preset?** It is a better teaching example on the metric that
   matters most (2 vs 5 degenerate residual blocks) — but K562 is the better-known cell line and pairs
   with the Basset K562 reader. This is a real trade-off I should not resolve for you.
2. **N3 — is the argmax selection rule deliberate?** "The shortcut preserves strong features untouched"
   is a legitimate and interesting lesson. If that is the intent, it needs to be *stated* as the lesson
   and paired with a counterexample, rather than being the silent default.
3. **What is the `[558, 1557]` slider range meant to represent?** If it is "coordinates where a profile
   prediction exists," that is defensible and should be labeled. If it was meant to bound the backbone,
   it is too narrow (N2).
4. **Is the count–profile correlation worth promoting to a finding?** GM21515's r = −0.33 (p ≈ 3e−15)
   says the two heads lean on partly different channels. That is more interesting than the null it is
   currently reported as — but it is n=1 locus for the activation-dependent half.
5. **Basset targets:** `top_predictions` holds only 15 entries. Was 15 chosen deliberately, and should
   the UI expose rank rather than a fixed slice?
6. **Schema versioning:** `basset-demo.json` uses `schema_version: 1` (number); `model-audit-summary.json`
   uses `"1.0.0"` (string). Worth unifying before either is consumed elsewhere.
7. **How much of the 110 MB should ship?** Every byte is referenced, so nothing is waste — but a
   `git clone` and a deploy both carry it. Is an LFS or fetch-on-demand split wanted before publication?
8. **Which medium leads?** The main page is a guided narrative, `/dilation-trace` an exploratory lab,
   `/model-audit` a technical workspace, `/basset` a second narrative. For a blog and a video these want
   different edits; for documentation they want merging.

---

### Bottom line

This is a substantially different and much stronger project than the snapshot I audited previously.
**Both architectures are correct** — I re-derived every ChromBPNet dimension and the complete Basset
forward chain, including the consequential channel-major flatten, and found **no architectural or
arithmetic error**. The Basset adapter's dual-implementation gate at 9.95 × 10⁻¹⁴, the evidence ladder,
the immutable channel registry, and the refusal to fabricate activation motifs are all better than
standard practice for public model explainers.

The defects are concentrated in **presentation and one hardcoded interpretation**:

- **one false statistical claim** that appears only on the second checkpoint (N1),
- **one units error** in the reach map (N4),
- **one navigation control that silently misreports** (N2),
- **a default view of the flagship panel that shows the model doing nothing** (N3),
- **one chart that still teaches the opposite of the truth** (N6),
- and **an undefined term that contradicts the page around it** (N5).

None require rearchitecting. N1, N2, N4, N5 and the lint fix are roughly an afternoon; they are also
the five that a hostile reader would find first. The most valuable single change is not on that list:
**switching the residual demo's selection rule (or the default preset)**, which converts the project's
weakest panel into its intended centerpiece.

I did **not** reproduce the crash I predicted in my previous audit. The hooks violation is real and
worth fixing, but it does not currently break anything — the correction is recorded in §1.
