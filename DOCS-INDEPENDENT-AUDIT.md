# Independent audit — documentation set

**Repository root:** `/Users/zengweiguo/Documents/sequence_model_explainer`
**Branch:** `codex/dilation-tensor-evolution-demo`
**HEAD examined:** `018cbdcf2b03e3e0fcb25a71cecfdf2474ca50a1` — *Document sequence CNN architecture and claim boundaries*
**Working tree:** clean except one untracked Microsoft Word lock file (`docs/~$CHNICAL-DOCUMENTATION.md`, 0 bytes)

**Files audited:** `docs/TECHNICAL-DOCUMENTATION.md` (385 lines), `docs/CLAIM-LEDGER.md` (145), `docs/VOCABULARY-AND-MENTAL-MODELS.md` (144).
**Audit only — no file was modified.** Verified independently against `app/`, `scripts/`, `app/data/*.json`, `public/data/`, and `tests/`. Prior audit reports were not used as evidence.

---

# 1. Verdict: READY WITH MINOR REVISIONS

The documentation set is unusually disciplined. Every architectural and arithmetic claim I checked
reproduced exactly, including the ones easiest to get wrong:

- all ten ChromBPNet stage shapes and the `output = input − d(k−1)` rule (2,114 → 2,094 → … → 1,074 → 1,000);
- receptive fields 21 → 1,041 bp and the profile footprint `1,041 + 74 = 1,115` bp;
- cross-correlation orientation and the no-rotation-of-the-base-axis rule;
- the residual equations `Z`, `T = ReLU(Z)`, `S[f,o] = X[f,o+d]`, `Y = T + S`, no post-merge ReLU;
- `predicted_total_count = exp(logcount) − 1` (the artifact is `expm1`, not `exp` — the doc is right);
- every Basset shape, receptive field and center spacing (19/21/51/60/132/168 bp; 1/3/3/12/12/48 bp) and `168 + 9 × 48 = 600`;
- the five pilot statistics in the table at TECHNICAL-DOCUMENTATION.md:259–265, all matching `model-audit-summary.json` to the digit;
- `npm run sync:browser-data` exists (`scripts/sync_browser_json.py`) and genuinely regenerates the `.json.gz` assets the app requests.

Activation / attribution / perturbation / biological evidence are kept separate consistently and
deliberately across all three documents. That separation is the strongest thing here.

**One finding (F1) is a genuine statistical error and should be corrected before publication.** The
rest are wording and placement fixes. Nothing requires re-running an analysis or changing the
application.

---

# 2. Consequential findings, most severe first

## F1 · "Low diagonal energy strongly rejects…" — the number does not support the word "low"
**Severity: HIGH · scientific/statistical**
`docs/TECHNICAL-DOCUMENTATION.md:271` · `docs/CLAIM-LEDGER.md:76` (C08)

> "The stored residual kernels also have **low** same-index diagonal energy—about 0.24–0.40% in the
> K562 checkpoint and 0.21–0.34% in GM21515—and large entropy-based effective input-channel counts.
> This **strongly rejects** the mental model 'each output channel reads only the same-numbered input
> channel.'"

Each residual kernel is `3 × 512 × 512 = 786,432` weights, of which `3 × 512 = 1,536` are same-index.
That is `1/512 = 0.1953%` of the entries. **If squared energy were spread uniformly, the diagonal
fraction would already be 0.195%.** So the observed values are not low — they are *above* the null:

| checkpoint | observed diagonal energy | ratio to the 0.195% chance baseline |
|---|---|---|
| K562 fold 0 | 0.237% – 0.403% | **1.21× – 2.07×** |
| GM21515 fold 0 | 0.213% – 0.340% | **1.09× – 1.74×** |

(recomputed from `app/data/model-audit-summary.json`, all eight blocks per checkpoint)

The *conclusion* is safe — a strictly diagonal kernel would put ~100% of its energy on the diagonal,
so 0.24% does refute that strawman. But the sentence attributes the refutation to the diagonal number
being **small**, and invites the reader to infer that the model actively *avoids* same-channel
connections. The data say the mild opposite: a **slight same-channel enrichment** sitting on top of
broad mixing. A specialist will notice this immediately, and the word "strongly" makes it worse.

The claim that actually does the work is already in both documents and is under-used: the
entropy-based effective input-channel count is **329–379 of 512 (K562)** and **329–406 of 512
(GM21515)**.

**Suggested replacement (TECHNICAL-DOCUMENTATION.md:271):**

> Same-index connections carry about 0.24–0.40% of squared weight energy in K562 and 0.21–0.34% in
> GM21515. Those figures are close to the 1/512 = 0.195% share expected if energy were spread evenly
> over all entries, so they indicate at most a slight same-channel preference — not avoidance. The
> decisive evidence against "each output channel reads only its same-numbered input channel" is the
> entropy-based effective input-channel count, which is 329–406 of 512 across blocks in both
> checkpoints.

**Suggested replacement (CLAIM-LEDGER.md:76, C08 approved wording):**

> "Same-index channel connections hold about 0.24–0.40% of K562 residual squared-weight energy,
> close to the 0.195% share expected under an even spread."
> *Add to "Do not say":* "The model avoids same-channel connections," or presenting 0.24% as evidence
> of depletion without the 1/512 baseline.

**Ledger disagreement:** C08's current status is *Qualified*, but its qualifier ("Only about…") is
itself the problem. The cross-channel conclusion should be moved onto **C09** (effective input
channels), which earns it.

---

## F2 · "Profile head: a sliding global-feature reader" — "global" invites exactly the wrong picture
**Severity: MEDIUM-HIGH · mental model**
`docs/VOCABULARY-AND-MENTAL-MODELS.md:59`

The body text underneath (line 61) is accurate: *"reads 75 neighboring final-tensor columns and all
512 channels… Sliding this reader produces 1,000 logits."* The **heading** is the problem.

"Global" here means global *across channels*. For a non-specialist — and headings are what people
skim, quote, and put on a slide — the dominant reading of "global" in a sequence model is global
*across the sequence*. That is the precise misconception `CLAIM-LEDGER.md:41` (A12) exists to prevent
("Do not say: The profile kernel reads only 75 DNA bases" — but equally, it does not read all 2,114).

It also inverts the contrast with its neighbour: the **count head is the genuinely global one** (it
averages all 1,074 positions), yet it is titled "summarize then read" while the local head gets
"global". A reader who takes both headings at face value forms the opposite of the truth.

**The user's proposed phrase, "a sliding 75-position, all-channel reader", is accurate and I recommend
it.** It states both extents explicitly, "sliding" carries the 1,000 repetitions, and it cannot be
misread as whole-sequence. It also matches the application's own wording at `app/page.tsx:750`
("A learned 75-position kernel reads all 512 channels and produces 1,000 logits").

**Suggested replacement (line 59):**

> ### Profile head: a sliding 75-position, all-channel reader

and consider adding one clarifying sentence after line 61:

> Its window is local along the sequence — 75 of the 1,074 final-tensor columns — and complete across
> channels. The count head is the opposite: complete along the sequence, one number per channel.

---

## F3 · The Basset K562 example is numerically right but has no denominator
**Severity: MEDIUM · claim stronger than its evidence**
`docs/TECHNICAL-DOCUMENTATION.md:305` · `docs/CLAIM-LEDGER.md:94` (B07)

> "The K562 output probability is approximately `0.9061` and ranks 13th among the 164 labels."

Both numbers are correct — I recomputed all 164 logits from `output_weights.f32.gz` × the stored
`dense2_activations` + `output_biases` and obtained `p(K562) = 0.906093` and **rank 13 exactly**. B07's
"Do not say" ("the sequence is experimentally 90.61% accessible") is well chosen.

What is missing is the distribution the rank sits in:

| statistic over all 164 labels | value |
|---|---|
| minimum | 0.5416 |
| **median** | **0.8010** |
| maximum | 0.9734 |
| labels above 0.80 | **82 of 164** |
| labels above 0.90 | 17 of 164 |

"0.9061, rank 13 of 164" reads as strong, specific evidence for K562. In fact half the cell types
exceed 0.80 on this sequence and K562 sits inside a crowded upper tail. Since this is the single
Basset number most likely to be quoted in a blog or on a slide, the denominator should travel with it.

**Suggested replacement (TECHNICAL-DOCUMENTATION.md:305):**

> The K562 output probability is approximately `0.9061`, ranking 13th of 164 labels. That rank should
> be read against the distribution for this sequence: the 164 probabilities run from 0.54 to 0.97 with
> a median of 0.80, and 82 labels exceed 0.80. The checkpoint predicts broadly high accessibility here
> rather than singling out K562. This is one checkpoint prediction, not measured accessibility and not
> an accuracy statistic.

**Ledger disagreement:** B07 is *Qualified*, but its qualifier does not require the distribution. I
would make the distribution part of the approved wording and add to "Do not say": *"The model
specifically identifies K562,"* or a rank quoted without the 164-label spread.

---

## F4 · The magnitude-scale guidance is filed under the wrong route and misstates the default
**Severity: MEDIUM · factual / navigational**
`docs/TECHNICAL-DOCUMENTATION.md:243`

Line 243 closes the "### Model Audit route `/model-audit`" section with guidance about the shared-vs-
per-layer magnitude scale and brightness gain. Those controls do not exist on `/model-audit` —
`grep -l "Magnitude scale"` matches **`app/page.tsx` only**, i.e. the tensor inspector on `/`.
`/model-audit` has only a "Weight contrast" gain slider for the profile-kernel heatmap.

Second problem in the same sentence: *"Shared magnitude scale is the correct default for cross-layer
comparisons."* The application's actual default is **per-layer** (`app/page.tsx:779`,
`useState(false)`; the browser suite asserts `toHaveValue("layer")`). Read as prescription the
sentence is fine; read as description — which its position in a route-walkthrough encourages — it is
wrong, and a reader will hunt for a control that is on a different page.

**Suggested fix:** move the paragraph into the "### Main route `/`" section (after line 205) and
rewrite the first clause:

> The tensor inspector on `/` opens with per-layer scaling, which reveals weak structure but hides
> absolute magnitude changes between layers. Switch to the shared magnitude scale before making any
> cross-layer magnitude comparison. Contrast and weak-value transforms affect display only; a higher
> brightness gain can saturate strong values and must never be read as a quantitative increase.

---

## F5 · The coordinate vocabulary omits the app's most-used term and one whole coordinate system
**Severity: MEDIUM-LOW · terminology consistency**
`docs/VOCABULARY-AND-MENTAL-MODELS.md:109–118`

The guide's purpose is to fix coordinate language, and `line 118` gives excellent advice ("Avoid
saying only 'position 500'"). But its four-item list does not match the running application:

| in the app | in the vocabulary guide |
|---|---|
| **"input-aligned coordinate"** (`app/page.tsx`, `app/dilation-trace/page.tsx`), "Shared input coordinate" | absent — the guide says "input-relative position" |
| "input-relative coordinates" | "input-relative position" ✅ |
| "tensor position" | ✅ |
| "profile position" | ✅ |
| **"relative position 1 … 75"** — the profile-**kernel** axis | **absent** |

Two consequences. First, the term a reader meets most often on screen ("input-aligned") is undefined,
and `TECHNICAL-DOCUMENTATION.md:224` uses it while the vocabulary guide does not — the two documents
are internally inconsistent. Second, the profile-kernel heatmap axis runs 1–75 and is labelled
"relative position", which collides head-on with "input-relative position" while meaning something
completely different (offset within the kernel, not a DNA coordinate). That is the single most
confusable pair of labels in the project and the guide is silent on it.

**Suggested addition after line 116:**

> - **input-aligned coordinate:** the same 1–2,114 input frame, used when a tensor cell is being
>   mapped back to the DNA base at the centre of its receptive field. The application uses
>   "input-aligned coordinate" and "input-relative position" interchangeably; prefer one in narration.
> - **kernel-relative position:** an offset *inside* a kernel — 1–21 for a stem filter, 1–75 for the
>   profile kernel. This is not a DNA coordinate. The profile-kernel heatmap's "relative position 37"
>   is the middle of the kernel, not input base 37.

**Minor, same section:** `line 102` states "**Red** = positive, blue = negative", but every on-screen
legend says **coral** (`app/page.tsx:343, 850`; `app/model-audit/page.tsx:233`;
`app/basset/page.tsx:250, 358`). Worth aligning the word, because base **A** is also drawn in a coral
tone (`#e96b54`) very close to the signed-positive coral (`#e15b45`). Line 103 already warns that base
colors are categorical, which is the right mitigation — it just needs the same noun the UI uses.

---

# 3. Other checks performed — no finding

- **Cross-document consistency of every shared number** (2,114 / 2,094 / 1,074 / 1,000 / 512 / 75 /
  38,400 / 1,041 / 1,115 / 164 / 600 / 2,000): consistent across all three documents and the app.
- **Provenance:** both checkpoint names, repositories, experiments, biosamples, assays and folds match
  `app/data/*-activations.json`. The `chrombpnet_nobias` explanation (TECH:82, ledger A17) is correct
  and appears in the app's hero text.
- **Synthetic preset** correctly described as a different input under the K562 model, not a third model (TECH:80).
- **K562/GM21515 confound** correctly flagged as cell-type *and* assay in both documents (TECH:78, ledger P06).
- **Softmax** (TECH:170–177, ledger A13): formula, the max-subtraction rationale, and "softmax does not
  invent the peak" are all correct; I confirmed identical argmax across logits, shifted logits,
  exponentials, probabilities and expected counts for all three presets.
- **Audit-metric definitions** (TECH:230–241): exact-zero, tolerance-zero (`10⁻⁷`), median active
  channels, positive-run counting, tap energy, effective channel count, count/profile correlation and
  the CKA sampling (128 positions, 64 highest-variance channels) all match
  `scripts/build_model_audit.py` exactly.
- **Reproducibility section** (TECH:345–375): `npm run sync:browser-data` exists and matches
  `scripts/sync_browser_json.py`; commit `364cc586` exists; the three audit files exist; the
  characterisation of the follow-up verdict is accurate.
- **Basset verification** (TECH:303, ledger B06): `9.947598e-14` in the artifact vs `9.95 × 10⁻¹⁴`
  in the docs ✅, and B06's "Do not say" correctly notes the browser uses float32.

---

# 4. Disagreement with the claim ledger

Three entries, in order:

1. **C08 (residual diagonal energy) — currently *Qualified*; the qualifier does not fix it.** See F1.
   Reword to include the 1/512 = 0.195% baseline, and move the cross-channel conclusion to C09.
2. **B07 (Basset K562 example) — currently *Qualified*; the required qualifier is the wrong one.**
   See F3. The distribution (median 0.80, 82/164 above 0.80) should be part of the approved wording,
   not optional context.
3. **C09 (effective input channels) — currently *Qualified* and under-stated.** "Entropy-based
   effective input-channel counts are high" is vaguer than the evidence warrants. This is the
   strongest structural result in the audit set and deserves its number:
   *"Entropy-based effective input-channel counts are 329–406 of 512 across blocks in both
   checkpoints, so output channels draw on a large fraction of the input channels."* Keeping the
   "Do not say" as-is.

Everything else in the ledger I would approve as written. In particular A12's "Do not say: *The
profile kernel reads only 75 DNA bases*" is exactly the right guard, and V11's refusal to let a
selected example imply a frequency ("Do not say: *Most cells have both paths active*") is the kind of
entry these ledgers usually lack.

---

# 5. Especially strong material — preserve as-is

1. **The four-level evidence table** (`TECHNICAL-DOCUMENTATION.md:18–27`), and the honest self-
   assessment at line 27 that the site is strong at levels 1–2, a pilot at level 3, and that level 4 is
   a roadmap. This frames everything else correctly.
2. **The "Do not say" column** throughout `CLAIM-LEDGER.md`. Naming the specific wrong sentence is far
   more effective than a general caution, and several entries (A06 on stored zeros, A15 on head
   independence, V09 "activation alone is attribution", C13 on long horizontal lines) pre-empt exactly
   the misreadings this kind of visualization produces.
3. **The residual block section** (`TECHNICAL-DOCUMENTATION.md:121–148`) — full equations, the
   `# · # · #` picture for dilation, "each tap is a complete `512 × 512` weight matrix", and the closing
   sentence "Reachability alone is not evidence that a particular feature affected a prediction."
4. **The receptive-field derivation** (`:150–164`), especially the closing paragraph separating
   theoretical reach from effective influence, and its ledger counterpart P08 ("Do not call theoretical
   reach 'the bases the model used'").
5. **"Not supported yet"** (`:319–327`). Seven specific prohibited conclusions, including the two most
   tempting ones — the cell-type-vs-assay confound and "ChromBPNet is biologically superior to Basset".
6. **The three stem-filter view definitions** (`:245–253`), which keep weight logo, activation motif
   and attribution motif distinct and state plainly that the activation motif is unavailable rather
   than substituting a proxy.
7. **The vocabulary guide's "Terms that must remain distinct" table** (`:71–86`) and the **approved
   narrative verbs** section (`:88–98`). Binding verb choice to evidence level is the most portable
   idea in the whole set.
8. **The musical-arrangement analogy with its stated limit** (`:135–144`). Introducing an analogy and
   immediately bounding it is the right pattern.
