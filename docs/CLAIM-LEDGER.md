# Sequence CNN Explainer — claim ledger

## Purpose

This ledger controls what the project may say in technical documentation, a blog, slides, narration, captions, and social posts. It prevents an attractive visualization from silently becoming a stronger scientific claim than its evidence supports.

Before publishing a claim, use the approved wording or wording of equal strength. If a claim is marked **pending**, do not convert it into a conclusion.

## Evidence and status codes

| Code | Meaning |
|---|---|
| **S — structural** | Read from the published checkpoint graph/weights and independently checked |
| **X — exact example** | Recomputed for one displayed sequence and checkpoint |
| **D — descriptive** | Summary of weights or observed activations; association, not causation |
| **M — mechanistic** | Prediction change under attribution, ablation, or controlled perturbation |
| **B — biological** | Supported by motif evidence and/or external experimental knowledge |

| Status | Publication rule |
|---|---|
| **Approved** | May be stated directly with its listed scope |
| **Qualified** | May be stated only with the required qualifier |
| **Pending** | Present as a question, hypothesis, or future analysis only |
| **Prohibited** | Do not use; it collapses distinct evidence levels or is false |

## Architecture and computation claims

| ID | Claim | Evidence | Status | Approved wording and required scope | Do not say |
|---|---|---|---|---|---|
| A01 | ChromBPNet input and outputs | S | Approved | “This checkpoint reads 2,114 one-hot DNA bases and predicts a 1,000-position profile plus one log-count.” | “It predicts a 2,114-position profile.” |
| A02 | ChromBPNet stem dimensions | S, X | Approved | “The 21-base stem convolution maps `4 × 2,114` one-hot DNA to `512 × 2,094` activations.” | “The picture is a downsampled 2-D image.” |
| A03 | Convolution orientation | S, X | Approved | “The model applies cross-correlation: stored kernel positions are used left to right without reversal.” | “Every CNN kernel must be rotated 180 degrees before interpretation.” |
| A04 | Stem intuition | S | Qualified | “Stem filters act as local sequence-feature detectors; some may respond to motif-like patterns.” | “Each stem channel is one complete transcription-factor motif.” |
| A05 | Residual kernel shape | S | Approved | “Each residual kernel has three spatial taps, 512 input channels, and 512 output channels: `3 × 512 × 512`.” | “It is a one-dimensional filter that reads one channel.” |
| A06 | Dilation | S, X | Approved | “At dilation `d`, the three taps read positions separated by `d`: `o`, `o+d`, and `o+2d`.” | “The stored kernel contains `d` learned zeros between taps.” |
| A07 | Cross-channel mixing | S, D | Approved | “Every residual output channel can mix all 512 input channels at each of its three sampled positions.” | “Motifs cannot communicate because the later convolution is one-dimensional.” |
| A08 | Shortcut equation | S, X | Approved | “The block adds its ReLU-clipped transform to the center-cropped, same-channel shortcut.” | “The shortcut is another convolutional filter.” |
| A09 | Monotone residual update | S, X | Approved | “For an aligned channel and coordinate, the block output is at least its shortcut value because the added correction is nonnegative.” | “Every visually aligned pixel must become brighter”; display scales and tensor cropping can differ. |
| A10 | No post-addition ReLU | S | Approved | “In these checkpoints, ReLU is applied to the transform before shortcut addition; there is no additional ReLU after the merge.” | “All residual architectures use this exact ordering.” |
| A11 | ChromBPNet backbone receptive field | S | Approved | “The final residual tensor has a theoretical receptive field of 1,041 input bases per cell.” | “Every one of those bases materially affects every cell.” |
| A12 | ChromBPNet profile footprint | S | Approved | “One profile logit reads 75 final-tensor positions × 512 channels and has a combined theoretical DNA footprint of 1,115 bp.” | “The profile kernel reads only 75 DNA bases.” |
| A13 | Profile head | S, X | Approved | “The profile head maps `512 × 1,074` to 1,000 logits, and softmax turns them into a positional distribution.” | “Softmax creates the location of the peak.” |
| A14 | Count head | S, X | Approved | “The count head averages each of 512 channels over 1,074 positions, then uses a dense layer to produce one log-count.” | “The count head reads the profile prediction.” |
| A15 | Parallel heads | S | Approved | “Profile and count are parallel readers of the same final feature tensor.” | “The profile output flows into the count output,” or the reverse. |
| A16 | Expected count profile | S, X | Approved | “Expected count at a position equals its profile probability multiplied by the predicted total count.” | “The profile probability alone is the expected cut count.” |
| A17 | `nobias` naming | S | Approved | “`chrombpnet_nobias` means the assay-specific enzyme-bias component is factored out; standard neural-layer biases remain.” | “This model has no bias parameters.” |

## Visualization and interpretation claims

| ID | Claim | Evidence | Status | Approved wording and required scope | Do not say |
|---|---|---|---|---|---|
| V01 | Bright activation | X | Qualified | “This channel is strongly active at this position for the displayed sequence.” | “This bright cell caused the prediction.” |
| V02 | A local product | X | Qualified | “This activation × weight term contributes to the selected internal preactivation or logit in the exact forward calculation.” | “This is a SHAP/DeepLIFT attribution” or “this proves biological importance.” |
| V03 | Whole-tensor comparison | X | Qualified | “With a shared magnitude scale, color intensity is comparable across the selected layers.” | “The same color is quantitatively comparable when per-layer scaling is enabled.” |
| V04 | Contrast control | X | Approved | “Brightness and weak-value transforms change visibility only; they do not change stored values.” | “Turning up contrast increases activation.” |
| V05 | Channel sorting | S | Approved | “Sorting changes presentation order only; immutable channel IDs and all associated axes remain synchronized.” | “Rank 1 is Channel 1,” unless original order is selected. |
| V06 | Cross-checkpoint channels | S | Approved | “Channel identity is checkpoint-specific; same-numbered channels from independently trained models are not assumed to correspond.” | “K562 Channel 118 is the same feature as GM21515 Channel 118.” |
| V07 | Weight logo | S, X | Approved | “The signed weight logo is an exact position-centered reparameterization when the removed means are transferred to the bias.” | “The weight logo is a probability matrix.” |
| V08 | Activation motif | — | Pending | “An activation motif will summarize a documented corpus of top-activating, aligned sequence windows once that artifact is generated.” | “The current weight logo shows sequences that activate the filter in the genome.” |
| V09 | Attribution motif | — | Pending | “Attribution motifs require separately generated prediction-contribution scores and motif discovery.” | “Activation alone is attribution.” |
| V10 | Reverse complement | S, X | Approved | “Reverse complement reverses positions and swaps A↔T and C↔G; applying it twice restores the original.” | “Reverse complement rotates the base axis without exchanging bases.” |
| V11 | Selected residual examples | X | Qualified | “The balanced example was deliberately selected to make both paths visible.” | “Most cells have both paths active,” unless corpus statistics establish it. |
| V12 | CKA | D | Qualified | “Pilot linear CKA compares patterns over 128 aligned positions and each layer's 64 highest-variance channels.” | “CKA proves that individual channels correspond,” or “this compares every tensor entry.” |

## Current ChromBPNet observations

| ID | Claim | Evidence | Status | Approved wording and required scope | Do not say |
|---|---|---|---|---|---|
| C01 | K562 sparsity trajectory | X, D | Qualified | “For the displayed K562 fold-0 locus, exact zeros decrease from 98.63% at the stem to 88.44% after Residual 8.” | “ChromBPNet sparsity always decreases this way.” |
| C02 | GM21515 sparsity trajectory | X, D | Qualified | “For the same displayed DNA window under the GM21515 fold-0 model, exact zeros decrease from 98.13% to 81.87%.” | “GM21515 is generally less sparse than K562.” |
| C03 | K562 active channels | X, D | Qualified | “At this K562 locus, median active channels per position increase from 7 at the stem to 58 after Residual 8.” | “Residual blocks activate 51 biologically meaningful motifs.” |
| C04 | GM21515 active channels | X, D | Qualified | “At this GM21515-model locus, the corresponding values are 9 and 92.” | “ATAC-seq models always activate more channels than DNase-seq models.” |
| C05 | Scope of C01–C04 | X | Approved | “These are single-locus descriptive observations, not population or biological conclusions.” | Any omission of the one-locus qualifier in a results claim. |
| C06 | Tap energy similarity | D | Qualified | “Across the complete stored kernels, each of the three taps carries roughly one third of squared weight energy.” | “All taps have the same values,” or “all taps contribute equally on every input.” |
| C07 | K562 tap-energy range | D | Qualified | “K562 tap-energy fractions range from about 32.2% to 35.6% across blocks.” | “Differences are large,” or a chart scale that visually exaggerates sub-percentage-point differences. |
| C08 | Residual diagonal energy | D | Qualified | “Same-index channel connections hold about 0.24–0.40% of K562 residual squared-weight energy, mildly above the 0.195% share expected if energy were spread evenly across channel pairs.” | “The model avoids same-channel connections,” or presenting 0.24% as depletion without the `1/512` baseline. |
| C09 | Effective input channels | D | Qualified | “Entropy-based effective input-channel counts span 329–379 of 512 across K562 blocks and 329–406 across GM21515 blocks, supporting broad weight distribution across input channels.” | “This is the exact number of biological features used,” or “every listed channel causally affects the prediction.” |
| C10 | K562 count/profile correlation | D | Qualified | “For the K562 checkpoint, correlation between absolute count weights and profile channel-energy magnitude is `r = −0.081`, a very weak negative association.” | “The two heads agree by −8.1%,” or “the heads make opposite predictions.” |
| C11 | GM21515 count/profile correlation | D | Qualified | “For the GM21515 checkpoint, the same weight-summary correlation is `r = −0.330`, a moderate negative association.” | “This proves biological specialization of the heads.” |
| C12 | Meaning of count/profile correlation | D | Approved | “The statistic compares two channel-level weight magnitudes; it does not compare outputs, accuracy, or causal importance.” | “Count/profile agreement measures prediction agreement.” |
| C13 | Long horizontal lines | X, D | Qualified | “A long line indicates one channel stays positive across many adjacent positions under the chosen state and display scale.” | “A long line is a long DNA motif,” or “a single enhancer.” |
| C14 | Positive-run count | D | Approved | “A positive run is one maximal contiguous segment above `10⁻⁷` within one channel; counts are summed over channels.” | “Positive runs count motifs.” |

## Basset claims

| ID | Claim | Evidence | Status | Approved wording and required scope | Do not say |
|---|---|---|---|---|---|
| B01 | Basset task | S | Approved | “The published checkpoint maps a 600-base sequence to 164 cell-type accessibility probabilities.” | “It predicts a base-resolution accessibility profile.” |
| B02 | Basset convolutional hierarchy | S, X | Approved | “Three valid convolutions mix channels; three max-pooling stages reduce the position axis from 600 bases to ten pooled positions.” | “Pooling keeps exact base-level coordinates.” |
| B03 | Basset long-range communication | S | Approved | “After pooling, Dense1 reads all `200 × 10 = 2,000` remaining cells, allowing features from all covered regions and channels to interact.” | “Ordinary CNNs cannot combine distant motifs without dilation.” |
| B04 | Why two dense layers | S | Qualified | “Dense1 creates 1,000 global combinations of the pooled convolutional representation; Dense2 recombines those into another 1,000-dimensional representation before 164 output readers.” | “The two dense layers correspond to two known biological stages.” |
| B05 | Basset receptive coverage | S | Approved | “The ten Pool3 positions jointly cover the complete 600-bp input; each has a 168-bp receptive field and centers are 48 bp apart.” | “Each pooled cell sees the whole sequence.” |
| B06 | Exact checkpoint adapter | X | Approved | “Independent NumPy and TensorFlow forward passes agree to a maximum absolute error of `9.95 × 10⁻¹⁴` in float64.” | “The browser itself computes with 10⁻¹⁴ precision”; browser tensors are float32. |
| B07 | K562 output example | X | Qualified | “For the official tutorial sequence, the K562 reader outputs approximately 0.9061 and ranks 13th among 164 labels; across all labels, probabilities range from about 0.54 to 0.97, the median is 0.80, and 82 exceed 0.80.” | “The sequence is experimentally 90.61% accessible in K562,” or “the model specifically identifies K562.” |
| B08 | Architecture comparison | S | Approved | “Basset uses pooling plus dense global mixing; ChromBPNet uses dilated residual convolutions while preserving one-base spacing for profile output.” | “One architecture is scientifically better based on these two demos.” |

## Mechanistic and biological claims not yet earned

| ID | Proposed claim | Needed evidence | Status | Safe current wording | Prohibited shortcut |
|---|---|---|---|---|---|
| P01 | A channel is important to count or profile | single-channel ablation plus effect distribution over a held-out corpus | Pending | “This channel is a candidate for ablation based on descriptive metrics.” | Calling high RMS, weight norm, or activation × weight “importance” without qualification |
| P02 | Two motifs cooperate | matched-background single and pair insertion/deletion; non-additive prediction effect; replication | Pending | “The model architecture permits separated features to interact.” | Inferring cooperation from coactivation, dilation reach, or one bright output |
| P03 | Two motifs compete | controlled perturbation and negative/non-additive effects | Pending | “Signed weights permit enhancing and suppressing terms before ReLU.” | Inferring competition solely from negative kernel weights |
| P04 | A stem filter matches a TF family | corpus activation motif or attribution motif, reverse-complement-aware database comparison, q-value and ambiguity | Pending | “The weight pattern suggests a testable sequence preference.” | Matching raw signed weights directly to JASPAR and naming a TF |
| P05 | A TF controls accessibility at the locus | motif evidence plus perturbation and external biological support | Pending | “Candidate TF-family hypothesis.” | Treating a database match as proof of TF occupancy or causal regulation |
| P06 | The K562 and GM21515 models use different regulatory logic | multiple matched sequences, folds, controlled assay/cell-type interpretation, interventions | Pending | “The two checkpoints show different descriptive responses on this shared sequence.” | Assigning the difference specifically to cell type or assay from the current comparison |
| P07 | Later layers contain motif syntax | corpus-level representation analysis plus controlled spacing/orientation tests | Pending | “Later layers can mathematically combine features over increasing distance.” | “Long lines are discovered regulatory grammar.” |
| P08 | Effective receptive field | gradients, perturbations, or another declared influence measure over a corpus | Pending | “The theoretical receptive field is 1,041 bp in the backbone and 1,115 bp per profile logit.” | Calling theoretical reach “the bases the model used” |
| P09 | Biological reverse-complement consistency | corpus predictions and defined tolerance, with strand-aware task interpretation | Pending | “The UI supports reverse-complement inspection; corpus consistency is planned.” | Assuming invariance from one double-RC implementation test |

## Required language for public artifacts

Every blog post, slide deck, and video script should include these statements near the first scientific result:

1. **The visualizer uses published, already trained checkpoints; it does not train a new model.**
2. **Architecture and displayed arithmetic were independently recomputed.**
3. **Current activation statistics are one-locus pilots unless a corpus sample count is stated.**
4. **Activation is not attribution, and neither alone proves biological causality.**
5. **Channel identities are checkpoint-specific.**

## Claim-upgrade protocol

To move a claim from pending to qualified or approved:

1. define the estimand and unit of analysis before computing it;
2. record checkpoint, fold, assay, biosample, genome build, coordinates, and sequence selection rule;
3. use held-out sequences and state the sample count;
4. add matched controls and reverse complements where appropriate;
5. report distributions and uncertainty, not only selected examples;
6. use an intervention when the wording implies mechanism;
7. replicate principal findings across folds or label them fold-specific;
8. add the generating script, artifact schema, numerical test, and ledger entry;
9. obtain human review before upgrading public wording.

## Source map

- Exact ChromBPNet forward export: [`scripts/run_chrombpnet_checkpoint.py`](../scripts/run_chrombpnet_checkpoint.py)
- ChromBPNet audit metrics: [`scripts/build_model_audit.py`](../scripts/build_model_audit.py)
- Kernel-logo and permutation invariants: [`scripts/verify_model_analysis.py`](../scripts/verify_model_analysis.py)
- Basset export: [`scripts/export_basset_demo.py`](../scripts/export_basset_demo.py)
- Basset independent checks: [`scripts/verify_basset_adapter.py`](../scripts/verify_basset_adapter.py)
- Source/numerical regression suite: [`tests/rendered-html.test.mjs`](../tests/rendered-html.test.mjs)
- Browser regression suite: [`tests/browser/explainer.spec.ts`](../tests/browser/explainer.spec.ts)
- Full technical interpretation: [`docs/TECHNICAL-DOCUMENTATION.md`](TECHNICAL-DOCUMENTATION.md)
- Independent audits: [`AUDIT-INDEPENDENT-V3-FOLLOWUP.md`](../AUDIT-INDEPENDENT-V3-FOLLOWUP.md)
