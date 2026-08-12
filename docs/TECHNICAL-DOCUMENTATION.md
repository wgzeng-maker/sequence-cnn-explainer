# Sequence CNN Explainer — technical documentation

## Status and scope

This document describes the implementation and scientific interpretation boundary of the Sequence CNN Explainer. The application evidence baseline is commit `364cc586`; later documentation-only commits do not change that numerical baseline.

The explainer currently contains two published sequence CNNs:

- **ChromBPNet**, a fully convolutional model that predicts a 1,000-position chromatin-accessibility profile and one total-count value from 2,114 DNA bases.
- **Basset**, a convolution-and-pooling model that predicts accessibility probabilities for 164 cell types from 600 DNA bases.

The intended reader already knows the basic image-CNN idea: a small kernel slides over an input, produces feature maps, and deeper layers combine earlier features. The explainer translates that mental model to one-dimensional DNA without pretending that a learned channel is automatically a named motif or biological mechanism.

This is an explainer and model-audit workspace, not a training pipeline. It uses published checkpoints and exact forward-pass tensors. It does not currently generate prediction attributions, corpus-scale activation motifs, or causal biological conclusions.

## Evidence discipline

Every statement should be read at one of four levels:

| Level | What it establishes | Examples in the current project |
|---|---|---|
| **Structural fact** | What the stored architecture and weights mathematically do | Tensor shapes, kernel widths, channel mixing, receptive fields |
| **Exact example** | What happened for one sequence in one checkpoint | A displayed activation, residual addition, profile logit, or count |
| **Descriptive pattern** | A summary of stored weights or observed tensors | Sparsity, tap-energy fractions, channel correlations, CKA |
| **Mechanistic or biological evidence** | What changes a prediction, or what is supported biologically | Ablations, sequence perturbations, attribution motifs, external validation |

The current site is strong at the first two levels and contains a limited pilot at the third. The fourth is a roadmap, not a completed result.

## Shared tensor convention

### DNA input

A DNA sequence of length `L` is represented as an `L × 4` one-hot tensor internally, with feature order `A, C, G, T`. The visualizer usually draws its transpose as **4 rows × L positions** because readers naturally scan a sequence from left to right.

At every position, exactly one of the four entries is 1 for an unambiguous base and the other three are 0. This input is not sparse because of biological inactivity; it is sparse only in the formal one-hot sense. Therefore the input view never compresses “quiet regions.”

### Convolution means cross-correlation in the checkpoints

The stored TensorFlow/Keras kernels are applied left to right without spatial reversal:

```text
output[o, f] = bias[f]
             + sum over kernel position p
             + sum over input channel c
               input[o + p, c] × kernel[p, c, f]
```

This operation is technically cross-correlation, although deep-learning libraries call it convolution. The Sequence CNN Explainer follows the stored-model convention. It never rotates the A/C/G/T axis because bases are feature channels, not a second spatial dimension. See the [TensorFlow `conv1d` documentation](https://www.tensorflow.org/api_docs/python/tf/nn/conv1d).

### What a channel means

- At the input, the four channels have fixed meanings: A, C, G, and T.
- After the stem convolution, each channel is the activation track of one learned filter.
- In later layers, a channel is a learned feature assembled from many earlier channels and positions. It should not be named after a transcription factor without additional evidence.

ChromBPNet channel IDs are immutable within a checkpoint: internal IDs `0–511` are displayed as `Channel 1–512`. A display sort changes only row order. It does not change the tensor, the calculation, or the model. Equal channel numbers in independently trained checkpoints do not imply equivalent features.

## ChromBPNet

### Provenance represented in the site

The default checkpoint is:

- model: `model.chrombpnet_nobias.fold_0.ENCSR000EOT.h5`
- repository: `kundajelab/encode-chrombpnet-DNASE-ENCSR000EOT-ENCSR296UHQ`
- experiment: ENCSR000EOT
- biosample and assay: K562 DNase-seq
- fold: 0

The replication checkpoint is:

- model: `model.chrombpnet_nobias.fold_0.ENCSR960KGO.h5`
- repository: `kundajelab/encode-chrombpnet-ATAC-ENCSR960KGO-ENCSR576OPE`
- experiment: ENCSR960KGO
- biosample and assay: GM21515 ATAC-seq
- fold: 0

The two checkpoints are shown on the same hg38 input window so their calculations can be compared. They differ in both cell type and assay, so a difference cannot be attributed to either factor alone.

The synthetic-sequence preset uses the K562 model. It is a different input, not a third independently trained model.

The model name `chrombpnet_nobias` means that the assay-specific enzyme sequence-bias component has been factored out. It does **not** mean that ordinary neural-network bias parameters are absent. The model still has stem, residual, profile-head, and count-head biases.

Primary references are the [ChromBPNet paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC11741299/) and [official repository](https://github.com/kundajelab/chrombpnet).

### Architecture and exact dimensions

All tensor shapes below are displayed as **channels × positions**.

| Stage | Operation | Kernel or setting | Output shape | Receptive field in input DNA |
|---|---|---:|---:|---:|
| Input | one-hot DNA | — | `4 × 2,114` | 1 bp |
| Stem | valid Conv1D, then ReLU | `21 × 4 → 512` | `512 × 2,094` | 21 bp |
| Residual 1 | dilated valid Conv1D, ReLU, shortcut add | width 3, `d=2` | `512 × 2,090` | 25 bp |
| Residual 2 | same | width 3, `d=4` | `512 × 2,082` | 33 bp |
| Residual 3 | same | width 3, `d=8` | `512 × 2,066` | 49 bp |
| Residual 4 | same | width 3, `d=16` | `512 × 2,034` | 81 bp |
| Residual 5 | same | width 3, `d=32` | `512 × 1,970` | 145 bp |
| Residual 6 | same | width 3, `d=64` | `512 × 1,842` | 273 bp |
| Residual 7 | same | width 3, `d=128` | `512 × 1,586` | 529 bp |
| Residual 8 | same | width 3, `d=256` | `512 × 1,074` | 1,041 bp |
| Profile head | valid Conv1D | `75 × 512 → 1` | `1 × 1,000` logits | 1,115 bp |
| Count head | global average pool, dense | `1,074 → 1`, then `512 → 1` | one log-count | whole final tensor |

For a valid width-`k` convolution with dilation `d` and stride 1,

```text
output length = input length − d × (k − 1).
```

The stem therefore changes `2,114` to `2,114 − 20 = 2,094`. Every residual convolution has three taps, so it removes `2d` positions. The profile kernel changes `1,074` to `1,074 − 74 = 1,000`.

### The stem convolution

Each of the 512 stem filters has shape `21 × 4`. At one output position, one filter reads 21 adjacent bases, selects one weight at each base through the one-hot input, sums the 21 selected weights, adds the filter bias, and applies ReLU.

The result is one number for that filter at that position. Repeating the calculation at every valid position produces one track of length 2,094; repeating it for all 512 filters produces the `512 × 2,094` stem tensor.

It is reasonable to use “local feature detector” as an intuition. It is not yet justified to call a particular filter a complete named DNA motif. CNNs can represent partial motifs across filters and assemble them later; see [Koo and Eddy](https://journals.plos.org/ploscompbiol/article?id=10.1371%2Fjournal.pcbi.1007560).

### A residual block

Let the block input be `X` with shape `512 × L`, and let dilation be `d`. For output channel `f` and output position `o`, the transform path computes:

```text
Z[f,o] = bias[f]
       + sum over input channel c = 1..512
         (X[c,o]     × W[left,c,f]
        + X[c,o+d]   × W[center,c,f]
        + X[c,o+2d]  × W[right,c,f])

T[f,o] = ReLU(Z[f,o])
```

The shortcut crops `d` positions from each edge so that it has the same length as the transform:

```text
S[f,o] = X[f,o+d]
Y[f,o] = T[f,o] + S[f,o]
```

Three distinctions are essential:

1. **Dilation changes spatial spacing.** The three taps read positions `o`, `o+d`, and `o+2d`. A dilation-2 kernel is conceptually `# · # · #`, not a wider stored kernel containing zeros.
2. **Each tap mixes all 512 channels.** A residual output channel does not read only its same-numbered input channel. Each tap is a complete `512 × 512` weight matrix.
3. **The shortcut is the same residual-connection idea used elsewhere in deep learning.** It preserves the aligned earlier feature value and adds a nonnegative learned correction. There is no extra ReLU after the addition in this checkpoint.

Thus a dilated residual block can combine local features that earlier stem filters detected in different channels and at separated positions. Dilation expands the possible distance efficiently; the weights decide which channel combinations are used. Reachability alone is not evidence that a particular feature affected a prediction.

### Receptive field

The theoretical receptive field is the span of input DNA that could affect one cell. The stem starts at 21 bp. Each width-3 residual block adds `2d` bases because it samples two dilation intervals beyond its left tap. Summing dilations `2 + 4 + … + 256 = 510` gives:

```text
21 + 2 × 510 = 1,041 bp.
```

The profile kernel reads 75 adjacent final-tensor positions, whose centers are one input base apart. Its combined DNA footprint is:

```text
1,041 + (75 − 1) = 1,115 bp.
```

Theoretical receptive field is not the same as effective influence. Some reachable inputs can receive near-zero weight or be blocked by ReLU. Establishing effective or causal influence requires measured contributions, attribution, or perturbation.

### Profile head

The profile kernel has shape `75 × 512 × 1`. For each of 1,000 output positions it reads 75 final-tensor positions across all 512 channels: `75 × 512 = 38,400` activation–weight products, plus one bias, produce one logit.

Softmax converts the 1,000 logits into a probability distribution:

```text
p[i] = exp(logit[i] − max(logit))
       / sum_j exp(logit[j] − max(logit)).
```

Subtracting one shared constant is a numerical-stability step and does not change probabilities or rank. Softmax does not invent the peak: the largest logit remains the largest probability.

The profile-kernel heatmap is `512 channels × 75 positions`. It is not a DNA logo because its input rows are learned features, not A/C/G/T.

### Count head

The count branch reads the same `512 × 1,074` final tensor in parallel with the profile branch:

```text
pooled[c] = mean over 1,074 positions of final[c, position]
logcount = count_bias + sum_c pooled[c] × count_weight[c]
predicted_total_count = exp(logcount) − 1
```

The expected count profile shown by the explainer is:

```text
expected_count[i] = profile_probability[i] × predicted_total_count.
```

The profile and count heads do not feed into one another. They answer different questions from a shared feature tensor: **where** accessibility is distributed and **how much** total accessibility is expected.

## Reading the ChromBPNet visualizations

### Main route `/`

The vertical layout follows information from DNA to prediction. It contains the model overview, exact stem calculation, residual calculation, tensor presentation studio, profile and count heads, and full output construction.

The whole-tensor views show all positions before providing a magnified region. Tensor width is proportional to the number of positions so valid-convolution cropping remains visible. The default residual state is `+ Shortcut`, because that is the actual block output passed to the next layer.

The residual teaching selector intentionally offers different questions:

- **Balanced merge:** maximizes the weaker of the ReLU correction and shortcut, so both paths are visible.
- **Strongest correction:** shows where the learned transform adds most, even if the shortcut is zero.
- **Strongest output:** shows the largest final block cell, which can be dominated by either path.

These are selected examples, not random samples and not frequency estimates.

### Dilation route `/dilation-trace`

This route keeps one input-aligned coordinate fixed while the reader steps from stem through Residual 8. It separates:

- possible geometric reach;
- the preserved shortcut;
- the nonnegative transform correction; and
- the final block output.

Moving the coordinate slider leaves the selected teaching example. The UI explicitly warns that a freely selected cell may have a zero correction, zero shortcut, or both.

### Model Audit route `/model-audit`

This route reports complete checkpoint-weight summaries and single-locus activation summaries. Important definitions are:

- **Exact-zero fraction:** fraction of tensor entries exactly equal to zero.
- **Tolerance-zero fraction:** fraction with absolute value at most `10⁻⁷`.
- **Median active channels per position:** for each position, count channels above `10⁻⁷`, then take the median over positions.
- **Positive run:** a maximal contiguous stretch of cells above the audit tolerance (`10⁻⁷`) within one channel; the reported count sums starts of such runs over all channels.
- **Channel occupancy:** fraction of positions at which a channel is active.
- **Channel RMS:** root mean square activation of one channel over positions.
- **Tap-energy fraction:** squared weights in one of the three spatial taps divided by squared weights in the entire residual kernel.
- **Diagonal energy:** squared weights connecting input channel `c` to output channel `c`, divided by all kernel squared weight. Low diagonal energy is evidence of broad cross-channel weights, not by itself evidence of causal cooperation.
- **Effective input-channel count:** entropy-based effective number of input channels represented in one output channel's squared weights.
- **Stable/effective rank:** summaries of how weight energy is distributed over singular directions. They do not count biological factors.
- **Count/profile agreement:** Pearson correlation across channels between absolute count-head weights and the square root of profile-kernel channel energy. It compares weight magnitudes, not predictions or causal importance.
- **CKA:** a representation-similarity statistic. This pilot uses 128 aligned positions and the 64 highest-variance channels per layer, so it is neither a full-tensor comparison nor a channel-identity map. See [Kornblith et al.](https://proceedings.mlr.press/v97/kornblith19a).

The heatmap scale and contrast controls affect display only. Shared magnitude scale is the correct default for cross-layer comparisons; per-layer scaling is useful for seeing weak structure but hides absolute magnitude changes. A higher brightness gain can saturate strong values, so it should never be read as a quantitative increase.

### Stem-filter views

The three names deliberately refer to different evidence:

- **Heatmap:** the stored signed `21 × 4` weights.
- **Weight logo:** position-centered signed weights. Subtracting each position's mean over A/C/G/T and adding the removed means to the bias reproduces the original filter output exactly for one-hot DNA.
- **Activation motif:** a PFM/information-content logo built from a documented corpus of top-activating 21-mers. This remains unavailable because the required corpus artifact has not been generated.

An attribution motif would answer a third question—what sequence pattern supports a selected prediction—and requires a separate method such as DeepLIFT/SHAP followed by motif discovery. It is not inferred from activations.

## Verified ChromBPNet pilot observations

The following are accurate descriptions of one exact locus per checkpoint. They must not be described as general properties of K562, GM21515, DNase-seq, ATAC-seq, or ChromBPNet as a model family.

| Observation | K562 DNase fold 0 | GM21515 ATAC fold 0 |
|---|---:|---:|
| Stem exact zeros | 98.63% | 98.13% |
| Residual 8 exact zeros | 88.44% | 81.87% |
| Median active channels per position, stem | 7 | 9 |
| Median active channels per position, Residual 8 | 58 | 92 |
| Count/profile weight correlation | −0.081 | −0.330 |

Within these two displayed examples, residual accumulation is associated with more active cells in later block outputs. This does not prove that every sequence, checkpoint, or cell type follows the same trajectory.

Across the complete stored residual kernels, the three taps carry similar fractions of squared weight energy. For K562, the tap fractions range from about 32.2% to 35.6%; for GM21515, from about 31.5% to 34.4%. That says the stored weights do not globally ignore a tap. It does not say all taps contribute equally for any particular input.

The stored residual kernels also have low same-index diagonal energy—about 0.24–0.40% in the K562 checkpoint and 0.21–0.34% in GM21515—and large entropy-based effective input-channel counts. This strongly rejects the mental model “each output channel reads only the same-numbered input channel.” It does not identify which channel combinations matter biologically.

## Basset

### Why Basset is the second model

Basset is a useful contrast because it solves long-range communication without dilated residual blocks. It repeatedly convolves and max-pools, then uses dense layers to combine the complete remaining representation. The comparison distinguishes two architecture strategies rather than implying that dilation is the only way to combine distant motifs.

Primary references are the [Basset paper](https://genome.cshlp.org/content/26/7/990) and [official repository](https://github.com/davek44/Basset). Exact checkpoint provenance and hashes are recorded in [the Basset adapter note](basset-checkpoint-adapter.md).

### Architecture and dimensions

| Stage | Operation | Output shape | Receptive field | Center spacing |
|---|---|---:|---:|---:|
| Input | one-hot DNA | `4 × 600` | 1 bp | 1 bp |
| Conv1 | width 19, valid; batch norm; ReLU | `300 × 582` | 19 bp | 1 bp |
| Pool1 | max, width/stride 3 | `300 × 194` | 21 bp | 3 bp |
| Conv2 | width 11, valid; batch norm; ReLU | `200 × 184` | 51 bp | 3 bp |
| Pool2 | max, width/stride 4 | `200 × 46` | 60 bp | 12 bp |
| Conv3 | width 7, valid; batch norm; ReLU | `200 × 40` | 132 bp | 12 bp |
| Pool3 | max, width/stride 4 | `200 × 10` | 168 bp | 48 bp |
| Flatten | channel-major order | `2,000` | all 600 bp jointly | — |
| Dense1 | linear; batch norm; ReLU | `1,000` | all 600 bp | — |
| Dense2 | linear; batch norm; ReLU | `1,000` | all 600 bp | — |
| Output | linear; sigmoid | `164` | all 600 bp | — |

The ten Pool3 positions jointly cover the full input because `168 + 9 × 48 = 600`. Flattening preserves every one of the `200 × 10` cells. Each Dense1 unit then has a separate weight for all 2,000 cells, so it can combine features from different channels and distant regions. Dense2 can recombine those 1,000 global features. The output layer has one separate reader for each of 164 cell types.

Pooling trades base-level location precision for compression and increasing context. ChromBPNet instead preserves one-base spacing through the backbone because it must produce a base-resolution profile. This is an architectural explanation, not a claim that one design is universally better.

### Exact adapter verification

The official legacy Torch7 checkpoint is evaluated independently in NumPy and TensorFlow. The maximum absolute disagreement over all compared intermediate states and outputs is `9.95 × 10⁻¹⁴` in float64. The browser then uses exported float32 tensors.

The displayed Basset sequence is the official tutorial sequence `chr7:27183235-27183835`. The K562 output probability is approximately `0.9061` and ranks 13th among the 164 labels for this sequence. This is one checkpoint prediction, not measured accessibility and not an accuracy statistic.

## What the current visual evidence can and cannot support

### Supported now

- exact architecture, tensor shapes, crop alignment, and receptive-field calculations;
- exact arithmetic for displayed stem, residual, profile, count, pooling, dense, and output cells;
- the fact that ChromBPNet residual taps mix all 512 channels;
- the fact that Basset's dense reader combines all final pooled positions and channels;
- descriptive properties of the complete stored kernels;
- descriptive properties of the exact displayed activation tensors;
- distinction among weight, activation, attribution, and perturbation evidence.

### Not supported yet

- a population claim based on the one displayed locus;
- a claim that a selected channel is a named transcription factor or complete motif;
- a claim that bright activation caused the final prediction;
- a claim about motif cooperation or competition without controlled perturbation;
- a claim that two same-index channels from different checkpoints correspond;
- a claim that K562–GM21515 differences are caused specifically by cell type rather than assay;
- a claim that ChromBPNet is biologically superior to Basset from these visual examples.

## Research-grade next analyses

The planned fast pass uses 5,000 held-out peaks, 5,000 GC/mappability/chromosome-matched inactive regions, and every sequence's reverse complement. The planned motif pass uses 30,000 peaks. Sequences should be streamed in minibatches; only aggregate statistics and bounded activation-window reservoirs should be retained.

Priority analyses are:

1. repeat sparsity, occupancy, run-length, dynamic-range, redundancy, and representation-similarity summaries over the corpus;
2. generate activation-derived first-layer logos from non-overlapping high-activation windows;
3. compare forward and reverse-complement predictions and filter responses;
4. measure empirical activation × downstream-weight terms, then perform single-channel ablations for shortlisted channels;
5. perform pair ablations only after principled shortlisting;
6. create profile-head and count-head attribution artifacts separately;
7. use TF-MoDISco and motif-database comparison for candidate motif families;
8. test candidate syntax with matched-background insertion/deletion, multiplicity, spacing, and orientation perturbations;
9. repeat principal findings across model folds before strong public claims.

## Reproducibility and release checks

The canonical browser-summary JSON files live in `app/data`. Compressed network copies under `public/data` are generated by:

```bash
npm run sync:browser-data
```

The main release gate is:

```bash
npm test
```

It runs linting, TypeScript checking, a production build, source/numerical tests, and Playwright browser tests. Independent scientific adapter checks are:

```bash
npm run verify:python
```

Key invariants include:

- an asymmetric `ACG` kernel peaks on `ACG`, while the deliberately reversed kernel peaks on `GCA`;
- centered stem weights plus adjusted bias reproduce raw filter outputs;
- applying reverse complement twice restores the original tensor;
- a consistent global channel permutation preserves the calculation;
- residual convolution, bias, ReLU, shortcut, and output reconcile numerically;
- full profile logits and the count head are recomputed from raw final tensors;
- the complete Basset forward chain agrees across independent implementations.

The independent audit history is preserved in `AUDIT-INDEPENDENT-V2.md`, `AUDIT-INDEPENDENT-V3.md`, and `AUDIT-INDEPENDENT-V3-FOLLOWUP.md`. The follow-up verdict was **ready with caveats**, with no scientific or numerical error found. The remaining pre-documentation teaching and packaging caveats were addressed in commit `364cc586`.

## References

- Pampari et al. [ChromBPNet: bias-factorized, base-resolution deep learning models of chromatin accessibility](https://pmc.ncbi.nlm.nih.gov/articles/PMC11741299/).
- Kundaje Lab. [Official ChromBPNet repository](https://github.com/kundajelab/chrombpnet).
- Kelley, Snoek, and Rinn. [Basset: learning the regulatory code of the accessible genome with deep convolutional neural networks](https://genome.cshlp.org/content/26/7/990).
- TensorFlow. [`tf.nn.conv1d`](https://www.tensorflow.org/api_docs/python/tf/nn/conv1d).
- Koo and Eddy. [Representation learning of genomic sequence motifs with convolutional neural networks](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1007560).
- Kornblith et al. [Similarity of Neural Network Representations Revisited](https://proceedings.mlr.press/v97/kornblith19a).
- Wang et al. [CNN Explainer](https://arxiv.org/abs/2004.15004) and the [interactive explainer](https://poloclub.github.io/cnn-explainer/).
