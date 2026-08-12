# Sequence CNN Explainer — vocabulary and mental-model guide

## The shortest correct story

For an image CNN, a filter slides over two spatial dimensions and produces a feature map. For a DNA CNN, the filter slides over **one spatial dimension: sequence position**. A/C/G/T are input feature channels, not a second spatial axis.

ChromBPNet first creates 512 local feature tracks. Its residual blocks repeatedly read three increasingly separated feature vectors, mix all 512 channels, create a nonnegative correction, and add that correction to a cropped copy of the earlier representation. Two parallel heads then read the final tensor: one predicts where accessibility lies, and one predicts its total amount.

Basset takes another route. Convolutions create feature tracks, max-pooling progressively compresses position, and dense layers globally mix every remaining channel-position cell before producing 164 cell-type probabilities.

## From image CNNs to sequence CNNs

| Image-CNN idea | Sequence-CNN translation | Important difference |
|---|---|---|
| image height × width × color | sequence length × A/C/G/T | DNA has one spatial axis |
| RGB channel | A/C/G/T input channel | exactly one base is active at an unambiguous position |
| learned feature map | learned feature track | one value per sequence position |
| 2-D sliding patch | 1-D sliding window | later kernels still mix all input channels |
| edge/texture detector | local sequence-feature detector | a first-layer filter need not be a complete motif |
| pooling over image regions | pooling over sequence positions | reduces coordinate precision |
| deeper visual feature | combination of earlier sequence features | biological identity requires extra evidence |

## Mental pictures to keep

### One-hot DNA: four rails

Imagine four horizontal rails labeled A, C, G, and T. At each sequence position, one rail is on and three are off. The sequence is still one-dimensional; the four rows describe which feature is present.

### Stem filter: a 21-base stencil

One ChromBPNet stem filter is a 21-column stencil with four weights per column. Sliding it over DNA produces one track. Five hundred twelve different stencils produce 512 tracks.

Safe intuition: **“Where does this learned local detector respond?”**

Unsafe shortcut: **“Which transcription factor is this channel?”**

### Dilated residual filter: three spaced feature snapshots

At dilation `d`, picture three vertical slices through the entire 512-row tensor:

```text
all 512 channels at position o
                 all 512 channels at position o+d
                                  all 512 channels at position o+2d
```

The filter weights and combines all three 512-number vectors to produce 512 new numbers. Dilation changes how far apart the slices are. It does not restrict communication to one channel.

### Residual connection: preserve, then edit

The shortcut is an aligned crop of the earlier tensor. The transform path proposes a nonnegative edit after ReLU. The output is:

```text
earlier aligned feature + learned correction.
```

This explains why a transform-path heatmap can become sparse again between blocks while the actual `+ Shortcut` block output preserves earlier positive cells. When comparing residual stages, use block output by default.

### Profile head: a sliding global-feature reader

At each output position, the profile head reads 75 neighboring final-tensor columns and all 512 channels. It turns 38,400 activation–weight products into one logit. Sliding this reader produces 1,000 logits.

### Count head: summarize then read

The count head averages each final channel across all 1,074 positions, leaving 512 summaries. One dense layer combines them into a scalar log-count.

### Basset dense layer: unfold the map

After pooling, Basset has 200 channels at 10 positions. Flattening lays those 2,000 cells into one vector. A dense unit has a different weight for every cell, so it can combine distant positions and different feature channels directly.

## Terms that must remain distinct

| Term | Meaning | What it does not establish |
|---|---|---|
| **Kernel/filter** | stored learned weights used at every valid position | biological identity |
| **Activation** | output value of a unit for a particular input | importance to the final prediction |
| **Feature track** | one learned channel across sequence positions | one known motif |
| **Weight logo** | signed visualization of first-layer weights | empirical sequence frequency |
| **Activation motif** | motif summary of sequences that strongly activate a filter | contribution to a selected output |
| **Attribution motif** | motif summary derived from prediction-attribution scores | proof of causal biology |
| **Perturbation effect** | prediction change after a controlled sequence edit or ablation | experimental biological effect unless externally validated |
| **Theoretical receptive field** | all input positions that could affect a unit through the graph | positions that actually mattered on this input |
| **Cross-channel mixing** | nonzero weights connect many input and output channels | cooperative biology |
| **Coactivation** | features are active together | cooperation or causality |
| **Correlation** | two quantities vary together | mechanism |
| **CKA** | similarity between multivariate representations | individual channel correspondence |

## Approved narrative verbs

Use verbs that match the evidence:

- Architecture: **reads, mixes, samples, adds, pools, maps, permits**.
- Exact forward pass: **produces, is active, contributes to this sum, is preserved**.
- Descriptive analysis: **is associated with, has higher occupancy, shows, suggests a candidate**.
- Intervention: **changes the prediction by, suppresses, increases, has a non-additive effect**.
- Biology: **matches a candidate motif family, is consistent with, is supported by**.

Avoid upgrading the evidence with verbs such as **proves, discovers, controls, cooperates, competes, explains**, or **causes** unless the corresponding intervention and biological validation exist.

## Color and scale language

- **Red = positive, blue = negative** for signed model values in this project.
- A/C/G/T identity colors are categorical and should not be read as positive or negative.
- White or near-white can mean zero or weak magnitude; it never means missing data unless explicitly labeled.
- A shared scale permits magnitude comparison across panels.
- A local scale or contrast gain reveals weak structure but can exaggerate visual differences.
- A saturated color means “at or beyond the current display limit,” not an infinite or exact maximum.

## Coordinate language

Always name the coordinate system:

- **input-relative position:** 1–2,114 for ChromBPNet or 1–600 for Basset;
- **tensor position:** index within the current cropped activation track;
- **profile position:** 1–1,000;
- **genomic coordinate:** chromosome and genome build, using a declared half-open or closed convention.

Avoid saying only “position 500.” After valid convolutions, position 500 in different tensors is not the same array index, even when it can be mapped to the same input-aligned center.

## A narration checklist

Before recording or publishing a section, ask:

1. Have I said whether the picture shows weights, activations, attributions, or perturbation effects?
2. Have I stated the tensor dimensions and which axis is position?
3. If I compare colors, are the panels on a shared scale?
4. If I show a selected cell, have I explained how it was selected?
5. If I say “motif,” is it a hypothesis, an activation-derived motif, or an attribution-derived motif?
6. If I imply mechanism, is there an intervention?
7. If I imply biology, is there external evidence?
8. If I quote a statistic, is the sample count visible?
9. If I compare checkpoints, have I avoided assuming channel correspondence?
10. Does the wording agree with [`CLAIM-LEDGER.md`](CLAIM-LEDGER.md)?

## One teaching analogy—and its limit

A useful analogy is a musical arrangement:

- stem filters detect short notes or phrases;
- deeper layers combine phrases across time and across instruments;
- a residual shortcut preserves the prior arrangement while adding an edit;
- output heads are different listeners asking “where is the emphasis?” and “how much total sound is there?”

The limit is important: channels are not guaranteed to be clean, human-named instruments. They can be distributed mixtures. The analogy explains information flow, not biological identity.
