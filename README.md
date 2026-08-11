# Sequence CNN Explainer

An interactive, top-to-bottom explanation of published ChromBPNet no-bias sequence models. The explainer follows one calculation from a 2,114-base one-hot input through:

1. the 21-base stem convolution;
2. eight dilated residual blocks;
3. the profile and total-count output heads; and
4. the `512 × 1,074 → 1 × 1,000` profile-head conversion; and
5. optional full `512 × N` tensor inspection and a whole-tensor dilation filmstrip magnifier.

The stem-filter panel includes three intentionally distinct views: the raw signed heatmap, an exactly reparameterized signed weight logo, and a corpus-derived activation-logo slot. The activation logo remains visibly unavailable until a real multi-sequence corpus artifact exists. A global channel registry keeps immutable channel IDs synchronized when every displayed layer and head is reordered.

Two supporting routes keep the main visual story focused:

- `/dilation-trace` follows one aligned tensor region through all eight residual blocks.
- `/model-audit` separates descriptive structure, model mechanism, and biological evidence while exposing layer statistics, channel rankings, kernel diagnostics, and representation similarity.

The default demo uses forward-pass activations extracted from the K562 DNase checkpoint `model.chrombpnet_nobias.fold_0.ENCSR000EOT.h5`. A second real checkpoint uses the published GM21515 ATAC model `model.chrombpnet_nobias.fold_0.ENCSR960KGO.h5` on the same DNA window, allowing a controlled model-to-model comparison. Raw browser heatmaps are stored as gzip-compressed, channel-major little-endian float32 files so weak nonzero activations are not lost to display quantization.

## Local use

```bash
npm run dev
```

Open [http://localhost:3000](http://localhost:3000).

## Verification

```bash
npm test
```

The tests check all three presets, tensor shapes and file sizes, residual convolution/ReLU/shortcut identities, profile-head convolution, profile normalization, expected-count totals, the count-head dense calculation, the ACG/GCA orientation convention, exact centered-weight bias compensation, reverse-complement involution, PFM information-content bounds, and global permutation invariance.

## Rebuild the compact audit artifact

The deployed audit JSON contains complete checkpoint-weight summaries and single-locus activation summaries—not the full activation tensors:

```bash
models/.extract-env/bin/python scripts/build_model_audit.py
models/.extract-env/bin/python scripts/verify_model_analysis.py
```

The current activation statistics are explicitly labeled as a one-locus descriptive pilot. The planned population pass uses 5,000 peaks plus 5,000 matched inactive regions and reverse complements; the motif pass uses 30,000 peaks and retains only aggregate statistics and bounded activation-window reservoirs.

When a genomic FASTA corpus is available, `scripts/build_activation_motifs.py` streams it through the exact TensorFlow stem cross-correlation, keeps a bounded top-activation reservoir, removes overlapping 21-mers, builds PFMs/information content, and can merge the results into the audit JSON. Until that command is run on a documented corpus, the site shows a deliberate empty state rather than a pseudo-motif.

The checkpoint extraction script is maintained at `scripts/run_chrombpnet_checkpoint.py`.
