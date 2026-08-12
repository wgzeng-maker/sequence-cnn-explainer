# Sequence CNN Explainer

An interactive, top-to-bottom explanation of published sequence CNN checkpoints. The main explainer follows a ChromBPNet calculation from a 2,114-base one-hot input through:

1. the 21-base stem convolution;
2. eight dilated residual blocks;
3. the profile and total-count output heads;
4. the `512 × 1,074 → 1 × 1,000` profile-head conversion; and
5. optional full `512 × N` tensor inspection and a whole-tensor dilation filmstrip magnifier.

The stem-filter panel includes three intentionally distinct views: the raw signed heatmap, an exactly reparameterized signed weight logo, and a corpus-derived activation-logo slot. The activation logo remains visibly unavailable until a real multi-sequence corpus artifact exists. A global channel registry keeps immutable channel IDs synchronized when every displayed layer and head is reordered.

Two supporting routes keep the main visual story focused:

- `/dilation-trace` follows one aligned tensor region through all eight residual blocks.
- `/model-audit` separates descriptive structure, model mechanism, and biological evidence while exposing layer statistics, channel rankings, kernel diagnostics, and representation similarity.

A third route, `/basset`, adapts the original published Basset Torch7 checkpoint. It connects a 600 bp input to three convolution/max-pooling stages, a `200 × 10 → 2,000` flattening step, two dense layers, and 164 cell-type accessibility probabilities. The page includes an exact sliding-filter calculation, a max-pooling microscope, complete tensor heatmaps with local zoom, a `300 channels × 11 positions` Conv2 mixing example, and the dense global readout.

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

This release gate runs linting, TypeScript checking, a production build, numerical/source tests, and rendered Playwright browser tests. The browser suite exercises checkpoint switching, both residual-example selectors, tensor loading and scaling, the shifted-logit softmax explanation, and the Basset K562 output.

The independent scientific/export checks remain available separately so `npm test` does not depend on a large local model environment:

```bash
npm run verify:python
```

Together, the checks cover ChromBPNet tensor shapes and raw values, residual convolution/ReLU/shortcut identities, both output heads, centered-weight bias compensation, reverse complements, information content, Basset graph/readout invariants, and key teaching claims.

## Rebuild the Basset adapter

The 1.7 GB decompressed Torch7 checkpoint is never committed or deployed. After downloading the official `pretrained_model.th.gz`, clone the official Basset repository for its tutorial sequence and target labels, then run:

```bash
models/.extract-env/bin/python scripts/export_basset_demo.py \
  --checkpoint /path/to/pretrained_model.th \
  --fasta /path/to/Basset/tutorials/satmut_eg/hoxa_boundary.fa \
  --targets /path/to/Basset/data/models/targets.txt
models/.extract-env/bin/python scripts/verify_basset_adapter.py
```

The exporter validates the decompressed checkpoint SHA-256, reads the stored module graph rather than guessing from a parameter file, evaluates every layer twice with independent NumPy and TensorFlow implementations, and exports about 1.1 MB of float32 browser tensors plus a compact JSON narrative artifact. See [`docs/basset-checkpoint-adapter.md`](docs/basset-checkpoint-adapter.md) for the provenance and operator conventions.

## Rebuild the compact audit artifact

The deployed audit JSON contains complete checkpoint-weight summaries and single-locus activation summaries—not the full activation tensors:

```bash
models/.extract-env/bin/python scripts/build_model_audit.py
models/.extract-env/bin/python scripts/verify_model_analysis.py
```

The current activation statistics are explicitly labeled as a one-locus descriptive pilot. The planned population pass uses 5,000 peaks plus 5,000 matched inactive regions and reverse complements; the motif pass uses 30,000 peaks and retains only aggregate statistics and bounded activation-window reservoirs.

When a genomic FASTA corpus is available, `scripts/build_activation_motifs.py` streams it through the exact TensorFlow stem cross-correlation, keeps a bounded top-activation reservoir, removes overlapping 21-mers, builds PFMs/information content, and can merge the results into the audit JSON. Until that command is run on a documented corpus, the site shows a deliberate empty state rather than a pseudo-motif.

The checkpoint extraction script is maintained at `scripts/run_chrombpnet_checkpoint.py`.

After changing a canonical JSON artifact in `app/data`, rebuild the compressed browser copies with `npm run sync:browser-data`. The generated files under `public/data` are the network representation; `app/data` remains the source used by numerical verification.

## Research and communication roadmap

[`docs/interpretability-and-claude-handoff-plan.md`](docs/interpretability-and-claude-handoff-plan.md) records the computer-vision interpretation methods still worth adapting, the decision to add Basset before Basenji, the multi-model acceptance gates, and a ready-to-use staged prompt for Claude Code. The prompt deliberately stops after the repository audit, again after the report, and again after the slide deck so scientific claims can be reviewed before video production.
