# Sequence CNN Explainer

See how ChromBPNet and Basset process a DNA sequence, layer by layer. The app uses published model checkpoints to show the calculations and intermediate values behind their predictions.

## Pages

| View | What it explains |
|---|---|
| Main explainer (`/`) | ChromBPNet stem convolution, residual blocks, and profile/count heads |
| Dilation trace (`/dilation-trace`) | One aligned tensor region through eight residual blocks |
| Model audit (`/model-audit`) | Layer statistics, channel rankings, kernel diagnostics, and representation similarity |
| Basset (`/basset`) | Convolution, pooling, flattening, and the 164-output accessibility readout |

- **Run locally:** [Local use](#local-use).
- **How it works:** [Technical documentation](docs/TECHNICAL-DOCUMENTATION.md).
- **Supported claims and limitations:** [Claim ledger](docs/CLAIM-LEDGER.md).
- **Tests:** [Verification](#verification) and [tests](tests/).
- **Basset model conversion:** [Basset provenance and conventions](docs/basset-checkpoint-adapter.md).

## Current limitations

The activation statistics currently describe one genomic region. Activation logos are not yet available; they require analysis of a documented set of sequences. The larger analyses planned for these views are described below.

## Local use

Requires Node.js **22.13.0 or newer**, as specified in `package.json`. From the repository root:

```bash
npm ci
npm run dev
```

Open [http://localhost:3000](http://localhost:3000).

## Verification

```bash
npm test
```

This command runs linting, TypeScript checking, a production build, numerical/source tests, and rendered Playwright browser tests. The browser tests cover checkpoint switching, both residual-example selectors, tensor loading and scaling, the shifted-logit softmax explanation, and the Basset K562 output.

The Python checks for model calculations and exported data run separately because they require a local model environment:

```bash
npm run verify:python
```

Together, the checks cover ChromBPNet tensor shapes and raw values, residual convolution/ReLU/shortcut identities, both output heads, centered-weight bias compensation, reverse complements, information content, Basset graph/readout invariants, and key teaching claims.

## Model walkthrough details

The main page follows a ChromBPNet calculation from a 2,114-base one-hot input through:

1. the 21-base stem convolution;
2. eight dilated residual blocks;
3. the profile and total-count output heads;
4. the `512 × 1,074 → 1 × 1,000` profile-head conversion; and
5. optional full `512 × N` tensor inspection and a whole-tensor dilation filmstrip magnifier.

The stem-filter panel includes three views: the raw signed heatmap, an exactly reparameterized signed weight logo, and a corpus-derived activation-logo slot. The activation logo is unavailable until the sequence-corpus analysis is complete. Channel IDs stay consistent across layers and heads when the display order changes.

Two pages provide more detail:

- `/dilation-trace` follows one aligned tensor region through all eight residual blocks.
- `/model-audit` separates descriptive structure, model mechanism, and biological evidence while exposing layer statistics, channel rankings, kernel diagnostics, and representation similarity.

A third route, `/basset`, adapts the original published Basset Torch7 checkpoint. It connects a 600 bp input to three convolution/max-pooling stages, a `200 × 10 → 2,000` flattening step, two dense layers, and 164 cell-type accessibility probabilities. The page includes an exact sliding-filter calculation, a max-pooling microscope, complete tensor heatmaps with local zoom, a `300 channels × 11 positions` Conv2 mixing example, and the dense global readout.

The default demo uses forward-pass activations extracted from the K562 DNase checkpoint `model.chrombpnet_nobias.fold_0.ENCSR000EOT.h5`. A second checkpoint uses the published GM21515 ATAC model `model.chrombpnet_nobias.fold_0.ENCSR960KGO.h5` on the same DNA window, allowing a controlled model-to-model comparison. Raw browser heatmaps are stored as gzip-compressed, channel-major little-endian float32 files so weak nonzero activations are not lost to display quantization.

## Rebuild the Basset adapter

The 1.7 GB decompressed Torch7 checkpoint is never committed or deployed. After downloading the official `pretrained_model.th.gz`, clone the official Basset repository for its tutorial sequence and target labels, then run:

```bash
models/.extract-env/bin/python scripts/export_basset_demo.py \
  --checkpoint /path/to/pretrained_model.th \
  --fasta /path/to/Basset/tutorials/satmut_eg/hoxa_boundary.fa \
  --targets /path/to/Basset/data/models/targets.txt
models/.extract-env/bin/python scripts/verify_basset_adapter.py
```

The exporter validates the decompressed checkpoint SHA-256, reads the stored module graph, evaluates every layer twice with independent NumPy and TensorFlow implementations, and exports about 1.1 MB of float32 browser tensors plus a compact JSON file for the walkthrough. See [`docs/basset-checkpoint-adapter.md`](docs/basset-checkpoint-adapter.md) for the provenance and operator conventions.

## Rebuild the audit data

The deployed audit JSON summarizes checkpoint weights and activations from one genomic region. Full activation tensors are stored separately:

```bash
models/.extract-env/bin/python scripts/build_model_audit.py
models/.extract-env/bin/python scripts/verify_model_analysis.py
```

The current activation statistics are labeled as a single-region analysis. The planned population analysis uses 5,000 peaks plus 5,000 matched inactive regions and reverse complements; the planned motif analysis uses 30,000 peaks and retains only aggregate statistics and bounded activation-window reservoirs.

When a genomic FASTA corpus is available, `scripts/build_activation_motifs.py` streams it through the exact TensorFlow stem cross-correlation, keeps a bounded top-activation reservoir, removes overlapping 21-mers, builds PFMs/information content, and can merge the results into the audit JSON. The activation-logo view stays empty until this command has been run on a documented corpus.

The checkpoint extraction script is maintained at `scripts/run_chrombpnet_checkpoint.py`.

After changing a source JSON file in `app/data`, rebuild the compressed browser copies with `npm run sync:browser-data`. The generated files under `public/data` are the network representation; `app/data` remains the source used by numerical verification.

## Planned work

[`docs/interpretability-and-claude-handoff-plan.md`](docs/interpretability-and-claude-handoff-plan.md) records proposed interpretation methods, the Basset-before-Basenji development order, checks for adding models, and instructions for Claude Code. The instructions call for human review after the repository audit, report, and slide deck, before video production.

## Further documentation

- [`docs/TECHNICAL-DOCUMENTATION.md`](docs/TECHNICAL-DOCUMENTATION.md) is the complete architecture, tensor-shape, visualization, metric, and reproducibility reference for ChromBPNet and Basset.
- [`docs/CLAIM-LEDGER.md`](docs/CLAIM-LEDGER.md) defines which structural, descriptive, mechanistic, and biological claims are currently supported and the qualifiers required for public communication.
- [`docs/VOCABULARY-AND-MENTAL-MODELS.md`](docs/VOCABULARY-AND-MENTAL-MODELS.md) translates image-CNN intuition to sequence CNNs and standardizes terms for the future blog, slides, and video narration.

The blog prompt and production handoff will be revised only after human review of these three documents.
