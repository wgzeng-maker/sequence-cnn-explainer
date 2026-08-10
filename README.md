# Sequence CNN Explainer

An interactive, top-to-bottom explanation of the published K562 ChromBPNet no-bias sequence model. The explainer follows one calculation from a 2,114-base one-hot input through:

1. the 21-base stem convolution;
2. eight dilated residual blocks;
3. the profile and total-count output heads; and
4. optional full `512 × N` tensor inspection.

The default demo uses forward-pass activations extracted from `model.chrombpnet_nobias.fold_0.ENCSR000EOT.h5`. Raw browser heatmaps are stored as gzip-compressed, channel-major little-endian float32 files so weak nonzero activations are not lost to display quantization.

## Local use

```bash
npm run dev
```

Open [http://localhost:3000](http://localhost:3000).

## Verification

```bash
npm test
npm run lint
```

The tests check tensor shapes and file sizes, residual convolution/ReLU/shortcut identities, profile normalization, expected-count totals, and the count-head dense calculation.

The checkpoint extraction script is maintained at `scripts/run_chrombpnet_checkpoint.py`.
