# Sequence CNN Explainer

An interactive, top-to-bottom explanation of published ChromBPNet no-bias sequence models. The explainer follows one calculation from a 2,114-base one-hot input through:

1. the 21-base stem convolution;
2. eight dilated residual blocks;
3. the profile and total-count output heads; and
4. the `512 × 1,074 → 1 × 1,000` profile-head conversion; and
5. optional full `512 × N` tensor inspection and a whole-tensor dilation filmstrip magnifier.

The default demo uses forward-pass activations extracted from the K562 DNase checkpoint `model.chrombpnet_nobias.fold_0.ENCSR000EOT.h5`. A second real checkpoint uses the published GM21515 ATAC model `model.chrombpnet_nobias.fold_0.ENCSR960KGO.h5` on the same DNA window, allowing a controlled model-to-model comparison. Raw browser heatmaps are stored as gzip-compressed, channel-major little-endian float32 files so weak nonzero activations are not lost to display quantization.

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

The tests check all three presets, tensor shapes and file sizes, residual convolution/ReLU/shortcut identities, profile-head convolution, profile normalization, expected-count totals, and the count-head dense calculation.

The checkpoint extraction script is maintained at `scripts/run_chrombpnet_checkpoint.py`.
