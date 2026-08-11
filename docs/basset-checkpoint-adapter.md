# Basset checkpoint adapter

## Provenance

- Official repository: <https://github.com/davek44/Basset>
- Repository commit inspected: `71cd8016b28b33e40357cac59ba5fbade3692ac2`
- Official pretrained archive URL, as referenced by `install_data.py`: <https://www.dropbox.com/s/rguytuztemctkf8/pretrained_model.th.gz>
- Compressed archive SHA-256: `cfc2cc76ef0f1d4570670b83fae191b956e262bec53547a1844fed1847f3ed91`
- Decompressed checkpoint SHA-256: `d228f99c8ca286f7f7684534fc66f8ad8c8099afa7380d35c15e88da341f1b94`
- Demonstration sequence: the repository’s `tutorials/satmut_eg/hoxa_boundary.fa`, header `chr7:27183235-27183835`
- Output labels: the repository’s `data/models/targets.txt`

The checkpoint is read-only input to `scripts/export_basset_demo.py`. It is not copied into the repository or the deployed website.

## Why the stored graph is authoritative

The repository’s current `convnet.lua` can construct padded convolutions. The published checkpoint stores `padW = 0` in all three convolution modules and therefore has these exact dimensions:

| Stage | Operation | Shape |
|---|---|---:|
| Input | one-hot DNA | `4 × 600` |
| Conv1 | width 19, valid; BN; ReLU | `300 × 582` |
| Pool1 | max, width/stride 3 | `300 × 194` |
| Conv2 | width 11, valid; BN; ReLU | `200 × 184` |
| Pool2 | max, width/stride 4 | `200 × 46` |
| Conv3 | width 7, valid; BN; ReLU | `200 × 40` |
| Pool3 | max, width/stride 4 | `200 × 10` |
| Flatten | channel-major Torch memory order | `2,000` |
| Dense1 | linear; BN; ReLU | `1,000` |
| Dense2 | linear; BN; ReLU | `1,000` |
| Output | linear; sigmoid | `164` |

Dropout modules are stored but are identity operations during evaluation.

## Operator conventions

Torch7 `SpatialConvolution` evaluates cross-correlation: the stored kernel’s left-to-right positions are not spatially reversed. The first layer reads A/C/G/T in that order. Later convolution weights have shape `output channel × input channel × width`; every output channel therefore mixes every input feature channel.

This legacy checkpoint uses the version-1 batch-normalization representation. Its `running_std` field is the stored inverse standard deviation, so evaluation is:

```text
normalized = (x - running_mean) × running_std × gamma + beta
```

The official Torch7 version-2 migration converts this value back into a running variance. The adapter follows the legacy inference expression directly.

## Numerical verification

The exporter performs the complete forward pass twice:

1. NumPy uses channel-major arrays that mirror Torch7.
2. TensorFlow uses independent `conv1d`, max-pooling, matrix multiplication, and sigmoid operators, with an explicit transpose before Torch-compatible flattening.

For the official tutorial sequence, the maximum absolute disagreement across all compared intermediate states and final predictions is `9.95 × 10⁻¹⁴` in float64. Browser tensors are exported afterward as little-endian float32.

## Receptive fields

The receptive field and center spacing evolve as follows:

| Stage | Receptive field | Center spacing |
|---|---:|---:|
| Conv1 | 19 bp | 1 bp |
| Pool1 | 21 bp | 3 bp |
| Conv2 | 51 bp | 3 bp |
| Pool2 | 60 bp | 12 bp |
| Conv3 | 132 bp | 12 bp |
| Pool3 | 168 bp | 48 bp |

The ten Pool3 positions jointly cover the complete 600 bp input because `168 + 9 × 48 = 600`. Dense1 then gives every hidden unit a separate weight for all `200 × 10 = 2,000` remaining cells.

## Scientific boundary

The selected stem and Conv2 examples are chosen by strong activation on one tutorial sequence. This is useful for explaining the computation, but it is not evidence that a channel corresponds to a named transcription factor or that the selected cell caused a final prediction. Those stronger claims require corpus-derived activation motifs, prediction attribution, and controlled sequence perturbations.
