#!/usr/bin/env python3
"""Run exact inference from published ChromBPNet HDF5 weights and export browser-sized tensors."""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import h5py
import numpy as np


BASES = np.array(list("ACGT"))
SELECTED_CHANNEL_COUNT = 12


def make_sequence() -> str:
    rng = np.random.default_rng(20260808)
    letters = rng.choice(BASES, size=2114)
    letters[75:82] = np.array(list("TGACTCA"))
    return "".join(letters.tolist())


def one_hot(sequence: str) -> np.ndarray:
    lookup = {base: index for index, base in enumerate(BASES)}
    encoded = np.zeros((1, len(sequence), 4), dtype=np.float32)
    encoded[0, np.arange(len(sequence)), [lookup[base] for base in sequence]] = 1
    return encoded


def weight(handle: h5py.File, layer: str, variable: str) -> np.ndarray:
    path = f"model_weights/{layer}/{layer}/{variable}:0"
    return np.asarray(handle[path][...], dtype=np.float32)


def conv1d(handle: h5py.File, x: np.ndarray, layer: str, dilation: int, relu: bool) -> np.ndarray:
    kernel = weight(handle, layer, "kernel")
    kernel_size = kernel.shape[0]
    output_length = x.shape[1] - dilation * (kernel_size - 1)
    windows = np.stack(
        [x[:, offset : offset + output_length, :] for offset in range(0, dilation * kernel_size, dilation)],
        axis=2,
    )
    result = np.tensordot(windows, kernel, axes=([2, 3], [0, 1]))
    result += weight(handle, layer, "bias")[None, None, :]
    return np.maximum(result, 0) if relu else result


def compact_tensor(
    value: np.ndarray,
    tensor_name: str,
    tensor_dir: Path,
    public_preset_id: str,
    channels: list[int] | None = None,
) -> dict[str, object]:
    array = value[0]
    length = array.shape[0]
    if array.ndim == 1:
        sampled = array[None, :]
        selected = [0]
        position_mean = array
        position_max = array
        position_active_fraction = (array > 0).astype(np.float32)
    else:
        if channels is None:
            channel_peaks = array.max(axis=0)
            selected = np.argsort(channel_peaks)[-SELECTED_CHANNEL_COUNT:][::-1].tolist()
        else:
            selected = [channel for channel in channels if channel < array.shape[1]]
        sampled = array[:, selected].T
        position_mean = array.mean(axis=1)
        position_max = array.max(axis=1)
        position_active_fraction = (array > 0).mean(axis=1)
    exported = {
        "shape_batch_positions_channels": list(value.shape),
        "selected_channels_zero_based": selected,
        "selected_channel_values": np.round(sampled, 6).tolist(),
        "position_mean": np.round(position_mean, 6).tolist(),
        "position_max": np.round(position_max, 6).tolist(),
        "position_active_fraction": np.round(position_active_fraction, 6).tolist(),
        "min": float(array.min()),
        "max": float(array.max()),
        "zero_fraction": float(np.mean(array == 0)),
    }
    if array.ndim == 2:
        tensor_dir.mkdir(parents=True, exist_ok=True)
        maximum = max(float(array.max()), 1e-8)
        # Keep the browser heatmap spatially complete and numerically faithful.
        # Float32 is larger than the earlier uint8 preview, but tensors are loaded
        # one layer at a time and weak non-zero activations no longer disappear.
        browser_values = np.asarray(array.T, dtype="<f4")
        binary_path = tensor_dir / f"{tensor_name}.f32.gz"
        binary_path.write_bytes(gzip.compress(browser_values.tobytes(order="C"), compresslevel=9, mtime=0))
        exported["full_heatmap"] = {
            "url": f"/data/tensors/{public_preset_id}/{tensor_name}.f32.gz",
            "dtype": "float32_le",
            "compression": "gzip",
            "layout": "channels_by_positions",
            "height_channels": int(array.shape[1]),
            "width_positions": int(array.shape[0]),
            "raw_max": maximum,
            "value_preservation": "raw float32 activations; no display quantization",
        }
    return exported


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--sequence-json", type=Path)
    parser.add_argument("--preset-id", default="synthetic")
    parser.add_argument("--locus-label", default="Synthetic teaching sequence")
    parser.add_argument("--coordinate-system", default="sequence positions 1–2,114")
    parser.add_argument("--tensor-dir", type=Path, help="Directory for browser tensor binaries")
    args = parser.parse_args()

    if args.sequence_json:
        sequence_payload = json.loads(args.sequence_json.read_text(encoding="utf-8"))
        sequence = sequence_payload["dna"].upper()
    else:
        sequence = make_sequence()
    if len(sequence) != 2114 or any(base not in "ACGT" for base in sequence):
        raise ValueError("ChromBPNet demo input must be exactly 2,114 unambiguous A/C/G/T bases")
    x = one_hot(sequence)
    tensors: dict[str, dict[str, object]] = {}
    tensor_dir = args.tensor_dir or args.output.parent / f"{args.preset_id}_tensor_bins"
    residual_kernel_demos: list[dict[str, object]] = []

    with h5py.File(args.model, "r") as handle:
        stem_kernels = weight(handle, "wo_bias_bpnet_1st_conv", "kernel")
        stem_biases = weight(handle, "wo_bias_bpnet_1st_conv", "bias")
        x = conv1d(handle, x, "wo_bias_bpnet_1st_conv", dilation=1, relu=True)
        stem_activations = x[0]
        tensors["stem"] = compact_tensor(x, "stem", tensor_dir, args.preset_id)

        demo_filters = np.argsort(stem_activations.max(axis=0))[-5:][::-1].tolist()

        for block, dilation in enumerate((2, 4, 8, 16, 32, 64, 128, 256), start=1):
            block_input = x
            transformed = conv1d(handle, block_input, f"wo_bias_bpnet_{block}conv", dilation=dilation, relu=True)
            shortcut = block_input[:, dilation:-dilation, :]
            x = transformed + shortcut
            tensor_name = f"res{block}"
            tensors[tensor_name] = compact_tensor(x, tensor_name, tensor_dir, args.preset_id)
            output_channel = int(tensors[tensor_name]["selected_channels_zero_based"][0])
            residual_kernel = weight(handle, f"wo_bias_bpnet_{block}conv", "kernel")
            residual_bias = weight(handle, f"wo_bias_bpnet_{block}conv", "bias")
            peak_position = int(np.argmax(x[0, :, output_channel]))
            tap_positions = [peak_position + tap * dilation for tap in range(3)]
            tap_vectors = np.stack([block_input[0, position, :] for position in tap_positions])
            tap_weights = residual_kernel[:, :, output_channel]
            weighted_products = tap_vectors * tap_weights
            tap_sums = weighted_products.sum(axis=1)
            convolution_sum = float(tap_sums.sum())
            before_relu = convolution_sum + float(residual_bias[output_channel])
            transformed_value = max(0.0, before_relu)
            shortcut_value = float(shortcut[0, peak_position, output_channel])
            residual_kernel_demos.append({
                "block": block,
                "dilation": dilation,
                "kernel_shape_positions_input_output_channels": list(residual_kernel.shape),
                "output_channel_zero_based": output_channel,
                "weights_input_channels_by_taps": np.round(residual_kernel[:, :, output_channel].T, 7).tolist(),
                "bias": float(residual_bias[output_channel]),
                "trace": {
                    "selection_rule": "output channel with the largest peak activation in this block; position of that channel's peak",
                    "output_position_zero_based": peak_position,
                    "input_tap_positions_zero_based": tap_positions,
                    "input_features_by_taps": np.round(tap_vectors, 7).tolist(),
                    "weighted_products_by_taps": np.round(weighted_products, 7).tolist(),
                    "tap_sums": np.round(tap_sums, 7).tolist(),
                    "convolution_sum": convolution_sum,
                    "before_relu": before_relu,
                    "transformed_after_relu": transformed_value,
                    "shortcut_value": shortcut_value,
                    "block_output": float(x[0, peak_position, output_channel]),
                },
            })

        profile = conv1d(handle, x, "wo_bias_bpnet_prof_out_precrop", dilation=1, relu=False)
        profile = np.squeeze(profile, axis=-1)
        pooled = np.mean(x, axis=1)
        count_kernel = weight(handle, "wo_bias_bpnet_logcount_predictions", "kernel")
        count_bias = weight(handle, "wo_bias_bpnet_logcount_predictions", "bias")
        logcount = pooled @ count_kernel
        logcount += count_bias[None, :]
        profile_kernel = weight(handle, "wo_bias_bpnet_prof_out_precrop", "kernel")

    shifted_profile = profile - np.max(profile, axis=1, keepdims=True)
    probabilities = np.exp(shifted_profile) / np.exp(shifted_profile).sum(axis=1, keepdims=True)
    predicted_count = np.exp(logcount) - 1
    base_signal = probabilities * predicted_count
    tensors["profile_logits"] = compact_tensor(profile, "profile_logits", tensor_dir, args.preset_id)
    tensors["profile_probabilities"] = compact_tensor(probabilities, "profile_probabilities", tensor_dir, args.preset_id)
    tensors["profile_signal"] = compact_tensor(base_signal, "profile_signal", tensor_dir, args.preset_id)

    exported = {
        "provenance": {
            "model": args.model.name,
            "repository": "kundajelab/encode-chrombpnet-DNASE-ENCSR000EOT-ENCSR296UHQ",
            "experiment": "ENCSR000EOT",
            "biosample": "K562",
            "assay": "DNASE-seq",
            "fold": 0,
            "model_variant": "chrombpnet_nobias",
        },
        "input": {
            "preset_id": args.preset_id,
            "locus_label": args.locus_label,
            "coordinate_system": args.coordinate_system,
            "sequence": sequence,
            "shape_batch_positions_channels": [1, 2114, 4],
            "base_order": BASES.tolist(),
        },
        "filter_demos": [
            {
                "layer": "wo_bias_bpnet_1st_conv",
                "filter_zero_based": filter_index,
                "filter_human_label": f"Filter {filter_index + 1}",
                "kernel_shape_positions_bases": [21, 4],
                "weights_base_rows_by_positions": np.round(stem_kernels[:, :, filter_index].T, 7).tolist(),
                "bias": float(stem_biases[filter_index]),
                "maximum_activation": float(stem_activations[:, filter_index].max()),
                "peak_position_zero_based": int(np.argmax(stem_activations[:, filter_index])),
                "positive_fraction": float(np.mean(stem_activations[:, filter_index] > 0)),
            }
            for filter_index in demo_filters
        ],
        "residual_kernel_demos": residual_kernel_demos,
        "head_demos": {
            "profile_kernel_shape_positions_input_output_channels": list(profile_kernel.shape),
            "count_pool_input_shape_positions_channels": [1074, 512],
            "count_pooled_features": np.round(pooled[0], 6).tolist(),
            "count_dense_weights": np.round(count_kernel[:, 0], 7).tolist(),
            "count_dense_bias": float(count_bias[0]),
        },
        "outputs": {
            "profile_logits_shape": list(profile.shape),
            "logcount_shape": list(logcount.shape),
            "logcount": float(logcount[0, 0]),
            "predicted_total_count": float(predicted_count[0, 0]),
        },
        "tensors": tensors,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(exported, separators=(",", ":")), encoding="utf-8")
    print(json.dumps(exported["outputs"], indent=2))
    print(f"Exported real checkpoint activations to {args.output}")


if __name__ == "__main__":
    main()
