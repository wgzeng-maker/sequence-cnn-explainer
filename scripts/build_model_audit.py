#!/usr/bin/env python3
"""Build compact, browser-facing model-audit summaries from exact exported tensors.

This script intentionally keeps full activations out of the JSON.  It records
aggregate layer/channel metrics and kernel diagnostics, with explicit evidence
provenance.  Corpus activation motifs are a separate artifact because one locus
is not an adequate motif corpus.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
from pathlib import Path
from typing import Any

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
LAYERS = ["stem", *[f"res{i}" for i in range(1, 9)]]
MODEL_PATHS = {
    "k562-peak": ROOT / "models/k562_dnase_fold0/model.chrombpnet_nobias.fold_0.ENCSR000EOT.h5",
    "gm21515": Path("/private/tmp/model.chrombpnet_nobias.fold_0.ENCSR960KGO.h5"),
}


def load_demo(preset: str) -> dict[str, Any]:
    return json.loads((ROOT / f"app/data/{preset}-activations.json").read_text())


def load_tensor(demo: dict[str, Any], layer: str) -> np.ndarray:
    metadata = demo["tensors"][layer]["full_heatmap"]
    path = ROOT / "public" / metadata["url"].lstrip("/")
    with gzip.open(path, "rb") as handle:
        data = np.frombuffer(handle.read(), dtype="<f4").copy()
    return data.reshape(metadata["height_channels"], metadata["width_positions"])


def quantiles(values: np.ndarray) -> list[float]:
    return np.quantile(values, [0, .01, .25, .5, .75, .99, 1]).astype(float).tolist()


def hoyer(values: np.ndarray, axis: int | None = None) -> np.ndarray:
    n = values.shape[axis] if axis is not None else values.size
    l1 = np.sum(np.abs(values), axis=axis)
    l2 = np.sqrt(np.sum(values * values, axis=axis))
    return (math.sqrt(n) - l1 / np.maximum(l2, 1e-12)) / max(math.sqrt(n) - 1, 1e-12)


def effective_count(energy: np.ndarray, axis: int = -1) -> np.ndarray:
    probability = energy / np.maximum(np.sum(energy, axis=axis, keepdims=True), 1e-12)
    entropy = -np.sum(np.where(probability > 0, probability * np.log(probability), 0), axis=axis)
    return np.exp(entropy)


def reverse_complement_stem(kernel: np.ndarray) -> np.ndarray:
    # kernel is position × base × output; reverse positions and complement A/C/G/T.
    return kernel[::-1, [3, 2, 1, 0], :]


def correlation_columns(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = left - left.mean(axis=0, keepdims=True)
    right = right - right.mean(axis=0, keepdims=True)
    numerator = np.sum(left * right, axis=0)
    denominator = np.sqrt(np.sum(left * left, axis=0) * np.sum(right * right, axis=0))
    return numerator / np.maximum(denominator, 1e-12)


def linear_cka(left: np.ndarray, right: np.ndarray) -> float:
    left = left - left.mean(axis=0, keepdims=True)
    right = right - right.mean(axis=0, keepdims=True)
    cross = left.T @ right
    numerator = np.sum(cross * cross)
    denominator = math.sqrt(float(np.sum((left.T @ left) ** 2) * np.sum((right.T @ right) ** 2)))
    return float(numerator / max(denominator, 1e-12))


def approximate_cka(tensors: dict[str, np.ndarray]) -> dict[str, Any]:
    # Align all stages to the final 1,074-position center, then use the same 128
    # positions and each layer's 64 most variable channels. This is a compact
    # position-level representation comparison, not a channel correspondence.
    rng = np.random.default_rng(1729)
    positions = np.sort(rng.choice(1074, 128, replace=False))
    reduced: list[np.ndarray] = []
    for layer in LAYERS:
        tensor = tensors[layer]
        crop = (tensor.shape[1] - 1074) // 2
        aligned = tensor[:, crop:crop + 1074][:, positions].T
        channels = np.argsort(np.var(aligned, axis=0))[-64:]
        reduced.append(aligned[:, channels].astype(np.float64))
    matrix = [[linear_cka(left, right) for right in reduced] for left in reduced]
    return {
        "method": "linear CKA on 128 shared positions and 64 highest-variance channels per layer",
        "evidence_level": "descriptive structure",
        "labels": LAYERS,
        "values": matrix,
    }


def activation_metrics(tensor: np.ndarray) -> dict[str, Any]:
    positive = tensor > 1e-7
    active_per_position = positive.sum(axis=0)
    occupancy = positive.mean(axis=1)
    rms = np.sqrt(np.mean(tensor.astype(np.float64) ** 2, axis=1))
    maximum = np.max(np.abs(tensor), axis=1)
    # Number of positive runs, computed independently per channel.
    starts = positive[:, :1].sum() + np.logical_and(positive[:, 1:], ~positive[:, :-1]).sum()
    return {
        "shape": [int(value) for value in tensor.shape],
        "exact_zero_fraction": float(np.mean(tensor == 0)),
        "tolerance_zero_fraction": float(np.mean(np.abs(tensor) <= 1e-7)),
        "value_quantiles": quantiles(tensor),
        "absolute_value_quantiles": quantiles(np.abs(tensor)),
        "dynamic_range": float(tensor.max() - tensor.min()),
        "median_active_channels_per_position": float(np.median(active_per_position)),
        "active_channels_per_position_quantiles": quantiles(active_per_position),
        "channel_occupancy_quantiles": quantiles(occupancy),
        "positive_run_count": int(starts),
        "channel_occupancy": occupancy.astype(float).tolist(),
        "channel_rms": rms.astype(float).tolist(),
        "channel_max_abs": maximum.astype(float).tolist(),
    }


def weight_group(handle: h5py.File, layer: str, item: str) -> np.ndarray:
    path = f"model_weights/{layer}/{layer}/{item}:0"
    return np.asarray(handle[path], dtype=np.float64)


def kernel_diagnostics(model_path: Path) -> dict[str, Any]:
    with h5py.File(model_path, "r") as handle:
        stem_name = "wo_bias_bpnet_1st_conv"
        stem = weight_group(handle, stem_name, "kernel")
        stem_flat = stem.reshape(84, 512)
        stem_rc = reverse_complement_stem(stem).reshape(84, 512)
        centered = stem - stem.mean(axis=1, keepdims=True)
        position_energy = np.sum(stem * stem, axis=1).T
        base_energy = np.sum(stem * stem, axis=0).T
        base_probability = base_energy / np.maximum(base_energy.sum(axis=1, keepdims=True), 1e-12)
        base_entropy = -np.sum(np.where(base_probability > 0, base_probability * np.log2(base_probability), 0), axis=1)
        result: dict[str, Any] = {
            "stem": {
                "shape": [21, 4, 512],
                "weight_quantiles": quantiles(stem),
                "positive_fraction": float(np.mean(stem > 0)),
                "l1_quantiles": quantiles(np.sum(np.abs(stem_flat), axis=0)),
                "l2_quantiles": quantiles(np.sqrt(np.sum(stem_flat * stem_flat, axis=0))),
                "hoyer_quantiles": quantiles(hoyer(stem_flat, axis=0)),
                "base_preference_entropy_bits_quantiles": quantiles(base_entropy),
                "reverse_complement_similarity_quantiles": quantiles(correlation_columns(stem_flat, stem_rc)),
                "centered_position_energy_quantiles": quantiles(np.sum(centered * centered, axis=1).T),
                "position_energy_mean": position_energy.mean(axis=0).astype(float).tolist(),
            },
            "residual": [],
        }
        for block in range(1, 9):
            name = f"wo_bias_bpnet_{block}conv"
            kernel = weight_group(handle, name, "kernel")
            tap_energy = np.sum(kernel * kernel, axis=(1, 2))
            channel_energy = np.sum(kernel * kernel, axis=0).T  # output × input
            diagonal = sum(np.sum(np.diag(kernel[tap]) ** 2) for tap in range(3))
            total = np.sum(kernel * kernel)
            # Flatten tap+input into rows; singular values summarize output directions.
            singular = np.linalg.svd(kernel.reshape(1536, 512), compute_uv=False)
            p = singular * singular / np.maximum(np.sum(singular * singular), 1e-12)
            result["residual"].append({
                "block": block,
                "dilation": 2 ** block,
                "shape": [3, 512, 512],
                "tap_energy_fraction": (tap_energy / total).astype(float).tolist(),
                "left_right_tap_cosine": float(np.sum(kernel[0] * kernel[2]) / max(np.linalg.norm(kernel[0]) * np.linalg.norm(kernel[2]), 1e-12)),
                "diagonal_energy_fraction": float(diagonal / max(total, 1e-12)),
                "effective_input_channels_quantiles": quantiles(effective_count(channel_energy, axis=1)),
                "stable_rank": float(total / max(singular[0] ** 2, 1e-12)),
                "effective_rank": float(np.exp(-np.sum(np.where(p > 0, p * np.log(p), 0)))),
                "singular_values_top20": singular[:20].astype(float).tolist(),
                "bias_quantiles": quantiles(weight_group(handle, name, "bias")),
            })
        profile_name = "wo_bias_bpnet_prof_out_precrop"
        profile = weight_group(handle, profile_name, "kernel")[:, :, 0]
        count_name = "wo_bias_bpnet_logcount_predictions"
        count = weight_group(handle, count_name, "kernel")[:, 0]
        profile_channel_energy = np.sum(profile * profile, axis=0)
        profile_position_energy = np.sum(profile * profile, axis=1)
        result["heads"] = {
            "profile_shape": [75, 512, 1],
            "profile_channel_energy": profile_channel_energy.astype(float).tolist(),
            "profile_position_energy": profile_position_energy.astype(float).tolist(),
            "profile_position_center_of_mass": float(np.sum(np.arange(75) * profile_position_energy) / max(profile_position_energy.sum(), 1e-12)),
            "profile_effective_input_channels": float(effective_count(profile_channel_energy[None, :], axis=1)[0]),
            "profile_weight_quantiles": quantiles(profile),
            "count_weight_quantiles": quantiles(count),
            "count_profile_absolute_weight_correlation": float(np.corrcoef(np.abs(count), np.sqrt(profile_channel_energy))[0, 1]),
            "count_weights": count.astype(float).tolist(),
        }
        return result


def checkpoint_summary(preset: str, model_path: Path) -> dict[str, Any]:
    demo = load_demo(preset)
    tensors = {layer: load_tensor(demo, layer) for layer in LAYERS}
    metrics = {layer: activation_metrics(tensor) for layer, tensor in tensors.items()}
    stem_occupancy = np.asarray(metrics["stem"]["channel_occupancy"])
    final_rms = np.asarray(metrics["res8"]["channel_rms"])
    final_mean = np.mean(tensors["res8"], axis=1)
    profile = np.asarray(demo["head_demos"]["profile_weights_input_channels_by_positions"], dtype=np.float64)
    count = np.asarray(demo["head_demos"]["count_dense_weights"], dtype=np.float64)
    profile_influence = final_rms * np.sqrt(np.sum(profile * profile, axis=1))
    count_influence = np.abs(final_mean * count)
    orders = {
        "original": list(range(512)),
        "stem_occupancy": np.argsort(-stem_occupancy, kind="stable").astype(int).tolist(),
        "final_rms": np.argsort(-final_rms, kind="stable").astype(int).tolist(),
        "profile_influence": np.argsort(-profile_influence, kind="stable").astype(int).tolist(),
        "count_influence": np.argsort(-count_influence, kind="stable").astype(int).tolist(),
    }
    registry = [{
        "id_zero_based": channel,
        "label": f"Channel {channel + 1}",
        "checkpoint_id": demo["provenance"]["experiment"] + "-fold0",
        "stem_occupancy": float(stem_occupancy[channel]),
        "final_rms": float(final_rms[channel]),
        "profile_influence": float(profile_influence[channel]),
        "count_influence": float(count_influence[channel]),
        "final_mean_activation": float(final_mean[channel]),
        "count_weight": float(count[channel]),
        "count_contribution": float(final_mean[channel] * count[channel]),
    } for channel in range(512)]
    kernels = kernel_diagnostics(model_path)
    kernels["heads"].update({
        "count_bias": float(demo["head_demos"]["count_dense_bias"]),
        "logcount": float(demo["outputs"]["logcount"]),
        "predicted_total_count": float(demo["outputs"]["predicted_total_count"]),
    })
    return {
        "preset_id": demo["input"]["preset_id"],
        "checkpoint": demo["provenance"],
        "evidence_scope": {
            "activation_sample_count": 1,
            "activation_corpus": demo["input"]["locus_label"],
            "warning": "Single-locus descriptive audit; not a population or biological conclusion.",
            "model_weights": "complete published fold-0 checkpoint",
        },
        "channel_registry": registry,
        "channel_orders": orders,
        "layers": metrics,
        "layer_similarity": approximate_cka(tensors),
        "kernels": kernels,
        "activation_motifs": {
            "status": "not_generated",
            "reason": "A genomic corpus and activation-window reservoir are required; one displayed locus is intentionally insufficient.",
            "target_site_count_per_filter": 500,
            "planned_corpus": "30,000 held-out peaks with matched inactive regions",
            "selection_rule": "top non-overlapping positive stem activations per filter",
            "motifs": [],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "app/data/model-audit-summary.json")
    args = parser.parse_args()
    checkpoints = {}
    for preset, model_path in MODEL_PATHS.items():
        if not model_path.exists():
            raise FileNotFoundError(f"Missing checkpoint for {preset}: {model_path}")
        print(f"Auditing {preset} …", flush=True)
        checkpoints[preset] = checkpoint_summary(preset, model_path)
    artifact = {
        "schema_version": "1.0.0",
        "generated_by": "scripts/build_model_audit.py",
        "evidence_levels": [
            {"id": "descriptive", "label": "Descriptive structure", "definition": "What the stored weights and observed tensors look like."},
            {"id": "mechanism", "label": "Model mechanism", "definition": "What changes predictions under contribution or ablation tests."},
            {"id": "biology", "label": "Biological evidence", "definition": "What survives motif, perturbation, and external experimental validation."},
        ],
        "checkpoints": checkpoints,
        "planned_corpus_audit": {
            "fast_pass": "5,000 held-out peaks + 5,000 matched inactive regions + reverse complements",
            "motif_pass": "30,000 peaks",
            "storage": "stream minibatches; retain aggregates and activation-window reservoirs",
            "replication": "repeat principal findings across folds before strong claims",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, separators=(",", ":")), encoding="utf-8")
    print(f"Wrote {args.output} ({args.output.stat().st_size / 1024:.1f} KiB)")


if __name__ == "__main__":
    main()
