#!/usr/bin/env python3
"""Export a compact, checkpoint-faithful Basset explainer artifact.

The official Basset checkpoint is a legacy Torch7 serialization.  This script
reads it without mutating it, evaluates one documented 600 bp sequence with a
small NumPy implementation of the exact stored module graph, independently
checks the result with TensorFlow operators, and exports only browser-sized
activations and diagnostics.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torchfile


BASES = "ACGT"
EXPECTED_CHECKPOINT_SHA256 = "d228f99c8ca286f7f7684534fc66f8ad8c8099afa7380d35c15e88da341f1b94"
CHECKPOINT_ARCHIVE_SHA256 = "cfc2cc76ef0f1d4570670b83fae191b956e262bec53547a1844fed1847f3ed91"
OFFICIAL_REPOSITORY_COMMIT = "71cd8016b28b33e40357cac59ba5fbade3692ac2"


def key(value: str) -> bytes:
    return value.encode("utf-8")


def obj_get(obj: Any, name: str) -> Any:
    storage = obj._obj if getattr(obj, "_obj", None) is not None else obj
    return storage[key(name)]


def read_fasta(path: Path) -> tuple[str, str]:
    header = ""
    sequence: list[str] = []
    for line in path.read_text().splitlines():
        if line.startswith(">"):
            header = line[1:]
        else:
            sequence.append(line.strip())
    result = "".join(sequence).upper()
    if len(result) != 600 or any(base not in BASES for base in result):
        raise ValueError(f"Expected one 600 bp A/C/G/T sequence, received {len(result)} bp")
    return header, result


def read_targets(path: Path) -> list[dict[str, str]]:
    targets = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        label, source = line.split(maxsplit=1)
        targets.append({"label": label, "source": source.strip()})
    if len(targets) != 164:
        raise ValueError(f"Expected 164 target labels, received {len(targets)}")
    return targets


def one_hot(sequence: str) -> np.ndarray:
    result = np.zeros((4, len(sequence)), dtype=np.float64)
    for position, base in enumerate(sequence):
        result[BASES.index(base), position] = 1.0
    return result


def conv_valid(values: np.ndarray, module: Any) -> np.ndarray:
    weights = obj_get(module, "weight")[:, :, 0, :]
    bias = obj_get(module, "bias")
    windows = np.lib.stride_tricks.sliding_window_view(values, weights.shape[-1], axis=1)
    return np.einsum("oik,ipk->op", weights, windows, optimize=True) + bias[:, None]


def batch_norm_old(values: np.ndarray, module: Any) -> np.ndarray:
    """Evaluate the Torch7 v1 BatchNormalization representation.

    Old checkpoints store ``running_std = 1 / sqrt(running_var + eps)``.  The
    Torch7 v2 migration source performs the inverse conversion, which makes
    this direct inference expression unambiguous.
    """

    mean = obj_get(module, "running_mean")
    inverse_std = obj_get(module, "running_std")
    gamma = obj_get(module, "weight")
    beta = obj_get(module, "bias")
    reshape = (values.shape[0],) + (1,) * (values.ndim - 1)
    return (values - mean.reshape(reshape)) * inverse_std.reshape(reshape) * gamma.reshape(reshape) + beta.reshape(reshape)


def max_pool(values: np.ndarray, module: Any) -> tuple[np.ndarray, np.ndarray]:
    width = int(obj_get(module, "kW"))
    stride = int(obj_get(module, "dW"))
    starts = np.arange(0, values.shape[1] - width + 1, stride)
    windows = np.stack([values[:, start : start + width] for start in starts], axis=1)
    return windows.max(axis=2), windows.argmax(axis=2)


def linear(values: np.ndarray, module: Any) -> np.ndarray:
    return obj_get(module, "weight") @ values + obj_get(module, "bias")


def sigmoid(values: np.ndarray) -> np.ndarray:
    positive = values >= 0
    result = np.empty_like(values)
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exp_values = np.exp(values[~positive])
    result[~positive] = exp_values / (1.0 + exp_values)
    return result


def forward_numpy(sequence_tensor: np.ndarray, modules: list[Any]) -> dict[str, np.ndarray]:
    states: dict[str, np.ndarray] = {"input": sequence_tensor}
    values = sequence_tensor
    conv_index = 0
    dense_index = 0
    for module in modules:
        typename = module._typename.decode("utf-8")
        if typename == "nn.SpatialConvolution":
            conv_index += 1
            values = conv_valid(values, module)
            states[f"conv{conv_index}_linear"] = values.copy()
        elif typename == "nn.SpatialBatchNormalization":
            values = batch_norm_old(values, module)
            states[f"conv{conv_index}_bn"] = values.copy()
        elif typename == "nn.ReLU":
            values = np.maximum(values, 0)
            name = f"conv{conv_index}_relu" if dense_index == 0 else f"dense{dense_index}_relu"
            states[name] = values.copy()
        elif typename == "nn.SpatialMaxPooling":
            values, indices = max_pool(values, module)
            states[f"pool{conv_index}"] = values.copy()
            states[f"pool{conv_index}_argmax"] = indices.copy()
        elif typename == "nn.Reshape":
            values = values.reshape(-1)
            states["flatten"] = values.copy()
        elif typename == "nn.Linear":
            dense_index += 1
            values = linear(values, module)
            states[f"dense{dense_index}_linear"] = values.copy()
        elif typename == "nn.BatchNormalization":
            values = batch_norm_old(values, module)
            states[f"dense{dense_index}_bn"] = values.copy()
        elif typename == "nn.Dropout":
            # The stored module graph is in evaluation mode, so dropout is identity.
            pass
        elif typename == "nn.Sigmoid":
            values = sigmoid(values)
            states["predictions"] = values.copy()
        else:
            raise ValueError(f"Unsupported Torch7 module {typename}")
    return states


def forward_tensorflow(sequence_tensor: np.ndarray, modules: list[Any]) -> dict[str, np.ndarray]:
    import tensorflow as tf

    tf.keras.backend.set_floatx("float64")
    values = tf.constant(sequence_tensor.T[None, :, :], dtype=tf.float64)
    states: dict[str, np.ndarray] = {}
    conv_index = 0
    dense_index = 0
    for module in modules:
        typename = module._typename.decode("utf-8")
        if typename == "nn.SpatialConvolution":
            conv_index += 1
            weights = obj_get(module, "weight")[:, :, 0, :].transpose(2, 1, 0)
            bias = obj_get(module, "bias")
            values = tf.nn.conv1d(values, tf.constant(weights), stride=1, padding="VALID")
            values = tf.nn.bias_add(values, tf.constant(bias))
            states[f"conv{conv_index}_linear"] = values.numpy()[0].T
        elif typename == "nn.SpatialBatchNormalization":
            mean = tf.constant(obj_get(module, "running_mean"))[None, None, :]
            inverse_std = tf.constant(obj_get(module, "running_std"))[None, None, :]
            gamma = tf.constant(obj_get(module, "weight"))[None, None, :]
            beta = tf.constant(obj_get(module, "bias"))[None, None, :]
            values = (values - mean) * inverse_std * gamma + beta
            states[f"conv{conv_index}_bn"] = values.numpy()[0].T
        elif typename == "nn.ReLU":
            values = tf.nn.relu(values)
            name = f"conv{conv_index}_relu" if dense_index == 0 else f"dense{dense_index}_relu"
            array = values.numpy()[0]
            states[name] = array.T if array.ndim == 2 and dense_index == 0 else array
        elif typename == "nn.SpatialMaxPooling":
            width = int(obj_get(module, "kW"))
            values = tf.nn.max_pool1d(values, ksize=width, strides=width, padding="VALID")
            states[f"pool{conv_index}"] = values.numpy()[0].T
        elif typename == "nn.Reshape":
            # Torch flattens channel-major NCHW memory, not TensorFlow's position-major NLC.
            values = tf.reshape(tf.transpose(values, [0, 2, 1]), [1, -1])
            states["flatten"] = values.numpy()[0]
        elif typename == "nn.Linear":
            dense_index += 1
            weights = tf.constant(obj_get(module, "weight").T)
            bias = tf.constant(obj_get(module, "bias"))
            values = tf.linalg.matmul(values, weights) + bias
            states[f"dense{dense_index}_linear"] = values.numpy()[0]
        elif typename == "nn.BatchNormalization":
            mean = tf.constant(obj_get(module, "running_mean"))[None, :]
            inverse_std = tf.constant(obj_get(module, "running_std"))[None, :]
            gamma = tf.constant(obj_get(module, "weight"))[None, :]
            beta = tf.constant(obj_get(module, "bias"))[None, :]
            values = (values - mean) * inverse_std * gamma + beta
            states[f"dense{dense_index}_bn"] = values.numpy()[0]
        elif typename == "nn.Dropout":
            pass
        elif typename == "nn.Sigmoid":
            values = tf.math.sigmoid(values)
            states["predictions"] = values.numpy()[0]
        else:
            raise ValueError(f"Unsupported Torch7 module {typename}")
    return states


def summarize(values: np.ndarray) -> dict[str, float | int]:
    flat = values.reshape(-1)
    return {
        "entries": int(flat.size),
        "exact_zero_fraction": float(np.mean(flat == 0)),
        "positive_fraction": float(np.mean(flat > 0)),
        "minimum": float(flat.min()),
        "p50": float(np.quantile(flat, 0.5)),
        "p90": float(np.quantile(flat, 0.9)),
        "p99": float(np.quantile(flat, 0.99)),
        "maximum": float(flat.max()),
        "rms": float(np.sqrt(np.mean(flat * flat))),
    }


def write_tensor(public_dir: Path, name: str, values: np.ndarray) -> dict[str, Any]:
    public_dir.mkdir(parents=True, exist_ok=True)
    path = public_dir / f"{name}.bin"
    np.asarray(values, dtype="<f4").tofile(path)
    return {
        "url": f"/data/basset/{name}.bin",
        "channels": int(values.shape[0]),
        "positions": int(values.shape[1]),
        "dtype": "float32-le",
        "values": int(values.size),
        "minimum": float(values.min()),
        "maximum": float(values.max()),
    }


def write_weight_matrix(public_dir: Path, name: str, values: np.ndarray) -> dict[str, Any]:
    """Write an exact row-major float32 matrix for progressive browser loading."""
    public_dir.mkdir(parents=True, exist_ok=True)
    matrix = np.asarray(values, dtype="<f4")
    path = public_dir / f"{name}.f32.gz"
    path.write_bytes(gzip.compress(matrix.tobytes(order="C"), compresslevel=9, mtime=0))
    absolute = np.abs(matrix)
    return {
        "url": f"/data/basset/{name}.f32.gz",
        "rows": int(matrix.shape[0]),
        "columns": int(matrix.shape[1]),
        "values": int(matrix.size),
        "dtype": "float32-le",
        "compression": "gzip",
        "layout": "output_units_by_input_features",
        "absolute_p995": float(np.quantile(absolute, .995)),
        "absolute_maximum": float(absolute.max()),
    }


def layer_entry(name: str, values: np.ndarray, asset: dict[str, Any], receptive_field: int, jump: int) -> dict[str, Any]:
    return {
        "id": name,
        "channels": int(values.shape[0]),
        "positions": int(values.shape[1]),
        "receptive_field_bp": receptive_field,
        "center_spacing_bp": jump,
        "stats": summarize(values),
        "asset": asset,
    }


def jsonable(array: np.ndarray, digits: int = 7) -> list[Any]:
    return np.round(array.astype(np.float64), digits).tolist()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--fasta", type=Path, required=True)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, default=Path("app/data/basset-demo.json"))
    parser.add_argument("--public-dir", type=Path, default=Path("public/data/basset"))
    args = parser.parse_args()

    checksum = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    if checksum != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError(f"Unexpected checkpoint SHA-256: {checksum}")

    # torchfile 0.1.0 retains a Python 2 name in one rarely exercised path.
    torchfile.xrange = range
    checkpoint = torchfile.load(str(args.checkpoint), force_8bytes_long=True)
    modules = obj_get(checkpoint, "model")._obj[key("modules")]
    module_types = [module._typename.decode("utf-8") for module in modules]
    expected_types = [
        "nn.SpatialConvolution", "nn.SpatialBatchNormalization", "nn.ReLU", "nn.SpatialMaxPooling",
        "nn.SpatialConvolution", "nn.SpatialBatchNormalization", "nn.ReLU", "nn.SpatialMaxPooling",
        "nn.SpatialConvolution", "nn.SpatialBatchNormalization", "nn.ReLU", "nn.SpatialMaxPooling",
        "nn.Reshape", "nn.Linear", "nn.BatchNormalization", "nn.ReLU", "nn.Dropout",
        "nn.Linear", "nn.BatchNormalization", "nn.ReLU", "nn.Dropout", "nn.Linear", "nn.Sigmoid",
    ]
    if module_types != expected_types:
        raise ValueError("Stored module graph does not match the audited Basset graph")

    header, sequence = read_fasta(args.fasta)
    targets = read_targets(args.targets)
    sequence_tensor = one_hot(sequence)
    states = forward_numpy(sequence_tensor, modules)
    tf_states = forward_tensorflow(sequence_tensor, modules)
    compared_states = [name for name in tf_states if name in states]
    state_errors = {name: float(np.max(np.abs(states[name] - tf_states[name]))) for name in compared_states}
    max_error = max(state_errors.values())
    if max_error > 1e-9:
        raise ValueError(f"Independent TensorFlow forward pass differs by {max_error}")

    heatmap_names = ["conv1_relu", "pool1", "conv2_relu", "pool2", "conv3_relu", "pool3"]
    assets = {name: write_tensor(args.public_dir, name, states[name]) for name in heatmap_names}
    receptive = {
        "conv1_relu": (19, 1), "pool1": (21, 3),
        "conv2_relu": (51, 3), "pool2": (60, 12),
        "conv3_relu": (132, 12), "pool3": (168, 48),
    }
    layers = [layer_entry(name, states[name], assets[name], *receptive[name]) for name in heatmap_names]

    conv_modules = [modules[index] for index in (0, 4, 8)]
    bn_modules = [modules[index] for index in (1, 5, 9)]
    dense_modules = [modules[index] for index in (13, 17, 21)]
    dense_bn_modules = [modules[index] for index in (14, 18)]
    dense_weight_matrices = [obj_get(module, "weight") for module in dense_modules]
    dense_weight_assets = {
        "dense1": write_weight_matrix(args.public_dir, "dense1_weights", dense_weight_matrices[0]),
        "dense2": write_weight_matrix(args.public_dir, "dense2_weights", dense_weight_matrices[1]),
        "output": write_weight_matrix(args.public_dir, "output_weights", dense_weight_matrices[2]),
    }

    conv1_strength = states["conv1_relu"].max(axis=1)
    selected_filters = np.argsort(conv1_strength)[-8:][::-1]
    stem_filters = []
    for filter_id in selected_filters:
        weights = obj_get(conv_modules[0], "weight")[filter_id, :, 0, :]
        peak = int(np.argmax(states["conv1_relu"][filter_id]))
        selected_weights = weights[np.argmax(sequence_tensor[:, peak : peak + 19], axis=0), np.arange(19)]
        pool_index = peak // 3
        pool_start = pool_index * 3
        stem_filters.append({
            "filter_id_zero_based": int(filter_id),
            "peak_position_zero_based": peak,
            "peak_activation": float(states["conv1_relu"][filter_id, peak]),
            "weights_bases_by_positions": jsonable(weights),
            "raw_bias": float(obj_get(conv_modules[0], "bias")[filter_id]),
            "batch_norm": {
                "running_mean": float(obj_get(bn_modules[0], "running_mean")[filter_id]),
                "running_inverse_std": float(obj_get(bn_modules[0], "running_std")[filter_id]),
                "gamma": float(obj_get(bn_modules[0], "weight")[filter_id]),
                "beta": float(obj_get(bn_modules[0], "bias")[filter_id]),
            },
            "selected_sequence": sequence[peak : peak + 19],
            "selected_weights": jsonable(selected_weights),
            "dot_product": float(selected_weights.sum()),
            "after_raw_bias": float(states["conv1_linear"][filter_id, peak]),
            "after_batch_norm": float(states["conv1_bn"][filter_id, peak]),
            "after_relu": float(states["conv1_relu"][filter_id, peak]),
            "activation_track": jsonable(states["conv1_relu"][filter_id]),
            "pooled_track": jsonable(states["pool1"][filter_id]),
            "pool_example": {
                "pool_index_zero_based": pool_index,
                "convolution_positions_zero_based": [pool_start, pool_start + 1, pool_start + 2],
                "values": jsonable(states["conv1_relu"][filter_id, pool_start : pool_start + 3]),
                "winner_offset_zero_based": int(states["pool1_argmax"][filter_id, pool_index]),
                "input_span_one_based": [pool_start + 1, pool_start + 21],
            },
        })

    conv2_channel = int(np.argmax(states["conv2_relu"].max(axis=1)))
    conv2_position = int(np.argmax(states["conv2_relu"][conv2_channel]))
    conv2_weights = obj_get(conv_modules[1], "weight")[conv2_channel, :, 0, :]
    conv2_window = states["pool1"][:, conv2_position : conv2_position + 11]
    conv2_contributions = conv2_window * conv2_weights
    contribution_per_channel = conv2_contributions.sum(axis=1)
    top_inputs = np.argsort(np.abs(contribution_per_channel))[-12:][::-1]

    dense1_linear = states["dense1_linear"]
    dense1_unit = int(np.argmax(states["dense1_relu"]))
    dense1_weights = obj_get(dense_modules[0], "weight")[dense1_unit].reshape(200, 10)
    dense1_contributions = dense1_weights * states["pool3"]
    top_dense_cells_flat = np.argsort(np.abs(dense1_contributions.reshape(-1)))[-15:][::-1]
    top_dense_cells = [{
        "channel_zero_based": int(index // 10),
        "position_zero_based": int(index % 10),
        "activation": float(states["pool3"].reshape(-1)[index]),
        "weight": float(dense1_weights.reshape(-1)[index]),
        "contribution": float(dense1_contributions.reshape(-1)[index]),
    } for index in top_dense_cells_flat]

    dense2_unit = int(np.argmax(states["dense2_relu"]))
    dense2_weights = dense_weight_matrices[1][dense2_unit]
    dense2_contributions = dense2_weights * states["dense1_relu"]

    predictions = states["predictions"]
    top_predictions = np.argsort(predictions)[-15:][::-1]
    k562_index = next(index for index, target in enumerate(targets) if target["label"] == "K562")

    architecture = [
        {"stage": "input", "operation": "one-hot DNA", "shape": [4, 600], "note": "four base channels × 600 positions"},
        {"stage": "conv1", "operation": "19-wide valid convolution + batch norm + ReLU", "shape": [300, 582]},
        {"stage": "pool1", "operation": "max pool width/stride 3", "shape": [300, 194]},
        {"stage": "conv2", "operation": "11-wide valid convolution + batch norm + ReLU", "shape": [200, 184]},
        {"stage": "pool2", "operation": "max pool width/stride 4", "shape": [200, 46]},
        {"stage": "conv3", "operation": "7-wide valid convolution + batch norm + ReLU", "shape": [200, 40]},
        {"stage": "pool3", "operation": "max pool width/stride 4", "shape": [200, 10]},
        {"stage": "flatten", "operation": "channel-major flatten", "shape": [2000]},
        {"stage": "dense1", "operation": "linear + batch norm + ReLU", "shape": [1000]},
        {"stage": "dense2", "operation": "linear + batch norm + ReLU", "shape": [1000]},
        {"stage": "output", "operation": "linear + sigmoid", "shape": [164], "note": "one accessibility probability per cell type"},
    ]

    artifact = {
        "schema_version": 1,
        "model": "Basset",
        "task": "DNase I hypersensitivity probability in 164 cell types",
        "provenance": {
            "official_repository": "https://github.com/davek44/Basset",
            "official_repository_commit_inspected": OFFICIAL_REPOSITORY_COMMIT,
            "checkpoint_url": "https://www.dropbox.com/s/rguytuztemctkf8/pretrained_model.th.gz",
            "checkpoint_sha256": checksum,
            "checkpoint_archive_sha256": CHECKPOINT_ARCHIVE_SHA256,
            "checkpoint_format": "Torch7 serialized ConvNet parameters (double precision)",
            "sequence_source": "official Basset tutorial: tutorials/satmut_eg/hoxa_boundary.fa",
            "sequence_header": header,
            "target_source": "official Basset repository: data/models/targets.txt",
        },
        "verification": {
            "operator_graph_source": "stored checkpoint modules, not inferred from parameter text",
            "independent_implementations": ["NumPy", "TensorFlow"],
            "compared_states": compared_states,
            "maximum_absolute_error": max_error,
            "per_state_maximum_absolute_error": state_errors,
            "dropout_mode": "evaluation / identity",
            "batch_norm_note": "legacy Torch7 running_std is stored inverse standard deviation",
        },
        "sequence": sequence,
        "architecture": architecture,
        "layers": layers,
        "stem_filters": stem_filters,
        "conv2_mixing_example": {
            "output_channel_zero_based": conv2_channel,
            "output_position_zero_based": conv2_position,
            "input_window_shape": [300, 11],
            "input_span_one_based": [conv2_position * 3 + 1, conv2_position * 3 + 51],
            "weights_input_channels_by_taps": jsonable(conv2_weights),
            "activations_input_channels_by_taps": jsonable(conv2_window),
            "contributions_input_channels_by_taps": jsonable(conv2_contributions),
            "top_input_channels": [{
                "channel_zero_based": int(channel),
                "signed_contribution": float(contribution_per_channel[channel]),
                "absolute_weight_energy": float(np.sum(conv2_weights[channel] ** 2)),
            } for channel in top_inputs],
            "sum_products": float(conv2_contributions.sum()),
            "raw_bias": float(obj_get(conv_modules[1], "bias")[conv2_channel]),
            "after_raw_bias": float(states["conv2_linear"][conv2_channel, conv2_position]),
            "after_batch_norm": float(states["conv2_bn"][conv2_channel, conv2_position]),
            "after_relu": float(states["conv2_relu"][conv2_channel, conv2_position]),
            "activation_track": jsonable(states["conv2_relu"][conv2_channel]),
            "pooled_track": jsonable(states["pool2"][conv2_channel]),
        },
        "dense_readout_example": {
            "unit_zero_based": dense1_unit,
            "input_shape": [200, 10],
            "input_receptive_field_bp": 168,
            "input_center_spacing_bp": 48,
            "input_coverage_bp": 600,
            "weights_channels_by_positions": jsonable(dense1_weights),
            "contributions_channels_by_positions": jsonable(dense1_contributions),
            "top_cells": top_dense_cells,
            "sum_products": float(dense1_contributions.sum()),
            "raw_bias": float(obj_get(dense_modules[0], "bias")[dense1_unit]),
            "after_raw_bias": float(dense1_linear[dense1_unit]),
            "after_batch_norm": float(states["dense1_bn"][dense1_unit]),
            "after_relu": float(states["dense1_relu"][dense1_unit]),
        },
        "dense2_readout_example": {
            "unit_zero_based": dense2_unit,
            "input_shape": [1000],
            "weights": jsonable(dense2_weights),
            "input_activations": jsonable(states["dense1_relu"]),
            "contributions": jsonable(dense2_contributions),
            "sum_products": float(dense2_contributions.sum()),
            "raw_bias": float(obj_get(dense_modules[1], "bias")[dense2_unit]),
            "after_raw_bias": float(states["dense2_linear"][dense2_unit]),
            "batch_norm": {
                "running_mean": float(obj_get(dense_bn_modules[1], "running_mean")[dense2_unit]),
                "running_inverse_std": float(obj_get(dense_bn_modules[1], "running_std")[dense2_unit]),
                "gamma": float(obj_get(dense_bn_modules[1], "weight")[dense2_unit]),
                "beta": float(obj_get(dense_bn_modules[1], "bias")[dense2_unit]),
            },
            "after_batch_norm": float(states["dense2_bn"][dense2_unit]),
            "after_relu": float(states["dense2_relu"][dense2_unit]),
        },
        "dense_weight_assets": dense_weight_assets,
        "outputs": {
            "k562_index_zero_based": k562_index,
            "k562_probability": float(predictions[k562_index]),
            "dense2_activations": jsonable(states["dense2_relu"]),
            "output_biases": jsonable(obj_get(dense_modules[2], "bias")),
            "k562_reader": {
                "weights": jsonable(dense_weight_matrices[2][k562_index]),
                "contributions": jsonable(dense_weight_matrices[2][k562_index] * states["dense2_relu"]),
                "sum_products": float(np.sum(dense_weight_matrices[2][k562_index] * states["dense2_relu"])),
                "bias": float(obj_get(dense_modules[2], "bias")[k562_index]),
                "logit": float(states["dense3_linear"][k562_index]),
                "probability": float(predictions[k562_index]),
            },
            "top_predictions": [{
                "target_index_zero_based": int(index),
                "label": targets[index]["label"],
                "probability": float(predictions[index]),
            } for index in top_predictions],
            "all_labels": [target["label"] for target in targets],
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(artifact, indent=2) + "\n")
    print(json.dumps({
        "output": str(args.output_json),
        "assets": len(assets),
        "maximum_absolute_error": max_error,
        "selected_stem_filters": [int(index) for index in selected_filters],
        "top_prediction": artifact["outputs"]["top_predictions"][0],
        "k562_probability": artifact["outputs"]["k562_probability"],
    }, indent=2))


if __name__ == "__main__":
    main()
