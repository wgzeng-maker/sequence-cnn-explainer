#!/usr/bin/env python3
"""Regression checks for the compact Basset checkpoint artifact."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "app" / "data" / "basset-demo.json"


def close(actual: float, expected: float, tolerance: float = 1e-6) -> None:
    if abs(actual - expected) > tolerance:
        raise AssertionError(f"{actual} != {expected} within {tolerance}")


def main() -> None:
    data = json.loads(ARTIFACT.read_text())
    assert data["schema_version"] == 1
    assert data["model"] == "Basset"
    assert len(data["sequence"]) == 600
    assert set(data["sequence"]) <= set("ACGT")
    assert data["verification"]["maximum_absolute_error"] < 1e-9

    expected_shapes = [
        [4, 600], [300, 582], [300, 194], [200, 184], [200, 46],
        [200, 40], [200, 10], [2000], [1000], [1000], [164],
    ]
    assert [stage["shape"] for stage in data["architecture"]] == expected_shapes

    expected_layers = {
        "conv1_relu": (300, 582, 19, 1),
        "pool1": (300, 194, 21, 3),
        "conv2_relu": (200, 184, 51, 3),
        "pool2": (200, 46, 60, 12),
        "conv3_relu": (200, 40, 132, 12),
        "pool3": (200, 10, 168, 48),
    }
    for layer in data["layers"]:
        channels, positions, receptive_field, jump = expected_layers[layer["id"]]
        assert (layer["asset"]["channels"], layer["asset"]["positions"]) == (channels, positions)
        assert (layer["receptive_field_bp"], layer["center_spacing_bp"]) == (receptive_field, jump)
        asset = ROOT / "public" / layer["asset"]["url"].lstrip("/")
        assert asset.stat().st_size == channels * positions * 4
        values = np.fromfile(asset, dtype="<f4")
        assert values.size == channels * positions
        close(float(values.max()), layer["asset"]["maximum"], 2e-6)

    sequence = data["sequence"]
    bases = "ACGT"
    for stem_filter in data["stem_filters"]:
        position = stem_filter["peak_position_zero_based"]
        weights = np.asarray(stem_filter["weights_bases_by_positions"], dtype=float)
        selected = np.asarray([weights[bases.index(sequence[position + offset]), offset] for offset in range(19)])
        close(float(selected.sum()), stem_filter["dot_product"])
        close(float(selected.sum() + stem_filter["raw_bias"]), stem_filter["after_raw_bias"])
        bn = stem_filter["batch_norm"]
        normalized = (stem_filter["after_raw_bias"] - bn["running_mean"]) * bn["running_inverse_std"] * bn["gamma"] + bn["beta"]
        close(normalized, stem_filter["after_batch_norm"])
        close(max(0.0, normalized), stem_filter["after_relu"])

    mixing = data["conv2_mixing_example"]
    contributions = np.asarray(mixing["contributions_input_channels_by_taps"], dtype=float)
    assert contributions.shape == (300, 11)
    close(float(contributions.sum()), mixing["sum_products"], 5e-5)
    close(mixing["sum_products"] + mixing["raw_bias"], mixing["after_raw_bias"], 5e-6)
    close(max(0.0, mixing["after_batch_norm"]), mixing["after_relu"])

    dense = data["dense_readout_example"]
    dense_contributions = np.asarray(dense["contributions_channels_by_positions"], dtype=float)
    assert dense_contributions.shape == (200, 10)
    assert dense["input_coverage_bp"] == 600
    close(float(dense_contributions.sum()), dense["sum_products"], 5e-5)
    close(dense["sum_products"] + dense["raw_bias"], dense["after_raw_bias"], 5e-6)
    close(max(0.0, dense["after_batch_norm"]), dense["after_relu"])

    outputs = data["outputs"]
    assert len(outputs["all_labels"]) == 164
    assert outputs["all_labels"][outputs["k562_index_zero_based"]] == "K562"
    assert 0 <= outputs["k562_probability"] <= 1
    assert all(0 <= item["probability"] <= 1 for item in outputs["top_predictions"])
    print("Basset adapter verification passed")


if __name__ == "__main__":
    main()
