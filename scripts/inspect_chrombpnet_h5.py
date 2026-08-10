#!/usr/bin/env python3
"""Extract architecture metadata from a published Keras HDF5 model without TensorFlow."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import h5py


def json_attr(handle: h5py.File, name: str) -> dict[str, Any]:
    value = handle.attrs[name]
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    return json.loads(value)


def dataset_shapes(handle: h5py.File) -> dict[str, list[int]]:
    shapes: dict[str, list[int]] = {}

    def visit(name: str, item: h5py.Dataset | h5py.Group) -> None:
        if isinstance(item, h5py.Dataset):
            shapes[name] = list(item.shape)

    handle.visititems(visit)
    return shapes


def compact_layer(layer: dict[str, Any], shapes: dict[str, list[int]]) -> dict[str, Any]:
    config = layer.get("config", {})
    name = config.get("name", layer.get("name", "unnamed"))
    relevant = {
        key: config[key]
        for key in (
            "filters",
            "kernel_size",
            "strides",
            "padding",
            "dilation_rate",
            "activation",
            "units",
            "axis",
            "keepdims",
            "cropping",
        )
        if key in config
    }
    weight_shapes = {
        path: shape
        for path, shape in shapes.items()
        if f"/{name}/" in f"/{path}/" or path.startswith(f"{name}/")
    }
    return {
        "name": name,
        "class_name": layer.get("class_name"),
        "config": relevant,
        "weight_shapes": weight_shapes,
        "inbound_nodes": layer.get("inbound_nodes", []),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    with h5py.File(args.model, "r") as handle:
        config = json_attr(handle, "model_config")
        shapes = dataset_shapes(handle)
        layers = config.get("config", {}).get("layers", [])
        extracted = {
            "source_model": args.model.name,
            "keras_class": config.get("class_name"),
            "input_layers": config.get("config", {}).get("input_layers"),
            "output_layers": config.get("config", {}).get("output_layers"),
            "layers": [compact_layer(layer, shapes) for layer in layers],
            "dataset_count": len(shapes),
            "all_dataset_shapes": shapes,
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(extracted, indent=2), encoding="utf-8")
    print(f"Extracted {len(extracted['layers'])} layers and {len(shapes)} datasets to {args.output}")


if __name__ == "__main__":
    main()
