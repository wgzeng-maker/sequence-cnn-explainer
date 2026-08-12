#!/usr/bin/env python3
"""Rebuild compressed browser JSON assets from their canonical tracked sources."""

from __future__ import annotations

import gzip
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ASSETS = {
    ROOT / "app/data/k562-peak-activations.json": ROOT / "public/data/demos/k562.json.gz",
    ROOT / "app/data/gm21515-activations.json": ROOT / "public/data/demos/gm21515.json.gz",
    ROOT / "app/data/synthetic-activations.json": ROOT / "public/data/demos/synthetic.json.gz",
    ROOT / "app/data/model-audit-summary.json": ROOT / "public/data/model-audit-summary.json.gz",
}


def main() -> None:
    for source, target in ASSETS.items():
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(gzip.compress(source.read_bytes(), compresslevel=9, mtime=0))
        print(f"{source.relative_to(ROOT)} -> {target.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
