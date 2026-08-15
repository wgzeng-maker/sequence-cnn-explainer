#!/usr/bin/env python3
"""Scan a FASTA corpus with the published stem convolution and build activation PFMs.

The script streams equal-length minibatches, retains a bounded candidate
reservoir, then greedily chooses non-overlapping top activations per filter.
It can merge the resulting compact motifs into model-audit-summary.json.
"""

from __future__ import annotations

import argparse
import heapq
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import h5py
import numpy as np


BASES = "ACGT"
COMPLEMENT = str.maketrans("ACGTN", "TGCAN")


def fasta_records(path: Path) -> Iterator[tuple[str, str]]:
    name: str | None = None
    parts: list[str] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if name is not None:
                    yield name, "".join(parts).upper()
                name, parts = line[1:].split()[0], []
            else:
                parts.append(line)
    if name is not None:
        yield name, "".join(parts).upper()


def reverse_complement(sequence: str) -> str:
    return sequence.translate(COMPLEMENT)[::-1]


def one_hot(sequence: str) -> np.ndarray:
    encoded = np.zeros((len(sequence), 4), dtype=np.float32)
    for base_index, base in enumerate(BASES):
        encoded[np.fromiter((letter == base for letter in sequence), dtype=bool), base_index] = 1
    return encoded


def load_stem(model: Path) -> tuple[np.ndarray, np.ndarray]:
    name = "wo_bias_bpnet_1st_conv"
    with h5py.File(model, "r") as handle:
        root = handle[f"model_weights/{name}/{name}"]
        return np.asarray(root["kernel:0"], dtype=np.float32), np.asarray(root["bias:0"], dtype=np.float32)


def scan_batch(records: list[tuple[str, str]], kernel: np.ndarray, bias: np.ndarray) -> np.ndarray:
    try:
        import tensorflow as tf
        batch = np.stack([one_hot(sequence) for _, sequence in records])
        return tf.nn.relu(tf.nn.conv1d(batch, kernel, stride=1, padding="VALID") + bias).numpy()
    except ImportError:
        # Correct but slower fallback, useful for small test corpora.
        outputs = []
        for _, sequence in records:
            encoded = one_hot(sequence)
            windows = np.lib.stride_tricks.sliding_window_view(encoded, (21, 4))[:, 0]
            outputs.append(np.maximum(0, np.tensordot(windows, kernel, axes=([1, 2], [0, 1])) + bias))
        return np.stack(outputs)


def information_content(pfm: np.ndarray, background: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        entropy_terms = np.where(pfm > 0, pfm * np.log2(pfm), 0)
    # Standard four-base sequence-logo information: 2 - H(column), bounded 0–2 bits.
    # Background frequencies are retained as provenance but not folded into this height.
    _ = background
    return 2 + entropy_terms.sum(axis=1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--fasta", type=Path, required=True, help="One genomic window per FASTA record; all records must have equal length.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--corpus-id", required=True)
    parser.add_argument("--checkpoint-key", default="k562-peak")
    parser.add_argument("--audit-json", type=Path, help="Optionally merge motifs into this audit artifact.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--sites-per-filter", type=int, default=500)
    parser.add_argument("--candidates-per-sequence", type=int, default=4)
    parser.add_argument("--include-reverse-complement", action="store_true")
    args = parser.parse_args()

    kernel, bias = load_stem(args.model)
    reservoir_size = args.sites_per_filter * 4
    reservoirs: list[list[tuple[float, str, int, str]]] = [[] for _ in range(512)]
    base_counts = np.ones(4, dtype=np.float64)  # weak pseudocount
    total_records = 0
    batch: list[tuple[str, str]] = []

    def consume(records: list[tuple[str, str]]) -> None:
        nonlocal total_records, base_counts
        activations = scan_batch(records, kernel, bias)
        for record_index, (name, sequence) in enumerate(records):
            total_records += 1
            for base_index, base in enumerate(BASES):
                base_counts[base_index] += sequence.count(base)
            for filter_id in range(512):
                track = activations[record_index, :, filter_id]
                candidate_count = min(args.candidates_per_sequence, len(track))
                positions = np.argpartition(track, -candidate_count)[-candidate_count:]
                for position in positions:
                    score = float(track[position])
                    if score <= 0:
                        continue
                    candidate = (score, name, int(position), sequence[position:position + 21])
                    heap = reservoirs[filter_id]
                    if len(heap) < reservoir_size:
                        heapq.heappush(heap, candidate)
                    elif score > heap[0][0]:
                        heapq.heapreplace(heap, candidate)

    for name, sequence in fasta_records(args.fasta):
        if len(sequence) < 21:
            continue
        batch.append((name, sequence))
        if args.include_reverse_complement:
            batch.append((f"{name}|rc", reverse_complement(sequence)))
        if len(batch) >= args.batch_size:
            lengths = {len(sequence) for _, sequence in batch}
            if len(lengths) != 1:
                raise ValueError("Each minibatch requires equal-length FASTA records; pre-pad or group the corpus by length.")
            consume(batch)
            batch = []
    if batch:
        lengths = {len(sequence) for _, sequence in batch}
        if len(lengths) != 1:
            raise ValueError("Each minibatch requires equal-length FASTA records; pre-pad or group the corpus by length.")
        consume(batch)

    background = base_counts / base_counts.sum()
    motifs: list[dict[str, Any]] = []
    for filter_id, heap in enumerate(reservoirs):
        selected: list[tuple[float, str, int, str]] = []
        occupied: dict[str, list[tuple[int, int]]] = {}
        for candidate in sorted(heap, reverse=True):
            score, record, position, sequence = candidate
            interval = (position, position + 21)
            if any(interval[0] < end and start < interval[1] for start, end in occupied.get(record, [])):
                continue
            selected.append(candidate)
            occupied.setdefault(record, []).append(interval)
            if len(selected) >= args.sites_per_filter:
                break
        if not selected:
            continue
        counts = np.ones((21, 4), dtype=np.float64) * .5
        for _, _, _, sequence in selected:
            for position, base in enumerate(sequence):
                if base in BASES:
                    counts[position, BASES.index(base)] += 1
        pfm = counts / counts.sum(axis=1, keepdims=True)
        motifs.append({
            "filter_id_zero_based": filter_id,
            "pfm_positions_by_bases": pfm.tolist(),
            "background_frequencies": background.tolist(),
            "information_content_bits": information_content(pfm, background).tolist(),
            "site_count": len(selected),
            "corpus_id": args.corpus_id,
            "activation_selection_rule": f"top positive stem activations; 21-bp non-overlap within each record; reservoir {reservoir_size}",
            "orientation": "model",
        })

    result = {
        "schema_version": "1.0.0",
        "model": args.model.name,
        "corpus_id": args.corpus_id,
        "records_scanned_including_rc": total_records,
        "include_reverse_complement": args.include_reverse_complement,
        "motifs": motifs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, separators=(",", ":")), encoding="utf-8")

    if args.audit_json:
        artifact = json.loads(args.audit_json.read_text())
        target = artifact["checkpoints"][args.checkpoint_key]["activation_motifs"]
        target.update({
            "status": "generated",
            "reason": "Generated from streamed genomic stem activations.",
            "target_site_count_per_filter": args.sites_per_filter,
            "planned_corpus": args.corpus_id,
            "selection_rule": "top positive non-overlapping stem activations",
            "motifs": motifs,
        })
        args.audit_json.write_text(json.dumps(artifact, separators=(",", ":")), encoding="utf-8")
    print(f"Scanned {total_records} records; wrote {len(motifs)} filter motifs to {args.output}")


if __name__ == "__main__":
    main()
