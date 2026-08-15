#!/usr/bin/env python3
"""Decisive toy checks for orientation, exact centering, RC, PFM, and permutations."""

from __future__ import annotations

import numpy as np


BASE_INDEX = {base: index for index, base in enumerate("ACGT")}


def one_hot(sequence: str) -> np.ndarray:
    result = np.zeros((len(sequence), 4), dtype=np.float64)
    for position, base in enumerate(sequence):
        result[position, BASE_INDEX[base]] = 1
    return result


def correlate(sequence: str, kernel: np.ndarray, bias: float = 0) -> np.ndarray:
    source = one_hot(sequence)
    return np.asarray([bias + np.sum(source[start:start + len(kernel)] * kernel) for start in range(len(sequence) - len(kernel) + 1)])


def reverse_complement(matrix: np.ndarray) -> np.ndarray:
    return matrix[::-1, [3, 2, 1, 0]]


def test_orientation() -> None:
    sequence = "TTACGTTGCATT"
    kernel = np.full((3, 4), -1.0)
    kernel[0, BASE_INDEX["A"]] = 2
    kernel[1, BASE_INDEX["C"]] = 2
    kernel[2, BASE_INDEX["G"]] = 2
    assert int(np.argmax(correlate(sequence, kernel))) == 2  # ACG
    assert int(np.argmax(correlate(sequence, kernel[::-1]))) == 7  # GCA


def test_exact_centering() -> None:
    rng = np.random.default_rng(7)
    kernel = rng.normal(size=(21, 4))
    bias = .37
    means = kernel.mean(axis=1)
    centered = kernel - means[:, None]
    adjusted_bias = bias + means.sum()
    sequence = "ACGTACGTACGTACGTACGTAACGTACGT"
    np.testing.assert_allclose(correlate(sequence, kernel, bias), correlate(sequence, centered, adjusted_bias), atol=1e-12)


def test_pfm_and_reverse_complement() -> None:
    pfm = np.asarray([[.7, .1, .1, .1], [.1, .7, .1, .1], [.1, .1, .7, .1]])
    np.testing.assert_allclose(pfm.sum(axis=1), 1)
    information = 2 + np.sum(np.where(pfm > 0, pfm * np.log2(pfm), 0), axis=1)
    assert np.all((0 <= information) & (information <= 2))
    np.testing.assert_allclose(reverse_complement(reverse_complement(pfm)), pfm)


def test_global_permutation_invariance() -> None:
    rng = np.random.default_rng(11)
    x = rng.normal(size=(17, 8))
    residual = rng.normal(size=(3, 8, 8))
    profile = rng.normal(size=(5, 8))
    count = rng.normal(size=8)
    permutation = rng.permutation(8)
    # One permutation must be applied to both the producer and consumer axes.
    residual_p = residual[:, permutation, :][:, :, permutation]
    x_p = x[:, permutation]
    profile_p = profile[:, permutation]
    count_p = count[permutation]
    for tap in range(3):
        np.testing.assert_allclose((x[0] @ residual[tap])[permutation], x_p[0] @ residual_p[tap], atol=1e-10)
    np.testing.assert_allclose(x @ count, x_p @ count_p)
    np.testing.assert_allclose(x @ profile.T, x_p @ profile_p.T)


if __name__ == "__main__":
    test_orientation()
    test_exact_centering()
    test_pfm_and_reverse_complement()
    test_global_permutation_invariance()
    print("model-analysis invariants: PASS")
