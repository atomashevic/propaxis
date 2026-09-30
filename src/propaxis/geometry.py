"""Pure NumPy geometry derived from the Populism Axis pole-mean construction."""

from __future__ import annotations

import re

import numpy as np


def matrix(value, rows: int, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 2 or result.shape[0] != rows or result.shape[1] == 0:
        raise ValueError(f"{name} must be a ({rows}, dimensions) matrix")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} contains non-finite values")
    return result


def unit_rows(value: np.ndarray) -> np.ndarray:
    return value / np.maximum(np.linalg.norm(value, axis=1, keepdims=True), 1e-12)


def pole_axis(positive, contrast) -> tuple[np.ndarray, float]:
    """normalize(mean(unit(positive)) - mean(unit(contrast)))."""
    difference = unit_rows(positive).mean(axis=0) - unit_rows(contrast).mean(axis=0)
    norm = float(np.linalg.norm(difference))
    if norm < 1e-10:
        raise ValueError("the two poles define a zero-length axis; revise the propositions")
    return difference / norm, norm


def activation(z: np.ndarray, temperature: float) -> np.ndarray:
    # logaddexp avoids overflow in log(1 + exp(z / t)).
    return np.maximum(np.logaddexp(0.0, z / temperature) ** 2, np.finfo(float).tiny)


def combine(values: np.ndarray, rule: str) -> np.ndarray:
    if rule == "geometric_mean":
        # Preserve genuine zero NLI support rather than inventing positive evidence.
        result = np.exp(np.log(np.maximum(values, np.finfo(float).tiny)).mean(axis=1))
        result[(values == 0).any(axis=1)] = 0
        return result
    if rule == "mean":
        return values.mean(axis=1)
    return values.min(axis=1)


def split_windows(text: str, window: int = 1, max_words: int = 60) -> list[str]:
    """Sentence windows with stride one and clause-aware word caps.

    The lightweight punctuation splitter is deterministic, not a linguistic parser;
    abbreviations and languages without these delimiters may need preprocessing.
    """
    parts = []
    for sentence in re.split(r"(?<=[.!?])\s+", text.strip()):
        words = sentence.split()
        if len(words) <= max_words:
            if words:
                parts.append(sentence)
            continue
        chunks: list[list[str]] = []
        for clause in re.split(r"(?<=[;:,])\s+|\s*\n+\s*", sentence):
            words = clause.split()
            for start in range(0, len(words), max_words):
                piece = words[start : start + max_words]
                if chunks and len(chunks[-1]) + len(piece) <= max_words:
                    chunks[-1].extend(piece)
                else:
                    chunks.append(piece)
        parts.extend(" ".join(chunk) for chunk in chunks)
    return (
        [" ".join(parts[i : i + window]) for i in range(max(1, len(parts) - window + 1))]
        if parts
        else []
    )


def select_windows(values: np.ndarray, docs: np.ndarray, count: int, k: int | None):
    selections = []
    for doc in range(count):
        indices = np.flatnonzero(docs == doc)
        if k is not None:
            indices = indices[np.argsort(-values[indices], kind="stable")[:k]]
        selections.append(indices)
    return selections


def pool(values: np.ndarray, selections: list[np.ndarray]) -> np.ndarray:
    return np.asarray([values[indices].mean(axis=0) for indices in selections])
