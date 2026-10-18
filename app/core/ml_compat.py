"""Compatibility helpers for numerical and ML operations when optional libraries are unavailable."""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any, Iterable, List, Sequence, Tuple

import numpy as np


def _normalize_labels(values: Sequence[Any]) -> List[str]:
    return [str(v) for v in values]


def cohen_kappa_score(y_true: Sequence[Any], y_pred: Sequence[Any]) -> float:
    """Compute Cohen's kappa coefficient without requiring scikit-learn."""
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    if not y_true:
        return 0.0

    labels = sorted(set(_normalize_labels(y_true)) | set(_normalize_labels(y_pred)))
    if len(labels) <= 1:
        return 0.0

    label_to_index = {label: idx for idx, label in enumerate(labels)}
    matrix = np.zeros((len(labels), len(labels)), dtype=float)

    for actual, predicted in zip(y_true, y_pred):
        row = label_to_index[str(actual)]
        col = label_to_index[str(predicted)]
        matrix[row, col] += 1.0

    total = matrix.sum()
    if total == 0:
        return 0.0

    observed = np.trace(matrix) / total
    row_totals = matrix.sum(axis=1)
    col_totals = matrix.sum(axis=0)
    expected = float(np.sum((row_totals[:, None] * col_totals[None, :]) / total))
    denominator = 1.0 - expected
    if math.isclose(denominator, 0.0):
        return 0.0
    return float((observed - expected) / denominator)


def safe_train_test_split(
    X: Sequence[Any],
    y: Sequence[Any] | None = None,
    train_size: float | int = 0.5,
    random_state: int | None = None,
    stratify: Sequence[Any] | None = None,
    **_: Any,
) -> Tuple[List[Any], List[Any]] | Tuple[List[Any], List[Any], List[Any], List[Any]]:
    """Simple train/test split fallback for environments without scikit-learn."""
    n_items = len(X)
    if n_items == 0:
        return [], [] if y is None else ([], [])

    if isinstance(train_size, float):
        train_count = max(1, min(n_items - 1, int(round(n_items * train_size))))
    else:
        train_count = max(1, min(n_items - 1, int(train_size)))

    rng = np.random.default_rng(random_state)
    indices = np.arange(n_items)
    rng.shuffle(indices)

    train_idx = indices[:train_count]
    test_idx = indices[train_count:]

    if stratify is not None:
        grouped: dict[str, list[int]] = {}
        for idx, label in enumerate(stratify):
            grouped.setdefault(str(label), []).append(idx)

        train_idx = []
        test_idx = []
        for label_indices in grouped.values():
            values = np.array(label_indices, dtype=int)
            rng.shuffle(values)
            group_count = max(1, min(len(values) - 1, int(round(len(values) * train_size)))) if len(values) > 1 else len(values)
            train_idx.extend(values[:group_count].tolist())
            test_idx.extend(values[group_count:].tolist())

        train_idx = sorted(train_idx)
        test_idx = sorted(test_idx)

    x_train = [X[idx] for idx in train_idx]
    x_test = [X[idx] for idx in test_idx]

    if y is None:
        return x_train, x_test

    y_train = [y[idx] for idx in train_idx]
    y_test = [y[idx] for idx in test_idx]
    return x_train, x_test, y_train, y_test


def semantic_similarity_fallback(query: str, references: Sequence[str]) -> float:
    """Token-overlap similarity fallback used when scikit-learn is unavailable."""
    if not query or not references:
        return 0.0

    valid_refs = [str(ref).strip() for ref in references if isinstance(ref, str) and str(ref).strip()]
    if not valid_refs:
        return 0.0

    query_tokens = Counter(re.findall(r"\w+", query.lower()))
    if not query_tokens:
        return 0.0

    best = 0.0
    for ref in valid_refs:
        ref_tokens = Counter(re.findall(r"\w+", ref.lower()))
        if not ref_tokens:
            continue

        union = set(query_tokens) | set(ref_tokens)
        if not union:
            continue

        overlap = sum(min(query_tokens[token], ref_tokens[token]) for token in union)
        score = overlap / max(1, sum(query_tokens.values()) + sum(ref_tokens.values()) - overlap)
        if score > best:
            best = score

    return float(best)
