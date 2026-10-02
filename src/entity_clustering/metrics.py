"""Distances and paper-defined quantile evaluation metrics."""

from __future__ import annotations

import fastnanquantile as fnq
import numpy as np
from scipy.special import digamma, gammaln


def _positive_dirichlet(alpha: np.ndarray) -> np.ndarray:
    values = np.asarray(alpha, dtype=float)
    if values.ndim != 2 or values.shape[1] == 0 or not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("Dirichlet parameters must be a finite, positive 2D array")
    return values


def kl_divergence_dirichlet(alpha_1: np.ndarray, alpha_2: np.ndarray) -> np.ndarray:
    """Pairwise KL(alpha_1 || alpha_2) for matching rows."""

    left = _positive_dirichlet(alpha_1)
    right = _positive_dirichlet(alpha_2)
    if left.shape != right.shape:
        raise ValueError("Dirichlet arrays must have identical shapes")
    left_sum = left.sum(axis=1)
    right_sum = right.sum(axis=1)
    return (
        gammaln(left_sum)
        - gammaln(left).sum(axis=1)
        - gammaln(right_sum)
        + gammaln(right).sum(axis=1)
        + ((left - right) * (digamma(left) - digamma(left_sum)[:, None])).sum(axis=1)
    )


def symmetric_kl_matrix(gamma: np.ndarray) -> np.ndarray:
    """Return the symmetric-KL matrix for entity Dirichlet parameters."""

    values = _positive_dirichlet(gamma)
    left = np.repeat(values, len(values), axis=0)
    right = np.tile(values, (len(values), 1))
    return (0.5 * (kl_divergence_dirichlet(left, right) + kl_divergence_dirichlet(right, left))).reshape(
        len(values), len(values)
    )


def bhattacharyya_matrix(gamma: np.ndarray) -> np.ndarray:
    """Return the Bhattacharyya distance matrix for entity Dirichlet parameters."""

    values = _positive_dirichlet(gamma)
    left = values[:, None, :]
    right = values[None, :, :]
    left_sum = left.sum(axis=2)
    right_sum = right.sum(axis=2)
    return (
        gammaln(0.5 * (left_sum + right_sum))
        + 0.5 * (gammaln(left).sum(axis=2) + gammaln(right).sum(axis=2))
        - gammaln(0.5 * (left + right)).sum(axis=2)
        - 0.5 * (gammaln(left_sum) + gammaln(right_sum))
    )


def quantile_loss(
    targets: np.ndarray,
    predictions: np.ndarray,
    quantiles: list[float],
    *,
    nonzero: bool = False,
) -> tuple[float, np.ndarray]:
    """Return the paper's mean quantile loss and its per-quantile/user/day values."""

    truth = np.asarray(targets, dtype=float)
    predicted = np.asarray(predictions, dtype=float)
    levels = np.asarray(quantiles, dtype=float)
    if predicted.shape != (len(levels), *truth.shape):
        raise ValueError("predictions must have shape (quantiles, *targets.shape)")
    errors = truth[None, ...] - predicted
    losses = np.maximum(levels[:, None, None, None] * errors, (levels[:, None, None, None] - 1) * errors)
    if nonzero:
        all_zero = np.all(np.isnan(losses) | (losses == 0), axis=0)
        losses[:, all_zero] = np.nan
    per_quantile = _nanmean_without_warning(losses, axis=-1)
    return float(_nanmean_without_warning(per_quantile, axis=None)), per_quantile


def _nanmean_without_warning(values: np.ndarray, axis: int | None) -> np.ndarray:
    count = np.sum(~np.isnan(values), axis=axis)
    total = np.nansum(values, axis=axis)
    return np.divide(total, count, out=np.full(np.shape(total), np.nan), where=count != 0)


def cluster_quantile_predictions(
    profiles: np.ndarray,
    labels: np.ndarray,
    quantiles: list[float],
    *,
    leave_one_out: bool,
) -> np.ndarray:
    """Build one quantile profile per system from its assigned cluster."""

    values = np.asarray(profiles, dtype=float)
    assignment = np.asarray(labels)
    if values.ndim != 3 or assignment.shape != (values.shape[0],):
        raise ValueError("profiles and labels must have shapes (systems, days, 96) and (systems,)")
    result = np.full((len(quantiles), *values.shape), np.nan)
    for user in range(values.shape[0]):
        members = np.flatnonzero(assignment == assignment[user])
        if leave_one_out:
            members = members[members != user]
        if len(members):
            result[:, user] = fnq.nanquantile(values[members], quantiles, axis=0)
    return result


def valid_cluster_assignment(labels: np.ndarray, min_cluster_size: int) -> tuple[bool, list[int]]:
    """Return whether every cluster has enough members for the requested evaluation."""

    assignment = np.asarray(labels)
    counts = [int(np.sum(assignment == label)) for label in np.unique(assignment)]
    return min(counts, default=0) >= min_cluster_size, counts
