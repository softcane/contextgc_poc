from __future__ import annotations

from math import comb
from random import Random
from statistics import mean


def paired_deltas(left: list[float], right: list[float]) -> list[float]:
    return [a - b for a, b in zip(left, right)]


def paired_bootstrap_ci(
    deltas: list[float],
    *,
    samples: int = 1000,
    confidence: float = 0.95,
    seed: int = 0,
) -> tuple[float, float]:
    if not deltas:
        return (0.0, 0.0)
    if len(deltas) == 1:
        return (deltas[0], deltas[0])

    rng = Random(seed)
    means = []
    for _ in range(samples):
        resample = [deltas[rng.randrange(len(deltas))] for _ in deltas]
        means.append(mean(resample))
    means.sort()
    alpha = (1.0 - confidence) / 2.0
    low_index = max(0, int(alpha * (len(means) - 1)))
    high_index = min(len(means) - 1, int((1.0 - alpha) * (len(means) - 1)))
    return (means[low_index], means[high_index])


def exact_sign_test_p_value(deltas: list[float]) -> float:
    wins = sum(1 for delta in deltas if delta > 0)
    losses = sum(1 for delta in deltas if delta < 0)
    trials = wins + losses
    if trials == 0:
        return 1.0
    tail = min(wins, losses)
    probability = sum(comb(trials, k) for k in range(0, tail + 1)) / (2 ** trials)
    return min(1.0, 2.0 * probability)


def win_tie_loss(deltas: list[float]) -> tuple[int, int, int]:
    wins = sum(1 for delta in deltas if delta > 0)
    ties = sum(1 for delta in deltas if delta == 0)
    losses = sum(1 for delta in deltas if delta < 0)
    return wins, ties, losses
