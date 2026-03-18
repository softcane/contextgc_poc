import math
from typing import Iterable, Set
from .registry import ContextChunk, ProtectionLevel

ROLE_WEIGHT = {
    "system": 1.0,
    "user": 0.9,
    "assistant": 0.7,
    "tool": 0.5,
}


def score_chunk(
    chunk: ContextChunk,
    current_turn: int,
    task_keywords: Set[str],
    decay_rate: float = 0.12,
) -> float:
    if chunk.protection == ProtectionLevel.PINNED:
        return 1.0

    delta = max(0, current_turn - chunk.turn)
    recency = math.exp(-decay_rate * delta)

    if task_keywords and chunk.keywords:
        intersection = chunk.keywords & task_keywords
        relevance = len(intersection) / min(len(task_keywords), len(chunk.keywords))
    else:
        relevance = 0.0

    role_weight = ROLE_WEIGHT.get(chunk.role, 0.5)
    raw_score = (0.40 * relevance) + (0.35 * recency) + (0.25 * role_weight)

    if chunk.protection == ProtectionLevel.CITED:
        citation_boost = min(0.15 * chunk.citation_count, 0.45)
        raw_score = min(raw_score + citation_boost, 1.0)

    return round(raw_score, 4)


def score_all_chunks(
    chunks: Iterable[ContextChunk],
    current_turn: int,
    task_keywords: Set[str],
) -> None:
    for chunk in chunks:
        if chunk.role == "system":
            continue
        chunk.score = score_chunk(
            chunk=chunk,
            current_turn=current_turn,
            task_keywords=task_keywords,
        )
