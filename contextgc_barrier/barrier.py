from dataclasses import dataclass
from typing import Iterable, List, Optional
from .extractor import extract, overlap_score
from .registry import ChunkRegistry, ProtectionLevel

CITATION_THRESHOLD = 0.15
CITATION_SCORE_BOOST = 0.35


@dataclass
class WriteBarrierResult:
    response_text: str
    response_keywords: set
    cited_chunks: List[str]
    cited_messages: List[str]
    missed_chunks: List[str]
    turn: int


class WriteBarrier:
    def __init__(
        self,
        registry: ChunkRegistry,
        citation_threshold: float = CITATION_THRESHOLD,
        score_boost: float = CITATION_SCORE_BOOST,
        citable_roles: Iterable[str] = ("user", "tool"),
    ) -> None:
        self.registry = registry
        self.citation_threshold = citation_threshold
        self.score_boost = score_boost
        self.citable_roles = tuple(citable_roles)
        self._events: List[WriteBarrierResult] = []

    def process(
        self,
        response_text: str,
        turn: int = -1,
        task_keywords: Optional[Iterable[str]] = None,
    ) -> WriteBarrierResult:
        response_extraction = extract(response_text)
        response_keywords = response_extraction.all_keywords
        active_task_keywords = set(task_keywords or [])

        cited_ids: List[str] = []
        cited_message_ids: set[str] = set()
        missed_ids: List[str] = []

        for chunk in self.registry.all():
            if chunk.role not in self.citable_roles:
                continue
            if active_task_keywords and not (chunk.keywords & active_task_keywords):
                missed_ids.append(chunk.id)
                continue

            score = overlap_score(chunk.keywords, response_keywords)
            if score >= self.citation_threshold:
                chunk.cite(turn=turn, score_boost=self.score_boost)
                cited_ids.append(chunk.id)
                cited_message_ids.add(chunk.message_id)
            else:
                missed_ids.append(chunk.id)

        result = WriteBarrierResult(
            response_text=response_text,
            response_keywords=response_keywords,
            cited_chunks=cited_ids,
            cited_messages=sorted(cited_message_ids),
            missed_chunks=missed_ids,
            turn=turn,
        )
        self._events.append(result)
        return result

    def history(self) -> List[WriteBarrierResult]:
        return self._events

    def summary(self) -> dict:
        total_citations = sum(len(event.cited_chunks) for event in self._events)
        return {
            "turns_processed": len(self._events),
            "total_citations": total_citations,
            "protected_chunks": len(
                [
                    chunk
                    for chunk in self.registry.all()
                    if chunk.protection == ProtectionLevel.CITED
                ]
            ),
            "protected_messages": len(self.registry.protected_message_ids()),
            "events": [
                {
                    "turn": event.turn,
                    "cited_chunks": event.cited_chunks,
                    "cited_messages": event.cited_messages,
                    "keyword_count": len(event.response_keywords),
                }
                for event in self._events
            ],
        }
