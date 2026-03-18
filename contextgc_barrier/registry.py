from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Set
import time


class ProtectionLevel(Enum):
    NORMAL = "normal"
    CITED = "cited"
    PINNED = "pinned"


@dataclass
class ContextChunk:
    id: str
    message_id: str
    message_index: int
    text: str
    role: str
    turn: int
    tokens: int
    keywords: Set[str]
    score: float = 0.0
    citation_count: int = 0
    protection: ProtectionLevel = ProtectionLevel.NORMAL
    created_at: float = field(default_factory=time.time)
    last_cited_turn: Optional[int] = None

    def is_protected(self) -> bool:
        return self.protection in (ProtectionLevel.CITED, ProtectionLevel.PINNED)

    def cite(self, turn: int, score_boost: float = 0.35) -> None:
        if self.protection != ProtectionLevel.PINNED:
            self.protection = ProtectionLevel.CITED
        self.citation_count += 1
        self.last_cited_turn = turn
        self.score = min(self.score + score_boost, 1.0)


@dataclass
class ContextMessage:
    id: str
    index: int
    role: str
    content: str
    turn: int
    tokens: int
    chunk_ids: List[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)


class ChunkRegistry:
    def __init__(self) -> None:
        self._chunks: dict[str, ContextChunk] = {}
        self._messages: dict[str, ContextMessage] = {}
        self._message_index: dict[int, str] = {}

    def register_message(
        self,
        message: ContextMessage,
        chunks: List[ContextChunk],
    ) -> None:
        self._messages[message.id] = message
        self._message_index[message.index] = message.id
        message.chunk_ids = [chunk.id for chunk in chunks]
        for chunk in chunks:
            self._chunks[chunk.id] = chunk

    def get(self, chunk_id: str) -> Optional[ContextChunk]:
        return self._chunks.get(chunk_id)

    def all(self) -> List[ContextChunk]:
        return list(self._chunks.values())

    def get_message(self, message_id: str) -> Optional[ContextMessage]:
        return self._messages.get(message_id)

    def get_message_by_index(self, index: int) -> Optional[ContextMessage]:
        message_id = self._message_index.get(index)
        if message_id is None:
            return None
        return self._messages.get(message_id)

    def all_messages(self) -> List[ContextMessage]:
        return sorted(self._messages.values(), key=lambda message: message.index)

    def message_chunks(self, message_id: str) -> List[ContextChunk]:
        message = self.get_message(message_id)
        if message is None:
            return []
        return [
            self._chunks[chunk_id]
            for chunk_id in message.chunk_ids
            if chunk_id in self._chunks
        ]

    def message_protection(self, message_id: str) -> ProtectionLevel:
        chunks = self.message_chunks(message_id)
        if any(chunk.protection == ProtectionLevel.PINNED for chunk in chunks):
            return ProtectionLevel.PINNED
        if any(chunk.protection == ProtectionLevel.CITED for chunk in chunks):
            return ProtectionLevel.CITED
        return ProtectionLevel.NORMAL

    def message_is_protected(self, message_id: str) -> bool:
        return self.message_protection(message_id) != ProtectionLevel.NORMAL

    def message_score(self, message_id: str) -> float:
        chunks = self.message_chunks(message_id)
        if not chunks:
            return 0.0
        return max(chunk.score for chunk in chunks)

    def message_citations(self, message_id: str) -> int:
        return sum(chunk.citation_count for chunk in self.message_chunks(message_id))

    def cited_message_ids(self) -> List[str]:
        return [
            message.id
            for message in self.all_messages()
            if self.message_citations(message.id) > 0
        ]

    def protected_message_ids(self) -> List[str]:
        return [
            message.id
            for message in self.all_messages()
            if self.message_is_protected(message.id)
        ]

    def stats(self) -> dict:
        messages = self.all_messages()
        chunks = self.all()
        return {
            "total_messages": len(messages),
            "total_chunks": len(chunks),
            "total_tokens": sum(message.tokens for message in messages),
            "cited_messages": len(self.cited_message_ids()),
            "protected_messages": len(self.protected_message_ids()),
        }
