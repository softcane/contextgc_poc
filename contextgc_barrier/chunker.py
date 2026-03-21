import uuid
from typing import List
from .extractor import ExtractorMode, extract
from .registry import ContextChunk


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def chunk_message(
    message_id: str,
    message_index: int,
    role: str,
    content: str,
    turn: int,
    message_tokens: int,
    extractor_mode: ExtractorMode = "spacy",
) -> List[ContextChunk]:
    chunks: List[ContextChunk] = []

    if role == "tool" and message_tokens > 500:
        segments = _split_tool_output(content)
        for i, segment in enumerate(segments):
            if not segment.strip():
                continue
            extraction = extract(segment, mode=extractor_mode)
            chunks.append(
                ContextChunk(
                    id=f"{uuid.uuid4().hex[:8]}_t{turn}_s{i}",
                    message_id=message_id,
                    message_index=message_index,
                    text=segment,
                    role=role,
                    turn=turn,
                    tokens=_scaled_tokens(segment, content, message_tokens),
                    keywords=extraction.all_keywords,
                    score=0.6 if i == 0 else 0.4,
                )
            )
    else:
        extraction = extract(content, mode=extractor_mode)
        chunks.append(
            ContextChunk(
                id=f"{uuid.uuid4().hex[:8]}_t{turn}",
                message_id=message_id,
                message_index=message_index,
                text=content,
                role=role,
                turn=turn,
                tokens=max(1, message_tokens),
                keywords=extraction.all_keywords,
                score=_base_score_for_role(role),
            )
        )

    return chunks


def _base_score_for_role(role: str) -> float:
    return {
        "system": 1.0,
        "user": 0.9,
        "assistant": 0.7,
        "tool": 0.5,
    }.get(role, 0.5)


def _scaled_tokens(segment: str, full_content: str, total_tokens: int) -> int:
    if not full_content:
        return 1
    ratio = len(segment) / max(len(full_content), 1)
    return max(1, round(total_tokens * ratio))


def _split_tool_output(content: str) -> List[str]:
    segments = []
    current = []
    in_code_block = False

    for line in content.split("\n"):
        if line.startswith("```"):
            in_code_block = not in_code_block

        current.append(line)

        if (
            not in_code_block
            and not line.strip()
            and estimate_tokens("\n".join(current)) > 300
        ):
            segments.append("\n".join(current))
            current = []

    if current:
        segments.append("\n".join(current))

    return segments
