import pytest

from contextgc_barrier import ContextGCBarrier, WriteBarrier, chunk_message
from contextgc_barrier.backend import make_response
from contextgc_barrier.extractor import extract, overlap_score
from contextgc_barrier.registry import ChunkRegistry, ContextMessage
from tests.fake_backend import FakeBackend


class _ReplayBackend:
    def __init__(self, reply_text: str) -> None:
        self.reply_text = reply_text
        self.create_calls = 0

    def count_tokens(self, messages, model):
        return sum(len(str(message.get("content", "")).split()) + 4 for message in messages)

    def create(self, model, messages, **kwargs):
        self.create_calls += 1
        return make_response(self.reply_text)


def test_write_barrier_promotes_cited_chunk_for_user_message():
    registry = ChunkRegistry()
    message = ContextMessage(
        id="m_0",
        index=0,
        role="user",
        content="memory leak in process_batch() when batch_size > 1000",
        turn=1,
        tokens=12,
    )
    chunks = chunk_message(
        message_id=message.id,
        message_index=message.index,
        role=message.role,
        content=message.content,
        turn=message.turn,
        message_tokens=message.tokens,
    )
    registry.register_message(message, chunks)

    barrier = WriteBarrier(registry=registry, citation_threshold=0.05, extractor_mode="lexical")
    barrier.process(
        response_text="The root cause is in process_batch() and the numpy cache.",
        turn=2,
    )

    chunk = registry.all()[0]
    assert chunk.citation_count == 1
    assert chunk.is_protected()


def test_write_barrier_does_not_promote_assistant_chunks_by_default():
    registry = ChunkRegistry()
    message = ContextMessage(
        id="m_0",
        index=0,
        role="assistant",
        content="I am still checking process_batch().",
        turn=1,
        tokens=8,
    )
    chunks = chunk_message(
        message_id=message.id,
        message_index=message.index,
        role=message.role,
        content=message.content,
        turn=message.turn,
        message_tokens=message.tokens,
    )
    registry.register_message(message, chunks)

    barrier = WriteBarrier(registry=registry)
    barrier.process(
        response_text="I am still checking process_batch().",
        turn=2,
    )

    chunk = registry.all()[0]
    assert chunk.citation_count == 0
    assert not chunk.is_protected()


def test_extract_function_names():
    result = extract("The bug is in process_batch() when batch_size > 1000")
    assert "process_batch()" in result.code_entities


def test_extract_version_strings():
    result = extract("Using numpy v1.24.2 which introduced the cache")
    assert "v1.24.2" in result.code_entities


def test_extract_file_paths():
    result = extract("See src/batch_processor.py line 247")
    assert "src/batch_processor.py" in result.code_entities


def test_overlap_score_zero_on_no_match():
    a = extract("memory leak in process_batch()")
    b = extract("unrelated discussion about weather")
    assert overlap_score(a.all_keywords, b.all_keywords) == 0.0


def test_overlap_score_high_on_match():
    a = extract("memory leak in process_batch()")
    b = extract("The root cause: process_batch() causes a memory leak")
    assert overlap_score(a.all_keywords, b.all_keywords) > 0.5


def test_lexical_extractor_mode_produces_overlap_without_spacy():
    a = extract("Reconfirm EXPORT_V2_GUARD in services/export_pipeline.py", mode="lexical")
    b = extract("The patch still uses EXPORT_V2_GUARD in services/export_pipeline.py", mode="lexical")
    assert overlap_score(a.all_keywords, b.all_keywords) > 0.5


def test_citation_disabled_prevents_protection():
    registry = ChunkRegistry()
    message = ContextMessage(
        id="m_0",
        index=0,
        role="user",
        content="memory leak in process_batch() when batch_size > 1000",
        turn=1,
        tokens=12,
    )
    chunks = chunk_message(
        message_id=message.id,
        message_index=message.index,
        role=message.role,
        content=message.content,
        turn=message.turn,
        message_tokens=message.tokens,
    )
    registry.register_message(message, chunks)

    barrier = WriteBarrier(registry=registry, citation_enabled=False)
    barrier.process(
        response_text="The root cause is in process_batch() and the numpy cache.",
        turn=2,
    )

    chunk = registry.all()[0]
    assert chunk.citation_count == 0
    assert not chunk.is_protected()


def test_anchor_behavior_keeps_system_and_current_user():
    backend = FakeBackend()
    cgc = ContextGCBarrier(
        backend=backend,
        window_budget=60,
        response_budget=10,
        strategy="summary80",
        sticky_recent_messages=2,
    )
    messages = [
        {"role": "system", "content": "You are a debugging assistant."},
        {"role": "user", "content": "Older context that may be dropped."},
        {"role": "assistant", "content": "Acknowledged."},
        {"role": "user", "content": "Current user request with important details."},
    ]

    cgc.chat(model="fake", messages=messages, max_tokens=10)
    state = cgc.context_state()
    selected_indexes = {item["index"] for item in state["selected_messages"]}
    assert 0 in selected_indexes
    assert 3 in selected_indexes


def test_replay_turn_matches_chat_selection_and_citations():
    reply_text = (
        "The active diagnosis points to process_batch() because the numpy cache retains the input array after warmup."
    )
    chat_backend = _ReplayBackend(reply_text)
    replay_backend = _ReplayBackend(reply_text)
    chat_cgc = ContextGCBarrier(
        backend=chat_backend,
        window_budget=90,
        response_budget=20,
        strategy="barrier",
        sticky_recent_messages=1,
        extractor_mode="lexical",
    )
    replay_cgc = ContextGCBarrier(
        backend=replay_backend,
        window_budget=90,
        response_budget=20,
        strategy="barrier",
        sticky_recent_messages=1,
        extractor_mode="lexical",
    )
    messages = [
        {"role": "system", "content": "You are a debugging assistant."},
        {"role": "user", "content": "Live incident path is process_batch() on v2.3.1 when queue_depth > 4096."},
        {"role": "tool", "content": "Diagnosis board: numpy cache retains the input array after warmup."},
        {"role": "user", "content": "Restate the live diagnosis in one line."},
    ]

    chat_response = chat_cgc.chat(model="fake", messages=messages, max_tokens=20)
    replay_response = replay_cgc.replay_turn(
        model="fake",
        messages_before_reply=messages,
        reply_text=reply_text,
        max_tokens=20,
    )

    assert chat_response._cgc_context_state["selected_messages"] == replay_response._cgc_context_state["selected_messages"]
    assert chat_response._cgc_context_state["protected_message_indexes"] == replay_response._cgc_context_state["protected_message_indexes"]
    assert chat_response._cgc_barrier_result.cited_messages == replay_response._cgc_barrier_result.cited_messages


@pytest.mark.parametrize("strategy", ["summary80", "barrier", "summary80_barrier"])
def test_replay_turn_does_not_call_backend_create(strategy: str):
    backend = _ReplayBackend("The active issue is process_batch().")
    cgc = ContextGCBarrier(
        backend=backend,
        window_budget=90,
        response_budget=20,
        strategy=strategy,
        extractor_mode="lexical",
    )
    messages = [
        {"role": "system", "content": "You are a debugging assistant."},
        {"role": "user", "content": "process_batch() is the active incident path."},
        {"role": "user", "content": "Repeat the active path."},
    ]

    cgc.replay_turn(model="fake", messages_before_reply=messages, reply_text="The active path is process_batch().")

    assert backend.create_calls == 0
