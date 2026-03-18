from contextgc_barrier import ContextGCBarrier, WriteBarrier, chunk_message
from contextgc_barrier.extractor import extract, overlap_score
from contextgc_barrier.registry import ChunkRegistry, ContextMessage
from tests.fake_backend import FakeBackend


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

    barrier = WriteBarrier(registry=registry)
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


def test_anchor_behavior_keeps_system_and_current_user():
    backend = FakeBackend()
    cgc = ContextGCBarrier(
        backend=backend,
        window_budget=60,
        response_budget=10,
        strategy="recency",
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
