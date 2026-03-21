from contextgc_barrier import ContextGCBarrier
from tests.fake_backend import FakeBackend

CRITICAL_DETAILS = [
    "process_batch",
    "batch_size > 1000",
    "numpy cache",
    "v2.3",
    "src/batch_processor.py",
    "200mb",
    "cache.clear()",
]


def score_response(text: str) -> float:
    lowered = text.lower()
    found = [detail for detail in CRITICAL_DETAILS if detail.lower() in lowered]
    return len(found) / len(CRITICAL_DETAILS)


def build_turns():
    log_payload = "\n".join(
        f"log {i} rss=200MB path=src/batch_processor.py line=247 trace=process_batch()"
        for i in range(35)
    )
    return [
        "The memory leak happens specifically in process_batch() when batch_size > 1000. RSS grows by 200MB per call and never releases. Started after numpy cache v2.3. File: src/batch_processor.py line 247.",
        "What looks suspicious from that report?",
        "Can you reason about whether the numpy cache is retaining references?",
        "Large production log dump follows:\n" + log_payload,
        "Where should I patch the code?",
        "Would cache.clear() be safe with multiple workers?",
        "Second large log dump follows:\n" + log_payload,
        "What details should I preserve in a ticket summary?",
        "Is the RSS growth enough evidence for the root cause?",
        "Another noisy payload:\n" + log_payload,
        "Before I close this ticket, summarize the root cause and the fix with all exact details.",
    ]


def run_session(strategy: str) -> tuple[str, dict]:
    backend = FakeBackend()
    cgc = ContextGCBarrier(
        backend=backend,
        window_budget=230,
        response_budget=40,
        strategy=strategy,
        sticky_recent_messages=2,
    )
    messages = [{"role": "system", "content": "You are a Python debugging assistant."}]

    final_response = ""
    for prompt in build_turns():
        messages.append({"role": "user", "content": prompt})
        response = cgc.chat(model="fake", messages=messages, max_tokens=40)
        assistant_text = response.choices[0].message.content
        messages.append({"role": "assistant", "content": assistant_text})
        final_response = assistant_text

    return final_response, cgc.context_state()


def test_fake_backend_smoke_keeps_anchor_under_barrier():
    recency_response, recency_state = run_session("recency")
    barrier_response, barrier_state = run_session("barrier")

    assert score_response(barrier_response) >= score_response(recency_response)
    assert any(item["index"] == 1 for item in barrier_state["selected_messages"])
    assert all(item["index"] != 1 for item in recency_state["selected_messages"])
