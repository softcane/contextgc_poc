from contextgc_barrier import ContextGCBarrier
from tests.fake_backend import FakeBackend


def build_turns():
    log_payload = "\n".join(
        f"log {i} rss=200MB process_batch() src/batch_processor.py line 247"
        for i in range(30)
    )
    return [
        (
            "The memory leak happens specifically in process_batch() when batch_size > 1000. "
            "RSS grows by 200MB per call. Started after numpy cache v2.3. "
            "File: src/batch_processor.py line 247."
        ),
        "What looks suspicious about that report?",
        "First noisy payload:\n" + log_payload,
        "Where would you patch the code?",
        "Second noisy payload:\n" + log_payload,
        (
            "Before I close this ticket, summarize the root cause and fix with the function, "
            "version, file path, symptoms, and remediation."
        ),
    ]


def run_session(strategy: str, window_budget: int):
    backend = FakeBackend()
    cgc = ContextGCBarrier(
        backend=backend,
        window_budget=window_budget,
        response_budget=40,
        strategy=strategy,
        sticky_recent_messages=2,
    )
    messages = [{"role": "system", "content": "You are a Python debugging assistant."}]

    for prompt in build_turns():
        messages.append({"role": "user", "content": prompt})
        response = cgc.chat(model="fake", messages=messages, max_tokens=40)
        messages.append({"role": "assistant", "content": response.choices[0].message.content})

    return cgc


def test_barrier_and_hybrid_keep_old_cited_message_under_budget():
    summary = run_session(strategy="summary80", window_budget=180)
    barrier = run_session(strategy="barrier", window_budget=180)
    hybrid = run_session(strategy="summary80_barrier", window_budget=180)

    summary_selected = {item["index"] for item in summary.context_state()["selected_messages"]}
    barrier_selected = {item["index"] for item in barrier.context_state()["selected_messages"]}
    hybrid_selected = {item["index"] for item in hybrid.context_state()["selected_messages"]}

    assert 1 not in summary_selected
    assert 1 in barrier_selected
    assert 1 in hybrid_selected
    assert 1 in hybrid.context_state()["protected_exception_indexes"]


def test_summary_strategy_respects_budget_and_reports_summary_metadata():
    cgc = run_session(strategy="summary80", window_budget=180)
    state = cgc.context_state()

    assert state["summary_active"]
    assert state["summary_tokens"] > 0
    assert state["prompt_tokens"] + cgc.response_budget <= cgc.window_budget


def test_ample_budget_keeps_full_transcript_for_all_strategies():
    total_messages = 1 + len(build_turns()) + (len(build_turns()) - 1)
    for strategy in ("summary80", "barrier", "summary80_barrier"):
        cgc = run_session(strategy=strategy, window_budget=5000)
        assert len(cgc.context_state()["selected_messages"]) == total_messages
