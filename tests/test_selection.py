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


def test_barrier_strategy_keeps_old_cited_message_under_budget():
    barrier = run_session(strategy="barrier", window_budget=180)
    recency = run_session(strategy="recency", window_budget=180)

    barrier_selected = {item["index"] for item in barrier.context_state()["selected_messages"]}
    recency_selected = {item["index"] for item in recency.context_state()["selected_messages"]}

    assert 1 in barrier_selected
    assert 1 not in recency_selected


def test_exact_token_budget_is_respected():
    cgc = run_session(strategy="recency", window_budget=180)
    state = cgc.context_state()
    assert state["prompt_tokens"] + cgc.response_budget <= cgc.window_budget


def test_ample_budget_keeps_full_transcript_for_both_strategies():
    barrier = run_session(strategy="barrier", window_budget=5000)
    recency = run_session(strategy="recency", window_budget=5000)

    total_messages = 1 + len(build_turns()) + (len(build_turns()) - 1)
    assert len(barrier.context_state()["selected_messages"]) == total_messages
    assert len(recency.context_state()["selected_messages"]) == total_messages
