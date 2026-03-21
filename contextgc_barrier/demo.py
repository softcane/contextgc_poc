import argparse
from typing import Any, Dict, List

from .backend import make_response
from .wrapper import ContextGCBarrier

CRITICAL_DETAILS = [
    ("process_batch", "process_batch()"),
    ("batch_size > 1000", "batch_size > 1000"),
    ("numpy cache", "numpy cache"),
    ("v2.3", "v2.3"),
    ("src/batch_processor.py", "src/batch_processor.py"),
    ("200mb", "200MB"),
    ("cache.clear()", "cache.clear()"),
]


class DemoBackend:
    def count_tokens(self, messages: List[Dict[str, str]], model: str) -> int:
        return sum(len(str(message.get("content", "")).split()) + 4 for message in messages)

    def create(self, model: str, messages: List[Dict[str, str]], **kwargs: Any):
        prompt_text = "\n".join(
            str(message.get("content", "")).lower()
            for message in messages
        )
        last_user = next(
            (
                str(message.get("content", "")).lower()
                for message in reversed(messages)
                if message.get("role") == "user"
            ),
            "",
        )

        if "compress conversation state" in prompt_text or "working-memory bullets" in prompt_text:
            found = [response_text for needle, response_text in CRITICAL_DETAILS if needle in prompt_text]
            bullets = found[:4] or ["earlier debugging context still matters"]
            return make_response("Rolling summary:\n" + "\n".join(f"- {item}" for item in bullets))

        if "before i close this ticket" in last_user:
            found = [response_text for needle, response_text in CRITICAL_DETAILS if needle in prompt_text]
            if not found:
                return make_response("I lost the earlier context and only remember a generic cache issue.")
            return make_response("Summary: " + "; ".join(found) + ".")

        if "memory leak happens specifically" in last_user:
            return make_response(
                "Initial read: process_batch() plus the numpy cache from v2.3 looks suspicious."
            )

        if "thread-safety" in last_user or "cache.clear()" in last_user:
            return make_response(
                "Use cache.clear() behind a lock, but the leak still points to process_batch()."
            )

        if "where should i patch" in last_user:
            return make_response(
                "Patch src/batch_processor.py around process_batch() and clear the numpy cache."
            )

        if "log dump" in last_user or "payload" in last_user:
            return make_response(
                "The logs are noisy, but process_batch() still matches the RSS growth pattern."
            )

        return make_response("I am still checking process_batch() and the numpy cache.")


def build_turns() -> List[str]:
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
        "Before I close this ticket, summarize the root cause and fix with all exact details.",
    ]


def score_response(text: str) -> float:
    lowered = text.lower()
    found = [detail for detail, _ in CRITICAL_DETAILS if detail.lower() in lowered]
    return len(found) / len(CRITICAL_DETAILS)


def run_session(strategy: str, window_budget: int) -> tuple[str, dict]:
    backend = DemoBackend()
    cgc = ContextGCBarrier(
        backend=backend,
        window_budget=window_budget,
        response_budget=40,
        strategy=strategy,
        sticky_recent_messages=2,
    )
    messages = [{"role": "system", "content": "You are a Python debugging assistant."}]
    final_response = ""

    for prompt in build_turns():
        messages.append({"role": "user", "content": prompt})
        response = cgc.chat(model="demo", messages=messages, max_tokens=40)
        assistant_text = response.choices[0].message.content
        messages.append({"role": "assistant", "content": assistant_text})
        final_response = assistant_text

    return final_response, cgc.context_state()


def print_run(strategy: str, window_budget: int) -> None:
    response, state = run_session(strategy=strategy, window_budget=window_budget)
    selected_indexes = [item["index"] for item in state["selected_messages"]]

    print(f"\n=== {strategy.upper()} ===")
    print(f"Recall score: {score_response(response):.0%}")
    print(f"Prompt tokens: {state['prompt_tokens']}")
    print(f"Selected message indexes: {selected_indexes}")
    print(f"Kept original bug report: {any(item['index'] == 1 for item in state['selected_messages'])}")
    print("Final response:")
    print(response)


def main() -> None:
    parser = argparse.ArgumentParser(description="Quick local demo for ContextGCBarrier.")
    parser.add_argument(
        "--strategy",
        choices=["summary80", "barrier", "summary80_barrier"],
        default="barrier",
        help="Which selection strategy to run.",
    )
    parser.add_argument(
        "--window-budget",
        type=int,
        default=230,
        help="Small prompt budget to force compaction in the demo.",
    )
    args = parser.parse_args()

    print_run(args.strategy, args.window_budget)


if __name__ == "__main__":
    main()
