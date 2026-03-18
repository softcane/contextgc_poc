import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from contextgc_barrier import ContextGCBarrier, MLXBackend

DEFAULT_MODEL = "mlx-community/Qwen3.5-4B-OptiQ-4bit"
DEFAULT_RESPONSE_BUDGET = 128
DEFAULT_OVERFLOW_RATIO = 1.10
INTERMEDIATE_MAX_TOKENS = 96

SYSTEM_PROMPT = (
    "You are a terse Python debugging assistant. "
    "Answer directly and keep intermediate replies under two sentences. "
    "Preserve exact literals from the transcript when asked."
)

BUG_REPORT = (
    "The memory leak happens specifically in process_batch() when batch_size > 1000. "
    "RSS grows by ~200MB per call and never releases. Started after we added the numpy "
    "cache in v2.3. File: src/batch_processor.py line 247."
)


@dataclass(frozen=True)
class CriticalFact:
    name: str
    patterns: tuple[re.Pattern[str], ...]


CRITICAL_FACTS = [
    CriticalFact("function", (re.compile(r"process_batch(?:\(\))?", re.IGNORECASE),)),
    CriticalFact(
        "trigger",
        (
            re.compile(r"batch[_ ]size\s*(?:>|>=|above|over|exceed(?:s|ing)?)\s*1000", re.IGNORECASE),
            re.compile(r"batch sizes?\s*(?:above|over)\s*1000", re.IGNORECASE),
        ),
    ),
    CriticalFact(
        "symptom",
        (
            re.compile(r"rss.{0,40}200\s*mb", re.IGNORECASE),
            re.compile(r"200\s*mb.{0,40}(?:per call|rss)", re.IGNORECASE),
        ),
    ),
    CriticalFact("version", (re.compile(r"v2\.3", re.IGNORECASE),)),
    CriticalFact(
        "file_line",
        (
            re.compile(r"src/batch_processor\.py.{0,25}(?:line\s*)?247", re.IGNORECASE),
            re.compile(r"(?:line\s*)?247.{0,25}src/batch_processor\.py", re.IGNORECASE),
        ),
    ),
    CriticalFact(
        "root_cause",
        (
            re.compile(r"numpy cache.{0,80}(?:retain|retaining|retains|hold|holding).{0,40}input array", re.IGNORECASE),
            re.compile(r"input array.{0,80}(?:retain|retaining|retains|held|holding).{0,40}numpy cache", re.IGNORECASE),
        ),
    ),
    CriticalFact(
        "remediation",
        (
            re.compile(r"cache\.clear\(\)", re.IGNORECASE),
            re.compile(r"clear the cache", re.IGNORECASE),
            re.compile(r"clear(?:ing)? (?:the )?numpy cache", re.IGNORECASE),
            re.compile(r"clear(?:ing)? (?:the )?cache after each batch", re.IGNORECASE),
        ),
    ),
]

PROOF_SESSION_TEMPLATE = [
    BUG_REPORT,
    (
        "Before we inspect any other evidence, restate the exact bug signature as six bullets. "
        "Preserve literals like the function, trigger, symptom, version, path, and line."
    ),
    "{noise:1}",
    "{noise:2}",
    "{noise:3}",
    "{noise:4}",
    "{noise:5}",
    "{noise:6}",
    "{noise:7}",
    "{noise:8}",
    "{noise:9}",
    "{noise:10}",
    "Ignore the unrelated telemetry. What category of memory bug still fits the original report best?",
    "I can only reproduce it after the warmup phase finishes. Does that narrow the theory?",
    "I confirmed the numpy cache retains the input array. State the root cause in one sentence.",
    "The proposed fix is cache.clear() after each batch. Does that solve the retention?",
    "Where exactly should the patch go in the codebase?",
    "Any thread-safety concern around the cleanup call?",
    "List the exact literals we must preserve in the ticket closure note.",
    (
        "Before I close this ticket, return exactly seven bullet lines with these labels:\n"
        "- function:\n"
        "- trigger:\n"
        "- symptom:\n"
        "- version:\n"
        "- file_line:\n"
        "- root_cause:\n"
        "- remediation:\n"
        "Preserve earlier literals exactly when known. If a field is unknown, write UNKNOWN."
    ),
]

MAX_CONTEXT_SESSION_TEMPLATE = [
    BUG_REPORT,
    (
        "Before we inspect anything else, restate the exact bug signature as six bullets. "
        "Preserve the literals verbatim."
    ),
    "Ignore unrelated queue chatter. What category of memory bug best fits the original report?",
    "I can only reproduce it after the warmup phase finishes. Does that narrow the theory?",
    "I confirmed the numpy cache retains the input array. State the root cause in one sentence.",
    "The proposed fix is cache.clear() after each batch. Does that solve the retention?",
    "Where exactly should the patch go in the codebase?",
    "Any thread-safety concern around the cleanup call?",
    "List the exact literals we must preserve in the ticket closure note.",
    "If I summarize this later, what symptom must not be dropped?",
    "What exact version boundary matters?",
    "What exact trigger condition matters?",
    "Reconfirm the root cause briefly.",
    "Reconfirm the remediation briefly.",
    "What exact file and line should I cite in the patch note?",
    "If I only keep one sentence about the bug, what should it be?",
    "The next message is a massive unrelated telemetry export. Ignore it when reasoning about the leak.",
    "{noise:1}",
    "After that telemetry export, what exact bug details are still the anchors?",
    (
        "Before I close this ticket, return exactly seven bullet lines with these labels:\n"
        "- function:\n"
        "- trigger:\n"
        "- symptom:\n"
        "- version:\n"
        "- file_line:\n"
        "- root_cause:\n"
        "- remediation:\n"
        "Preserve earlier literals exactly when known. If a field is unknown, write UNKNOWN."
    ),
]


def score_response(text: str) -> dict:
    found = [
        fact.name
        for fact in CRITICAL_FACTS
        if any(pattern.search(text) for pattern in fact.patterns)
    ]
    return {
        "found": found,
        "missed": [fact.name for fact in CRITICAL_FACTS if fact.name not in found],
        "score": len(found) / len(CRITICAL_FACTS),
    }


def build_noise_message(
    backend: MLXBackend,
    model: str,
    dump_id: int,
    target_tokens: int,
) -> str:
    header = (
        f"Unrelated retry-queue telemetry dump {dump_id}. The identifiers below come from a different "
        "service and should not replace the original leak report.\n"
    )
    header_tokens = backend.count_tokens([{"role": "user", "content": header}], model=model)

    sample_count = 200
    sample_body = "\n".join(_noise_line(dump_id, row) for row in range(sample_count))
    sample_tokens = backend.count_tokens(
        [{"role": "user", "content": header + sample_body}],
        model=model,
    )
    per_line_tokens = max(1.0, (sample_tokens - header_tokens) / sample_count)
    line_count = max(1, math.ceil((target_tokens - header_tokens) / per_line_tokens))

    while True:
        body = "\n".join(_noise_line(dump_id, row) for row in range(line_count))
        content = header + body
        current_tokens = backend.count_tokens([{"role": "user", "content": content}], model=model)
        if current_tokens >= target_tokens:
            return content
        shortfall = target_tokens - current_tokens
        line_count += max(1, math.ceil(shortfall / per_line_tokens))


def _noise_line(dump_id: int, row: int) -> str:
    return (
        f"[dump={dump_id} row={row}] svc=retry_queue fn=parse_retry_window() "
        f"path=services/retry_queue.py line={100 + (row % 700)} "
        f"build=v9.1.{row % 9} heap={64 + (row % 11)}MB fd={row % 64} "
        f"shard={row % 17} timeout_ms={80 + (row % 300)} jitter={row % 23} "
        f"status=backoff trace=retry_window_parser"
    )


def build_session(
    backend: MLXBackend,
    model: str,
    window_budget: int,
    overflow_ratio: float,
    profile: str,
) -> tuple[list[tuple[str, str]], dict]:
    if profile == "max":
        session_template = MAX_CONTEXT_SESSION_TEMPLATE
    else:
        session_template = PROOF_SESSION_TEMPLATE

    placeholder_messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    placeholder_messages.extend(
        {"role": "user", "content": "Unrelated retry-queue telemetry dump placeholder."}
        if item.startswith("{noise:")
        else {"role": "user", "content": item}
        for item in session_template
    )
    base_tokens = backend.count_tokens(placeholder_messages, model=model)
    target_tokens = max(base_tokens + 1, int(window_budget * overflow_ratio))
    noise_slots = sum(1 for item in session_template if item.startswith("{noise:"))
    noise_budget = max(2_000, target_tokens - base_tokens)
    per_noise_tokens = max(2_000, math.ceil(noise_budget / max(noise_slots, 1)))

    session = []
    for item in session_template:
        if item.startswith("{noise:"):
            dump_id = int(item.removeprefix("{noise:").removesuffix("}"))
            print(
                f"[session] building noise dump {dump_id}/{noise_slots} at ~{per_noise_tokens} tokens",
                flush=True,
            )
            content = build_noise_message(
                backend=backend,
                model=model,
                dump_id=dump_id,
                target_tokens=per_noise_tokens,
            )
        else:
            content = item
        session.append(("user", content))

    transcript_messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    transcript_messages.extend({"role": role, "content": content} for role, content in session)
    source_tokens = backend.count_tokens(transcript_messages, model=model)
    print(
        f"[session] built 20-turn transcript with {source_tokens} source prompt tokens",
        flush=True,
    )
    return session, {
        "profile": profile,
        "source_prompt_tokens": source_tokens,
        "target_source_tokens": target_tokens,
        "source_overflow_ratio": round(source_tokens / window_budget, 3),
        "noise_turns": noise_slots,
        "noise_tokens_per_turn": per_noise_tokens,
    }


def run_session(
    backend: MLXBackend,
    model: str,
    strategy: str,
    session: list[tuple[str, str]],
    window_budget: int,
    response_budget: int,
) -> dict:
    cgc = ContextGCBarrier(
        backend=backend,
        window_budget=window_budget,
        response_budget=response_budget,
        strategy=strategy,
    )
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    final_response = ""
    total_turns = len(session)
    for turn_index, (role, content) in enumerate(session, start=1):
        print(f"[{strategy}] turn {turn_index}/{total_turns}", flush=True)
        messages.append({"role": role, "content": content})
        max_tokens = response_budget if turn_index == total_turns else min(response_budget, INTERMEDIATE_MAX_TOKENS)
        response = cgc.chat(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.0,
        )
        assistant_message = response.choices[0].message.content
        messages.append({"role": "assistant", "content": assistant_message})
        final_response = assistant_message

    state = cgc.context_state()
    score = score_response(final_response)
    protected_bug_report_selected = any(
        item["index"] == 1
        for item in state["selected_messages"]
    )
    return {
        "strategy": strategy,
        "final_response": final_response,
        "score": score,
        "state": state,
        "protected_bug_report_selected": protected_bug_report_selected,
    }


def write_results(output_dir: Path, results: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "latest_run.json"
    md_path = output_dir / "latest_run.md"

    json_path.write_text(json.dumps(results, indent=2))

    recency = results["runs"]["recency"]
    barrier = results["runs"]["barrier"]
    conclusion = results["conclusion"]

    lines = [
        "# ContextGC Controlled-Budget Benchmark",
        "",
        f"- Model: `{results['model']}`",
        f"- Profile: `{results['session']['profile']}`",
        f"- Window budget: `{results['window_budget']}`",
        f"- Response budget: `{results['response_budget']}`",
        f"- Source prompt tokens before assistant replies: `{results['session']['source_prompt_tokens']}`",
        f"- Source/window ratio: `{results['session']['source_overflow_ratio']}`",
        f"- Noise turns: `{results['session']['noise_turns']}` @ `{results['session']['noise_tokens_per_turn']}` tokens each",
        f"- Conclusion: `{conclusion}`",
        "",
        "## Score Summary",
        "",
        "| Strategy | Recall | Bug Report Selected | Final Prompt Tokens |",
        "|---|---:|---:|---:|",
        f"| Recency | {recency['score']['score']:.0%} | {recency['protected_bug_report_selected']} | {recency['state']['prompt_tokens']} |",
        f"| Barrier | {barrier['score']['score']:.0%} | {barrier['protected_bug_report_selected']} | {barrier['state']['prompt_tokens']} |",
        "",
        "## Recency Final Response",
        "",
        recency["final_response"],
        "",
        "## Barrier Final Response",
        "",
        barrier["final_response"],
        "",
    ]

    md_path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the controlled-budget ContextGC benchmark.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--profile", choices=("proof", "max"), default="proof")
    parser.add_argument("--window-budget", type=int)
    parser.add_argument("--response-budget", type=int, default=DEFAULT_RESPONSE_BUDGET)
    parser.add_argument("--overflow-ratio", type=float, default=DEFAULT_OVERFLOW_RATIO)
    parser.add_argument("--output-dir", default=str(Path(__file__).with_name("results")))
    args = parser.parse_args()

    backend = MLXBackend(default_model=args.model)
    window_budget = args.window_budget or backend.model_max_length(args.model)
    session, session_stats = build_session(
        backend=backend,
        model=args.model,
        window_budget=window_budget,
        overflow_ratio=args.overflow_ratio,
        profile=args.profile,
    )
    recency = run_session(
        backend=backend,
        model=args.model,
        strategy="recency",
        session=session,
        window_budget=window_budget,
        response_budget=args.response_budget,
    )
    barrier = run_session(
        backend=backend,
        model=args.model,
        strategy="barrier",
        session=session,
        window_budget=window_budget,
        response_budget=args.response_budget,
    )

    if barrier["score"]["score"] == 1.0 and barrier["score"]["score"] > recency["score"]["score"]:
        conclusion = "success"
    elif barrier["score"]["score"] > recency["score"]["score"]:
        conclusion = "partial"
    else:
        conclusion = "inconclusive"

    results = {
        "model": args.model,
        "window_budget": window_budget,
        "response_budget": args.response_budget,
        "session": session_stats,
        "conclusion": conclusion,
        "runs": {
            "recency": recency,
            "barrier": barrier,
        },
    }
    write_results(Path(args.output_dir), results)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
