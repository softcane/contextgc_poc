from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from .backend import ContextBackend

SUMMARY_GENERATION_MAX_TOKENS = 96


@dataclass
class SummaryState:
    summary_message: Optional[dict[str, str]]
    summarized_through_index: int
    tail_messages: list[dict[str, Any]]


def ensure_summary_fit(
    *,
    backend: ContextBackend,
    model_name: str,
    system_prompt: str,
    history: list[dict[str, Any]],
    window_budget: int,
    response_budget: int,
    summary_trigger: int,
    summary_cap: int,
    prior_state: Optional[SummaryState],
    allow_generation: bool = True,
) -> SummaryState:
    summarized_through_index = prior_state.summarized_through_index if prior_state else 0
    summary_message = prior_state.summary_message if prior_state else None
    tail_messages = [message for message in history if message["index"] > summarized_through_index]
    prompt_messages = compose_summary_prompt(system_prompt, summary_message, tail_messages)
    prompt_tokens = backend.count_tokens(prompt_messages, model=model_name)

    while prompt_tokens > summary_trigger and len(tail_messages) > 4:
        summarized_through_index, summary_message, tail_messages, prompt_messages, prompt_tokens = fold_oldest_tail_block(
            backend=backend,
            model_name=model_name,
            history=history,
            summary_message=summary_message,
            tail_messages=tail_messages,
            summarized_through_index=summarized_through_index,
            summary_cap=summary_cap,
            system_prompt=system_prompt,
            allow_generation=allow_generation,
        )

    while prompt_tokens + response_budget > window_budget and len(tail_messages) > 2:
        summarized_through_index, summary_message, tail_messages, prompt_messages, prompt_tokens = fold_oldest_tail_block(
            backend=backend,
            model_name=model_name,
            history=history,
            summary_message=summary_message,
            tail_messages=tail_messages,
            summarized_through_index=summarized_through_index,
            summary_cap=summary_cap,
            system_prompt=system_prompt,
            allow_generation=allow_generation,
        )

    if prompt_tokens + response_budget > window_budget:
        trimmed_tail = list(tail_messages)
        while prompt_tokens + response_budget > window_budget and trimmed_tail:
            trimmed_tail.pop(0)
            prompt_messages = compose_summary_prompt(system_prompt, summary_message, trimmed_tail)
            prompt_tokens = backend.count_tokens(prompt_messages, model=model_name)
        tail_messages = trimmed_tail

    return SummaryState(
        summary_message=summary_message,
        summarized_through_index=summarized_through_index,
        tail_messages=tail_messages,
    )


def fold_oldest_tail_block(
    *,
    backend: ContextBackend,
    model_name: str,
    history: list[dict[str, Any]],
    summary_message: Optional[dict[str, str]],
    tail_messages: list[dict[str, Any]],
    summarized_through_index: int,
    summary_cap: int,
    system_prompt: str,
    allow_generation: bool = True,
) -> tuple[int, dict[str, str], list[dict[str, Any]], list[dict[str, str]], int]:
    block_size = min(4, max(1, len(tail_messages) - 2))
    block = tail_messages[:block_size]
    summarized_through_index = block[-1]["index"]
    summary_message = {
        "role": "assistant",
        "content": build_rolling_summary(
            backend=backend,
            model_name=model_name,
            existing_summary=summary_message["content"] if summary_message else "",
            messages=block,
            summary_cap=summary_cap,
            allow_generation=allow_generation,
        ),
    }
    tail_messages = [message for message in history if message["index"] > summarized_through_index]
    prompt_messages = compose_summary_prompt(system_prompt, summary_message, tail_messages)
    prompt_tokens = backend.count_tokens(prompt_messages, model=model_name)
    return summarized_through_index, summary_message, tail_messages, prompt_messages, prompt_tokens


def compose_summary_prompt(
    system_prompt: str,
    summary_message: Optional[dict[str, str]],
    tail_messages: list[dict[str, Any]],
) -> list[dict[str, str]]:
    prompt_messages = [{"role": "system", "content": system_prompt}]
    if summary_message is not None:
        prompt_messages.append(summary_message)
    prompt_messages.extend(
        {"role": message["role"], "content": message["content"]}
        for message in tail_messages
    )
    return prompt_messages


def build_rolling_summary(
    *,
    backend: ContextBackend,
    model_name: str,
    existing_summary: str,
    messages: list[dict[str, Any]],
    summary_cap: int,
    allow_generation: bool = True,
) -> str:
    if not allow_generation:
        return _heuristic_summary_from_messages(
            backend=backend,
            model_name=model_name,
            existing_summary=existing_summary,
            messages=messages,
            summary_cap=summary_cap,
        )

    block_lines = []
    for message in messages:
        content = " ".join(str(message["content"]).split())
        block_lines.append(f"{message['role'].upper()}: {content[:220]}")

    summary_instructions = [
        "Compress the conversation state into at most six short bullets.",
        "Preserve the active goal, resolved decisions, unresolved questions, and any concrete details that still look operationally important.",
        "Keep exact literals for identifiers, paths, dates, numeric thresholds, versions, form IDs, order IDs, SKUs, emails, or API routes when they may matter later.",
        "Drop obvious noise, repetition, and details that are clearly unrelated to the main task.",
        "Do not invent or normalize facts.",
    ]
    prompt_parts = [
        "Existing summary:",
        existing_summary or "None.",
        "",
        "New messages:",
        "\n".join(block_lines),
        "",
        "\n".join(summary_instructions),
    ]
    prompt = "\n".join(prompt_parts)
    response = backend.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "You compress conversation state into concise working-memory bullets."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=min(summary_cap, SUMMARY_GENERATION_MAX_TOKENS),
        temperature=0.0,
    )
    content = (response.choices[0].message.content or "").strip()
    if not content:
        return "Rolling summary:\n- Earlier context was compressed at a high level."

    lines = [line.strip() for line in content.splitlines() if line.strip()]
    if not any(line.startswith("-") for line in lines):
        lines = [f"- {line}" for line in lines[:4]]
    summary_lines = ["Rolling summary:"] + lines[:4]
    while summary_lines:
        candidate = "\n".join(summary_lines)
        tokens = backend.count_tokens([{"role": "assistant", "content": candidate}], model=model_name)
        if tokens <= summary_cap:
            return candidate
        summary_lines.pop()
    return "Rolling summary:\n- Earlier context was compressed at a high level."


def _heuristic_summary_from_messages(
    *,
    backend: ContextBackend,
    model_name: str,
    existing_summary: str,
    messages: list[dict[str, Any]],
    summary_cap: int,
) -> str:
    summary_lines = ["Rolling summary:"]
    if existing_summary:
        existing_lines = [line.strip() for line in existing_summary.splitlines()[1:] if line.strip()]
        summary_lines.extend(existing_lines[:2])
    for message in messages[:3]:
        content = " ".join(str(message["content"]).split())
        summary_lines.append(f"- {message['role']}: {content[:120]}")
    while summary_lines:
        candidate = "\n".join(summary_lines[:5])
        tokens = backend.count_tokens([{"role": "assistant", "content": candidate}], model=model_name)
        if tokens <= summary_cap:
            return candidate
        summary_lines.pop()
    return "Rolling summary:\n- Earlier context was compressed at a high level."
