from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from contextgc_barrier import ContextGCBarrier
from contextgc_barrier.backend import ContextBackend

from .specs import FrozenReplayTask, ModelSpec, RunResult, StrategySpec, TaskInstance

INTERMEDIATE_MAX_TOKENS = 48
SUMMARY_GENERATION_MAX_TOKENS = 96


def run_internal_strategy(
    *,
    backend: ContextBackend,
    task: TaskInstance,
    model: ModelSpec,
    strategy: StrategySpec,
    window_budget: int,
    response_budget: int,
) -> RunResult:
    cgc = ContextGCBarrier(
        backend=backend,
        window_budget=window_budget,
        response_budget=response_budget,
        strategy=strategy.internal_strategy or "barrier",
        citation_enabled=strategy.citation_enabled,
        extractor_mode=strategy.extractor_mode,
    )
    messages = [{"role": "system", "content": task.system_prompt}]
    final_response = ""

    for turn in task.turns:
        messages.append({"role": turn["role"], "content": turn["content"]})
        max_tokens = response_budget if turn is task.turns[-1] else min(response_budget, INTERMEDIATE_MAX_TOKENS)
        response = cgc.chat(
            model=model.model_name,
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.0,
        )
        assistant_text = response.choices[0].message.content or ""
        messages.append({"role": "assistant", "content": assistant_text})
        final_response = assistant_text

    return _build_internal_run_result(
        task=task,
        model=model,
        strategy=strategy,
        window_budget=window_budget,
        response_budget=response_budget,
        final_response=final_response,
        state=cgc.context_state(),
    )


def run_internal_replay_strategy(
    *,
    backend: ContextBackend,
    task: FrozenReplayTask,
    model: ModelSpec,
    strategy: StrategySpec,
    window_budget: int,
    response_budget: int,
) -> RunResult:
    cgc = ContextGCBarrier(
        backend=backend,
        window_budget=window_budget,
        response_budget=response_budget,
        strategy=strategy.internal_strategy or "barrier",
        citation_enabled=strategy.citation_enabled,
        extractor_mode=strategy.extractor_mode,
    )
    messages = [{"role": "system", "content": task.system_prompt}]

    for message in task.messages:
        if message["role"] != "assistant":
            messages.append({"role": message["role"], "content": message["content"]})
            continue
        cgc.replay_turn(
            model=model.model_name,
            messages_before_reply=messages,
            reply_text=message["content"],
            max_tokens=min(response_budget, INTERMEDIATE_MAX_TOKENS),
        )
        messages.append({"role": "assistant", "content": message["content"]})

    response = cgc.chat(
        model=model.model_name,
        messages=messages,
        max_tokens=response_budget,
        temperature=0.0,
    )
    final_response = response.choices[0].message.content or ""
    return _build_internal_run_result(
        task=task,
        model=model,
        strategy=strategy,
        window_budget=window_budget,
        response_budget=response_budget,
        final_response=final_response,
        state=cgc.context_state(),
    )


def _build_internal_run_result(
    *,
    task: TaskInstance | FrozenReplayTask,
    model: ModelSpec,
    strategy: StrategySpec,
    window_budget: int,
    response_budget: int,
    final_response: str,
    state: dict[str, Any],
) -> RunResult:
    selected_indexes = [item["index"] for item in state["selected_messages"]]
    protected_selected_indexes = [
        item["index"] for item in state["selected_messages"]
        if item.get("protected")
    ]
    cited_selected_indexes = [
        item["index"] for item in state["selected_messages"]
        if item.get("citations", 0) > 0
    ]
    protected_message_indexes = state.get("protected_message_indexes", [])
    cited_message_indexes = state.get("cited_message_indexes", [])
    score = task.score_response(final_response)
    turn_count = len(task.turns) if isinstance(task, TaskInstance) else int(task.metadata.get("turn_count", 0))
    return RunResult(
        task=task.name,
        model=model.alias,
        model_name=model.model_name,
        provider=model.provider,
        strategy=strategy.name,
        window_budget=window_budget,
        seed=int(task.metadata.get("seed", 0)),
        score=score["score"],
        secondary_score=score["secondary_score"],
        found=score["found"],
        wrong=score["wrong"],
        missing=score["missing"],
        missed=score["missed"],
        fact_results=score["fact_results"],
        contamination=score["contamination"],
        contamination_count=score["contamination_count"],
        scorer_agreement=score["scorer_agreement"],
        final_response=final_response,
        prompt_tokens=state["prompt_tokens"],
        usable_prompt_budget=window_budget - response_budget,
        selected_indexes=selected_indexes,
        protected_selected_indexes=protected_selected_indexes,
        cited_selected_indexes=cited_selected_indexes,
        protected_message_indexes=protected_message_indexes,
        cited_message_indexes=cited_message_indexes,
        retained_anchor=_all_present(task.metadata.get("anchor_indexes", [task.metadata.get("anchor_index", 1)]), selected_indexes),
        retained_distractor=_any_present(task.metadata.get("distractor_indexes", []), selected_indexes),
        anchor_protected=_any_present(task.metadata.get("anchor_indexes", [task.metadata.get("anchor_index", 1)]), protected_message_indexes),
        turn_count=turn_count,
        session_metadata=task.metadata,
        strategy_metadata={
            "citation_enabled": strategy.citation_enabled,
            "extractor_mode": strategy.extractor_mode,
            "barrier_summary": state["barrier_summary"],
        },
        final_prompt_anchor_overlap=float(task.metadata.get("final_prompt_anchor_overlap", 0.0)),
        final_prompt_distractor_overlap=float(task.metadata.get("final_prompt_distractor_overlap", 0.0)),
    )


def run_summary80_strategy(
    *,
    backend: ContextBackend,
    task: TaskInstance,
    model: ModelSpec,
    strategy: StrategySpec,
    window_budget: int,
    response_budget: int,
) -> RunResult:
    usable_prompt_budget = window_budget - response_budget
    summary_trigger = int(usable_prompt_budget * 0.80)
    summary_cap = max(32, int(usable_prompt_budget * 0.15))
    history: list[dict[str, Any]] = []
    final_response = ""
    final_prompt_tokens = 0
    final_selected_indexes: list[int] = []
    next_index = 1
    summary_state: Optional[_SummaryState] = None

    for turn in task.turns:
        history.append({"index": next_index, "role": turn["role"], "content": turn["content"]})
        next_index += 1
        summary_state = _ensure_summary_fit(
            backend=backend,
            model_name=model.model_name,
            system_prompt=task.system_prompt,
            history=history,
            window_budget=window_budget,
            response_budget=response_budget,
            summary_trigger=summary_trigger,
            summary_cap=summary_cap,
            prior_state=summary_state,
        )
        prompt_messages = [{"role": "system", "content": task.system_prompt}]
        if summary_state.summary_message is not None:
            prompt_messages.append(summary_state.summary_message)
        prompt_messages.extend(
            {"role": message["role"], "content": message["content"]}
            for message in summary_state.tail_messages
        )
        final_selected_indexes = [message["index"] for message in summary_state.tail_messages]
        final_prompt_tokens = backend.count_tokens(prompt_messages, model=model.model_name)

        response = backend.create(
            model=model.model_name,
            messages=prompt_messages,
            max_tokens=response_budget if turn is task.turns[-1] else min(response_budget, INTERMEDIATE_MAX_TOKENS),
            temperature=0.0,
        )
        assistant_text = response.choices[0].message.content or ""
        history.append({"index": next_index, "role": "assistant", "content": assistant_text})
        next_index += 1
        final_response = assistant_text

    return _build_summary_run_result(
        task=task,
        model=model,
        strategy=strategy,
        window_budget=window_budget,
        response_budget=response_budget,
        final_response=final_response,
        final_prompt_tokens=final_prompt_tokens,
        final_selected_indexes=final_selected_indexes,
        summary_state=summary_state,
    )


def run_summary80_replay_strategy(
    *,
    backend: ContextBackend,
    task: FrozenReplayTask,
    model: ModelSpec,
    strategy: StrategySpec,
    window_budget: int,
    response_budget: int,
) -> RunResult:
    usable_prompt_budget = window_budget - response_budget
    summary_trigger = int(usable_prompt_budget * 0.80)
    summary_cap = max(32, int(usable_prompt_budget * 0.15))
    history: list[dict[str, Any]] = []
    final_prompt_tokens = 0
    final_selected_indexes: list[int] = []
    next_index = 1
    summary_state: Optional[_SummaryState] = None

    for message in task.messages:
        history.append({"index": next_index, "role": message["role"], "content": message["content"]})
        next_index += 1
        if message["role"] == "assistant":
            continue
        summary_state = _ensure_summary_fit(
            backend=backend,
            model_name=model.model_name,
            system_prompt=task.system_prompt,
            history=history,
            window_budget=window_budget,
            response_budget=response_budget,
            summary_trigger=summary_trigger,
            summary_cap=summary_cap,
            prior_state=summary_state,
        )
    prompt_messages = [{"role": "system", "content": task.system_prompt}]
    if summary_state.summary_message is not None:
        prompt_messages.append(summary_state.summary_message)
    prompt_messages.extend(
        {"role": message["role"], "content": message["content"]}
        for message in summary_state.tail_messages
    )
    final_selected_indexes = [message["index"] for message in summary_state.tail_messages]
    final_prompt_tokens = backend.count_tokens(prompt_messages, model=model.model_name)
    response = backend.create(
        model=model.model_name,
        messages=prompt_messages,
        max_tokens=response_budget,
        temperature=0.0,
    )
    final_response = response.choices[0].message.content or ""
    return _build_summary_run_result(
        task=task,
        model=model,
        strategy=strategy,
        window_budget=window_budget,
        response_budget=response_budget,
        final_response=final_response,
        final_prompt_tokens=final_prompt_tokens,
        final_selected_indexes=final_selected_indexes,
        summary_state=summary_state,
    )


def run_full_history_replay_strategy(
    *,
    backend: ContextBackend,
    task: FrozenReplayTask,
    model: ModelSpec,
    strategy: StrategySpec,
    window_budget: int,
    response_budget: int,
) -> RunResult:
    prompt_messages = [{"role": "system", "content": task.system_prompt}, *task.messages]
    prompt_tokens = backend.count_tokens(prompt_messages, model=model.model_name)
    response = backend.create(
        model=model.model_name,
        messages=prompt_messages,
        max_tokens=response_budget,
        temperature=0.0,
    )
    final_response = response.choices[0].message.content or ""
    score = task.score_response(final_response)
    return RunResult(
        task=task.name,
        model=model.alias,
        model_name=model.model_name,
        provider=model.provider,
        strategy=strategy.name,
        window_budget=window_budget,
        seed=int(task.metadata.get("seed", 0)),
        score=score["score"],
        secondary_score=score["secondary_score"],
        found=score["found"],
        wrong=score["wrong"],
        missing=score["missing"],
        missed=score["missed"],
        fact_results=score["fact_results"],
        contamination=score["contamination"],
        contamination_count=score["contamination_count"],
        scorer_agreement=score["scorer_agreement"],
        final_response=final_response,
        prompt_tokens=prompt_tokens,
        usable_prompt_budget=window_budget - response_budget,
        selected_indexes=list(range(len(prompt_messages))),
        protected_selected_indexes=[],
        cited_selected_indexes=[],
        protected_message_indexes=[],
        cited_message_indexes=[],
        retained_anchor=_all_present(task.metadata.get("anchor_indexes", [task.metadata.get("anchor_index", 1)]), list(range(len(prompt_messages)))),
        retained_distractor=_any_present(task.metadata.get("distractor_indexes", []), list(range(len(prompt_messages)))),
        anchor_protected=False,
        turn_count=int(task.metadata.get("turn_count", 0)),
        session_metadata=task.metadata,
        strategy_metadata={
            "replay_mode": "full_history",
        },
        final_prompt_anchor_overlap=float(task.metadata.get("final_prompt_anchor_overlap", 0.0)),
        final_prompt_distractor_overlap=float(task.metadata.get("final_prompt_distractor_overlap", 0.0)),
    )


def _build_summary_run_result(
    *,
    task: TaskInstance | FrozenReplayTask,
    model: ModelSpec,
    strategy: StrategySpec,
    window_budget: int,
    response_budget: int,
    final_response: str,
    final_prompt_tokens: int,
    final_selected_indexes: list[int],
    summary_state: Optional["_SummaryState"],
) -> RunResult:
    score = task.score_response(final_response)
    turn_count = len(task.turns) if isinstance(task, TaskInstance) else int(task.metadata.get("turn_count", 0))
    return RunResult(
        task=task.name,
        model=model.alias,
        model_name=model.model_name,
        provider=model.provider,
        strategy=strategy.name,
        window_budget=window_budget,
        seed=int(task.metadata.get("seed", 0)),
        score=score["score"],
        secondary_score=score["secondary_score"],
        found=score["found"],
        wrong=score["wrong"],
        missing=score["missing"],
        missed=score["missed"],
        fact_results=score["fact_results"],
        contamination=score["contamination"],
        contamination_count=score["contamination_count"],
        scorer_agreement=score["scorer_agreement"],
        final_response=final_response,
        prompt_tokens=final_prompt_tokens,
        usable_prompt_budget=window_budget - response_budget,
        selected_indexes=final_selected_indexes,
        protected_selected_indexes=[],
        cited_selected_indexes=[],
        protected_message_indexes=[],
        cited_message_indexes=[],
        retained_anchor=_all_present(task.metadata.get("anchor_indexes", [task.metadata.get("anchor_index", 1)]), final_selected_indexes),
        retained_distractor=_any_present(task.metadata.get("distractor_indexes", []), final_selected_indexes),
        anchor_protected=False,
        turn_count=turn_count,
        session_metadata=task.metadata,
        strategy_metadata={
            "summary_trigger": int((window_budget - response_budget) * 0.80),
            "summary_cap": max(32, int((window_budget - response_budget) * 0.15)),
            "summarized_through_index": summary_state.summarized_through_index if summary_state else 0,
        },
        final_prompt_anchor_overlap=float(task.metadata.get("final_prompt_anchor_overlap", 0.0)),
        final_prompt_distractor_overlap=float(task.metadata.get("final_prompt_distractor_overlap", 0.0)),
    )


@dataclass
class _SummaryState:
    summary_message: Optional[dict[str, str]]
    summarized_through_index: int
    tail_messages: list[dict[str, Any]]


def _ensure_summary_fit(
    *,
    backend: ContextBackend,
    model_name: str,
    system_prompt: str,
    history: list[dict[str, Any]],
    window_budget: int,
    response_budget: int,
    summary_trigger: int,
    summary_cap: int,
    prior_state: Optional[_SummaryState],
) -> _SummaryState:
    summarized_through_index = prior_state.summarized_through_index if prior_state else 0
    summary_message = prior_state.summary_message if prior_state else None
    tail_messages = [message for message in history if message["index"] > summarized_through_index]
    prompt_messages = _compose_summary_prompt(system_prompt, summary_message, tail_messages)
    prompt_tokens = backend.count_tokens(prompt_messages, model=model_name)

    while prompt_tokens > summary_trigger and len(tail_messages) > 4:
        summarized_through_index, summary_message, tail_messages, prompt_messages, prompt_tokens = _fold_oldest_tail_block(
            backend=backend,
            model_name=model_name,
            history=history,
            summary_message=summary_message,
            tail_messages=tail_messages,
            summarized_through_index=summarized_through_index,
            summary_cap=summary_cap,
            system_prompt=system_prompt,
        )

    while prompt_tokens + response_budget > window_budget and len(tail_messages) > 2:
        summarized_through_index, summary_message, tail_messages, prompt_messages, prompt_tokens = _fold_oldest_tail_block(
            backend=backend,
            model_name=model_name,
            history=history,
            summary_message=summary_message,
            tail_messages=tail_messages,
            summarized_through_index=summarized_through_index,
            summary_cap=summary_cap,
            system_prompt=system_prompt,
        )

    if prompt_tokens + response_budget > window_budget:
        trimmed_tail = list(tail_messages)
        while prompt_tokens + response_budget > window_budget and trimmed_tail:
            trimmed_tail.pop(0)
            prompt_messages = _compose_summary_prompt(system_prompt, summary_message, trimmed_tail)
            prompt_tokens = backend.count_tokens(prompt_messages, model=model_name)
        tail_messages = trimmed_tail

    return _SummaryState(
        summary_message=summary_message,
        summarized_through_index=summarized_through_index,
        tail_messages=tail_messages,
    )


def _fold_oldest_tail_block(
    *,
    backend: ContextBackend,
    model_name: str,
    history: list[dict[str, Any]],
    summary_message: Optional[dict[str, str]],
    tail_messages: list[dict[str, Any]],
    summarized_through_index: int,
    summary_cap: int,
    system_prompt: str,
) -> tuple[int, dict[str, str], list[dict[str, Any]], list[dict[str, str]], int]:
    block_size = min(4, max(1, len(tail_messages) - 2))
    block = tail_messages[:block_size]
    summarized_through_index = block[-1]["index"]
    summary_message = {
        "role": "assistant",
        "content": _build_rolling_summary(
            backend=backend,
            model_name=model_name,
            existing_summary=summary_message["content"] if summary_message else "",
            messages=block,
            summary_cap=summary_cap,
        ),
    }
    tail_messages = [message for message in history if message["index"] > summarized_through_index]
    prompt_messages = _compose_summary_prompt(system_prompt, summary_message, tail_messages)
    prompt_tokens = backend.count_tokens(prompt_messages, model=model_name)
    return summarized_through_index, summary_message, tail_messages, prompt_messages, prompt_tokens


def _compose_summary_prompt(
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


def _build_rolling_summary(
    *,
    backend: ContextBackend,
    model_name: str,
    existing_summary: str,
    messages: list[dict[str, Any]],
    summary_cap: int,
) -> str:
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


def _all_present(indexes: list[int] | tuple[int, ...], selected_indexes: list[int]) -> bool:
    if not indexes:
        return False
    selected = set(selected_indexes)
    return all(index in selected for index in indexes)


def _any_present(indexes: list[int] | tuple[int, ...], selected_indexes: list[int]) -> bool:
    if not indexes:
        return False
    selected = set(selected_indexes)
    return any(index in selected for index in indexes)
