from __future__ import annotations

from typing import Any

from contextgc_barrier import ContextGCBarrier
from contextgc_barrier.backend import ContextBackend

from .specs import FrozenReplayTask, ModelSpec, RunResult, StrategySpec, TaskInstance

INTERMEDIATE_MAX_TOKENS = 48


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
        strategy=strategy.internal_strategy or strategy.name,
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
        strategy=strategy.internal_strategy or strategy.name,
        citation_enabled=strategy.citation_enabled,
        extractor_mode=strategy.extractor_mode,
    )
    messages = [{"role": "system", "content": task.system_prompt}]

    for message in task.messages:
        prompt_message = {"role": message["role"], "content": message["content"]}
        if message["role"] != "assistant":
            messages.append(prompt_message)
            continue
        cgc.replay_turn(
            model=model.model_name,
            messages_before_reply=messages,
            reply_text=message["content"],
            max_tokens=min(response_budget, INTERMEDIATE_MAX_TOKENS),
        )
        messages.append(prompt_message)

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
            "summary_active": state.get("summary_active", False),
            "summarized_through_index": state.get("summarized_through_index", 0),
            "summary_tokens": state.get("summary_tokens", 0),
            "protected_exception_indexes": state.get("protected_exception_indexes", []),
            "barrier_summary": state["barrier_summary"],
        },
        final_prompt_anchor_overlap=float(task.metadata.get("final_prompt_anchor_overlap", 0.0)),
        final_prompt_distractor_overlap=float(task.metadata.get("final_prompt_distractor_overlap", 0.0)),
    )


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
