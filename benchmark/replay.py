from __future__ import annotations

import csv
import json
from dataclasses import replace
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Iterable

from .runner import build_default_model_specs
from .specs import FactSpec, FrozenReplayTask, ModelSpec, RunResult, StrategySpec, choose_audit_sample
from .stats import exact_sign_test_p_value, paired_bootstrap_ci, win_tie_loss
from .strategies import (
    INTERMEDIATE_MAX_TOKENS,
    run_full_history_replay_strategy,
    run_internal_replay_strategy,
    run_summary80_replay_strategy,
)
from .tasks import (
    DEBUGGING_REPLAY_TARGET_TOKENS,
    DEBUGGING_REPLAY_TOKEN_TOLERANCE,
    build_debugging_replay_seed_task,
)

DEBUGGING_REPLAY_PROFILE = "debugging_replay_v1"
DEBUGGING_REPLAY_SMOKE_PROFILE = "debugging_replay_smoke"
DEBUGGING_REPLAY_TASK_NAME = "debugging_replay_v1"
DEBUGGING_REPLAY_TEMPLATES = ("incident_closeout", "root_cause_recap")
DEBUGGING_REPLAY_WINDOWS = (3072, 4096)
DEBUGGING_REPLAY_ORACLE_WINDOW = 16_384
DEBUGGING_REPLAY_SMOKE_WINDOWS = (3072,)
DEBUGGING_REPLAY_SMOKE_ACCEPTED_PER_TEMPLATE = 1
DEBUGGING_REPLAY_SMOKE_TARGET_TOKENS = DEBUGGING_REPLAY_TARGET_TOKENS
DEBUGGING_REPLAY_SMOKE_TOKEN_TOLERANCE = 500
DEBUGGING_REPLAY_SMOKE_MAX_SOURCE_SEED = 20
FROZEN_TRANSCRIPT_CALIBRATION_ATTEMPTS = 4


def build_debugging_replay_strategy_specs() -> list[StrategySpec]:
    return [
        StrategySpec(
            name="recency",
            label="Recency",
            internal_strategy="recency",
            citation_enabled=False,
        ),
        StrategySpec(
            name="summary80",
            label="Summary @80%",
        ),
        StrategySpec(
            name="score_only",
            label="Score Only",
            internal_strategy="barrier",
            citation_enabled=False,
        ),
        StrategySpec(
            name="barrier",
            label="Barrier",
            internal_strategy="barrier",
        ),
        StrategySpec(
            name="full_history",
            label="Full History",
            citation_enabled=False,
        ),
    ]


def run_debugging_replay_v1(
    *,
    output_dir: Path,
    primary_local_model: str,
    response_budget: int = 128,
    accepted_per_template: int = 10,
    target_total_tokens: int = DEBUGGING_REPLAY_TARGET_TOKENS,
    target_token_tolerance: int = DEBUGGING_REPLAY_TOKEN_TOLERANCE,
    constrained_window_budgets: tuple[int, ...] = DEBUGGING_REPLAY_WINDOWS,
    oracle_window: int = DEBUGGING_REPLAY_ORACLE_WINDOW,
    max_source_seed: int = 500,
    profile_name: str = DEBUGGING_REPLAY_PROFILE,
    model_specs: list[ModelSpec] | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_log_path = output_dir / "progress.log"
    transcripts_path = output_dir / "transcripts.jsonl"
    progress_log_path.unlink(missing_ok=True)
    transcripts_path.unlink(missing_ok=True)
    model_spec = (model_specs or build_default_model_specs(primary_local_model=primary_local_model))[0]
    backend = model_spec.backend_factory()
    strategies = {strategy.name: strategy for strategy in build_debugging_replay_strategy_specs()}
    _log_progress(
        progress_log_path,
        (
            f"[start] profile={profile_name} model={model_spec.model_name} response_budget={response_budget} "
            f"accepted_per_template={accepted_per_template} target_total_tokens={target_total_tokens} "
            f"target_token_tolerance={target_token_tolerance} constrained_windows={','.join(str(item) for item in constrained_window_budgets)} "
            f"oracle_window={oracle_window} max_source_seed={max_source_seed}"
        ),
    )

    transcripts = generate_frozen_debugging_replay_corpus(
        backend=backend,
        model=model_spec,
        response_budget=response_budget,
        accepted_per_template=accepted_per_template,
        target_total_tokens=target_total_tokens,
        target_token_tolerance=target_token_tolerance,
        oracle_window=oracle_window,
        full_history_strategy=strategies["full_history"],
        max_source_seed=max_source_seed,
        transcripts_path=transcripts_path,
        progress_log_path=progress_log_path,
    )
    _write_transcripts(transcripts_path, transcripts)

    results: list[RunResult] = []
    for transcript in transcripts:
        transcript_id = str(transcript.metadata.get("transcript_id", "unknown"))
        for window_budget in constrained_window_budgets:
            for strategy_name in ("recency", "summary80", "score_only", "barrier"):
                _log_progress(
                    progress_log_path,
                    f"[eval] transcript={transcript_id} window={window_budget} strategy={strategy_name} status=running",
                )
                result = _run_replay_strategy(
                    backend=backend,
                    task=transcript,
                    model=model_spec,
                    strategy=strategies[strategy_name],
                    window_budget=window_budget,
                    response_budget=response_budget,
                )
                results.append(result)
                _log_progress(
                    progress_log_path,
                    (
                        f"[eval] transcript={transcript_id} window={window_budget} strategy={strategy_name} "
                        f"status=completed score={result.score:.3f} prompt_tokens={result.prompt_tokens}"
                    ),
                )
        _log_progress(
            progress_log_path,
            f"[eval] transcript={transcript_id} window={oracle_window} strategy=full_history status=running",
        )
        oracle_result = _run_replay_strategy(
            backend=backend,
            task=transcript,
            model=model_spec,
            strategy=strategies["full_history"],
            window_budget=oracle_window,
            response_budget=response_budget,
        )
        results.append(oracle_result)
        _log_progress(
            progress_log_path,
            (
                f"[eval] transcript={transcript_id} window={oracle_window} strategy=full_history "
                f"status=completed score={oracle_result.score:.3f} prompt_tokens={oracle_result.prompt_tokens}"
            ),
        )

    finalized = _finalize_results(results)
    aggregate_records = aggregate_debugging_replay_results(finalized)
    _write_result_artifacts(
        output_dir=output_dir,
        results=finalized,
        aggregate_records=aggregate_records,
        response_budget=response_budget,
        transcript_count=len(transcripts),
        constrained_window_budgets=constrained_window_budgets,
        oracle_window=oracle_window,
        profile_name=profile_name,
    )
    _log_progress(
        progress_log_path,
        f"[done] profile={profile_name} transcripts={len(transcripts)} runs={len(finalized)} artifacts_written=yes",
    )
    return {
        "profile": profile_name,
        "runs": [result.to_record() for result in finalized],
        "aggregates": aggregate_records,
        "transcripts": [transcript.metadata["transcript_id"] for transcript in transcripts],
    }


def run_debugging_replay_smoke(
    *,
    output_dir: Path,
    primary_local_model: str,
    response_budget: int = 128,
    target_total_tokens: int = DEBUGGING_REPLAY_SMOKE_TARGET_TOKENS,
    target_token_tolerance: int = DEBUGGING_REPLAY_SMOKE_TOKEN_TOLERANCE,
    max_source_seed: int = DEBUGGING_REPLAY_SMOKE_MAX_SOURCE_SEED,
    constrained_window_budgets: tuple[int, ...] = DEBUGGING_REPLAY_SMOKE_WINDOWS,
    model_specs: list[ModelSpec] | None = None,
) -> dict[str, Any]:
    return run_debugging_replay_v1(
        output_dir=output_dir,
        primary_local_model=primary_local_model,
        response_budget=response_budget,
        accepted_per_template=DEBUGGING_REPLAY_SMOKE_ACCEPTED_PER_TEMPLATE,
        target_total_tokens=target_total_tokens,
        target_token_tolerance=target_token_tolerance,
        constrained_window_budgets=constrained_window_budgets,
        oracle_window=DEBUGGING_REPLAY_ORACLE_WINDOW,
        max_source_seed=max_source_seed,
        profile_name=DEBUGGING_REPLAY_SMOKE_PROFILE,
        model_specs=model_specs,
    )


def generate_frozen_debugging_replay_corpus(
    *,
    backend,
    model: ModelSpec,
    response_budget: int,
    accepted_per_template: int,
    target_total_tokens: int,
    target_token_tolerance: int,
    oracle_window: int,
    full_history_strategy: StrategySpec,
    max_source_seed: int = 500,
    transcripts_path: Path | None = None,
    progress_log_path: Path | None = None,
) -> list[FrozenReplayTask]:
    transcripts: list[FrozenReplayTask] = []
    eval_seed = 1
    for template_name in DEBUGGING_REPLAY_TEMPLATES:
        _log_progress(
            progress_log_path,
            (
                f"[corpus] template={template_name} status=starting accepted_target={accepted_per_template} "
                f"target_total_tokens={target_total_tokens} oracle_window={oracle_window}"
            ),
        )
        accepted = 0
        source_seed = 1
        while accepted < accepted_per_template:
            if source_seed > max_source_seed:
                raise RuntimeError(
                    "Unable to build enough oracle-valid replay transcripts for "
                    f"template '{template_name}' within source seeds 1..{max_source_seed}. "
                    f"Accepted {accepted} of {accepted_per_template}."
                )
            _log_progress(
                progress_log_path,
                f"[corpus] template={template_name} source_seed={source_seed} status=materializing",
            )
            frozen, prompt_tokens, calibrated_target_total_tokens, calibration_attempts = _build_calibrated_frozen_transcript(
                backend=backend,
                model_name=model.model_name,
                source_seed=source_seed,
                template_name=template_name,
                target_total_tokens=target_total_tokens,
                response_budget=response_budget,
                eval_seed=eval_seed,
                target_token_tolerance=target_token_tolerance,
            )
            if abs(prompt_tokens - target_total_tokens) > target_token_tolerance:
                _log_progress(
                    progress_log_path,
                    (
                        f"[corpus] template={template_name} source_seed={source_seed} status=rejected "
                        f"reason=token_drift prompt_tokens={prompt_tokens} target_total_tokens={target_total_tokens} "
                        f"calibrated_source_target={calibrated_target_total_tokens} calibration_attempts={calibration_attempts}"
                    ),
                )
                source_seed += 1
                continue
            oracle_result = run_full_history_replay_strategy(
                backend=backend,
                task=frozen,
                model=model,
                strategy=full_history_strategy,
                window_budget=oracle_window,
                response_budget=response_budget,
            )
            if oracle_result.prompt_tokens + response_budget > oracle_window:
                _log_progress(
                    progress_log_path,
                    (
                        f"[corpus] template={template_name} source_seed={source_seed} status=rejected "
                        f"reason=oracle_window_overflow prompt_tokens={oracle_result.prompt_tokens} "
                        f"window_budget={oracle_window}"
                    ),
                )
                source_seed += 1
                continue
            if not _passes_oracle_gate(oracle_result):
                _log_progress(
                    progress_log_path,
                    (
                        f"[corpus] template={template_name} source_seed={source_seed} status=rejected "
                        f"reason=oracle_gate score={oracle_result.score:.3f} "
                        f"wrong={','.join(oracle_result.wrong) or '-'} missing={','.join(oracle_result.missing) or '-'} "
                        f"contamination={oracle_result.contamination_count}"
                    ),
                )
                source_seed += 1
                continue
            transcript_id = f"{template_name}-{accepted + 1:02d}"
            metadata = dict(frozen.metadata)
            metadata.update(
                {
                    "seed": eval_seed,
                    "source_seed": source_seed,
                    "transcript_id": transcript_id,
                    "transcript_prompt_tokens": prompt_tokens,
                    "calibrated_source_target_tokens": calibrated_target_total_tokens,
                    "calibration_attempts": calibration_attempts,
                    "oracle_score": oracle_result.score,
                    "oracle_found": oracle_result.found,
                    "oracle_missing": oracle_result.missing,
                    "oracle_contamination_count": oracle_result.contamination_count,
                }
            )
            accepted_transcript = replace(frozen, metadata=metadata)
            transcripts.append(accepted_transcript)
            if transcripts_path is not None:
                _append_transcript(transcripts_path, accepted_transcript)
            _log_progress(
                progress_log_path,
                (
                    f"[corpus] template={template_name} source_seed={source_seed} status=accepted "
                    f"transcript_id={transcript_id} prompt_tokens={prompt_tokens} oracle_score={oracle_result.score:.3f} "
                    f"calibrated_source_target={calibrated_target_total_tokens} calibration_attempts={calibration_attempts}"
                ),
            )
            accepted += 1
            eval_seed += 1
            source_seed += 1
        _log_progress(
            progress_log_path,
            f"[corpus] template={template_name} status=completed accepted={accepted}",
        )
    return transcripts


def _build_calibrated_frozen_transcript(
    *,
    backend,
    model_name: str,
    source_seed: int,
    template_name: str,
    target_total_tokens: int,
    target_token_tolerance: int,
    response_budget: int,
    eval_seed: int,
) -> tuple[FrozenReplayTask, int, int, int]:
    calibrated_target_total_tokens = target_total_tokens
    last_frozen: FrozenReplayTask | None = None
    last_prompt_tokens = 0
    last_attempt_target_total_tokens = calibrated_target_total_tokens

    for attempt in range(1, FROZEN_TRANSCRIPT_CALIBRATION_ATTEMPTS + 1):
        last_attempt_target_total_tokens = calibrated_target_total_tokens
        candidate = build_debugging_replay_seed_task(
            backend=backend,
            model=model_name,
            source_seed=source_seed,
            template_name=template_name,
            target_total_tokens=calibrated_target_total_tokens,
        )
        frozen = _freeze_task_transcript(
            backend=backend,
            model_name=model_name,
            candidate=candidate,
            template_name=template_name,
            response_budget=response_budget,
            eval_seed=eval_seed,
            source_seed=source_seed,
        )
        prompt_tokens = backend.count_tokens(
            [{"role": "system", "content": frozen.system_prompt}, *frozen.messages],
            model=model_name,
        )
        last_frozen = frozen
        last_prompt_tokens = prompt_tokens
        if abs(prompt_tokens - target_total_tokens) <= target_token_tolerance:
            return frozen, prompt_tokens, last_attempt_target_total_tokens, attempt

        calibrated_target_total_tokens = max(
            900,
            calibrated_target_total_tokens - (prompt_tokens - target_total_tokens),
        )

    if last_frozen is None:
        raise RuntimeError("Failed to build a frozen replay transcript")
    return last_frozen, last_prompt_tokens, last_attempt_target_total_tokens, FROZEN_TRANSCRIPT_CALIBRATION_ATTEMPTS


def _freeze_task_transcript(
    *,
    backend,
    model_name: str,
    candidate,
    template_name: str,
    response_budget: int,
    eval_seed: int,
    source_seed: int,
) -> FrozenReplayTask:
    messages: list[dict[str, str]] = []
    prompt_messages = [{"role": "system", "content": candidate.system_prompt}]
    for index, turn in enumerate(candidate.turns):
        message = {"role": turn["role"], "content": turn["content"]}
        messages.append(message)
        prompt_messages.append(message)
        if index == len(candidate.turns) - 1:
            break
        response = backend.create(
            model=model_name,
            messages=prompt_messages,
            max_tokens=min(response_budget, INTERMEDIATE_MAX_TOKENS),
            temperature=0.0,
        )
        assistant_text = response.choices[0].message.content or ""
        assistant_message = {"role": "assistant", "content": assistant_text}
        messages.append(assistant_message)
        prompt_messages.append(assistant_message)

    metadata = dict(candidate.metadata)
    metadata.update(
        {
            "seed": eval_seed,
            "source_seed": source_seed,
            "template_name": template_name,
            "turn_count": len(candidate.turns),
        }
    )
    return FrozenReplayTask(
        name=DEBUGGING_REPLAY_TASK_NAME,
        system_prompt=candidate.system_prompt,
        messages=tuple(messages),
        final_user_index=len(messages) - 1,
        facts=candidate.facts,
        metadata=metadata,
    )


def _passes_oracle_gate(result: RunResult) -> bool:
    if result.score < (6 / 7):
        return False
    required = {"root_cause", "file_line", "remediation"}
    statuses = {fact["name"]: fact["status"] for fact in result.fact_results}
    return all(statuses.get(name) == "correct" for name in required)


def _run_replay_strategy(
    *,
    backend,
    task: FrozenReplayTask,
    model: ModelSpec,
    strategy: StrategySpec,
    window_budget: int,
    response_budget: int,
) -> RunResult:
    if strategy.name == "summary80":
        return run_summary80_replay_strategy(
            backend=backend,
            task=task,
            model=model,
            strategy=strategy,
            window_budget=window_budget,
            response_budget=response_budget,
        )
    if strategy.name == "full_history":
        return run_full_history_replay_strategy(
            backend=backend,
            task=task,
            model=model,
            strategy=strategy,
            window_budget=window_budget,
            response_budget=response_budget,
        )
    return run_internal_replay_strategy(
        backend=backend,
        task=task,
        model=model,
        strategy=strategy,
        window_budget=window_budget,
        response_budget=response_budget,
    )


def aggregate_debugging_replay_results(results: list[RunResult]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, int], dict[str, list[RunResult]]] = {}
    for result in results:
        grouped.setdefault((result.task, result.model, result.window_budget), {}).setdefault(result.strategy, []).append(result)

    oracle_means: dict[tuple[str, str], float] = {}
    for (task_name, model_alias, window_budget), by_strategy in grouped.items():
        oracle_runs = by_strategy.get("full_history")
        if window_budget == DEBUGGING_REPLAY_ORACLE_WINDOW and oracle_runs:
            oracle_means[(task_name, model_alias)] = mean(run.score for run in oracle_runs)

    records: list[dict[str, Any]] = []
    for (task_name, model_alias, window_budget), by_strategy in sorted(grouped.items()):
        summary_runs = {run.seed: run.score for run in by_strategy.get("summary80", [])}
        for strategy_name, runs in sorted(by_strategy.items()):
            ordered_runs = sorted(runs, key=lambda item: item.seed)
            if strategy_name == "summary80" or window_budget == DEBUGGING_REPLAY_ORACLE_WINDOW:
                paired = [0.0 for _ in ordered_runs]
            else:
                paired = [
                    run.score - summary_runs[run.seed]
                    for run in ordered_runs
                    if run.seed in summary_runs
                ]
            ci_low, ci_high = paired_bootstrap_ci(paired, seed=window_budget + len(ordered_runs))
            wins, ties, losses = win_tie_loss(paired)
            oracle_gap = oracle_means.get((task_name, model_alias), 0.0) - (mean(run.score for run in ordered_runs) if ordered_runs else 0.0)
            records.append(
                {
                    "task": task_name,
                    "model": model_alias,
                    "model_name": ordered_runs[0].model_name if ordered_runs else model_alias,
                    "provider": ordered_runs[0].provider if ordered_runs else "",
                    "strategy": strategy_name,
                    "window_budget": window_budget,
                    "mean_score": mean(run.score for run in ordered_runs) if ordered_runs else 0.0,
                    "stddev_score": pstdev([run.score for run in ordered_runs]) if len(ordered_runs) > 1 else 0.0,
                    "mean_secondary_score": mean(run.secondary_score for run in ordered_runs) if ordered_runs else 0.0,
                    "n": len(ordered_runs),
                    "retained_anchor_rate": _rate(ordered_runs, lambda run: run.retained_anchor),
                    "retained_distractor_rate": _rate(ordered_runs, lambda run: run.retained_distractor),
                    "contamination_rate": _rate(ordered_runs, lambda run: run.contamination_count > 0),
                    "scorer_agreement_rate": _rate(ordered_runs, lambda run: run.scorer_agreement),
                    "root_cause_accuracy": _fact_accuracy(ordered_runs, "root_cause"),
                    "file_line_accuracy": _fact_accuracy(ordered_runs, "file_line"),
                    "remediation_accuracy": _fact_accuracy(ordered_runs, "remediation"),
                    "source_index_3_selected_rate": _rate(ordered_runs, lambda run: 3 in run.selected_indexes),
                    "oracle_gap": oracle_gap,
                    "delta_vs_summary80": mean(paired) if paired else 0.0,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "p_value": exact_sign_test_p_value(paired),
                    "wins": wins,
                    "ties": ties,
                    "losses": losses,
                }
            )
    return records


def _rate(runs: Iterable[RunResult], predicate) -> float:
    runs = list(runs)
    if not runs:
        return 0.0
    return sum(1 for run in runs if predicate(run)) / len(runs)


def _fact_accuracy(runs: Iterable[RunResult], fact_name: str) -> float:
    runs = list(runs)
    if not runs:
        return 0.0
    correct = 0
    for run in runs:
        statuses = {fact["name"]: fact["status"] for fact in run.fact_results}
        if statuses.get(fact_name) == "correct":
            correct += 1
    return correct / len(runs)


def _finalize_results(results: list[RunResult]) -> list[RunResult]:
    deduped = {result.key(): result for result in results}
    finalized = sorted(
        deduped.values(),
        key=lambda item: (item.task, item.model, item.window_budget, item.seed, item.strategy),
    )
    for index, result in enumerate(finalized, start=1):
        result.blind_id = f"audit-{index:04d}"
        result.barrier_extra_selected = False
        result.barrier_extra_selected_indexes = []
        result.barrier_rescue = False
        result.barrier_rescue_facts = []
        result.audit_required = False

    grouped: dict[tuple[str, str, int, int], dict[str, RunResult]] = {}
    for result in finalized:
        grouped.setdefault((result.task, result.model, result.window_budget, result.seed), {})[result.strategy] = result
    for runs in grouped.values():
        score_only = runs.get("score_only")
        barrier = runs.get("barrier")
        if score_only is None or barrier is None:
            continue
        barrier_only_indexes = sorted(set(barrier.selected_indexes) - set(score_only.selected_indexes))
        barrier.barrier_extra_selected_indexes = barrier_only_indexes
        barrier.barrier_extra_selected = bool(barrier_only_indexes)

        score_only_facts = {fact["name"]: fact["status"] for fact in score_only.fact_results}
        rescue_facts: list[str] = []
        for fact in barrier.fact_results:
            if fact["status"] != "correct":
                continue
            if score_only_facts.get(fact["name"]) == "correct":
                continue
            source_indexes = set(fact.get("source_message_indexes", []))
            if not source_indexes:
                continue
            if not (source_indexes & set(barrier.protected_message_indexes)):
                continue
            if source_indexes <= set(score_only.selected_indexes):
                continue
            rescue_facts.append(fact["name"])
        barrier.barrier_rescue_facts = sorted(rescue_facts)
        barrier.barrier_rescue = bool(rescue_facts)

    for result in choose_audit_sample(finalized, seed=0):
        result.audit_required = True
    return finalized


def _write_result_artifacts(
    *,
    output_dir: Path,
    results: list[RunResult],
    aggregate_records: list[dict[str, Any]],
    response_budget: int,
    transcript_count: int,
    constrained_window_budgets: tuple[int, ...],
    oracle_window: int,
    profile_name: str,
) -> None:
    run_path = output_dir / "runs.jsonl"
    aggregate_json_path = output_dir / "aggregate.json"
    aggregate_csv_path = output_dir / "aggregate.csv"
    summary_path = output_dir / "summary.md"
    audit_queue_path = output_dir / "audit_queue.jsonl"

    with run_path.open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps(result.to_record()) + "\n")

    aggregate_json_path.write_text(json.dumps(aggregate_records, indent=2))
    with aggregate_csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(aggregate_records[0].keys()))
        writer.writeheader()
        writer.writerows(aggregate_records)

    with audit_queue_path.open("w", encoding="utf-8") as handle:
        for result in results:
            if not result.audit_required:
                continue
            handle.write(json.dumps(_audit_queue_record(result)) + "\n")

    summary_lines = [
        "# ContextGC Debugging Replay Benchmark",
        "",
        f"- Profile: `{profile_name}`",
        f"- Runs: `{len(results)}`",
        f"- Transcripts: `{transcript_count}`",
        f"- Constrained windows: `{','.join(str(item) for item in constrained_window_budgets)}`",
        f"- Oracle window: `{oracle_window}`",
        f"- Response budget: `{response_budget}`",
        f"- Audit queue size: `{sum(1 for result in results if result.audit_required)}`",
        "",
        "## Aggregate Summary",
        "",
        "| Window | Strategy | Mean | Secondary | Root Cause | File Line | Remediation | Anchor | SrcIdx3 | Distractor | Contam | Agree | Oracle Gap | Delta vs Summary80 | 95% CI | p-value |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|",
    ]
    for aggregate in aggregate_records:
        summary_lines.append(
            f"| {aggregate['window_budget']} | {aggregate['strategy']} | {aggregate['mean_score']:.3f} | "
            f"{aggregate['mean_secondary_score']:.3f} | {aggregate['root_cause_accuracy']:.0%} | "
            f"{aggregate['file_line_accuracy']:.0%} | {aggregate['remediation_accuracy']:.0%} | "
            f"{aggregate['retained_anchor_rate']:.0%} | {aggregate['source_index_3_selected_rate']:.0%} | "
            f"{aggregate['retained_distractor_rate']:.0%} | {aggregate['contamination_rate']:.0%} | "
            f"{aggregate['scorer_agreement_rate']:.0%} | {aggregate['oracle_gap']:.3f} | "
            f"{aggregate['delta_vs_summary80']:.3f} | [{aggregate['ci_low']:.3f}, {aggregate['ci_high']:.3f}] | {aggregate['p_value']:.3f} |"
        )

    summary_lines.extend(["", "## Barrier vs Summary80", ""])
    for window_budget in constrained_window_budgets:
        barrier = next(
            aggregate for aggregate in aggregate_records
            if aggregate["window_budget"] == window_budget and aggregate["strategy"] == "barrier"
        )
        summary_lines.append(
            f"- `{window_budget}`: mean `{barrier['mean_score']:.3f}`, delta vs summary80 `{barrier['delta_vs_summary80']:.3f}`, "
            f"root-cause accuracy `{barrier['root_cause_accuracy']:.0%}`, contamination `{barrier['contamination_rate']:.0%}`, "
            f"agreement `{barrier['scorer_agreement_rate']:.0%}`"
        )
    summary_path.write_text("\n".join(summary_lines))


def _write_transcripts(path: Path, transcripts: list[FrozenReplayTask]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for transcript in transcripts:
            handle.write(json.dumps(_transcript_record(transcript)) + "\n")


def _append_transcript(path: Path, transcript: FrozenReplayTask) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_transcript_record(transcript)) + "\n")


def _log_progress(path: Path | None, message: str) -> None:
    print(message, flush=True)
    if path is None:
        return
    with path.open("a", encoding="utf-8") as handle:
        handle.write(message + "\n")


def _transcript_record(transcript: FrozenReplayTask) -> dict[str, Any]:
    return {
        "task": transcript.name,
        "template_name": transcript.metadata.get("template_name"),
        "transcript_id": transcript.metadata.get("transcript_id"),
        "seed": transcript.metadata.get("seed"),
        "source_seed": transcript.metadata.get("source_seed"),
        "final_user_index": transcript.final_user_index,
        "system_prompt": transcript.system_prompt,
        "messages": list(transcript.messages),
        "facts": [_fact_record(fact) for fact in transcript.facts],
        "metadata": transcript.metadata,
    }


def _fact_record(fact: FactSpec) -> dict[str, Any]:
    return {
        "name": fact.name,
        "canonical_value": fact.canonical_value,
        "allowed_aliases": list(fact.allowed_aliases),
        "wrong_aliases": list(fact.wrong_aliases),
        "normalizer_id": fact.normalizer_id,
        "parser_id": fact.parser_id,
        "source_message_indexes": list(fact.source_message_indexes),
        "distractor_message_indexes": list(fact.distractor_message_indexes),
    }


def _audit_queue_record(result: RunResult) -> dict[str, Any]:
    reasons = []
    if not result.scorer_agreement:
        reasons.append("scorer_disagreement")
    if result.contamination_count > 0:
        reasons.append("contamination")
    if not reasons:
        reasons.append("random_sample")
    return {
        "blind_id": result.blind_id,
        "task": result.task,
        "model": result.model,
        "window_budget": result.window_budget,
        "seed": result.seed,
        "review_reasons": reasons,
        "official_score": result.score,
        "secondary_score": result.secondary_score,
        "fact_results": result.fact_results,
        "contamination_count": result.contamination_count,
        "final_response": result.final_response,
    }
