from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Iterable

from .runner import build_default_model_specs, build_default_strategy_specs
from .specs import FactSpec, FrozenReplayTask, ModelSpec, RunResult, build_phrase_pattern, choose_audit_sample
from .stats import exact_sign_test_p_value, paired_bootstrap_ci, win_tie_loss
from .strategies import run_internal_replay_strategy
from .tasks import (
    DEBUGGING_REPLAY_TARGET_TOKENS,
    DEBUGGING_REPLAY_TOKEN_TOLERANCE,
    build_debugging_replay_seed_task,
)

DEBUGGING_REPLAY_PROFILE = "debugging_replay"
DEBUGGING_REPLAY_SMOKE_PROFILE = "debugging_replay_smoke"
DEBUGGING_REPLAY_TASK_NAME = "debugging_replay"
DEBUGGING_REPLAY_TEMPLATES = ("incident_closeout", "root_cause_recap")
DEBUGGING_REPLAY_WINDOWS = (3072, 4096, 16_384)
DEBUGGING_REPLAY_SMOKE_WINDOWS = (3072,)
DEBUGGING_REPLAY_SMOKE_ACCEPTED_PER_TEMPLATE = 1
DEBUGGING_REPLAY_SMOKE_TARGET_TOKENS = DEBUGGING_REPLAY_TARGET_TOKENS
DEBUGGING_REPLAY_SMOKE_TOKEN_TOLERANCE = 500
FROZEN_TRANSCRIPT_CALIBRATION_ATTEMPTS = 4

def run_debugging_replay(
    *,
    output_dir: Path,
    primary_local_model: str,
    response_budget: int = 128,
    accepted_per_template: int = 10,
    target_total_tokens: int = DEBUGGING_REPLAY_TARGET_TOKENS,
    target_token_tolerance: int = DEBUGGING_REPLAY_TOKEN_TOLERANCE,
    window_budgets: tuple[int, ...] = DEBUGGING_REPLAY_WINDOWS,
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
    strategies = {strategy.name: strategy for strategy in build_default_strategy_specs()}
    _log_progress(
        progress_log_path,
        (
            f"[start] profile={profile_name} model={model_spec.model_name} response_budget={response_budget} "
            f"accepted_per_template={accepted_per_template} target_total_tokens={target_total_tokens} "
            f"target_token_tolerance={target_token_tolerance} windows={','.join(str(item) for item in window_budgets)}"
        ),
    )

    transcripts = generate_scripted_debugging_replay_corpus(
        backend=backend,
        model=model_spec,
        accepted_per_template=accepted_per_template,
        target_total_tokens=target_total_tokens,
        target_token_tolerance=target_token_tolerance,
        transcripts_path=transcripts_path,
        progress_log_path=progress_log_path,
    )
    _write_transcripts(transcripts_path, transcripts)

    results: list[RunResult] = []
    for transcript in transcripts:
        transcript_id = str(transcript.metadata.get("transcript_id", "unknown"))
        for window_budget in window_budgets:
            for strategy_name in ("summary80", "barrier", "summary80_barrier"):
                _log_progress(
                    progress_log_path,
                    f"[eval] transcript={transcript_id} window={window_budget} strategy={strategy_name} status=running",
                )
                result = run_internal_replay_strategy(
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

    finalized = _finalize_results(results)
    aggregate_records = aggregate_debugging_replay_results(finalized)
    _write_result_artifacts(
        output_dir=output_dir,
        results=finalized,
        aggregate_records=aggregate_records,
        response_budget=response_budget,
        transcript_count=len(transcripts),
        window_budgets=window_budgets,
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
    window_budgets: tuple[int, ...] = DEBUGGING_REPLAY_SMOKE_WINDOWS,
    model_specs: list[ModelSpec] | None = None,
) -> dict[str, Any]:
    return run_debugging_replay(
        output_dir=output_dir,
        primary_local_model=primary_local_model,
        response_budget=response_budget,
        accepted_per_template=DEBUGGING_REPLAY_SMOKE_ACCEPTED_PER_TEMPLATE,
        target_total_tokens=target_total_tokens,
        target_token_tolerance=target_token_tolerance,
        window_budgets=window_budgets,
        profile_name=DEBUGGING_REPLAY_SMOKE_PROFILE,
        model_specs=model_specs,
    )


def generate_scripted_debugging_replay_corpus(
    *,
    backend,
    model: ModelSpec,
    accepted_per_template: int,
    target_total_tokens: int,
    target_token_tolerance: int,
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
                f"target_total_tokens={target_total_tokens}"
            ),
        )
        for source_seed in range(1, accepted_per_template + 1):
            _log_progress(
                progress_log_path,
                f"[corpus] template={template_name} source_seed={source_seed} status=materializing",
            )
            frozen, prompt_tokens, calibrated_target_total_tokens, calibration_attempts = _build_calibrated_scripted_transcript(
                backend=backend,
                model_name=model.model_name,
                source_seed=source_seed,
                template_name=template_name,
                target_total_tokens=target_total_tokens,
                target_token_tolerance=target_token_tolerance,
                eval_seed=eval_seed,
            )
            transcript_id = f"{template_name}-{source_seed:02d}"
            metadata = dict(frozen.metadata)
            metadata.update(
                {
                    "seed": eval_seed,
                    "source_seed": source_seed,
                    "transcript_id": transcript_id,
                    "transcript_prompt_tokens": prompt_tokens,
                    "calibrated_source_target_tokens": calibrated_target_total_tokens,
                    "calibration_attempts": calibration_attempts,
                    "scripted_replay": True,
                }
            )
            accepted_transcript = FrozenReplayTask(
                name=frozen.name,
                system_prompt=frozen.system_prompt,
                messages=frozen.messages,
                final_user_index=frozen.final_user_index,
                facts=frozen.facts,
                metadata=metadata,
            )
            transcripts.append(accepted_transcript)
            if transcripts_path is not None:
                _append_transcript(transcripts_path, accepted_transcript)
            _log_progress(
                progress_log_path,
                (
                    f"[corpus] template={template_name} source_seed={source_seed} status=accepted "
                    f"transcript_id={transcript_id} prompt_tokens={prompt_tokens} "
                    f"calibrated_source_target={calibrated_target_total_tokens} calibration_attempts={calibration_attempts}"
                ),
            )
            eval_seed += 1
        _log_progress(
            progress_log_path,
            f"[corpus] template={template_name} status=completed accepted={accepted_per_template}",
        )
    return transcripts


def _build_calibrated_scripted_transcript(
    *,
    backend,
    model_name: str,
    source_seed: int,
    template_name: str,
    target_total_tokens: int,
    target_token_tolerance: int,
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
        frozen = _freeze_scripted_task_transcript(
            candidate=candidate,
            template_name=template_name,
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
        raise RuntimeError("Failed to build a scripted replay transcript")
    return last_frozen, last_prompt_tokens, last_attempt_target_total_tokens, FROZEN_TRANSCRIPT_CALIBRATION_ATTEMPTS


def _freeze_scripted_task_transcript(
    *,
    candidate,
    template_name: str,
    eval_seed: int,
    source_seed: int,
) -> FrozenReplayTask:
    messages: list[dict[str, str]] = []
    for index, turn in enumerate(candidate.turns):
        message = {"role": turn["role"], "content": turn["content"]}
        messages.append(message)
        if index == len(candidate.turns) - 1:
            break
        assistant_message = {
            "role": "assistant",
            "content": _scripted_debugging_assistant_reply(
                template_name=template_name,
                turn_index=index,
            ),
        }
        _ensure_reply_has_no_fact_aliases(assistant_message["content"], candidate.facts)
        messages.append(assistant_message)

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


def _scripted_debugging_assistant_reply(
    *,
    template_name: str,
    turn_index: int,
) -> str:
    if template_name == "root_cause_recap":
        replies = {
            0: "I have the active incident from the first escalation pinned as the live case.",
            1: "The diagnosis board and patch note from earlier are still the source of truth.",
            2: "Acknowledged. I will preserve the exact shipped cleanup step from the earlier note.",
            3: "Short recap: the earlier diagnosis and patch note still apply.",
            4: "The telemetry is noisy, but it does not replace the first incident record.",
            5: "I will keep the earlier diagnosis and cleanup step together in the handoff.",
            6: "Acknowledged. I will ignore the stale rollback notes.",
            7: "The live incident still comes from the same earlier diagnosis, code change, and shipped cleanup step.",
            8: "The memory graph adds noise, not a new live explanation.",
            9: "Understood. The archived remediation chatter is stale and should be ignored.",
            10: "I am still writing from the original live notes, not the stale case.",
            11: "The profiling dump supports the same earlier code change.",
        }
        return replies.get(
            turn_index,
            "The live incident still follows the first diagnosis note.",
        )

    replies = {
        0: "I have the live incident from the opening handoff pinned as the active case.",
        1: "The trace note and working diagnosis from earlier are still the source of truth.",
        2: "Acknowledged. I will keep the shipped corrective step from the first notes.",
        3: "The first handoff still defines the live routine and threshold.",
        4: "The retry telemetry is noisy, but it does not replace the original incident report.",
        5: "I will keep the exact patch note and cleanup step tied to the live incident.",
        6: "Acknowledged. I will ignore the archived side-case.",
        7: "In plain English, the earlier live diagnosis still stands.",
        8: "The profiling dump still lines up with the same production issue from the first report.",
        9: "The shipped build note and the production blocker are unchanged from the opening report.",
        10: "Those meeting notes are about the old canary and not the live incident.",
        11: "The live closeout still comes from the original handoff, not the archived case.",
    }
    return replies.get(
        turn_index,
        "The active incident still follows the opening handoff.",
    )


def _ensure_reply_has_no_fact_aliases(reply_text: str, facts: tuple[FactSpec, ...]) -> None:
    lowered_reply = reply_text.lower()
    for fact in facts:
        for alias in (fact.canonical_value, *fact.allowed_aliases):
            if not alias:
                continue
            if build_phrase_pattern(alias).search(reply_text):
                raise ValueError(f"Scripted replay reply leaked fact alias: {alias}")
            if alias.lower() in lowered_reply:
                raise ValueError(f"Scripted replay reply leaked fact alias: {alias}")


def aggregate_debugging_replay_results(results: list[RunResult]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, int], dict[str, list[RunResult]]] = {}
    for result in results:
        grouped.setdefault((result.task, result.model, result.window_budget), {}).setdefault(result.strategy, []).append(result)

    records: list[dict[str, Any]] = []
    for (task_name, model_alias, window_budget), by_strategy in sorted(grouped.items()):
        summary_runs = {run.seed: run.score for run in by_strategy.get("summary80", [])}
        for strategy_name, runs in sorted(by_strategy.items()):
            ordered_runs = sorted(runs, key=lambda item: item.seed)
            if strategy_name == "summary80":
                paired = [0.0 for _ in ordered_runs]
            else:
                paired = [
                    run.score - summary_runs[run.seed]
                    for run in ordered_runs
                    if run.seed in summary_runs
                ]
            ci_low, ci_high = paired_bootstrap_ci(paired, seed=window_budget + len(ordered_runs))
            wins, ties, losses = win_tie_loss(paired)
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


def _paired_comparison_records(
    results: list[RunResult],
    *,
    left: str,
    right: str,
    seed_offset: int = 0,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, int], dict[str, list[RunResult]]] = {}
    for result in results:
        grouped.setdefault((result.task, result.model, result.window_budget), {}).setdefault(result.strategy, []).append(result)

    records: list[dict[str, Any]] = []
    for (task_name, model_alias, window_budget), by_strategy in sorted(grouped.items()):
        left_runs = sorted(by_strategy.get(left, []), key=lambda item: item.seed)
        right_runs = sorted(by_strategy.get(right, []), key=lambda item: item.seed)
        if not left_runs or not right_runs:
            continue
        right_scores = {run.seed: run.score for run in right_runs}
        paired = [run.score - right_scores[run.seed] for run in left_runs if run.seed in right_scores]
        if not paired:
            continue
        ci_low, ci_high = paired_bootstrap_ci(paired, seed=window_budget + seed_offset)
        wins, ties, losses = win_tie_loss(paired)
        records.append(
            {
                "task": task_name,
                "model": model_alias,
                "window_budget": window_budget,
                "left": left,
                "right": right,
                "left_mean": mean(run.score for run in left_runs),
                "right_mean": mean(run.score for run in right_runs),
                "delta": mean(paired),
                "ci_low": ci_low,
                "ci_high": ci_high,
                "p_value": exact_sign_test_p_value(paired),
                "wins": wins,
                "ties": ties,
                "losses": losses,
                "left_anchor_rate": _rate(left_runs, lambda run: run.retained_anchor),
                "left_contamination_rate": _rate(left_runs, lambda run: run.contamination_count > 0),
                "left_scorer_agreement_rate": _rate(left_runs, lambda run: run.scorer_agreement),
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
        result.audit_required = False

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
    window_budgets: tuple[int, ...],
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
        f"- Windows: `{','.join(str(item) for item in window_budgets)}`",
        f"- Response budget: `{response_budget}`",
        f"- Audit queue size: `{sum(1 for result in results if result.audit_required)}`",
        "",
        "## Aggregate Summary",
        "",
        "| Window | Strategy | Mean | Secondary | Root Cause | File Line | Remediation | Anchor | SrcIdx3 | Distractor | Contam | Agree | Delta vs Summary80 | 95% CI | p-value |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|",
    ]
    for aggregate in aggregate_records:
        summary_lines.append(
            f"| {aggregate['window_budget']} | {aggregate['strategy']} | {aggregate['mean_score']:.3f} | "
            f"{aggregate['mean_secondary_score']:.3f} | {aggregate['root_cause_accuracy']:.0%} | "
            f"{aggregate['file_line_accuracy']:.0%} | {aggregate['remediation_accuracy']:.0%} | "
            f"{aggregate['retained_anchor_rate']:.0%} | {aggregate['source_index_3_selected_rate']:.0%} | "
            f"{aggregate['retained_distractor_rate']:.0%} | {aggregate['contamination_rate']:.0%} | "
            f"{aggregate['scorer_agreement_rate']:.0%} | {aggregate['delta_vs_summary80']:.3f} | "
            f"[{aggregate['ci_low']:.3f}, {aggregate['ci_high']:.3f}] | {aggregate['p_value']:.3f} |"
        )

    summary_lines.extend(["", "## Barrier vs Summary80", ""])
    summary_lines.extend(_paired_comparison_lines(results, left="barrier", right="summary80", seed_offset=0))
    summary_lines.extend(["", "## Summary80 + Barrier vs Barrier", ""])
    summary_lines.extend(_paired_comparison_lines(results, left="summary80_barrier", right="barrier", seed_offset=1_000))
    summary_path.write_text("\n".join(summary_lines))


def _paired_comparison_lines(
    results: list[RunResult],
    *,
    left: str,
    right: str,
    seed_offset: int,
) -> list[str]:
    lines: list[str] = []
    for comparison in _paired_comparison_records(results, left=left, right=right, seed_offset=seed_offset):
        lines.append(
            f"- `{comparison['task']}` @ `{comparison['window_budget']}`: "
            f"`{left}` mean `{comparison['left_mean']:.3f}` vs `{right}` `{comparison['right_mean']:.3f}` "
            f"(delta `{comparison['delta']:.3f}`, 95% CI `[{comparison['ci_low']:.3f}, {comparison['ci_high']:.3f}]`, "
            f"p `{comparison['p_value']:.3f}`, wins/ties/losses `{comparison['wins']}/{comparison['ties']}/{comparison['losses']}`), "
            f"anchor `{comparison['left_anchor_rate']:.0%}`, contamination `{comparison['left_contamination_rate']:.0%}`, "
            f"agreement `{comparison['left_scorer_agreement_rate']:.0%}`"
        )
    return lines


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
        "strategy": result.strategy,
        "window_budget": result.window_budget,
        "seed": result.seed,
        "review_reasons": reasons,
        "official_score": result.score,
        "secondary_score": result.secondary_score,
        "fact_results": result.fact_results,
        "contamination_count": result.contamination_count,
        "final_response": result.final_response,
    }
