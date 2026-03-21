from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Optional

from contextgc_barrier import MLXBackend

from .specs import (
    AggregateResult,
    ModelSpec,
    RunResult,
    StrategySpec,
    TaskSpec,
    choose_audit_sample,
)
from .stats import exact_sign_test_p_value, paired_bootstrap_ci, win_tie_loss
from .strategies import run_internal_strategy

DEFAULT_PRIMARY_LOCAL_MODEL = "mlx-community/Qwen3.5-4B-OptiQ-4bit"
DEFAULT_WINDOW_BUDGETS = (3072, 16384)


def build_default_model_specs(
    *,
    primary_local_model: str = DEFAULT_PRIMARY_LOCAL_MODEL,
) -> list[ModelSpec]:
    return [
        ModelSpec(
            alias="qwen_local",
            provider="mlx",
            model_name=primary_local_model,
            window_limit=16_384,
            backend_factory=lambda: MLXBackend(default_model=primary_local_model),
            tags=("local", "primary_local"),
        ),
    ]


def build_default_strategy_specs() -> list[StrategySpec]:
    return [
        StrategySpec(
            name="summary80",
            label="Summary @80%",
            include_in_adaptive_gate=True,
            internal_strategy="summary80",
            citation_enabled=False,
        ),
        StrategySpec(
            name="barrier",
            label="Barrier",
            include_in_adaptive_gate=True,
            internal_strategy="barrier",
        ),
        StrategySpec(
            name="summary80_barrier",
            label="Summary @80% + Barrier",
            include_in_adaptive_gate=True,
            internal_strategy="summary80_barrier",
        ),
    ]


class BenchmarkRunner:
    def __init__(
        self,
        *,
        task_specs: Iterable[TaskSpec],
        model_specs: Iterable[ModelSpec],
        strategy_specs: Iterable[StrategySpec],
        output_dir: Path,
        response_budget: int = 128,
        overflow_ratio: float = 1.15,
        initial_seed_count: int = 5,
        max_seed_count: int = 10,
        resume: bool = False,
    ) -> None:
        self.task_specs = {task.name: task for task in task_specs}
        self.model_specs = {model.alias: model for model in model_specs}
        self.strategy_specs = {strategy.name: strategy for strategy in strategy_specs}
        self.output_dir = output_dir
        self.response_budget = response_budget
        self.overflow_ratio = overflow_ratio
        self.initial_seed_count = initial_seed_count
        self.max_seed_count = max_seed_count
        self.resume = resume
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.run_jsonl_path = self.output_dir / "runs.jsonl"
        self.aggregate_json_path = self.output_dir / "aggregate.json"
        self.aggregate_csv_path = self.output_dir / "aggregate.csv"
        self.summary_path = self.output_dir / "summary.md"
        self.audit_queue_path = self.output_dir / "audit_queue.jsonl"
        if not resume:
            for path in (
                self.run_jsonl_path,
                self.aggregate_json_path,
                self.aggregate_csv_path,
                self.summary_path,
                self.audit_queue_path,
            ):
                path.unlink(missing_ok=True)
        self._backend_cache: dict[str, Any] = {}
        self._results = self._load_existing_results() if resume else []

    def run(
        self,
        *,
        task_names: Optional[list[str]] = None,
        model_names: Optional[list[str]] = None,
        strategy_names: Optional[list[str]] = None,
        window_budgets: Iterable[int] = DEFAULT_WINDOW_BUDGETS,
    ) -> dict[str, Any]:
        selected_tasks = self._select(self.task_specs, task_names)
        selected_models = self._select(self.model_specs, model_names)
        selected_strategies = self._select(self.strategy_specs, strategy_names)
        budgets = list(window_budgets)

        self._run_seed_range(
            selected_tasks=selected_tasks,
            selected_models=selected_models,
            selected_strategies=selected_strategies,
            window_budgets=budgets,
            start_seed=1,
            end_seed=self.initial_seed_count,
        )

        extension_groups = self._groups_requiring_extension(
            selected_tasks=selected_tasks,
            selected_models=selected_models,
            selected_strategies=selected_strategies,
            window_budgets=budgets,
        )
        if extension_groups:
            self._run_seed_range(
                selected_tasks=selected_tasks,
                selected_models=selected_models,
                selected_strategies=selected_strategies,
                window_budgets=budgets,
                start_seed=self.initial_seed_count + 1,
                end_seed=self.max_seed_count,
                only_groups=extension_groups,
            )

        self._finalize_results()
        aggregates = self.aggregate_results()
        self._write_aggregate_artifacts(aggregates)
        return {
            "runs": [result.to_record() for result in self._results],
            "aggregates": [aggregate.to_record() for aggregate in aggregates],
        }

    def aggregate_results(self) -> list[AggregateResult]:
        grouped: dict[tuple[str, str, int], dict[str, list[RunResult]]] = defaultdict(lambda: defaultdict(list))
        for result in self._results:
            grouped[(result.task, result.model, result.window_budget)][result.strategy].append(result)

        aggregates: list[AggregateResult] = []
        for (task_name, model_alias, window_budget), by_strategy in sorted(grouped.items()):
            summary_runs = sorted(by_strategy.get("summary80", []), key=lambda item: item.seed)
            summary_scores = {run.seed: run.score for run in summary_runs}
            model_spec = self.model_specs[model_alias]
            for strategy_name, runs in sorted(by_strategy.items()):
                ordered_runs = sorted(runs, key=lambda item: item.seed)
                if strategy_name == "summary80":
                    paired = [0.0 for _ in ordered_runs]
                else:
                    paired = [
                        run.score - summary_scores[run.seed]
                        for run in ordered_runs
                        if run.seed in summary_scores
                    ]
                ci_low, ci_high = paired_bootstrap_ci(paired, seed=window_budget + len(ordered_runs))
                wins, ties, losses = win_tie_loss(paired)
                aggregates.append(
                    AggregateResult.from_runs(
                        task=task_name,
                        model=model_alias,
                        model_name=model_spec.model_name,
                        provider=model_spec.provider,
                        strategy=strategy_name,
                        window_budget=window_budget,
                        runs=ordered_runs,
                        delta_vs_summary80=mean(paired) if paired else 0.0,
                        ci_low=ci_low,
                        ci_high=ci_high,
                        p_value=exact_sign_test_p_value(paired),
                        wins=wins,
                        ties=ties,
                        losses=losses,
                    )
                )
        return sorted(aggregates, key=lambda item: (item.task, item.model, item.window_budget, item.strategy))

    def _run_seed_range(
        self,
        *,
        selected_tasks: dict[str, TaskSpec],
        selected_models: dict[str, ModelSpec],
        selected_strategies: dict[str, StrategySpec],
        window_budgets: list[int],
        start_seed: int,
        end_seed: int,
        only_groups: Optional[set[tuple[str, str, int]]] = None,
    ) -> None:
        existing_keys = {result.key() for result in self._results}
        for task_name, task_spec in selected_tasks.items():
            for model_alias, model_spec in selected_models.items():
                backend = self._backend_for(model_spec)
                for window_budget in window_budgets:
                    if window_budget > model_spec.window_limit:
                        continue
                    group_key = (task_name, model_alias, window_budget)
                    if only_groups is not None and group_key not in only_groups:
                        continue
                    for seed in range(start_seed, end_seed + 1):
                        task_instance = task_spec.build(
                            backend=backend,
                            model=model_spec.model_name,
                            window_budget=window_budget,
                            overflow_ratio=self.overflow_ratio,
                            seed=seed,
                        )
                        for strategy_name, strategy_spec in selected_strategies.items():
                            if not strategy_spec.is_applicable(model_spec, window_budget):
                                continue
                            run_key = (task_name, model_alias, strategy_name, window_budget, seed)
                            if run_key in existing_keys:
                                continue
                            result = self._run_single(
                                backend=backend,
                                task=task_instance,
                                model=model_spec,
                                strategy=strategy_spec,
                                window_budget=window_budget,
                            )
                            self._results.append(result)
                            existing_keys.add(run_key)
                            self._append_run_record(result)

    def _run_single(
        self,
        *,
        backend: Any,
        task,
        model: ModelSpec,
        strategy: StrategySpec,
        window_budget: int,
    ) -> RunResult:
        return run_internal_strategy(
            backend=backend,
            task=task,
            model=model,
            strategy=strategy,
            window_budget=window_budget,
            response_budget=self.response_budget,
        )

    def _groups_requiring_extension(
        self,
        *,
        selected_tasks: dict[str, TaskSpec],
        selected_models: dict[str, ModelSpec],
        selected_strategies: dict[str, StrategySpec],
        window_budgets: list[int],
    ) -> set[tuple[str, str, int]]:
        grouped: dict[tuple[str, str, int], dict[str, list[RunResult]]] = defaultdict(lambda: defaultdict(list))
        for result in self._results:
            group_key = (result.task, result.model, result.window_budget)
            grouped[group_key][result.strategy].append(result)

        extension_groups: set[tuple[str, str, int]] = set()
        for task_name in selected_tasks:
            for model_alias in selected_models:
                for window_budget in window_budgets:
                    group_key = (task_name, model_alias, window_budget)
                    runs_by_strategy = grouped.get(group_key)
                    if not runs_by_strategy:
                        continue
                    summary_runs = sorted(runs_by_strategy.get("summary80", []), key=lambda item: item.seed)
                    barrier_runs = sorted(runs_by_strategy.get("barrier", []), key=lambda item: item.seed)
                    hybrid_runs = sorted(runs_by_strategy.get("summary80_barrier", []), key=lambda item: item.seed)
                    if len(summary_runs) < self.initial_seed_count:
                        continue
                    scorer_disagreement_rate = self._scorer_disagreement_rate(group_key)
                    needs_extension = False
                    if len(barrier_runs) >= self.initial_seed_count:
                        needs_extension = needs_extension or self._pair_is_inconclusive(
                            left_runs=barrier_runs,
                            right_runs=summary_runs,
                            seed=window_budget,
                            scorer_disagreement_rate=scorer_disagreement_rate,
                        )
                    if len(barrier_runs) >= self.initial_seed_count and len(hybrid_runs) >= self.initial_seed_count:
                        needs_extension = needs_extension or self._pair_is_inconclusive(
                            left_runs=hybrid_runs,
                            right_runs=barrier_runs,
                            seed=window_budget + 1_000,
                            scorer_disagreement_rate=scorer_disagreement_rate,
                        )
                    if needs_extension and self.max_completed_seed(group_key) < self.max_seed_count:
                        extension_groups.add(group_key)
        return extension_groups

    def max_completed_seed(self, group_key: tuple[str, str, int]) -> int:
        seeds = [
            result.seed
            for result in self._results
            if (result.task, result.model, result.window_budget) == group_key
        ]
        return max(seeds) if seeds else 0

    def _pair_is_inconclusive(
        self,
        *,
        left_runs: list[RunResult],
        right_runs: list[RunResult],
        seed: int,
        scorer_disagreement_rate: float,
    ) -> bool:
        right_scores = {run.seed: run.score for run in right_runs}
        paired = [
            run.score - right_scores[run.seed]
            for run in sorted(left_runs, key=lambda item: item.seed)
            if run.seed in right_scores
        ]
        if len(paired) < self.initial_seed_count:
            return True
        delta_mean = mean(paired)
        ci_low, ci_high = paired_bootstrap_ci(paired, seed=seed)
        p_value = exact_sign_test_p_value(paired)
        return (
            ci_low <= 0.0 <= ci_high
            or delta_mean < 0.10
            or p_value > 0.05
            or scorer_disagreement_rate > 0.05
        )

    def _scorer_disagreement_rate(self, group_key: tuple[str, str, int]) -> float:
        group_runs = [
            result for result in self._results
            if (result.task, result.model, result.window_budget) == group_key
        ]
        if not group_runs:
            return 0.0
        disagreements = sum(1 for result in group_runs if not result.scorer_agreement)
        return disagreements / len(group_runs)

    def _backend_for(self, model: ModelSpec):
        if model.alias not in self._backend_cache:
            self._backend_cache[model.alias] = model.backend_factory()
        return self._backend_cache[model.alias]

    def _select(self, items: dict[str, Any], names: Optional[list[str]]) -> dict[str, Any]:
        if not names:
            return dict(items)
        selected = {}
        for name in names:
            if name not in items:
                raise ValueError(f"Unknown name: {name}")
            selected[name] = items[name]
        return selected

    def _load_existing_results(self) -> list[RunResult]:
        if not self.run_jsonl_path.exists():
            return []
        deduped: dict[tuple[str, str, str, int, int], RunResult] = {}
        for line in self.run_jsonl_path.read_text().splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            result = RunResult(
                task=record["task"],
                model=record["model"],
                model_name=record.get("model_name", record["model"]),
                provider=record["provider"],
                strategy=record["strategy"],
                window_budget=record["window_budget"],
                seed=record["seed"],
                score=record["score"],
                secondary_score=record.get("secondary_score", record["score"]),
                found=record.get("found", []),
                wrong=record.get("wrong", []),
                missing=record.get("missing", []),
                missed=record.get("missed", []),
                fact_results=record.get("fact_results", []),
                contamination=record.get("contamination", []),
                contamination_count=record.get("contamination_count", 0),
                scorer_agreement=record.get("scorer_agreement", True),
                final_response=record["final_response"],
                prompt_tokens=record["prompt_tokens"],
                usable_prompt_budget=record["usable_prompt_budget"],
                selected_indexes=record.get("selected_indexes", []),
                protected_selected_indexes=record.get("protected_selected_indexes", []),
                cited_selected_indexes=record.get("cited_selected_indexes", []),
                protected_message_indexes=record.get("protected_message_indexes", []),
                cited_message_indexes=record.get("cited_message_indexes", []),
                retained_anchor=record.get("retained_anchor", False),
                retained_distractor=record.get("retained_distractor", False),
                anchor_protected=record.get("anchor_protected", False),
                turn_count=record["turn_count"],
                session_metadata=record["session_metadata"],
                strategy_metadata=record["strategy_metadata"],
                final_prompt_anchor_overlap=record.get("final_prompt_anchor_overlap", 0.0),
                final_prompt_distractor_overlap=record.get("final_prompt_distractor_overlap", 0.0),
                blind_id=record.get("blind_id"),
                audit_required=record.get("audit_required", False)
                or (not record.get("scorer_agreement", True))
                or record.get("contamination_count", 0) > 0,
            )
            deduped[result.key()] = result
        return sorted(deduped.values(), key=lambda item: (item.task, item.model, item.window_budget, item.strategy, item.seed))

    def _append_run_record(self, result: RunResult) -> None:
        result.audit_required = self._requires_immediate_audit(result)
        with self.run_jsonl_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(result.to_record()) + "\n")
        if result.audit_required:
            with self.audit_queue_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(self._audit_queue_record(result)) + "\n")

    def _requires_immediate_audit(self, result: RunResult) -> bool:
        return (not result.scorer_agreement) or result.contamination_count > 0

    def _audit_queue_record(self, result: RunResult) -> dict[str, Any]:
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

    def _finalize_results(self) -> None:
        deduped = {result.key(): result for result in self._results}
        self._results = sorted(
            deduped.values(),
            key=lambda item: (item.task, item.model, item.window_budget, item.seed, item.strategy),
        )

        for index, result in enumerate(self._results, start=1):
            result.blind_id = f"audit-{index:04d}"
            result.audit_required = False

        for result in choose_audit_sample(self._results, seed=0):
            result.audit_required = True

    def _write_aggregate_artifacts(self, aggregates: list[AggregateResult]) -> None:
        self._rewrite_run_records()
        self._write_audit_queue()

        aggregate_records = [aggregate.to_record() for aggregate in aggregates]
        self.aggregate_json_path.write_text(json.dumps(aggregate_records, indent=2))
        with self.aggregate_csv_path.open("w", newline="", encoding="utf-8") as handle:
            if aggregate_records:
                writer = csv.DictWriter(handle, fieldnames=list(aggregate_records[0].keys()))
                writer.writeheader()
                writer.writerows(aggregate_records)

        summary_lines = [
            "# ContextGC Benchmark Matrix",
            "",
            f"- Runs: `{len(self._results)}`",
            f"- Initial seeds: `{self.initial_seed_count}`",
            f"- Max seeds: `{self.max_seed_count}`",
            f"- Response budget: `{self.response_budget}`",
            f"- Audit queue size: `{sum(1 for result in self._results if result.audit_required)}`",
            "",
            "## Aggregate Summary",
            "",
            "| Task | Model | Window | Strategy | Mean | Secondary | Contam | Agree | Anchor | Distractor | Delta vs Summary80 | 95% CI | p-value |",
            "|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|---:|",
        ]
        for aggregate in aggregates:
            summary_lines.append(
                f"| {aggregate.task} | {aggregate.model} | {aggregate.window_budget} | {aggregate.strategy} | "
                f"{aggregate.mean_score:.3f} | {aggregate.mean_secondary_score:.3f} | "
                f"{aggregate.contamination_rate:.0%} | {aggregate.scorer_agreement_rate:.0%} | "
                f"{aggregate.retained_anchor_rate:.0%} | {aggregate.retained_distractor_rate:.0%} | "
                f"{aggregate.delta_vs_summary80:.3f} | "
                f"[{aggregate.ci_low:.3f}, {aggregate.ci_high:.3f}] | {aggregate.p_value:.3f} |"
            )
        summary_lines.extend(["", "## Barrier vs Summary80", ""])
        summary_lines.extend(self._paired_comparison_lines(aggregates, left="barrier", right="summary80"))
        summary_lines.extend(["", "## Summary80 + Barrier vs Barrier", ""])
        summary_lines.extend(self._paired_comparison_lines(aggregates, left="summary80_barrier", right="barrier"))
        self.summary_path.write_text("\n".join(summary_lines))

    def _paired_comparison_lines(
        self,
        aggregates: list[AggregateResult],
        *,
        left: str,
        right: str,
    ) -> list[str]:
        aggregate_lookup = {
            (aggregate.task, aggregate.model, aggregate.window_budget, aggregate.strategy): aggregate
            for aggregate in aggregates
        }
        lines: list[str] = []
        for aggregate in aggregates:
            if aggregate.strategy != left:
                continue
            right_aggregate = aggregate_lookup.get((aggregate.task, aggregate.model, aggregate.window_budget, right))
            if right_aggregate is None:
                continue
            left_runs = sorted(
                [
                    result
                    for result in self._results
                    if (
                        result.task == aggregate.task
                        and result.model == aggregate.model
                        and result.window_budget == aggregate.window_budget
                        and result.strategy == left
                    )
                ],
                key=lambda item: item.seed,
            )
            right_scores = {
                result.seed: result.score
                for result in self._results
                if (
                    result.task == aggregate.task
                    and result.model == aggregate.model
                    and result.window_budget == aggregate.window_budget
                    and result.strategy == right
                )
            }
            paired = [run.score - right_scores[run.seed] for run in left_runs if run.seed in right_scores]
            ci_low, ci_high = paired_bootstrap_ci(paired, seed=aggregate.window_budget + (1_000 if left == "summary80_barrier" else 0))
            wins, ties, losses = win_tie_loss(paired)
            delta = mean(paired) if paired else 0.0
            p_value = exact_sign_test_p_value(paired)
            lines.append(
                f"- `{aggregate.task}` @ `{aggregate.window_budget}`: "
                f"`{left}` mean `{aggregate.mean_score:.3f}` vs `{right}` `{right_aggregate.mean_score:.3f}` "
                f"(delta `{delta:.3f}`, 95% CI `[{ci_low:.3f}, {ci_high:.3f}]`, p `{p_value:.3f}`, "
                f"wins/ties/losses `{wins}/{ties}/{losses}`), anchor `{aggregate.retained_anchor_rate:.0%}`, "
                f"contamination `{aggregate.contamination_rate:.0%}`, agreement `{aggregate.scorer_agreement_rate:.0%}`"
            )
        return lines

    def _rewrite_run_records(self) -> None:
        with self.run_jsonl_path.open("w", encoding="utf-8") as handle:
            for result in self._results:
                handle.write(json.dumps(result.to_record()) + "\n")

    def _write_audit_queue(self) -> None:
        with self.audit_queue_path.open("w", encoding="utf-8") as handle:
            for result in self._results:
                if not result.audit_required:
                    continue
                handle.write(json.dumps(self._audit_queue_record(result)) + "\n")
