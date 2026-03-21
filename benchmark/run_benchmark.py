from __future__ import annotations

import argparse
import json
from pathlib import Path

from benchmark.replay import run_debugging_replay_smoke, run_debugging_replay_v1
from benchmark.runner import (
    DEFAULT_PRIMARY_LOCAL_MODEL,
    DEFAULT_WINDOW_BUDGETS,
    BenchmarkRunner,
    build_default_model_specs,
    build_default_strategy_specs,
)
from benchmark.tasks import build_task_specs


def parse_csv(value: str | None) -> list[str] | None:
    if value is None:
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_int_csv(value: str | None) -> list[int] | None:
    if value is None:
        return None
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the ContextGC benchmark matrix.")
    parser.add_argument("--profile", choices=("matrix", "proof", "debugging_replay_v1", "debugging_replay_smoke"), default="matrix")
    parser.add_argument("--tasks", help="Comma-separated task names.")
    parser.add_argument("--models", help="Comma-separated model aliases.")
    parser.add_argument("--strategies", help="Comma-separated strategy names.")
    parser.add_argument("--window-budgets", help="Comma-separated prompt budgets.")
    parser.add_argument("--window-budget", type=int, help="Compatibility shorthand for the proof profile.")
    parser.add_argument("--response-budget", type=int, default=128)
    parser.add_argument("--overflow-ratio", type=float, default=1.15)
    parser.add_argument("--seed-count", type=int, default=5)
    parser.add_argument("--max-seed-count", type=int, default=10)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output-dir", default=str(Path(__file__).with_name("results")))
    parser.add_argument("--primary-local-model", default=DEFAULT_PRIMARY_LOCAL_MODEL)
    args = parser.parse_args()

    model_specs = build_default_model_specs(
        primary_local_model=args.primary_local_model,
    )
    strategy_specs = build_default_strategy_specs()
    task_specs = build_task_specs()

    if args.profile in {"debugging_replay_v1", "debugging_replay_smoke"}:
        replay_runner = run_debugging_replay_smoke if args.profile == "debugging_replay_smoke" else run_debugging_replay_v1
        results = replay_runner(
            output_dir=Path(args.output_dir),
            primary_local_model=args.primary_local_model,
            response_budget=args.response_budget,
        )
        print(json.dumps(results, indent=2))
        return

    selected_tasks = parse_csv(args.tasks)
    selected_models = parse_csv(args.models)
    selected_strategies = parse_csv(args.strategies)
    window_budgets = parse_int_csv(args.window_budgets)

    if args.profile == "proof":
        selected_tasks = selected_tasks or ["debugging"]
        selected_models = selected_models or ["qwen_local"]
        selected_strategies = selected_strategies or ["summary80", "barrier"]
        window_budgets = window_budgets or [args.window_budget or 3072]
        seed_count = 1
        max_seed_count = 1
    else:
        available_model_aliases = [model.alias for model in model_specs]
        available_strategy_names = [strategy.name for strategy in strategy_specs]
        selected_models = selected_models or available_model_aliases
        selected_strategies = selected_strategies or available_strategy_names
        selected_tasks = selected_tasks or [task.name for task in task_specs]
        window_budgets = window_budgets or list(DEFAULT_WINDOW_BUDGETS)
        seed_count = args.seed_count
        max_seed_count = args.max_seed_count

    runner = BenchmarkRunner(
        task_specs=task_specs,
        model_specs=model_specs,
        strategy_specs=strategy_specs,
        output_dir=Path(args.output_dir),
        response_budget=args.response_budget,
        overflow_ratio=args.overflow_ratio,
        initial_seed_count=seed_count,
        max_seed_count=max_seed_count,
        resume=args.resume,
    )
    results = runner.run(
        task_names=selected_tasks,
        model_names=selected_models,
        strategy_names=selected_strategies,
        window_budgets=window_budgets,
    )

    if args.profile == "proof":
        _write_proof_compatibility_artifacts(Path(args.output_dir), results)

    print(json.dumps(results, indent=2))


def _write_proof_compatibility_artifacts(output_dir: Path, results: dict) -> None:
    run_lookup = {
        (record["strategy"], record["seed"]): record
        for record in results["runs"]
    }
    summary80 = run_lookup.get(("summary80", 1))
    barrier = run_lookup.get(("barrier", 1))
    if summary80 is None or barrier is None:
        return

    if barrier["score"] > summary80["score"] and barrier["score"] == 1.0:
        conclusion = "success"
    elif barrier["score"] > summary80["score"]:
        conclusion = "partial"
    else:
        conclusion = "inconclusive"

    compatibility = {
        "model": summary80["model_name"],
        "window_budget": summary80["window_budget"],
        "response_budget": summary80["window_budget"] - summary80["usable_prompt_budget"],
        "session": {
            "profile": "proof",
            "source_prompt_tokens": summary80["session_metadata"].get("source_prompt_tokens", 0),
            "target_source_tokens": summary80["session_metadata"].get("target_source_tokens", 0),
            "source_overflow_ratio": summary80["session_metadata"].get("source_overflow_ratio", 0),
            "noise_turns": summary80["session_metadata"].get("noise_turns", 0),
            "noise_tokens_per_turn": summary80["session_metadata"].get("noise_tokens_per_turn", 0),
        },
        "conclusion": conclusion,
        "runs": {
            "summary80": {
                "strategy": "summary80",
                "final_response": summary80["final_response"],
                "score": {"score": summary80["score"], "found": summary80["found"], "missed": summary80["missed"]},
                "state": {"prompt_tokens": summary80["prompt_tokens"]},
                "protected_bug_report_selected": summary80["retained_anchor"],
            },
            "barrier": {
                "strategy": "barrier",
                "final_response": barrier["final_response"],
                "score": {"score": barrier["score"], "found": barrier["found"], "missed": barrier["missed"]},
                "state": {"prompt_tokens": barrier["prompt_tokens"]},
                "protected_bug_report_selected": barrier["retained_anchor"],
            },
        },
    }

    json_path = output_dir / "latest_run.json"
    md_path = output_dir / "latest_run.md"
    json_path.write_text(json.dumps(compatibility, indent=2))
    lines = [
        "# ContextGC Controlled-Budget Benchmark",
        "",
        f"- Model: `{compatibility['model']}`",
        f"- Profile: `{compatibility['session']['profile']}`",
        f"- Window budget: `{compatibility['window_budget']}`",
        f"- Conclusion: `{compatibility['conclusion']}`",
        "",
        "## Score Summary",
        "",
        "| Strategy | Recall | Anchor Selected | Final Prompt Tokens |",
        "|---|---:|---:|---:|",
        f"| Summary @80% | {summary80['score']:.0%} | {summary80['retained_anchor']} | {summary80['prompt_tokens']} |",
        f"| Barrier | {barrier['score']:.0%} | {barrier['retained_anchor']} | {barrier['prompt_tokens']} |",
        "",
        "## Summary80 Final Response",
        "",
        summary80["final_response"],
        "",
        "## Barrier Final Response",
        "",
        barrier["final_response"],
    ]
    md_path.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
