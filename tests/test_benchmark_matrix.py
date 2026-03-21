from pathlib import Path

import pytest

from benchmark.replay import (
    aggregate_debugging_replay_results,
    generate_scripted_debugging_replay_corpus,
    run_debugging_replay,
    run_debugging_replay_smoke,
)
from benchmark.runner import BenchmarkRunner, build_default_model_specs, build_default_strategy_specs
from benchmark.specs import FactSpec, ModelSpec, RunResult, TaskInstance, TaskSpec, build_phrase_pattern, normalize_value
from benchmark.tasks import (
    DEBUGGING_VALUES,
    DEBUGGING_REPLAY_TARGET_TOKENS,
    LEAKAGE_THRESHOLD,
    REPLAY_LEAKAGE_THRESHOLD,
    _debugging_final_prompt_variants,
    _choose_prompt_variant,
    build_debugging_replay_seed_task,
    build_task_specs,
)
from contextgc_barrier.backend import make_response
from contextgc_barrier.summary import ensure_summary_fit
from tests.fake_backend import FakeBackend


def _fake_model_specs():
    return [
        ModelSpec(
            alias="primary_fake",
            provider="fake",
            model_name="fake",
            window_limit=16_384,
            backend_factory=FakeBackend,
            tags=("local", "primary_local"),
        ),
    ]


def _minimal_run_result(**overrides):
    base = dict(
        task="win_task",
        model="primary_fake",
        model_name="fake",
        provider="fake",
        strategy="summary80",
        window_budget=3072,
        seed=1,
        score=0.0,
        secondary_score=0.0,
        found=[],
        wrong=[],
        missing=[],
        missed=[],
        fact_results=[],
        contamination=[],
        contamination_count=0,
        scorer_agreement=True,
        final_response="",
        prompt_tokens=10,
        usable_prompt_budget=100,
        selected_indexes=[],
        protected_selected_indexes=[],
        cited_selected_indexes=[],
        protected_message_indexes=[],
        cited_message_indexes=[],
        retained_anchor=False,
        retained_distractor=False,
        anchor_protected=False,
        turn_count=1,
        session_metadata={"anchor_indexes": [1], "distractor_indexes": [5], "seed": 1},
        strategy_metadata={},
        final_prompt_anchor_overlap=0.01,
        final_prompt_distractor_overlap=0.01,
    )
    base.update(overrides)
    return RunResult(**base)


def test_task_generation_is_seed_stable_across_builds():
    backend = FakeBackend()
    task_spec = build_task_specs()[0]
    task_a = task_spec.build(backend=backend, model="fake", window_budget=3072, overflow_ratio=1.15, seed=3)
    task_b = task_spec.build(backend=backend, model="fake", window_budget=3072, overflow_ratio=1.15, seed=3)

    assert task_a.turns == task_b.turns
    assert [fact.canonical_value for fact in task_a.facts] == [fact.canonical_value for fact in task_b.facts]


def test_all_default_tasks_start_with_user_turn_and_pass_leakage_check():
    backend = FakeBackend()
    for task_spec in build_task_specs():
        task = task_spec.build(backend=backend, model="fake", window_budget=3072, overflow_ratio=1.15, seed=1)
        assert task.turns[0]["role"] == "user"
        assert task.metadata["final_prompt_anchor_overlap"] < LEAKAGE_THRESHOLD
        assert task.metadata["final_prompt_distractor_overlap"] < LEAKAGE_THRESHOLD


def test_matrix_runner_writes_three_strategy_outputs(tmp_path: Path):
    runner = BenchmarkRunner(
        task_specs=build_task_specs()[:1],
        model_specs=_fake_model_specs()[:1],
        strategy_specs=build_default_strategy_specs(),
        output_dir=tmp_path,
        initial_seed_count=2,
        max_seed_count=2,
        response_budget=40,
    )

    results = runner.run(
        task_names=["debugging"],
        model_names=["primary_fake"],
        strategy_names=["summary80", "barrier", "summary80_barrier"],
        window_budgets=[3072],
    )

    assert tmp_path.joinpath("runs.jsonl").exists()
    assert tmp_path.joinpath("aggregate.json").exists()
    assert tmp_path.joinpath("aggregate.csv").exists()
    assert tmp_path.joinpath("summary.md").exists()
    assert tmp_path.joinpath("audit_queue.jsonl").exists()
    assert results["aggregates"]
    first = results["aggregates"][0]
    assert {
        "mean_score",
        "mean_secondary_score",
        "n",
        "ci_low",
        "ci_high",
        "p_value",
        "contamination_rate",
        "scorer_agreement_rate",
        "delta_vs_summary80",
    } <= set(first.keys())
    assert "barrier_rescue_rate" not in first
    summary_text = tmp_path.joinpath("summary.md").read_text()
    assert "## Barrier vs Summary80" in summary_text
    assert "## Summary80 + Barrier vs Barrier" in summary_text
    assert "wins/ties/losses" in summary_text


def test_adaptive_extension_promotes_group_when_barrier_pair_is_inconclusive(tmp_path: Path):
    task_spec = TaskSpec(
        name="win_task",
        description="Barrier beats the baseline, but not yet conclusively after five seeds.",
        builder=lambda backend, model, window_budget, overflow_ratio, seed: TaskInstance(
            name="win_task",
            system_prompt="Return labels.",
            turns=({"role": "user", "content": "placeholder"},),
            facts=(),
            metadata={"anchor_indexes": [1], "seed": seed},
        ),
    )
    runner = BenchmarkRunner(
        task_specs=[task_spec],
        model_specs=_fake_model_specs()[:1],
        strategy_specs=build_default_strategy_specs(),
        output_dir=tmp_path,
        initial_seed_count=5,
        max_seed_count=10,
        response_budget=20,
    )

    for seed in range(1, 6):
        runner._results.append(_minimal_run_result(seed=seed, strategy="summary80", score=0.0))
        runner._results.append(_minimal_run_result(seed=seed, strategy="barrier", score=1.0, found=["fact"]))
        runner._results.append(_minimal_run_result(seed=seed, strategy="summary80_barrier", score=1.0, found=["fact"]))

    extension_groups = runner._groups_requiring_extension(
        selected_tasks={"win_task": task_spec},
        selected_models={"primary_fake": _fake_model_specs()[0]},
        selected_strategies={strategy.name: strategy for strategy in build_default_strategy_specs()},
        window_budgets=[3072],
    )

    assert ("win_task", "primary_fake", 3072) in extension_groups


class _SummaryBackend:
    def count_tokens(self, messages, model):
        return sum(len(str(message.get("content", "")).split()) + 4 for message in messages)

    def create(self, model, messages, **kwargs):
        return make_response("Rolling summary:\n- preserved anchor details")


def test_summary80_uses_additional_summary_before_dropping_tail_messages():
    backend = _SummaryBackend()
    history = [
        {"index": index, "role": "user" if index % 2 else "assistant", "content": f"message {index} " + "token " * 18}
        for index in range(1, 9)
    ]

    state = ensure_summary_fit(
        backend=backend,
        model_name="fake",
        system_prompt="Summarize faithfully.",
        history=history,
        window_budget=80,
        response_budget=20,
        summary_trigger=10_000,
        summary_cap=40,
        prior_state=None,
    )

    assert state.summary_message is not None
    assert state.summarized_through_index > 0
    prompt_messages = [{"role": "system", "content": "Summarize faithfully."}, state.summary_message]
    prompt_messages.extend({"role": message["role"], "content": message["content"]} for message in state.tail_messages)
    assert backend.count_tokens(prompt_messages, model="fake") + 20 <= 80


def test_default_matrix_scope_is_qwen_vs_three_strategies():
    model_specs = build_default_model_specs()
    strategy_specs = build_default_strategy_specs()

    assert [model.alias for model in model_specs] == ["qwen_local"]
    assert [strategy.name for strategy in strategy_specs] == ["summary80", "barrier", "summary80_barrier"]


def test_non_resume_run_clears_existing_artifacts(tmp_path: Path):
    runs_path = tmp_path / "runs.jsonl"
    runs_path.write_text('{"stale": true}\n')
    (tmp_path / "aggregate.json").write_text("{}")
    (tmp_path / "aggregate.csv").write_text("stale")
    (tmp_path / "summary.md").write_text("stale")
    (tmp_path / "audit_queue.jsonl").write_text("stale")

    BenchmarkRunner(
        task_specs=build_task_specs()[:1],
        model_specs=_fake_model_specs()[:1],
        strategy_specs=build_default_strategy_specs(),
        output_dir=tmp_path,
        initial_seed_count=1,
        max_seed_count=1,
        response_budget=40,
        resume=False,
    )

    assert not runs_path.exists()
    assert not tmp_path.joinpath("aggregate.json").exists()
    assert not tmp_path.joinpath("aggregate.csv").exists()
    assert not tmp_path.joinpath("summary.md").exists()
    assert not tmp_path.joinpath("audit_queue.jsonl").exists()


def test_scoring_marks_distractor_values_wrong():
    task = TaskInstance(
        name="scoring",
        system_prompt="",
        turns=(),
        facts=(
            FactSpec(
                name="policy_name",
                canonical_value="Data Export Exception Policy",
                wrong_aliases=("Restricted Dataset Release Policy",),
                parser_id="phrase",
            ),
        ),
    )

    score = task.score_response("The controlling rule is Restricted Dataset Release Policy.")
    assert score["wrong"] == ["policy_name"]
    assert score["found"] == []
    assert score["contamination_count"] == 1


def test_normalizers_handle_dates_ids_and_paths():
    assert normalize_value("September 15, 2025", "date") == "2025-09-15"
    assert normalize_value("form-ex17", "identifier") == "FORM-EX17"
    assert normalize_value("src/retry_worker.py line 188", "path_line") == "src/retry_worker.py:188"


def test_debugging_root_cause_retry_buffer_identifier_alias_scores_correct():
    value = "retry buffer keeps the warmed arrays pinned in the numpy cache"
    task = TaskInstance(
        name="debugging",
        system_prompt="",
        turns=(),
        facts=(
            FactSpec(
                name="root_cause",
                canonical_value=value,
                allowed_aliases=tuple(alias for alias in DEBUGGING_VALUES["root_cause"][value] if alias != value),
                wrong_aliases=DEBUGGING_VALUES["root_cause"]["numpy cache retains the input array after warmup"],
                parser_id="phrase",
                normalizer_id="phrase",
            ),
        ),
    )

    score = task.score_response("The real issue was `retry_buffer` pinning warmed arrays in the numpy cache.")
    assert score["score"] == 1.0
    assert score["wrong"] == []


def test_task_construction_rejects_overly_keyword_aligned_prompt():
    with pytest.raises(ValueError):
        _choose_prompt_variant(
            seed=1,
            anchor_values=["process_batch()", "v2.3.1"],
            distractor_values=["flush_backlog()", "v2.3.4"],
            prompt_variants=("Use process_batch() on v2.3.1 in the final note.",),
        )


def test_debugging_replay_seed_task_is_fixed_size_and_uses_tighter_leakage():
    backend = FakeBackend()
    task_a = build_debugging_replay_seed_task(
        backend=backend,
        model="fake",
        source_seed=2,
        template_name="incident_closeout",
        target_total_tokens=DEBUGGING_REPLAY_TARGET_TOKENS,
    )
    task_b = build_debugging_replay_seed_task(
        backend=backend,
        model="fake",
        source_seed=2,
        template_name="incident_closeout",
        target_total_tokens=DEBUGGING_REPLAY_TARGET_TOKENS,
    )

    assert task_a.turns == task_b.turns
    assert task_a.metadata["final_prompt_anchor_overlap"] <= REPLAY_LEAKAGE_THRESHOLD
    assert task_a.metadata["final_prompt_distractor_overlap"] <= REPLAY_LEAKAGE_THRESHOLD


def test_root_cause_recap_prompts_force_short_plain_text_output():
    variants = _debugging_final_prompt_variants("root_cause_recap")
    assert all("plain-text sentence" in variant for variant in variants)
    assert any("no heading or bullets" in variant for variant in variants)
    assert any("no markdown" in variant for variant in variants)


def test_generate_scripted_debugging_replay_corpus_produces_reusable_transcripts():
    model = _fake_model_specs()[0]
    backend = model.backend_factory()

    transcripts = generate_scripted_debugging_replay_corpus(
        backend=backend,
        model=model,
        accepted_per_template=1,
        target_total_tokens=900,
        target_token_tolerance=300,
    )

    assert len(transcripts) == 2
    assert [transcript.metadata["seed"] for transcript in transcripts] == [1, 2]
    for transcript in transcripts:
        assert transcript.messages[transcript.final_user_index]["role"] == "user"
        assert transcript.messages[transcript.final_user_index - 1]["role"] == "assistant"
        assert abs(transcript.metadata["transcript_prompt_tokens"] - 900) <= 300
        assert transcript.metadata["scripted_replay"] is True


def test_scripted_debugging_replay_assistant_replies_do_not_leak_scored_aliases():
    model = _fake_model_specs()[0]
    backend = model.backend_factory()

    transcripts = generate_scripted_debugging_replay_corpus(
        backend=backend,
        model=model,
        accepted_per_template=1,
        target_total_tokens=900,
        target_token_tolerance=300,
    )

    for transcript in transcripts:
        assistant_messages = [message["content"] for message in transcript.messages if message["role"] == "assistant"]
        for assistant_text in assistant_messages:
            for fact in transcript.facts:
                for alias in (fact.canonical_value, *fact.allowed_aliases):
                    assert not build_phrase_pattern(alias).search(assistant_text)


def test_debugging_replay_aggregate_tracks_fact_accuracy_and_deltas():
    runs = [
        _minimal_run_result(
            task="debugging_replay",
            strategy="summary80",
            window_budget=3072,
            seed=1,
            score=0.3,
            fact_results=[
                {"name": "root_cause", "status": "missing"},
                {"name": "file_line", "status": "correct"},
                {"name": "remediation", "status": "missing"},
            ],
            selected_indexes=[5],
        ),
        _minimal_run_result(
            task="debugging_replay",
            strategy="barrier",
            window_budget=3072,
            seed=1,
            score=0.8,
            fact_results=[
                {"name": "root_cause", "status": "correct"},
                {"name": "file_line", "status": "correct"},
                {"name": "remediation", "status": "correct"},
            ],
            selected_indexes=[3, 5],
        ),
        _minimal_run_result(
            task="debugging_replay",
            strategy="summary80_barrier",
            window_budget=3072,
            seed=1,
            score=0.9,
            fact_results=[
                {"name": "root_cause", "status": "correct"},
                {"name": "file_line", "status": "correct"},
                {"name": "remediation", "status": "correct"},
            ],
            selected_indexes=[1, 3, 5],
        ),
    ]

    aggregates = aggregate_debugging_replay_results(runs)
    barrier = next(record for record in aggregates if record["strategy"] == "barrier")
    hybrid = next(record for record in aggregates if record["strategy"] == "summary80_barrier")
    summary = next(record for record in aggregates if record["strategy"] == "summary80")

    assert barrier["root_cause_accuracy"] == 1.0
    assert barrier["source_index_3_selected_rate"] == 1.0
    assert barrier["delta_vs_summary80"] == pytest.approx(0.5)
    assert hybrid["delta_vs_summary80"] == pytest.approx(0.6)
    assert summary["root_cause_accuracy"] == 0.0


def test_debugging_replay_profile_writes_expected_artifacts(tmp_path: Path):
    results = run_debugging_replay(
        output_dir=tmp_path,
        primary_local_model="fake",
        response_budget=40,
        accepted_per_template=1,
        target_total_tokens=900,
        target_token_tolerance=300,
        window_budgets=(240, 360),
        model_specs=_fake_model_specs(),
    )

    assert tmp_path.joinpath("transcripts.jsonl").exists()
    assert tmp_path.joinpath("runs.jsonl").exists()
    assert tmp_path.joinpath("aggregate.json").exists()
    assert tmp_path.joinpath("aggregate.csv").exists()
    assert tmp_path.joinpath("summary.md").exists()
    assert tmp_path.joinpath("audit_queue.jsonl").exists()
    assert tmp_path.joinpath("progress.log").exists()
    assert len(results["runs"]) == 12
    assert len(results["aggregates"]) == 6
    assert results["profile"] == "debugging_replay"
    progress_log = tmp_path.joinpath("progress.log").read_text()
    assert "status=accepted" in progress_log
    assert "status=completed" in progress_log
    assert "wins/ties/losses" in tmp_path.joinpath("summary.md").read_text()
    audit_record = next(iter(tmp_path.joinpath("audit_queue.jsonl").read_text().splitlines()), None)
    assert audit_record is not None
    assert '"strategy":' in audit_record

    transcript_tokens_by_seed = {}
    for run in results["runs"]:
        transcript_tokens_by_seed.setdefault(run["seed"], set()).add(run["session_metadata"]["transcript_prompt_tokens"])
    assert all(len(values) == 1 for values in transcript_tokens_by_seed.values())


def test_debugging_replay_smoke_profile_uses_smoke_defaults(tmp_path: Path):
    results = run_debugging_replay_smoke(
        output_dir=tmp_path,
        primary_local_model="fake",
        response_budget=40,
        target_total_tokens=4000,
        target_token_tolerance=400,
        model_specs=_fake_model_specs(),
    )

    assert results["profile"] == "debugging_replay_smoke"
    assert len(results["runs"]) == 6
    assert len(results["aggregates"]) == 3
    summary_text = tmp_path.joinpath("summary.md").read_text()
    assert "- Profile: `debugging_replay_smoke`" in summary_text
