# ContextGC Barrier

ContextGC Barrier is a small research repo about one question:

When a conversation gets too long for the prompt window, which old messages should stay in the prompt?

The name is inspired by ZGC. The idea here is simple: be careful about what stays live when space is tight.

This repo compares three strategies:

- `summary80`
- `barrier`
- `summary80_barrier`

The focus is narrow: keep exact old facts available when the prompt window is tight.

## At A Glance

If you only want the short version:

| Strategy | What it does | Current replay result |
|---|---|---|
| `summary80` | Compress older context into a rolling summary. | Weak baseline under tight budgets. |
| `barrier` | Keep the most relevant older raw messages. | Best performer in the current debugging replay benchmark. |
| `summary80_barrier` | Summarize first, then add back protected raw exceptions. | Better than `summary80`, but not better than `barrier` on the current replay. |

## What Each Strategy Does

| Strategy | Simple description |
|---|---|
| `summary80` | When the prompt gets tight, older context is compressed into one rolling summary. |
| `barrier` | Older raw messages are scored, and the most useful ones stay in the prompt. |
| `summary80_barrier` | Older context is summarized first, then important raw `user` and `tool` messages are added back. |

## What The Benchmark Measures

There are two benchmark styles:

- `matrix`: live multi-turn tasks for debugging, document QA, coding, and support
- `debugging_replay`: a scripted debugging replay benchmark where every strategy sees the exact same frozen transcript

The main benchmark to read is `debugging_replay`.

Every strategy sees the same frozen debugging transcript. The model stays the same. The prompt budget stays the same. Only the retention strategy changes.

That makes it a clean comparison of context handling.

## Runtime Notes

The shared runtime is [ContextGCBarrier](contextgc_barrier/wrapper.py).

Valid strategy ids are:

- `summary80`
- `barrier`
- `summary80_barrier`

`barrier` is still the default.

For the summary strategies, `context_state()` also reports:

- `summary_active`
- `summarized_through_index`
- `summary_tokens`
- `protected_exception_indexes`

## Repository Layout

- [contextgc_barrier](contextgc_barrier): runtime selection, scoring, summary logic, local demo
- [benchmark](benchmark): task generation, matrix runner, replay runner, stats, CLI
- [tests](tests): unit and integration coverage
- [benchmark/results/debugging_replay](benchmark/results/debugging_replay): current benchmark output

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e ".[dev]"
```

For live local Qwen runs on Apple Silicon:

```bash
python -m pip install -e ".[mlx]"
```

Optional spaCy model:

```bash
python -m spacy download en_core_web_sm
```

## Quick Start

Run the local demo for one strategy:

```bash
./.venv/bin/python -m contextgc_barrier.demo --strategy barrier
```

Run the small proof profile:

```bash
./.venv/bin/python benchmark/run_benchmark.py \
  --profile proof \
  --window-budget 3072
```

Run the scripted debugging replay smoke benchmark:

```bash
./.venv/bin/python benchmark/run_benchmark.py \
  --profile debugging_replay_smoke \
  --output-dir benchmark/results/debugging_replay_smoke
```

Run the full scripted debugging replay benchmark:

```bash
./.venv/bin/python benchmark/run_benchmark.py \
  --profile debugging_replay \
  --output-dir benchmark/results/debugging_replay
```

Run the full task matrix:

```bash
./.venv/bin/python benchmark/run_benchmark.py \
  --profile matrix \
  --strategies summary80,barrier,summary80_barrier
```

## Replay Artifacts

The replay benchmark writes:

- `transcripts.jsonl`: the frozen scripted transcripts
- `runs.jsonl`: one run per transcript/window/strategy
- `aggregate.json` and `aggregate.csv`: grouped metrics
- `summary.md`: human-readable summary
- `audit_queue.jsonl`: rows flagged for manual review
- `progress.log`: corpus generation and evaluation progress

The frozen assistant turns can refer back to the earlier incident, but they do not restate the answer facts. That keeps the replay focused on retention.

## Current Result

Latest full run: [summary.md](benchmark/results/debugging_replay/summary.md)

This is the current benchmark to read. It uses `20` frozen debugging transcripts, `3` strategies, and `3` prompt windows.

| Window | `summary80` | `barrier` | `summary80_barrier` | Honest read |
|---:|---:|---:|---:|---|
| `3072` | `0.211` | `0.829` | `0.786` | `barrier` is clearly best. The hybrid beats summary but still trails raw retention. |
| `4096` | `0.200` | `0.818` | `0.793` | Same result. `barrier` stays ahead. |
| `16384` | `0.825` | `0.825` | `0.825` | Once the full transcript fits, the strategies tie. |

The paired comparisons in [summary.md](benchmark/results/debugging_replay/summary.md) support this reading:

| Comparison | `3072` | `4096` | What to say publicly |
|---|---|---|---|
| `barrier` vs `summary80` | delta `+0.618`, wins `20/20`, `p=0.000` | delta `+0.618`, wins `20/20`, `p=0.000` | `barrier` clearly beats `summary80` on this debugging replay benchmark. |
| `summary80_barrier` vs `barrier` | delta `-0.043`, `p=0.388` | delta `-0.025`, `p=1.000` | The hybrid does not beat plain `barrier` on this benchmark. |

The short takeaway is simple: keeping the right raw context beats replacing old context with a lossy rolling summary.

## Audit Notes

I checked the `24` flagged rows in [audit_queue.jsonl](benchmark/results/debugging_replay/audit_queue.jsonl) against [runs.jsonl](benchmark/results/debugging_replay/runs.jsonl).

| Audit finding | Count | What it means |
|---|---:|---|
| `contamination` | `17` | Mostly `summary80` pulling stale side-case facts under tight budgets. This strengthens the main benchmark story. |
| `scorer_disagreement` | `4` | Small parser misses, mainly around `remediation` or `file_line`, not a new benchmark failure mode. |
| `random_sample` | `3` | Spot-check rows chosen for manual review; no new issue pattern found. |

| Strategy | Flagged rows | Contamination | Scorer disagreement |
|---|---:|---:|---:|
| `summary80` | `15` | `14` | `1` |
| `barrier` | `6` | `2` | `2` |
| `summary80_barrier` | `3` | `1` | `1` |

The flagged rows do not change the main result. The only caveat is that a few rows still depend on scorer interpretation, so the claim should stay narrow and benchmark-specific.
