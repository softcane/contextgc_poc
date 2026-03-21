# ContextGC Barrier

ContextGC Barrier is a small research prototype for long-context chat sessions.

The core problem is simple: when a conversation gets long, most systems compress or drop older messages. That often removes the exact detail that still matters, such as the original bug report, the real policy, the right file path, or the correct customer case note.

This repo tests whether a simple retention policy can do better than a rolling summary baseline under a fixed prompt budget.

![Minimal example output](assets/demo-output.svg)

## Problem

Long chat sessions usually fail in a boring way:

- the window fills up
- older messages are compressed or dropped
- the model loses the one detail that still matters

This project studies that failure mode under a fixed prompt budget.

Instead of keeping history by recency alone, the repo compares three ways to manage old context:

- summarize it
- score it and keep the best pieces
- score it and also protect context that the model has already relied on

## Strategies

The benchmark compares three strategies.

### `summary80`

This is the baseline that tries to mimic a common real setup.

How it works:

- the usable prompt budget is `window_budget - response_budget`
- when the live prompt reaches about `80%` of that usable budget, the oldest block is summarized with an extra LLM call from the same model
- that summary is capped at about `15%` of the usable prompt budget
- the prompt then contains:
  - the system prompt
  - one rolling summary message for older context
  - the recent unsummarized tail

So `summary80` trades exact old context for a shorter model-written note.

### `score_only`

This is the simpler selective-retention method.

It does not protect old messages just because they were used before. It just scores them and keeps the best ones that fit.

How it works:

- the system always keeps the system prompt
- it also keeps the latest user turn
- it tries to keep a few recent non-system messages as soft anchors
- for older messages, it builds keywords from the latest meaningful user request
- it scores each candidate message and fills the remaining budget with the highest-ranked ones

The score uses three signals:

- relevance overlap with the current task keywords
- recency decay by turn distance
- a small role weight

In code, the base score is:

- `0.40 * relevance`
- `0.35 * recency`
- `0.25 * role_weight`

So `score_only` is still simple, but it is not just recency truncation.

### `barrier`

This is the full method.

It starts from the same selection logic as `score_only`, then adds a write-barrier style protection step.

How it works:

- after each assistant reply, the system extracts keywords from that reply
- it compares those keywords against older `user` and `tool` messages
- if an older message looks like something the model just used, that message is marked as cited
- cited messages get protection and a score boost on later turns

The current implementation uses the default extractor pipeline from the repo, gives cited chunks an extra score boost of up to `0.45`, and orders protected messages ahead of normal candidates.

So the practical difference is:

- `summary80` replaces old raw context with a summary
- `score_only` keeps old raw context only when it still scores well
- `barrier` does the same thing, but tries harder to keep old context that the model already depended on

## How Prompt Selection Works

For `score_only` and `barrier`, prompt selection follows the same general pipeline:

1. Register new messages and chunk them.
2. Extract keywords from the latest meaningful user turn.
3. Score all stored chunks.
4. Always keep:
   - the system message
   - the latest user message
   - a few recent non-system messages when they still fit
5. Fill the remaining budget with the best older candidates.

The main difference is candidate ordering:

- `score_only` ranks by overlap, score, and recency
- `barrier` also prefers protected or cited messages

If a message has no task overlap and is not protected, the barrier path does not try to re-add it when space is tight.

## What the Benchmark Is Testing

This benchmark asks a simple question:

When the prompt window is tight, which policy keeps the right old information alive?

More specifically, it compares three strategies on the same seeded conversations:

- `summary80`: summarize old context
- `score_only`: keep the older messages that score as most relevant
- `barrier`: do the same scoring, but also protect messages the model seems to have already used

Each run uses:

- one task type
- one window budget
- one random seed
- one strategy

The seed matters because it fixes the exact transcript for that condition. That means `summary80`, `score_only`, and `barrier` all see the same base conversation for the same seed. Only the context-management strategy changes.

Current benchmark scope:

- model: local Qwen 3.5 on MLX
- tasks: debugging, document Q&A, multi-step coding, customer support
- window budgets: `3072` and `16384`
- strategies: `summary80`, `score_only`, `barrier`
- decoding: deterministic (`temperature = 0`)
- replication: start at `5` seeds and extend to `10` when the result is still unstable

The task templates are built so old information can realistically be lost:

- key facts are placed early in the conversation
- some of those facts are split across multiple messages
- later distractor messages use similar domain words and similar entity types
- the final user request asks for the real answer indirectly, not by copying the original field names

For each run, the benchmark also records:

- whether the real early anchor messages were still in the final prompt
- whether distractor messages were still in the final prompt
- which messages were selected
- whether the barrier protected any earlier message
- whether barrier kept anything that `score_only` would have dropped

## How Evaluation Works

Each task comes with a small fact rubric. You can think of it as the answer key.

Example:

- a debugging task may require the real function name, trigger condition, version, file location, root cause, and fix
- a customer support task may require the real order id, defect, refund policy, and shipping details

When the model gives its final answer, the benchmark checks each fact and labels it as:

- `correct`: the answer contains the right value
- `wrong`: the answer contains a distractor or conflicting value
- `missing`: the answer does not contain the fact clearly enough

The official score is deterministic. It is computed from the primary scorer as:

- `score = (correct_count - 0.5 * wrong_count) / total_fact_count`

So:

- correct facts increase the score
- wrong facts are penalized
- missing facts get no credit

The benchmark also runs a second independent scorer on the same answer. This gives:

- `official score`: the primary score used in the tables
- `secondary score`: an independent cross-check
- `scorer agreement`: whether both scorers reached the same fact labels

The reason for the second scorer is simple: if both scoring methods agree, the result is more trustworthy. If they disagree, that run needs attention.

Each run records at least these evaluation fields:

- official score
- secondary score
- per-fact status
- contamination count
- scorer agreement
- anchor retention
- distractor retention
- barrier rescue metadata

Rows with scorer disagreement or contamination are written to the blinded audit queue for manual review.

## Repository Layout

- [contextgc_barrier](contextgc_barrier): barrier logic, scoring, extraction, backend wrappers
- [benchmark](benchmark): task generation, strategies, runner, stats, CLI
- [tests](tests): unit and integration coverage
- [spec.md](spec.md): plain-English project framing

## Setup

Create a virtual environment and install the project:

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

Optional spaCy model for stronger extraction:

```bash
python -m spacy download en_core_web_sm
```

## Quick Start

Run the small local demo:

```bash
python -m contextgc_barrier.demo
```

Run the one-seed proof profile:

```bash
./.venv/bin/python benchmark/run_benchmark.py \
  --profile proof \
  --window-budget 3072
```

Run the strict debugging replay smoke check:

```bash
./.venv/bin/python benchmark/run_benchmark.py \
  --profile debugging_replay_smoke \
  --output-dir benchmark/results/debugging_replay_smoke
```

Run the full debugging replay benchmark:

```bash
./.venv/bin/python benchmark/run_benchmark.py \
  --profile debugging_replay_v1 \
  --output-dir benchmark/results/debugging_replay_v1
```

Run the full matrix with the default reviewer-focused scope:

```bash
./.venv/bin/python benchmark/run_benchmark.py \
  --profile matrix \
  --tasks debugging,document_qa,multi_step_coding,customer_support \
  --models qwen_local \
  --strategies summary80,score_only,barrier \
  --window-budgets 3072,16384 \
  --seed-count 5 \
  --max-seed-count 10 \
  --output-dir benchmark/results/<run-name>
```

Resume an interrupted run:

```bash
./.venv/bin/python benchmark/run_benchmark.py \
  --profile matrix \
  --tasks debugging,document_qa,multi_step_coding,customer_support \
  --models qwen_local \
  --strategies summary80,score_only,barrier \
  --window-budgets 3072,16384 \
  --seed-count 5 \
  --max-seed-count 10 \
  --output-dir benchmark/results/<run-name> \
  --resume
```

## Output Artifacts

Each matrix run writes artifacts under the chosen output directory:

- `runs.jsonl`: one row per run
- `aggregate.json`: aggregate metrics per task/window/strategy
- `aggregate.csv`: spreadsheet-friendly aggregate table
- `summary.md`: human-readable benchmark summary
- `audit_queue.jsonl`: blinded rows that need manual review

The proof profile also writes:

- `latest_run.json`
- `latest_run.md`

The replay profiles also write:

- `transcripts.jsonl`: the frozen replay corpus used for all strategies
- `progress.log`: corpus-building and evaluation progress

## Results

There are now two useful result sets in the repo:

- a broad older matrix across four task families
- a stricter newer debugging replay benchmark that replays the same frozen transcript to every strategy

### Latest strict replay benchmark

Artifacts:

- [aggregate.json](benchmark/results/debugging_replay_v1_after_fix_v2/aggregate.json)
- [aggregate.csv](benchmark/results/debugging_replay_v1_after_fix_v2/aggregate.csv)
- [summary.md](benchmark/results/debugging_replay_v1_after_fix_v2/summary.md)
- [audit_queue.jsonl](benchmark/results/debugging_replay_v1_after_fix_v2/audit_queue.jsonl)

Simple English summary:

- `barrier` clearly beats `summary80` on the strict debugging replay benchmark
- at `3072`, `barrier` scores `0.875` vs `summary80` `0.486`
- at `4096`, `barrier` scores `0.886` vs `summary80` `0.471`
- in matched pairs, `barrier` beats `summary80` `18` times, ties `2`, and loses `0` at both windows
- `full_history` at `16384` is the ceiling at `0.964`

The more important interpretation is:

- this benchmark strongly supports `raw relevant context > rolling summary`
- it does **not** yet prove that the write barrier itself is the reason
- `score_only` ties `barrier` on this replay benchmark
- `recency` is also very close and is slightly better at `4096`

So the safest claim is:

- the weak baseline is `summary80`
- the main win comes from keeping the right raw context alive
- the extra value of write-barrier protection still needs stronger proof

One caveat still matters:

- the audit queue is smaller than before, but scorer agreement is still only about `85%` to `90%` for the raw-context strategies
- most disagreements are secondary-parser misses on `root_cause`, not major benchmark failures
- that means the result is useful, but the scorer should still be hardened before using these numbers in a polished public claim

### Earlier broad matrix

Artifacts:

- [aggregate.json](benchmark/results/latest/aggregate.json)
- [aggregate.csv](benchmark/results/latest/aggregate.csv)
- [summary.md](benchmark/results/latest/summary.md)
- [audit_queue.jsonl](benchmark/results/latest/audit_queue.jsonl)

This older matrix is still useful for exploration across debugging, document Q&A, multi-step coding, and customer support, but the stricter replay benchmark above is the better proof test for the core question.


## Notes

- This repo is still a research harness, not a production memory system.
- The strongest honest claim should follow the benchmark outcome.
- If `barrier > score_only`, citation protection adds value.
- If `barrier ~= score_only`, scoring is the main win.
- If `barrier < score_only`, the simpler method is better in this setup.
