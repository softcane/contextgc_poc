# ContextGC Barrier

ContextGC Barrier is a small Python proof-of-concept for a simple problem:

In long coding chats, the most important message is often the first one. It has the real bug report, the file path, the version number, and the exact symptom. When the prompt gets too big, most systems keep only the newest messages. That is how the useful stuff gets lost.

I spent years as a Java developer, so I kept thinking about garbage collection. In the JVM, some objects survive because something still points to them. This project applies the same idea to LLM context. If the model keeps referring back to an older message, that message should be harder to throw away.

![Minimal example output](assets/demo-output.svg)

## What it does

- Tracks when the model effectively cites older `user` or `tool` messages
- Keeps those cited messages alive when the next prompt would overflow
- Compares that strategy against simple recency-based truncation

## Two strategies

`recency` is the simple baseline.  
When the prompt gets too large, it keeps the newest messages and drops older ones first.

`barrier` is the ContextGC strategy.  
When the model keeps referring back to an older message, that message becomes harder to drop. So even if newer noisy messages arrive later, the original bug report can stay in the prompt.

In simple terms:
- `recency` = keep what is newest
- `barrier` = keep what is newest, plus older messages that are still clearly important

## Result

On the main 20-turn local Qwen3.5 proof benchmark, the simple `recency` baseline forgot the original bug report.

The `barrier` strategy kept it.

Under the same prompt budget:

- `recency` recall: `29%`
- `barrier` recall: `100%`

The short benchmark summary is in `benchmark/results/latest_run.md`.

## Try it

Quick demo, no model download:

```bash
python -m contextgc_barrier.demo
```

Real local benchmark:

```bash
pip install ".[mlx]"
python benchmark/run_benchmark.py --profile proof --window-budget 3072
```

Optional for better keyword extraction:

```bash
python -m spacy download en_core_web_sm
```
