# ContextGC Proof Result

This is the main 20-turn local proof run.

- Model: `mlx-community/Qwen3.5-4B-OptiQ-4bit`
- Profile: `proof`
- Window budget: `3072`
- Response budget: `128`
- Conclusion: `success`

## Summary

| Strategy | Recall | Kept original bug report? |
|---|---:|---:|
| Recency | 29% | No |
| Barrier | 100% | Yes |

## In plain English

With the same prompt budget, the recency baseline lost the original bug report and invented wrong details from later noise.

The barrier strategy kept the original bug report alive and returned all seven critical facts in the final answer.
