# ContextGC Debugging Replay Benchmark

- Profile: `debugging_replay_v1`
- Runs: `180`
- Transcripts: `20`
- Constrained windows: `3072,4096`
- Oracle window: `16384`
- Response budget: `128`
- Audit queue size: `27`

## Aggregate Summary

| Window | Strategy | Mean | Secondary | Root Cause | File Line | Remediation | Anchor | SrcIdx3 | Distractor | Contam | Agree | Oracle Gap | Delta vs Summary80 | 95% CI | p-value |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|
| 3072 | barrier | 0.875 | 0.868 | 80% | 90% | 95% | 100% | 100% | 100% | 5% | 85% | 0.089 | 0.389 | [0.279, 0.489] | 0.000 |
| 3072 | recency | 0.857 | 0.843 | 80% | 95% | 85% | 100% | 100% | 100% | 10% | 90% | 0.107 | 0.371 | [0.257, 0.486] | 0.000 |
| 3072 | score_only | 0.875 | 0.868 | 80% | 90% | 95% | 100% | 100% | 100% | 5% | 85% | 0.089 | 0.389 | [0.279, 0.489] | 0.000 |
| 3072 | summary80 | 0.486 | 0.486 | 50% | 5% | 45% | 0% | 0% | 100% | 5% | 100% | 0.479 | 0.000 | [0.000, 0.000] | 1.000 |
| 4096 | barrier | 0.886 | 0.864 | 80% | 95% | 100% | 100% | 100% | 100% | 0% | 85% | 0.079 | 0.414 | [0.286, 0.525] | 0.000 |
| 4096 | recency | 0.893 | 0.879 | 85% | 95% | 95% | 100% | 100% | 100% | 0% | 90% | 0.071 | 0.421 | [0.293, 0.529] | 0.000 |
| 4096 | score_only | 0.886 | 0.864 | 80% | 95% | 100% | 100% | 100% | 100% | 0% | 85% | 0.079 | 0.414 | [0.286, 0.525] | 0.000 |
| 4096 | summary80 | 0.471 | 0.471 | 50% | 5% | 45% | 0% | 0% | 100% | 15% | 100% | 0.493 | 0.000 | [0.000, 0.000] | 1.000 |
| 16384 | full_history | 0.964 | 0.943 | 100% | 100% | 100% | 100% | 100% | 100% | 0% | 85% | 0.000 | 0.000 | [0.000, 0.000] | 1.000 |

## Barrier vs Summary80

- `3072`: mean `0.875`, delta vs summary80 `0.389`, root-cause accuracy `80%`, contamination `5%`, agreement `85%`
- `4096`: mean `0.886`, delta vs summary80 `0.414`, root-cause accuracy `80%`, contamination `0%`, agreement `85%`