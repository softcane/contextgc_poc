# ContextGC Debugging Replay Benchmark

- Profile: `debugging_replay`
- Runs: `180`
- Transcripts: `20`
- Windows: `3072,4096,16384`
- Response budget: `128`
- Audit queue size: `24`

## Aggregate Summary

| Window | Strategy | Mean | Secondary | Root Cause | File Line | Remediation | Anchor | SrcIdx3 | Distractor | Contam | Agree | Delta vs Summary80 | 95% CI | p-value |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|
| 3072 | barrier | 0.829 | 0.821 | 35% | 90% | 90% | 100% | 100% | 100% | 0% | 95% | 0.618 | [0.532, 0.707] | 0.000 |
| 3072 | summary80 | 0.211 | 0.211 | 0% | 0% | 0% | 0% | 0% | 100% | 30% | 100% | 0.000 | [0.000, 0.000] | 1.000 |
| 3072 | summary80_barrier | 0.786 | 0.786 | 55% | 95% | 40% | 50% | 100% | 100% | 0% | 100% | 0.575 | [0.507, 0.646] | 0.000 |
| 4096 | barrier | 0.818 | 0.818 | 45% | 95% | 65% | 100% | 100% | 100% | 5% | 100% | 0.618 | [0.536, 0.707] | 0.000 |
| 4096 | summary80 | 0.200 | 0.200 | 0% | 0% | 0% | 0% | 0% | 100% | 35% | 100% | 0.000 | [0.000, 0.000] | 1.000 |
| 4096 | summary80_barrier | 0.793 | 0.793 | 60% | 95% | 40% | 50% | 100% | 100% | 0% | 100% | 0.593 | [0.521, 0.664] | 0.000 |
| 16384 | barrier | 0.825 | 0.832 | 45% | 70% | 85% | 100% | 100% | 100% | 5% | 95% | 0.000 | [0.000, 0.000] | 1.000 |
| 16384 | summary80 | 0.825 | 0.832 | 45% | 70% | 85% | 100% | 100% | 100% | 5% | 95% | 0.000 | [0.000, 0.000] | 1.000 |
| 16384 | summary80_barrier | 0.825 | 0.832 | 45% | 70% | 85% | 100% | 100% | 100% | 5% | 95% | 0.000 | [0.000, 0.000] | 1.000 |

## Barrier vs Summary80

- `debugging_replay` @ `3072`: `barrier` mean `0.829` vs `summary80` `0.211` (delta `0.618`, 95% CI `[0.536, 0.711]`, p `0.000`, wins/ties/losses `20/0/0`), anchor `100%`, contamination `0%`, agreement `95%`
- `debugging_replay` @ `4096`: `barrier` mean `0.818` vs `summary80` `0.200` (delta `0.618`, 95% CI `[0.536, 0.711]`, p `0.000`, wins/ties/losses `20/0/0`), anchor `100%`, contamination `5%`, agreement `100%`
- `debugging_replay` @ `16384`: `barrier` mean `0.825` vs `summary80` `0.825` (delta `0.000`, 95% CI `[0.000, 0.000]`, p `1.000`, wins/ties/losses `0/20/0`), anchor `100%`, contamination `5%`, agreement `95%`

## Summary80 + Barrier vs Barrier

- `debugging_replay` @ `3072`: `summary80_barrier` mean `0.786` vs `barrier` `0.829` (delta `-0.043`, 95% CI `[-0.107, 0.029]`, p `0.388`, wins/ties/losses `4/8/8`), anchor `50%`, contamination `0%`, agreement `100%`
- `debugging_replay` @ `4096`: `summary80_barrier` mean `0.793` vs `barrier` `0.818` (delta `-0.025`, 95% CI `[-0.100, 0.043]`, p `1.000`, wins/ties/losses `4/12/4`), anchor `50%`, contamination `0%`, agreement `100%`
- `debugging_replay` @ `16384`: `summary80_barrier` mean `0.825` vs `barrier` `0.825` (delta `0.000`, 95% CI `[0.000, 0.000]`, p `1.000`, wins/ties/losses `0/20/0`), anchor `100%`, contamination `5%`, agreement `95%`