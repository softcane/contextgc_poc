# ContextGC Debugging Replay Benchmark

- Profile: `debugging_replay`
- Runs: `180`
- Transcripts: `20`
- Windows: `3072,4096,16384`
- Response budget: `128`
- Audit queue size: `24`

## Read This First

- `barrier` beats `summary80` at `3072` and `4096`.
- `summary80_barrier` is better than `summary80`, but it does not beat `barrier`.
- At `16384`, all three strategies tie because the full transcript fits.

## Aggregate Summary

| Window | Strategy | Mean | Delta vs Summary80 | 95% CI | p-value |
|---:|---|---:|---:|---|---:|
| 3072 | barrier | 0.829 | 0.618 | [0.532, 0.707] | 0.000 |
| 3072 | summary80 | 0.211 | 0.000 | [0.000, 0.000] | 1.000 |
| 3072 | summary80_barrier | 0.786 | 0.575 | [0.507, 0.646] | 0.000 |
| 4096 | barrier | 0.818 | 0.618 | [0.536, 0.707] | 0.000 |
| 4096 | summary80 | 0.200 | 0.000 | [0.000, 0.000] | 1.000 |
| 4096 | summary80_barrier | 0.793 | 0.593 | [0.521, 0.664] | 0.000 |
| 16384 | barrier | 0.825 | 0.000 | [0.000, 0.000] | 1.000 |
| 16384 | summary80 | 0.825 | 0.000 | [0.000, 0.000] | 1.000 |
| 16384 | summary80_barrier | 0.825 | 0.000 | [0.000, 0.000] | 1.000 |

## Barrier vs Summary80

- `debugging_replay` @ `3072`: `barrier` mean `0.829` vs `summary80` `0.211` (delta `0.618`, 95% CI `[0.536, 0.711]`, p `0.000`, wins/ties/losses `20/0/0`)
- `debugging_replay` @ `4096`: `barrier` mean `0.818` vs `summary80` `0.200` (delta `0.618`, 95% CI `[0.536, 0.711]`, p `0.000`, wins/ties/losses `20/0/0`)
- `debugging_replay` @ `16384`: `barrier` mean `0.825` vs `summary80` `0.825` (delta `0.000`, 95% CI `[0.000, 0.000]`, p `1.000`, wins/ties/losses `0/20/0`)

## Summary80 + Barrier vs Barrier

- `debugging_replay` @ `3072`: `summary80_barrier` mean `0.786` vs `barrier` `0.829` (delta `-0.043`, 95% CI `[-0.107, 0.029]`, p `0.388`, wins/ties/losses `4/8/8`)
- `debugging_replay` @ `4096`: `summary80_barrier` mean `0.793` vs `barrier` `0.818` (delta `-0.025`, 95% CI `[-0.100, 0.043]`, p `1.000`, wins/ties/losses `4/12/4`)
- `debugging_replay` @ `16384`: `summary80_barrier` mean `0.825` vs `barrier` `0.825` (delta `0.000`, 95% CI `[0.000, 0.000]`, p `1.000`, wins/ties/losses `0/20/0`)