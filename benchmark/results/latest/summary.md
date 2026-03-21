# ContextGC Benchmark Matrix

- Runs: `240`
- Initial seeds: `5`
- Max seeds: `10`
- Response budget: `128`
- Audit queue size: `137`

## Aggregate Summary

| Task | Model | Window | Strategy | Mean | Secondary | Contam | Agree | Anchor | Distractor | Rescue | Delta vs Barrier | 95% CI | p-value |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|
| customer_support | qwen_local | 3072 | barrier | 0.693 | 0.507 | 10% | 0% | 100% | 100% | 40% | 0.000 | [0.000, 0.000] | 1.000 |
| customer_support | qwen_local | 3072 | score_only | 0.643 | 0.486 | 20% | 0% | 20% | 100% | 0% | 0.050 | [-0.086, 0.193] | 0.727 |
| customer_support | qwen_local | 3072 | summary80 | 0.593 | 0.421 | 10% | 0% | 0% | 100% | 0% | 0.100 | [0.021, 0.157] | 0.039 |
| customer_support | qwen_local | 16384 | barrier | 0.671 | 0.529 | 0% | 0% | 100% | 100% | 0% | 0.000 | [0.000, 0.000] | 1.000 |
| customer_support | qwen_local | 16384 | score_only | 0.743 | 0.600 | 0% | 0% | 100% | 100% | 0% | -0.071 | [-0.186, 0.029] | 0.453 |
| customer_support | qwen_local | 16384 | summary80 | 0.586 | 0.421 | 20% | 0% | 0% | 100% | 0% | 0.086 | [-0.036, 0.207] | 0.344 |
| debugging | qwen_local | 3072 | barrier | 0.907 | 0.907 | 10% | 100% | 100% | 100% | 40% | 0.000 | [0.000, 0.000] | 1.000 |
| debugging | qwen_local | 3072 | score_only | 0.850 | 0.850 | 10% | 100% | 40% | 100% | 0% | 0.057 | [0.014, 0.100] | 0.125 |
| debugging | qwen_local | 3072 | summary80 | 0.307 | 0.307 | 10% | 100% | 0% | 100% | 0% | 0.600 | [0.529, 0.679] | 0.002 |
| debugging | qwen_local | 16384 | barrier | 0.943 | 0.971 | 0% | 60% | 100% | 100% | 0% | 0.000 | [0.000, 0.000] | 1.000 |
| debugging | qwen_local | 16384 | score_only | 0.943 | 0.971 | 0% | 60% | 100% | 100% | 0% | 0.000 | [0.000, 0.000] | 1.000 |
| debugging | qwen_local | 16384 | summary80 | 0.379 | 0.379 | 30% | 100% | 0% | 100% | 0% | 0.564 | [0.507, 0.614] | 0.002 |
| document_qa | qwen_local | 3072 | barrier | 0.886 | 0.886 | 0% | 100% | 100% | 100% | 100% | 0.000 | [0.000, 0.000] | 1.000 |
| document_qa | qwen_local | 3072 | score_only | 0.714 | 0.714 | 0% | 100% | 0% | 100% | 0% | 0.171 | [0.086, 0.243] | 0.016 |
| document_qa | qwen_local | 3072 | summary80 | 0.607 | 0.607 | 10% | 100% | 0% | 100% | 0% | 0.279 | [0.193, 0.357] | 0.004 |
| document_qa | qwen_local | 16384 | barrier | 0.871 | 0.871 | 0% | 100% | 100% | 100% | 0% | 0.000 | [0.000, 0.000] | 1.000 |
| document_qa | qwen_local | 16384 | score_only | 0.900 | 0.900 | 0% | 100% | 100% | 100% | 0% | -0.029 | [-0.071, 0.000] | 0.500 |
| document_qa | qwen_local | 16384 | summary80 | 0.600 | 0.600 | 20% | 100% | 0% | 100% | 0% | 0.271 | [0.164, 0.371] | 0.008 |
| multi_step_coding | qwen_local | 3072 | barrier | 0.771 | 0.386 | 0% | 0% | 100% | 100% | 50% | 0.000 | [0.000, 0.000] | 1.000 |
| multi_step_coding | qwen_local | 3072 | score_only | 0.679 | 0.371 | 10% | 0% | 10% | 100% | 0% | 0.093 | [0.007, 0.179] | 0.219 |
| multi_step_coding | qwen_local | 3072 | summary80 | 0.514 | 0.286 | 40% | 0% | 0% | 100% | 0% | 0.257 | [0.179, 0.329] | 0.004 |
| multi_step_coding | qwen_local | 16384 | barrier | 0.800 | 0.400 | 0% | 0% | 100% | 100% | 0% | 0.000 | [0.000, 0.000] | 1.000 |
| multi_step_coding | qwen_local | 16384 | score_only | 0.800 | 0.414 | 0% | 0% | 100% | 100% | 0% | 0.000 | [-0.057, 0.057] | 1.000 |
| multi_step_coding | qwen_local | 16384 | summary80 | 0.664 | 0.400 | 10% | 0% | 0% | 100% | 0% | 0.136 | [0.043, 0.243] | 0.070 |

## Barrier Rescue Summary

- `customer_support` @ `3072`: rescue rate `40%`, anchor protected `100%`, scorer agreement `0%`
- `customer_support` @ `16384`: rescue rate `0%`, anchor protected `100%`, scorer agreement `0%`
- `debugging` @ `3072`: rescue rate `40%`, anchor protected `100%`, scorer agreement `100%`
- `debugging` @ `16384`: rescue rate `0%`, anchor protected `100%`, scorer agreement `60%`
- `document_qa` @ `3072`: rescue rate `100%`, anchor protected `100%`, scorer agreement `100%`
- `document_qa` @ `16384`: rescue rate `0%`, anchor protected `100%`, scorer agreement `100%`
- `multi_step_coding` @ `3072`: rescue rate `50%`, anchor protected `100%`, scorer agreement `0%`
- `multi_step_coding` @ `16384`: rescue rate `0%`, anchor protected `100%`, scorer agreement `0%`