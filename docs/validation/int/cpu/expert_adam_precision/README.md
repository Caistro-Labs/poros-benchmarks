# The expert adapters' optimizer, studied on the CPU

`scripts/expert_adam_precision.py` and `scripts/moe_precision/metal_exact_arith.py` at commit 1a8d7e912b166c373fcfec862ad7297221d33a71 (`commit.txt`; each command is in `command.txt`).
The verdicts are `summary_2000.json` (the study: each arm against the fp32 reference) and `metal_exact_arith.json` (the Metal kernel's arithmetic against the correctly rounded results); each has its stdout beside it.
