# ExpertAdamW self-test on the RTX 4060 Laptop GPU (8 GB)

`scripts/moe_precision/device_selftest.py --device cuda --time` at commit 1a8d7e912b166c373fcfec862ad7297221d33a71 (`commit.txt`).
The verdict is `selftest_cuda.txt`: every check line reads `ok`; `selftest_last_line.json` is its last line, the same result as JSON, with `"pass": true`.
