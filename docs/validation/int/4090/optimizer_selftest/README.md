# ExpertAdamW self-test on the RTX 4090

`scripts/moe_precision/device_selftest.py --device cuda --time` at commit 1a8d7e912b166c373fcfec862ad7297221d33a71 (`commit.txt` says where that is recorded).
The verdict is `selftest_cuda_time.txt`: every check line reads `ok`, and its last line is the same result as JSON, with `"pass": true`.
