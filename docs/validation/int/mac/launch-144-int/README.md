# Qwen2.5-72B-Instruct on the Mac, 144 steps from the internal SSD

`poros train Qwen/Qwen2.5-72B-Instruct` for 144 steps on an M4 Mac with 16 GB, rows of 128 tokens, its 4-bit copy read from the Mac's internal SSD (`model-drive.txt`), at commit 1a8d7e912b166c373fcfec862ad7297221d33a71 (`commit.txt`).
The results are `receipt.json` and `events.jsonl` (the run, with its held-out loss every 25 steps) and `eval.json` (`poros eval --json` of the finished run on the 4 held-out rows).
The Mac's job wrote these files with the home folder's path in them; in this copy the store folder under it reads `~/<store>` and the user name `<user>`, the marks the certification driver uses. Nothing else is changed.
