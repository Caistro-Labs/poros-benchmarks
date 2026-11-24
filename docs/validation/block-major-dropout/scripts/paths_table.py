#!/usr/bin/env python3
"""paths_table.py RECEIPTS_DIR RUN... -- the README's path table from the runs' records: for each
run, its loop and block size (config.resolved.json) and its mean s/step, peak VRAM allocated and
reserved and peak host RAM (receipt.json), as one markdown row."""
import json
import os
import sys

root, runs = sys.argv[1], sys.argv[2:]
print("| Run | model, tokens | loop | block | s/step | allocated / reserved (GB) | host RAM (GB) |")
print("|---|---|---|---|---|---|---|")
for run in runs:
    said = json.load(open(os.path.join(root, f"{run}.config.resolved.json")))
    got = json.load(open(os.path.join(root, f"{run}.receipt.json")))
    loop = said.get("path") or "sequential"
    model = said["model_name_or_path"].split("/")[-1]
    print(f"| `{run}` | {model}, {said['max_seq_length']} | {loop} | {said['block_size'] or 4} | "
          f"{got['s_per_step']:.2f} | {got['peak_vram_allocated_gb']:.2f} / "
          f"{got['peak_vram_reserved_gb']:.2f} | {got['peak_host_ram_gb']:.1f} |")
