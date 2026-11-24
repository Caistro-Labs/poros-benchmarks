#!/usr/bin/env python3
"""refcheck.py RECEIPTS_DIR OUT.json -- the MoE loss check: each run's per-update losses
(events.jsonl) side by side, and how far the streams' curves sit apart.

moe_ref_base: the earlier tree at lora_dropout 0.05 (the sequential loop, its stream drawn in that
loop's order); moe_ref_new: this tree (block-major, its stream keyed by layer); moe_ref_d0: this
tree at dropout 0, the curve without masks. For each pair: the mean and largest |difference| of
the per-update losses, and the mean loss over the last 10 updates of each."""
import json
import os
import sys

root, out = sys.argv[1], sys.argv[2]


def losses(run):
    path = os.path.join(root, f"{run}.events.jsonl")
    steps = [json.loads(line) for line in open(path)]
    return [e["data"]["loss"] for e in steps if e.get("event_type") == "step"]


runs = {r: losses(r) for r in ("moe_ref_base", "moe_ref_new", "moe_ref_d0")}
pairs = {}
for a, b in (("moe_ref_base", "moe_ref_new"), ("moe_ref_base", "moe_ref_d0"),
             ("moe_ref_new", "moe_ref_d0")):
    d = [abs(x - y) for x, y in zip(runs[a], runs[b])]
    pairs[f"{a} vs {b}"] = {"updates": len(d), "mean_abs_diff": sum(d) / len(d),
                            "max_abs_diff": max(d)}
summary = {"losses": runs, "pairs": pairs,
           "last10_mean": {r: sum(v[-10:]) / len(v[-10:]) for r, v in runs.items()}}
json.dump(summary, open(out, "w"), indent=1)
print(json.dumps({"pairs": pairs, "last10_mean": summary["last10_mean"]}, indent=1))
