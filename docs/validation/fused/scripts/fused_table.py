#!/usr/bin/env python3
"""fused_table.py ROOT KEY... -- the per-family table of the fused-compute receipts in ROOT/<KEY>/.

For each family:
  probe     ROOT/<KEY>/<KEY>_probe.json (famprobe.py): activation, layer/MLP classes, fused routes;
  refcheck  per-update losses (events.jsonl) of three 100-update runs: PEFT's modules at micro-batch 1
            (rc_<KEY>_peft_mb1), the fused default at micro-batch 1 (rc_<KEY>_fused_mb1) and PEFT's
            modules at micro-batch 2 (rc_<KEY>_peft_mb2, the same 8 rows an update split 2 x 4).
            deviation = |fused mb1 - PEFT mb1|, noise = |PEFT mb2 - PEFT mb1|, update by update.
            PASS when the deviation's mean and max are both at or under the noise's mean and max;
  bitwise   compare_runs.py results (cmp_pa_<KEY>_<a>_<b>.json): sequential == efficient,
            sequential == fast, fast streamed == fast fully resident (every step's loss and grad
            norm, every adapter tensor torch.equal; a sha256 over all adapter tensors).
Writes ROOT/table.json and prints one Markdown row per family."""
import json
import os
import statistics
import sys

root, keys = sys.argv[1], sys.argv[2:]


def steps(path):
    out = []
    for line in open(path):
        e = json.loads(line)
        if e.get("event_type") == "step":
            out.append(e["data"])
    return out


rows = []
for k in keys:
    d = os.path.join(root, k)
    probe = json.load(open(os.path.join(d, f"{k}_probe.json")))
    runs = {n: steps(os.path.join(d, f"rc_{k}_{n}_events.jsonl")) for n in ("peft_mb1", "fused_mb1", "peft_mb2")}
    loss = {n: [s["loss"] for s in v] for n, v in runs.items()}
    dev = [abs(a - b) for a, b in zip(loss["fused_mb1"], loss["peft_mb1"])]
    noise = [abs(a - b) for a, b in zip(loss["peft_mb2"], loss["peft_mb1"])]
    assert len(dev) == len(noise) == len(loss["peft_mb1"]) == len(loss["fused_mb1"]) == len(loss["peft_mb2"])
    rec = {n: json.load(open(os.path.join(d, f"rc_{k}_{n}_receipt.json"))) for n in runs}
    s_med = {n: statistics.median([s["s_per_step"] for s in v][1:]) for n, v in runs.items()}
    cfg = {n: json.load(open(os.path.join(d, f"{n}_config.resolved.json")))
           for n in [f"rc_{k}_{x}" for x in runs] + [f"pa_{k}_{x}" for x in ("seq", "eff", "fast", "fast_res")]}
    cfg = {n[len("rc_") + len(k) + 1:] if n.startswith("rc_") else n: c for n, c in cfg.items()}
    # what ran: PEFT's modules where fused_compute was set false, the fused default everywhere else
    assert [cfg[n]["fused_compute"] for n in ("peft_mb1", "fused_mb1", "peft_mb2")] == [False, True, False], k
    assert all(cfg[f"pa_{k}_{x}"]["fused_compute"] for x in ("seq", "eff", "fast", "fast_res")), k
    assert cfg[f"pa_{k}_fast_res"]["resident_blocks"] >= 999 and cfg[f"pa_{k}_fast"]["resident_blocks"] == 0, k
    # each run trained on the path it asked for (ran_path: no fallback)
    assert [cfg[f"pa_{k}_{x}"]["ran_path"] for x in ("seq", "eff", "fast", "fast_res")] == \
        ["sequential", "efficient", "fast", "fast"], k
    assert all(cfg[n]["ran_path"] == "fast" for n in ("peft_mb1", "fused_mb1", "peft_mb2")), k
    cmp = {}
    for p in ("seq_eff", "seq_fast", "fast_fast_res"):
        c = json.load(open(os.path.join(d, f"cmp_pa_{k}_{p}.json")))
        cmp[p] = {"pass": c["pass"], "steps": c["steps_ref"], "adapter_tensors": c["adapter_tensors"],
                  "adapter_unequal_n": c["adapter_unequal_n"], "ref_digest": c["ref_digest"],
                  "cand_digest": c["cand_digest"]}
    r = {
        "key": k, "model": probe["model"], "model_type": probe["model_type"], "activation": probe["activation"],
        "act_fn_classes": probe["act_fn_classes"], "decoder_layer_classes": probe["decoder_layer_classes"],
        "mlp_classes": probe["mlp_classes"], "layer_types": probe["layer_types"],
        "probe_fused_installed": probe["fused_installed"], "probe_fused_refusals": probe["fused_refusals"],
        "run_lora_target_modules": cfg["fused_mb1"]["lora_target_modules"],
        "run_lora_linears": cmp["seq_fast"]["adapter_tensors"] // 2,
        "fused_compute_as_run": {n: c["fused_compute"] for n, c in cfg.items()},
        "refcheck": {
            "updates": len(dev),
            "dev_mean": statistics.mean(dev), "dev_max": max(dev),
            "noise_mean": statistics.mean(noise), "noise_max": max(noise),
            "last20_mean_loss": {n: statistics.mean(v[-20:]) for n, v in loss.items()},
            "pass": statistics.mean(dev) <= statistics.mean(noise) and max(dev) <= max(noise),
        },
        "mb1_s_per_step_median_2_100": {n: s_med[n] for n in ("peft_mb1", "fused_mb1")},
        "mb1_peak_vram_gb": {n: {"allocated": rec[n]["peak_vram_allocated_gb"], "reserved": rec[n]["peak_vram_reserved_gb"]}
                             for n in ("peft_mb1", "fused_mb1")},
        "bitwise": cmp,
        "bitwise_pass": all(v["pass"] and v["ref_digest"] == v["cand_digest"] for v in cmp.values()),
    }
    rows.append(r)
    rc = r["refcheck"]
    print(f"| {k} | {r['activation']} | {r['probe_fused_installed']} | {rc['dev_mean']:.2e} / {rc['dev_max']:.2e} "
          f"| {rc['noise_mean']:.2e} / {rc['noise_max']:.2e} | {'PASS' if rc['pass'] else 'FAIL'} "
          f"| {cmp['seq_fast']['ref_digest'][:12]} | {'PASS' if r['bitwise_pass'] else 'FAIL'} |")
json.dump(rows, open(os.path.join(root, "table.json"), "w"), indent=1)
