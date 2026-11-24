#!/usr/bin/env python3
"""gate30.py SSS_DIR OUT.json [--mbs 4,2,1] [--steps 2] -- streamed == resident on the REAL Qwen3-30B-A3B, fused.

Both arms load the model with the experts from the quantized .sss store under SSS_DIR (store_path: no bf16 expert
copy in host RAM; the file is written by a store=ssd run first), install the fused compute and its extras
(attention, RMSNorm, loss) as the trainer does, and take --steps plain-SGD steps (poros.moe.gates' G-TRAIN loop,
lr 0.05) on the first mb rows of the h2h rows. Resident first: the largest mb of --mbs whose steps fit in VRAM
(each OOM is recorded with its peak); then streamed (streaming_config: one layer of experts plus one slice of
look-ahead, the dense blocks streamed) at that mb. torch.equal on every step's loss and grad norm and on every
final adapter tensor. Prints GATE30 {json}; exit 0 on pass."""
import argparse, gc, json, os, sys, time

ap = argparse.ArgumentParser()
ap.add_argument("sss"); ap.add_argument("out")
ap.add_argument("--mbs", default="4,2,1"); ap.add_argument("--steps", type=int, default=2)
ap.add_argument("--lr", type=float, default=0.05)
ap.add_argument("--rows", default=os.path.expanduser("~/h2h/data30/rows_real64.pt"),
                help="the rows .pt scripts/make_data.py writes (64 oasst1 rows of 512 tokens)")
a = ap.parse_args()
import torch

from poros.core.fused.extras import install_extras
from poros.moe.gates import _train_adapter_steps
from poros.moe.load import load_moe_streamed
from poros.moe.prepare import MoeConfig, prepare_moe, streaming_config

M, REV = "Qwen/Qwen3-30B-A3B", "ad44e777bcd18fa416d9da3bd8f70d33ebb85d39"
G = 2**30
rows = torch.load(a.rows)["input_ids"]


def arm(streamed, mb):
    t0 = time.time()
    model = load_moe_streamed(M, torch.bfloat16, revision=REV, store="ram", store_path=a.sss)
    load_s = time.time() - t0
    if streamed:
        cfg = streaming_config(model.config, device="cuda", expert_compute="fused")
    else:
        cfg = MoeConfig(streamed=False, device="cuda", expert_compute="fused")
    rep = prepare_moe(model, cfg)
    install_extras(model)
    torch.cuda.reset_peak_memory_stats()
    rec = {"streamed": streamed, "mb": mb, "load_s": round(load_s, 1), "guarantee": rep.guarantee,
           "dense_streamed": rep.dense_streamed}
    trace = None
    try:
        t = _train_adapter_steps(model, rows[:mb].cuda(), steps=a.steps, lr=a.lr)
        trace = ([x.cpu() for x in t.losses], [x.cpu() for x in t.grad_norms],
                 {k: v.cpu() for k, v in t.final_adapters.items()})
        del t
        rec["ok"] = True
    except torch.OutOfMemoryError as e:
        rec["ok"] = False
        rec["error"] = str(e)[:200]
    rec["peak_alloc_gib"] = round(torch.cuda.max_memory_allocated() / G, 2)
    rec["peak_reserved_gib"] = round(torch.cuda.max_memory_reserved() / G, 2)
    rec["s"] = round(time.time() - t0, 1)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return rec, trace


out = {"model": M, "revision": REV, "steps": a.steps, "lr": a.lr, "resident_tries": []}
res = None
for mb in [int(x) for x in a.mbs.split(",")]:
    rec, trace = arm(False, mb)
    out["resident_tries"].append(rec)
    print("RESIDENT " + json.dumps(rec), flush=True)
    if rec["ok"]:
        res = (mb, trace)
        break
if res is None:
    out["pass"] = None
    out["note"] = "no mb of --mbs fits resident"
else:
    mb, r = res
    srec, s = arm(True, mb)
    out["streamed"] = srec
    print("STREAMED " + json.dumps(srec), flush=True)
    if srec["ok"]:
        (sl, sg, sa), (rl, rg, ra) = s, r
        out["losses_streamed"] = [float(x) for x in sl]
        out["losses_resident"] = [float(x) for x in rl]
        out["losses_equal"] = all(torch.equal(x, y) for x, y in zip(sl, rl, strict=True))
        out["grad_norms_equal"] = all(torch.equal(x, y) for x, y in zip(sg, rg, strict=True))
        out["adapters_equal"] = set(sa) == set(ra) and all(torch.equal(sa[k], ra[k]) for k in sa)
        out["adapter_tensors"] = len(sa)
        out["pass"] = bool(out["losses_equal"] and out["grad_norms_equal"] and out["adapters_equal"])
    else:
        out["pass"] = None
json.dump(out, open(a.out, "w"), indent=1)
print("GATE30 " + json.dumps({k: v for k, v in out.items() if k not in ("losses_streamed", "losses_resident")}),
      flush=True)
sys.exit(0 if out.get("pass") else 1)
