#!/usr/bin/env python3
"""prof128.py CONFIG.yaml OUT_PREFIX -- torch.profiler over one whole optimizer update (the 3rd of 3) of a
`poros train` config through PorosTrainer; per-kernel self CUDA time -> OUT_ks.json / OUT_kavg.txt. Read-only:
only _adapter_optimizer_step is wrapped (to start/stop the profiler between updates)."""
import json, sys, time
import torch, yaml
from torch.profiler import ProfilerActivity, profile
from poros.core.config import PorosConfig
from poros.train.trainer import PorosTrainer
cfgp, out = sys.argv[1:3]
raw = yaml.safe_load(open(cfgp)); raw.update(output_dir=out + ".scratch", progress=False, max_steps=3, save_steps=0)
cfg = PorosConfig(**raw)
tr = PorosTrainer.from_pretrained(cfg.model_name_or_path, config=cfg)
st = {"n": 0, "p": None, "t": None, "wall": []}
orig = tr._adapter_optimizer_step
def opt(*a, **k):
    r = orig(*a, **k); torch.cuda.synchronize(); st["n"] += 1
    now = time.time()
    if st["t"] is not None: st["wall"].append(round(now - st["t"], 3))
    if st["p"] is not None:
        st["p"].__exit__(None, None, None); st["done"] = st["p"]; st["p"] = None
    if st["n"] == 2:
        st["p"] = profile(activities=[ProfilerActivity.CUDA]); st["p"].__enter__()
    st["t"] = time.time()
    return r
tr._adapter_optimizer_step = opt
tr.train(cfg.dataset, steps=3, seq_len=cfg.max_seq_length)
ka = st["done"].key_averages()
rows = []
for e in ka:
    d = getattr(e, "self_device_time_total", 0)
    if d: rows.append({"name": e.key[:160], "n": e.count, "self_ms": round(d / 1e3, 3)})
rows.sort(key=lambda r: -r["self_ms"])
tot = sum(r["self_ms"] for r in rows)
json.dump({"wall_s_after_update1": st["wall"], "device_ms_total": round(tot, 1), "kernels": rows}, open(out + "_ks.json", "w"), indent=0)
print("PROF", out, "wall", st["wall"], "device_ms", round(tot, 1), "peak GiB", round(torch.cuda.max_memory_allocated() / 2**30, 3))
for r in rows[:28]: print(f"  {r['self_ms']:9.1f} ms {r['n']:6d}  {r['name'][:120]}")
