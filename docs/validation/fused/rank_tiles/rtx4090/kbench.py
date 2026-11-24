#!/usr/bin/env python3
"""kbench.py OUT.json KERNEL[,KERNEL..] [R] -- each fused kernel's launch candidates at padded rank R (default 128),
timed through poros.core.fused.lora's own helpers (partial sums included) at Qwen2.5-72B's widths (4,096 rows,
hidden 8,192, MLP 29,568, q 64 x 128, k/v 8 x 128) and Qwen2.5-0.5B's (512 rows, 896, 4,864, q 14 x 64, k/v 2 x 64).
Per candidate: compiled shared memory, registers, spills (sm of this card), median ms of 10 calls. KERNEL: out_fwd
out_bwd swiglu_fwd swiglu_bwd rope_fwd rope_bwd partial base (base: the current tiles of every kernel)."""
import itertools, json, statistics, sys
import torch
from poros.core import fused_tiles as FT
from poros.core.fused import kernels as K, lora as L
out, which = sys.argv[1], sys.argv[2].split(",")
R = int(sys.argv[3]) if len(sys.argv) > 3 else 128
dev = torch.device("cuda"); bf = torch.bfloat16; f32 = torch.float32
LIMIT = torch.cuda.get_device_properties(0).shared_memory_per_block_optin
meta = {}
def launch(kernel, grid, *a, **kw):
    st = kw.pop("num_stages", 3)
    ck = kernel[grid](*a, num_stages=st, **kw)
    meta[kernel.fn.__name__ if hasattr(kernel, "fn") else str(kernel)] = (ck.metadata.shared, ck.n_regs, ck.n_spills)
K.launch = launch
SHAPES = {"72b": dict(M=4096, H=8192, I=29568, D=128, q=8192, kv=1024),
          "05b": dict(M=512, H=896, I=4864, D=64, q=896, kv=128)}
g = torch.Generator(device=dev).manual_seed(0)
def rn(*s, dt=f32, sc=0.02): return (torch.randn(*s, generator=g, device=dev, dtype=f32) * sc).to(dt)
T = {}
for k, s in SHAPES.items():
    M, H, I, D = s["M"], s["H"], s["I"], s["D"]
    pos = torch.arange(M, device=dev, dtype=f32)[:, None] * torch.arange(D, device=dev, dtype=f32)[None] * 1e-3
    T[k] = dict(s=s, bg=rn(M, I, dt=bf, sc=1), bu=rn(M, I, dt=bf, sc=1), ag=rn(M, R), au=rn(M, R), Bg=rn(I, R), Bu=rn(I, R),
                Ad=rn(R, I), dzb=rn(M, I, dt=bf, sc=1), dad=rn(M, R), md=torch.rand(M, I, generator=g, device=dev) > 0.05,
                yH=rn(M, H, dt=bf, sc=1), aH=rn(M, R), BH=rn(H, R), res=rn(M, H, sc=1), gH=rn(M, H, sc=1),
                ykv=rn(M, s["kv"], dt=bf, sc=1), Bkv=rn(s["kv"], R), gkv=rn(M, s["kv"], sc=1),
                yq=rn(M, s["q"], dt=bf, sc=1), Bq=rn(s["q"], R), gq=rn(M, s["q"], sc=1),
                rope=L.Rope(torch.cos(pos)[None], torch.sin(pos)[None], 2))
def calls(k):
    t = T[k]; s = t["s"]; M, I = s["M"], s["I"]
    hb = torch.empty(M, I, dtype=bf, device=dev)
    return {
      "swiglu_fwd": lambda: L.swiglu_fwd(K, t["bg"], t["bu"], t["ag"], t["au"], t["Bg"], t["Bu"], t["Ad"], 2.0, hb, t["Ad"]),
      "swiglu_fwd_hf": lambda: L.swiglu_fwd(K, t["bg"], t["bu"], t["ag"], t["au"], t["Bg"], t["Bu"], t["Ad"], 2.0, hb, None),
      "swiglu_bwd": lambda: L.swiglu_bwd(K, t["bg"], t["bu"], t["ag"], t["au"], t["Bg"], t["Bu"], t["dzb"], t["dad"], t["Ad"], None, 2.0, 1.0, bf),
      "swiglu_bwd_drop": lambda: L.swiglu_bwd(K, t["bg"], t["bu"], t["ag"], t["au"], t["Bg"], t["Bu"], t["dzb"], t["dad"], t["Ad"], t["md"], 2.0, 1/0.95, bf),
      "out_fwd_H": lambda: L._out_fwd(K, t["yH"], t["aH"], t["BH"], 2.0, t["res"]),
      "out_fwd_kv": lambda: L._out_fwd(K, t["ykv"], t["aH"], t["Bkv"], 2.0),
      "out_bwd_H": lambda: L._out_bwd(K, t["gH"], t["aH"], t["BH"], 2.0, bf),
      "out_bwd_kv": lambda: L._out_bwd(K, t["gkv"], t["aH"], t["Bkv"], 2.0, bf),
      "rope_fwd_q": lambda: L._rope_fwd(K, t["yq"], t["aH"], t["Bq"], 2.0, t["rope"]),
      "rope_fwd_kv": lambda: L._rope_fwd(K, t["ykv"], t["aH"], t["Bkv"], 2.0, t["rope"]),
      "rope_bwd_q": lambda: L._rope_bwd(K, t["gq"], t["aH"], t["Bq"], 2.0, bf, t["rope"]),
      "rope_bwd_kv": lambda: L._rope_bwd(K, t["gkv"], t["aH"], t["Bkv"], 2.0, bf, t["rope"]),
    }
# a layer's calls: forward + rematerialized forward (x2), backward x1; q/k rotary, v plain, o + down with the residual
LAYER = {"swiglu_fwd": 2, "out_fwd_H": 4, "out_fwd_kv": 2, "rope_fwd_q": 2, "rope_fwd_kv": 2,
         "swiglu_bwd": 1, "out_bwd_H": 2, "out_bwd_kv": 1, "rope_bwd_q": 1, "rope_bwd_kv": 1}
USE = {"out_fwd": ["out_fwd_H", "out_fwd_kv"], "out_bwd": ["out_bwd_H", "out_bwd_kv"], "swiglu_fwd": ["swiglu_fwd", "swiglu_fwd_hf"],
       "swiglu_bwd": ["swiglu_bwd", "swiglu_bwd_drop"], "rope_fwd": ["rope_fwd_q", "rope_fwd_kv"], "rope_bwd": ["rope_bwd_q", "rope_bwd_kv"],
       "partial": ["swiglu_fwd", "swiglu_bwd", "out_bwd_H"], "base": list(LAYER) + ["swiglu_fwd_hf", "swiglu_bwd_drop"]}
KNAME = {"out_fwd": "_lora_out_fwd", "out_bwd": "_lora_out_bwd", "swiglu_fwd": "_swiglu_fwd", "swiglu_bwd": "_swiglu_bwd",
         "rope_fwd": "_lora_rope_fwd", "rope_bwd": "_lora_rope_bwd"}
def timeit(fn, n=10):
    fn(); torch.cuda.synchronize()
    ev = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(n)]
    for a, b in ev:
        a.record(); fn(); b.record()
    torch.cuda.synchronize()
    return round(statistics.median(a.elapsed_time(b) for a, b in ev), 3)
CANDS = {
 "out_fwd": [(bm, bn, w) for bm in (16, 32, 64) for bn in (32, 64, 128) for w in (4, 8)],
 "out_bwd": [(bm, bn, w, s) for bm in (16, 32, 64) for bn in (32, 64, 128) for w in (4, 8) for s in (1, 2)],
 "swiglu_fwd": [(bm, bn, w) for bm in (16, 32, 64) for bn in (16, 32, 64, 128) for w in (4, 8)],
 "swiglu_bwd": [(bm, bn, w, s) for bm in (16, 32, 64) for bn in (16, 32, 64) for w in (4, 8) for s in (1, 2)],
 "rope_fwd": [(bm, w) for bm in (16, 32, 64) for w in (4, 8)],
 "rope_bwd": [(bm, w, s) for bm in (16, 32, 64) for w in (4, 8) for s in (1, 2)],
 "partial": [1 << 24, 1 << 26, 1 << 27],
}
base = FT.tiles(R)
res = {"R": R, "limit": LIMIT, "base": base._asdict(), "dev": torch.cuda.get_device_name(0), "rows": []}
for kern in which:
    cands = [None] if kern == "base" else CANDS[kern]
    for c in cands:
        t = base if c is None else base._replace(partial_elems=c) if kern == "partial" else base._replace(**{kern: c})
        FT._TILES_128 = t; FT._TILES_64 = t if R == 64 else FT._TILES_64
        row = {"kernel": kern, "cand": c}
        for k in SHAPES:
            cs = calls(k)
            for name in USE[kern]:
                meta.clear()
                try:
                    row[f"{k}:{name}"] = timeit(cs[name])
                    kn = KNAME.get(kern)
                    if kn and kn in meta: row["smem_regs_spills"] = meta[kn]
                except Exception as e:
                    row[f"{k}:{name}"] = f"{type(e).__name__}: {str(e)[:120]}"
        torch.cuda.empty_cache()
        res["rows"].append(row); print(json.dumps(row), flush=True)
    FT._TILES_128 = base
json.dump(res, open(out, "w"), indent=0)
