#!/usr/bin/env python3
"""quantcheck.py CHECKPOINT_DIR OUT.json [--moe] -- quant.quantize_tensor on the CPU vs on CUDA, every tensor of a
checkpoint: sha256 of each blob both ways, and of all blobs in order.

--moe (Qwen3-30B-A3B): each expert's gate_up (torch.cat of gate and up, as poros.moe.load builds it) and down, the
tensors the streamed loader quantizes; else (a dense model, Qwen2.5-0.5B): every 2-D tensor of the checkpoint. Each
tensor is read in bf16 as the loader reads it. Prints QUANTCHECK {json}: tensors, mismatches (first 10 names), the
two whole-model digests, CPU and CUDA seconds."""
import argparse, glob, hashlib, json, os, re, sys, time

ap = argparse.ArgumentParser()
ap.add_argument("ckpt"); ap.add_argument("out"); ap.add_argument("--moe", action="store_true")
a = ap.parse_args()
import torch
from safetensors import safe_open

from poros.moe import quant

EXPERT = re.compile(r"model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$")


def tensors():
    halves = {}
    for shard in sorted(glob.glob(os.path.join(a.ckpt, "*.safetensors"))):
        with safe_open(shard, framework="pt", device="cpu") as f:
            for name in sorted(f.keys()):
                m = EXPERT.match(name)
                if a.moe:
                    if m is None:
                        continue
                    key = (int(m[1]), int(m[2]))
                    t = f.get_tensor(name).to(torch.bfloat16, copy=True)
                    if m[3] == "down_proj":
                        yield f"{key}.down", t
                        continue
                    pair = halves.setdefault(key, {})
                    pair[m[3]] = t
                    if len(pair) == 2:
                        yield f"{key}.gate_up", torch.cat([pair["gate_proj"], pair["up_proj"]])
                        del halves[key]
                else:
                    t = f.get_tensor(name)
                    if t.dim() == 2:
                        yield name, t.to(torch.bfloat16)


hc, hg = hashlib.sha256(), hashlib.sha256()
n, bad, tc, tg = 0, [], 0.0, 0.0
for name, t in tensors():
    s = time.perf_counter(); bc = quant.quantize_tensor(t); tc += time.perf_counter() - s
    s = time.perf_counter(); bg = quant.quantize_tensor(t, "cuda"); tg += time.perf_counter() - s
    c, g = bc.numpy().tobytes(), bg.numpy().tobytes()
    hc.update(c); hg.update(g)
    if c != g:
        bad.append(name)
    n += 1
rec = {"checkpoint": a.ckpt, "moe": a.moe, "tensors": n, "mismatches": len(bad), "first_mismatches": bad[:10],
       "sha256_cpu_all": hc.hexdigest(), "sha256_cuda_all": hg.hexdigest(), "identical": not bad and n > 0,
       "cpu_s": round(tc, 1), "cuda_s": round(tg, 1), "torch": torch.__version__}
json.dump(rec, open(a.out, "w"), indent=1)
print("QUANTCHECK " + json.dumps(rec), flush=True)
sys.exit(0 if rec["identical"] else 1)
