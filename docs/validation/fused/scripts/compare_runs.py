#!/usr/bin/env python3
"""compare_runs.py REF_DIR CAND_DIR OUT.json -- two `poros train` output dirs, bit for bit: every step's
loss and grad_norm (events.jsonl, exact float equality) and every adapter tensor (torch.equal), plus a
sha256 digest over all adapter tensors (sorted keys, raw bytes)."""
import glob, hashlib, json, os, sys
import torch
from safetensors.torch import load_file
ref, cand, out = sys.argv[1:4]
def steps(d):
    ev = [json.loads(l) for l in open(os.path.join(d, "events.jsonl"))]
    return [(e["data"].get("loss"), e["data"].get("grad_norm")) for e in ev if e.get("event_type") == "step"]
def flat(o, p=""):
    if isinstance(o, torch.Tensor): return {p: o}
    out = {}
    if isinstance(o, dict): it = o.items()
    elif isinstance(o, (list, tuple)): it = enumerate(o)
    else: return out
    for k, v in it: out.update(flat(v, f"{p}.{k}" if p else str(k)))
    return out
def adapters(d):
    """adapter_model.safetensors (dense runs), else adapter_weights.pt (poros.moe's expert LoRA state), flattened."""
    f = sorted(glob.glob(os.path.join(d, "**", "adapter_model.safetensors"), recursive=True))
    f = [x for x in f if "checkpoint-" not in x] or f
    if f: return load_file(f[0]), f[0]
    f = sorted(x for x in glob.glob(os.path.join(d, "**", "adapter_weights.pt"), recursive=True) if "checkpoint-" not in x)
    return flat(torch.load(f[0], map_location="cpu", weights_only=True)), f[0]
def digest(t):
    h = hashlib.sha256()
    for k in sorted(t): h.update(k.encode()); h.update(t[k].contiguous().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()
sr, sc = steps(ref), steps(cand)
(ar, fr), (ac, fc) = adapters(ref), adapters(cand)
uneq = [k for k in sorted(ar) if k not in ac or not torch.equal(ar[k], ac[k])]
rec = {"ref": ref, "cand": cand, "ref_file": fr, "cand_file": fc, "steps_ref": len(sr), "steps_cand": len(sc),
       "losses_equal": [a[0] for a in sr] == [b[0] for b in sc], "grad_norms_equal": [a[1] for a in sr] == [b[1] for b in sc],
       "adapter_tensors": len(ar), "adapter_keys_equal": sorted(ar) == sorted(ac), "adapter_unequal_n": len(uneq), "adapter_unequal": uneq[:10],
       "ref_digest": digest(ar), "cand_digest": digest(ac), "ref_losses": [a[0] for a in sr], "cand_losses": [b[0] for b in sc],
       "ref_grad_norms": [a[1] for a in sr], "cand_grad_norms": [b[1] for b in sc]}
rec["pass"] = bool(rec["losses_equal"] and rec["grad_norms_equal"] and rec["adapter_keys_equal"] and not uneq and len(sr) == len(sc) > 0)
json.dump(rec, open(out, "w"), indent=1)
print("COMPARE", json.dumps({k: rec[k] for k in ("pass", "losses_equal", "grad_norms_equal", "adapter_tensors", "adapter_unequal_n", "ref_digest", "cand_digest")}))
