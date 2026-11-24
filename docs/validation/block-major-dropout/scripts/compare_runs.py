#!/usr/bin/env python3
"""compare_runs.py REF_DIR CAND_DIR OUT.json -- two `poros train` output folders, bit for bit.

Every step CAND ran (events.jsonl, by step number; a resumed run has only its later steps) has
REF's loss and grad_norm, and every tensor of the final adapter is equal (torch.equal): PEFT's
adapter_model.safetensors, or a native adapter's adapter_weights.pt."""
import glob
import hashlib
import json
import os
import sys

import torch


def steps(d):
    rows = [json.loads(line) for line in open(os.path.join(d, "events.jsonl"))]
    return {e["step"]: (e["data"].get("loss"), e["data"].get("grad_norm"))
            for e in rows if e.get("event_type") == "step"}


def adapters(d):
    for name in ("adapter_model.safetensors", "adapter_weights.pt"):
        found = sorted(glob.glob(os.path.join(d, "**", name), recursive=True))
        final = [f for f in found if "checkpoint-" not in f]
        if final:
            if name.endswith(".pt"):
                return torch.load(final[0], map_location="cpu"), final[0]
            from safetensors.torch import load_file

            return load_file(final[0]), final[0]
    raise SystemExit(f"no final adapter in {d}")


def digest(t):
    h = hashlib.sha256()
    for k in sorted(t):
        h.update(k.encode())
        h.update(t[k].contiguous().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


ref, cand, out = sys.argv[1:4]
sr, sc = steps(ref), steps(cand)
(ar, _), (ac, _) = adapters(ref), adapters(cand)
uneq = [k for k in sorted(ar) if k not in ac or not torch.equal(ar[k], ac[k])]
rec = {"ref": os.path.basename(ref.rstrip("/")), "cand": os.path.basename(cand.rstrip("/")),
       "steps_compared": sorted(sc), "steps_in_ref": bool(sc) and set(sc) <= set(sr),
       "losses_equal": all(sr.get(s, (None,))[0] == v[0] for s, v in sc.items()),
       "grad_norms_equal": all(sr.get(s, (None, None))[1] == v[1] for s, v in sc.items()),
       "adapter_tensors": len(ar), "adapter_keys_equal": sorted(ar) == sorted(ac),
       "adapter_unequal": uneq[:10], "ref_digest": digest(ar), "cand_digest": digest(ac),
       "ref_losses": [sr[s][0] for s in sorted(sr)], "cand_losses": [sc[s][0] for s in sorted(sc)]}
rec["pass"] = (rec["steps_in_ref"] and rec["losses_equal"] and rec["grad_norms_equal"]
               and rec["adapter_keys_equal"] and not uneq)
json.dump(rec, open(out, "w"), indent=1)
print("COMPARE", json.dumps({k: rec[k] for k in ("ref", "cand", "pass", "steps_compared",
                                                 "adapter_tensors", "ref_digest")}))
