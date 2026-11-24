#!/usr/bin/env python3
"""prec64.py MODEL OUT.json [--layers all|N,...] [--rows 8] [--revision REV] -- the fused and the reference MoE expert
compute against an fp64 computation of the same math, on identical inputs.

RULE (fixed before any run of this script; binding for making fused the default): for every layer checked -- each
of the 6 layers of the 6-layer random-weight Qwen3-30B-A3B checkpoint (scripts/mk30.py --layers 6), and layer 24
of the real Qwen3-30B-A3B (rev ad44e777) -- the fused arm's error against fp64 is <= the reference arm's, for the
block output AND for the layer's adapter grads (its four LoRA tensors together), on BOTH max-abs and relative-L2
(||arm - fp64|| / ||fp64||). Any comparison fused > reference: FAIL, fused stays opt-in.

Per layer, one set of inputs serves all three arms: the layer's block input captured from a forward of the model on
the first --rows rows of the h2h rows (rows_real64.pt), the router's top-k and weights on it (bf16, as the model
computes them), the layer's LoRA (A as initialized, B set to seeded N(0, 1e-3): B = 0 would make A's grad 0), and a
seeded output grad (N(0, 1e-3), bf16). Arms: fused (poros.moe.fused.FusedExpertFunction), reference
(poros.moe.modules._ManagedExpertFunction, pow2 grouping), fp64 (transformers' Qwen3MoeExperts math -- per expert:
gate_up = x W^T + s (x A^T) B^T, SwiGLU, down likewise, times the routing weight, summed per token -- every operand
the bf16 value upcast to fp64, the experts' weights the NF4 decode the arms use; autograd in fp64 for the grads).
Also reported (not in the rule): the input grad. Prints PREC64 {json} and writes OUT.json."""
import argparse, json, os, sys, time

ap = argparse.ArgumentParser()
ap.add_argument("model"); ap.add_argument("out")
ap.add_argument("--layers", default="all"); ap.add_argument("--rows", type=int, default=8)
ap.add_argument("--revision", default=None)
ap.add_argument("--rows-pt", default=os.path.expanduser("~/h2h/data30/rows_real64.pt"),
                help="the rows .pt scripts/make_data.py writes (64 oasst1 rows of 512 tokens)")
a = ap.parse_args()
import torch
import torch.nn.functional as F

from poros.moe.fused import FusedExpertFunction
from poros.moe.modules import ManagedMoeBlock, _ManagedExpertFunction
from poros.moe.prepare import MoeConfig, prepare_moe, streaming_config

torch.backends.cuda.matmul.allow_tf32 = False
ids = torch.load(a.rows_pt)["input_ids"][: a.rows].cuda()
t0 = time.time()
if os.path.isdir(a.model):  # a small checkpoint: whole and resident
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16)
    prepare_moe(model, MoeConfig(streamed=False, device="cuda", expert_compute="reference"))
else:  # the real model: streamed load and residency
    from poros.moe.load import load_moe_streamed

    model = load_moe_streamed(a.model, torch.bfloat16, revision=a.revision)
    prepare_moe(model, streaming_config(model.config, device="cuda", expert_compute="reference"))
load_s = time.time() - t0
blocks = [m for m in model.modules() if isinstance(m, ManagedMoeBlock)]
want = range(len(blocks)) if a.layers == "all" else [int(x) for x in a.layers.split(",")]
last = max(want)
captured = {}


class _Stop(Exception):
    pass


def hook(i):
    def pre(module, args, kwargs):
        x = args[0] if args else kwargs["hidden_states"]
        captured[i] = x.detach().reshape(-1, x.shape[-1]).clone()
        if i == last:
            raise _Stop
    return pre


hs = [blocks[i].register_forward_pre_hook(hook(i), with_kwargs=True) for i in want]
with torch.no_grad():
    try:
        model(input_ids=ids)
    except _Stop:
        pass
for h in hs:
    h.remove()


def errs(arm, ref64):
    d = arm.double() - ref64
    return {"max_abs": float(d.abs().max()), "rel_l2": float(d.norm() / ref64.norm())}


def lora_params(ad):
    return [ad.lora_A_gate_up, ad.lora_B_gate_up, ad.lora_A_down, ad.lora_B_down]


def run_arm(block, fn, x, idx, w, gy):
    for p in lora_params(block.adapter):
        p.grad = None
    xi = x.clone().requires_grad_(True)
    out = fn.apply(xi, idx, w, block)
    out.backward(gy)
    grads = [p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p) for p in lora_params(block.adapter)]
    return out.detach(), xi.grad.detach(), grads


def fp64_arm(block, x, idx, w, gy):
    ad = block.adapter
    s = float(ad.scaling)
    P = [p.detach().double().clone().requires_grad_(True) for p in lora_params(ad)]
    x64 = x.double().clone().requires_grad_(True)
    w64 = w.double()
    out = torch.zeros_like(x64)
    hit = sorted(set(idx.reshape(-1).tolist()))
    block.residency.ensure([(block.layer_idx, e) for e in hit])
    for e in hit:
        key = (block.layer_idx, e)
        slab = block.residency.resident_qbytes(key)
        wgu = block.store.dequantize_tensor(key, "gate_up_proj", slab).double()
        wd = block.store.dequantize_tensor(key, "down_proj", slab).double()
        tok, slot = torch.where(idx == e)
        xe = x64[tok]
        gu = xe @ wgu.T + s * ((xe @ P[0][e].T) @ P[1][e].T)
        gate, up = gu.chunk(2, dim=-1)
        h = F.silu(gate) * up
        d = h @ wd.T + s * ((h @ P[2][e].T) @ P[3][e].T)
        out = out.index_add(0, tok, d * w64[tok, slot, None])
    out.backward(gy.double())
    return out.detach(), x64.grad.detach(), [p.grad.detach() for p in P]


res = {"model": a.model, "rows": a.rows, "load_s": round(load_s, 1), "layers": {}}
for i in want:
    block = blocks[i]
    x = captured[i]
    ad = block.adapter
    g = torch.Generator(device="cpu").manual_seed(1000 + i)
    with torch.no_grad():
        for p in (ad.lora_B_gate_up, ad.lora_B_down):
            p.copy_((torch.randn(p.shape, generator=g) * 1e-3).to(p.dtype))
    with torch.no_grad():
        _, w, idx = block.gate(x)
    gy = (torch.randn(x.shape, generator=g) * 1e-3).to(torch.bfloat16).cuda()
    o64, dx64, g64 = fp64_arm(block, x, idx, w, gy)
    G64 = torch.cat([t.flatten() for t in g64])
    rec = {"tokens": int(x.shape[0])}
    for name, fn, compute in (("fused", FusedExpertFunction, "fused"), ("reference", _ManagedExpertFunction, "reference")):
        block.expert_compute = compute
        o, dx, gr = run_arm(block, fn, x, idx, w, gy)
        rec[name] = {"output": errs(o, o64), "adapter_grads": errs(torch.cat([t.flatten() for t in gr]), G64),
                     "input_grad": errs(dx, dx64)}
    block.expert_compute = "reference"
    checks = {f"{t}.{m}": rec["fused"][t][m] <= rec["reference"][t][m]
              for t in ("output", "adapter_grads") for m in ("max_abs", "rel_l2")}
    rec["fused_le_reference"] = checks
    rec["pass"] = all(checks.values())
    res["layers"][str(block.layer_idx)] = rec
    print(f"layer {block.layer_idx}: " + json.dumps({k: rec[k] for k in ("fused", "reference", "pass")}), flush=True)
res["pass"] = all(v["pass"] for v in res["layers"].values())
json.dump(res, open(a.out, "w"), indent=1)
print("PREC64 " + json.dumps({"pass": res["pass"], "layers": {k: v["pass"] for k, v in res["layers"].items()}}),
      flush=True)
