#!/usr/bin/env python3
"""gradcheck.py MODEL_DIR ROWS_PT [--rows 8] -- the expert adapters' gradient of one backward, per expert compute.

The checkpoint loaded resident and prepared (prepare_moe, seed 1234: the same LoRA A in every arm); every LoRA B
set to the same seeded random values (B starts at zero, which makes A's grad zero); one forward with labels on the
first --rows rows of ROWS_PT and one backward. Arms: ref (reference, pow2), ref_single (reference, one group),
stock (the reference with transformers' Qwen3MoeExperts loop, scripts/refcheck_hooks.py), fused (the fused experts only),
fused_all (fused + the fused attention, RMSNorm and chunked loss), exact (the stock loop with the whole model in
fp32 over the same dequantized weights: the gradient the bf16 arms round). Per arm against ref, per adapter kind
(A/B x gate_up/down) and overall: relative error ||g - g_ref|| / ||g_ref||, norm ratio ||g|| / ||g_ref||, cosine,
and the mean of g_ref . (g - g_ref) / ||g_ref||^2 (a bias along the reference gradient); and per arm against exact:
relative error, and the fraction of elements whose sign matches exact's (all, and the smaller half by |exact|): what
an Adam step, close to sign(g) early, takes from each. Prints GRADCHECK {json}."""
import argparse, json, os, sys

ap = argparse.ArgumentParser()
ap.add_argument("model"); ap.add_argument("rows"); ap.add_argument("--rows-n", type=int, default=8)
a = ap.parse_args()
import torch
from transformers import AutoModelForCausalLM

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import poros.moe.modules as M
from poros.core.fused.extras import install_extras
from poros.moe.prepare import MoeConfig, prepare_moe

ids = torch.load(a.rows)["input_ids"][: a.rows_n].cuda()
orig_compute = M._experts_compute
os.environ["REFCHECK_STOCK"] = "1"
import refcheck_hooks  # noqa: F401  (installs the stock loop; restored per arm below)
stock_compute = M._experts_compute
M._experts_compute = orig_compute


def exact_compute(hidden_states, top_k_index, top_k_weights, block):
    """The stock loop in fp32: the dequantized bf16 weights upcast (the values every arm trains on)."""
    deq = block.store.dequantize_tensor
    block.store.dequantize_tensor = lambda *args: deq(*args).float()
    try:
        return stock_compute(hidden_states, top_k_index, top_k_weights, block)
    finally:
        block.store.dequantize_tensor = deq


def grads(arm):
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16)
    compute = "fused" if arm.startswith("fused") else "reference"
    exact = arm == "exact"
    grouping = "single" if arm == "ref_single" else "pow2"
    prepare_moe(model, MoeConfig(streamed=False, device="cuda", expert_compute=compute, expert_grouping=grouping))
    if arm == "fused_all":
        install_extras(model)
    M._experts_compute = exact_compute if exact else stock_compute if arm == "stock" else orig_compute
    if exact:
        model.float()
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    with torch.no_grad():
        for i, (n, p) in enumerate(named):
            if "lora_B" in n:
                g = torch.Generator(device="cpu").manual_seed(1000 + i)
                p.copy_((torch.randn(p.shape, generator=g) * 1e-3).to(p.dtype))
    out = model(input_ids=ids, labels=ids)
    out.loss.backward()
    res = {n: p.grad.detach().float().clone() for n, p in named}
    loss = float(out.loss)
    M._experts_compute = orig_compute
    del model, out
    torch.cuda.empty_cache()
    return loss, res


arms = ["ref", "ref_single", "stock", "fused", "fused_all", "exact"]
G, losses = {}, {}
for arm in arms:
    losses[arm], G[arm] = grads(arm)
kinds = {"A_gate_up": "lora_A_gate_up", "B_gate_up": "lora_B_gate_up", "A_down": "lora_A_down",
         "B_down": "lora_B_down", "all": "lora_"}


def stats(g, r):
    g = torch.cat([x.flatten() for x in g]); r = torch.cat([x.flatten() for x in r])
    rn = r.norm()
    return {"rel_err": float((g - r).norm() / rn), "norm_ratio": float(g.norm() / rn),
            "cos": float(torch.dot(g, r) / (g.norm() * rn)), "bias_along_ref": float(torch.dot(r, g - r) / rn**2)}


rec = {"rows": a.rows_n, "loss": losses, "vs_ref": {}}
def vs_exact(g, x):
    g = torch.cat([t.flatten() for t in g]); x = torch.cat([t.flatten() for t in x])
    small = x.abs() <= x.abs().median()
    same = torch.sign(g) == torch.sign(x)
    return {"rel_err": float((g - x).norm() / x.norm()), "sign_agree": float(same.float().mean()),
            "sign_agree_small_half": float(same[small].float().mean())}


rec["vs_exact"] = {}
for arm in arms[:-1]:
    rec["vs_exact"][arm] = {}
    for k, frag in kinds.items():
        names = [n for n in G["ref"] if frag in n]
        rec["vs_exact"][arm][k] = {s: round(v, 6) for s, v in vs_exact([G[arm][n] for n in names],
                                                                         [G["exact"][n] for n in names]).items()}
for arm in arms[1:]:
    rec["vs_ref"][arm] = {}
    for k, frag in kinds.items():
        names = [n for n in G["ref"] if frag in n]
        rec["vs_ref"][arm][k] = {s: round(v, 6) for s, v in stats([G[arm][n] for n in names],
                                                                    [G["ref"][n] for n in names]).items()}
print("GRADCHECK " + json.dumps(rec), flush=True)
