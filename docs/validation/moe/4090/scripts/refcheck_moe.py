#!/usr/bin/env python3
"""refcheck_moe.py REC_DIR PREFIX -- the fused MoE path against the reference compute and stock transformers, on
real data: per-update training loss of each arm (PREFIX_<arm>_events.jsonl), the final eval loss, and the pairs:
  fused vs reference, fused vs stock          -- the lever (same batch shape, mb8 x 1)
  reference vs stock                          -- the reference compute's own distance from stock (baseline)
  reference mb8x1 vs mb4x2, fused mb8x1 vs mb4x2, reference pow2 vs single -- the same update with other roundings:
                                                  the run-to-run noise scale
Rule (no quality loss): fused vs reference within the largest noise pair (mean |loss diff| and |final eval loss
diff|); fused vs stock within the larger of that and reference vs stock. Prints REFCHECK {json}; exit 0 on pass.
Also printed, beside the two-sided rule (added after the 6-layer dry run): every pair of arms
("pairs"), and "one_sided": the reference family's spread (largest mean |diff| and |eval diff| over pairs of
ref_mb8, ref_mb4, refs_mb8, stock_mb8) and, per fused arm against each of them, the mean signed loss diff
(fused - other) and eval diff -- quality loss is fused HIGHER than the reference family by more than its spread."""
import json, os, statistics, sys
D, P = sys.argv[1], sys.argv[2]
ARMS = ("fz_mb8", "fz_mb4", "ref_mb8", "ref_mb4", "refs_mb8", "stock_mb8")
def load(arm):
    p = os.path.join(D, f"{P}_{arm}_events.jsonl")
    if not os.path.exists(p):
        return None
    ev = [json.loads(l) for l in open(p)]
    loss = [e["data"]["loss"] for e in ev if e.get("event_type") == "step"]
    pe = os.path.join(D, f"{P}_{arm}_refcheck_eval.json")
    el = json.load(open(pe)).get("eval_loss") if os.path.exists(pe) else None
    return {"loss": loss, "eval_loss": el}
arms = {a: load(a) for a in ARMS}
arms = {a: v for a, v in arms.items() if v}
def pair(a, b):
    x, y = arms[a]["loss"], arms[b]["loss"]
    d = [abs(u - v) for u, v in zip(x, y)]
    ea, eb = arms[a]["eval_loss"], arms[b]["eval_loss"]
    return {"updates": len(d), "mean_abs": statistics.mean(d), "max_abs": max(d),
            "mean_signed": statistics.mean(u - v for u, v in zip(x, y)),
            "eval_abs": abs(ea - eb) if ea is not None and eb is not None else None}
lever = [("fz_mb8", "ref_mb8"), ("fz_mb8", "stock_mb8")]
noise = [("ref_mb8", "ref_mb4"), ("fz_mb8", "fz_mb4"), ("ref_mb8", "refs_mb8")]
out = {"lever": {}, "noise": {}, "baseline": {}}
if "ref_mb8" in arms and "stock_mb8" in arms:
    out["baseline"]["ref_mb8 vs stock_mb8"] = pair("ref_mb8", "stock_mb8")
for a, b in lever:
    if a in arms and b in arms:
        out["lever"][f"{a} vs {b}"] = pair(a, b)
for a, b in noise:
    if a in arms and b in arms:
        out["noise"][f"{a} vs {b}"] = pair(a, b)
nm = max((v["mean_abs"] for v in out["noise"].values()), default=None)
ne = max((v["eval_abs"] for v in out["noise"].values() if v["eval_abs"] is not None), default=None)
base = out["baseline"].get("ref_mb8 vs stock_mb8")
def within(v, m, e):
    return v["mean_abs"] <= m and (e is None or v["eval_abs"] is None or v["eval_abs"] <= e)
ok = nm is not None and len(out["lever"]) == 2 and base is not None
if ok:
    ok = within(out["lever"]["fz_mb8 vs ref_mb8"], nm, ne) and within(
        out["lever"]["fz_mb8 vs stock_mb8"], max(nm, base["mean_abs"]),
        None if ne is None or base["eval_abs"] is None else max(ne, base["eval_abs"]))
out.update(noise_mean_abs=nm, noise_eval_abs=ne, last10_mean_loss={a: statistics.mean(v["loss"][-10:]) for a, v in arms.items()},
           eval_loss={a: v["eval_loss"] for a, v in arms.items()}, pass_=ok)
import itertools
out["pairs"] = {f"{a} vs {b}": pair(a, b) for a, b in itertools.combinations(arms, 2)}
fam = [a for a in ("ref_mb8", "ref_mb4", "refs_mb8", "stock_mb8") if a in arms]
fz = [a for a in ("fz_mb8", "fz_mb4") if a in arms]
fp = [pair(a, b) for a, b in itertools.combinations(fam, 2)]
spread = max((v["mean_abs"] for v in fp), default=None)
espread = max((v["eval_abs"] for v in fp if v["eval_abs"] is not None), default=None)
cross = {f"{f} - {r}": {"mean_signed": pair(f, r)["mean_signed"],
                        "eval_diff": (arms[f]["eval_loss"] - arms[r]["eval_loss"]
                                      if arms[f]["eval_loss"] is not None and arms[r]["eval_loss"] is not None
                                      else None)} for f in fz for r in fam}
worse = spread is not None and any(
    v["mean_signed"] > spread or (espread is not None and v["eval_diff"] is not None and v["eval_diff"] > espread)
    for v in cross.values())
out["one_sided"] = {"family_spread_mean_abs": spread, "family_spread_eval_abs": espread, "fused_minus_family": cross,
                    "fused_worse_than_family_spread": worse if spread is not None else None}
print("REFCHECK " + json.dumps(out))
sys.exit(0 if ok else 1)
