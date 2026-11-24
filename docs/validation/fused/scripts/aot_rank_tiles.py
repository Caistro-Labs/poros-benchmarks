"""Compile every fused Triton kernel ahead of time, at each LoRA rank's own tiles, for each GPU
architecture, and compare its shared memory with the architecture's per-block limit (no GPU).

    python aot_rank_tiles.py <tree> <out.json> [--caps 70,75,...] [--ranks 8,16,...]

Dense kernels (``poros.core.fused.kernels``, ``tiles(R)``) in bf16 on every arch and in fp16 on
sm_70 / sm_75 (the precision those cards run); the MoE expert kernels (``poros.moe.kernels``,
``tiles(rp)``, bf16 only) on sm_80 and newer. Each loop kernel at its configured stages and at one
(``launch`` cuts stages to fit, the same values). Triton's ``triton.compile`` with
``GPUTarget("cuda", cap, 32)``; nothing runs. A rank "fits" an arch when every kernel the fused
compute launches at it fits at one stage or more.
"""
import argparse
import json
import sys

import triton
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource

ap = argparse.ArgumentParser()
ap.add_argument("tree")
ap.add_argument("out")
ap.add_argument("--caps", default="70,75,80,86,89,90,100,120")
ap.add_argument("--ranks", default="8,16,32,64,128,256")
args = ap.parse_args()
sys.path.insert(0, args.tree)
from poros.core.fused import kernels as K  # noqa: E402
from poros.moe import kernels as MK  # noqa: E402

# per-block shared memory opt-in limits (poros.core.gpu.shared_memory_limit)
LIMIT = {70: 98304, 75: 65536, 80: 166912, 86: 101376, 89: 101376, 90: 232448, 100: 232448,
         120: 101376}
rows = []


def P(t):
    return "*" + t


def comp(name, fn, sig, cx, warps, stages, cap, tag, extra=None):
    sig = {**sig, **{k: "constexpr" for k in cx}}
    opts = {"num_warps": warps, "num_stages": stages, **(extra or {})}
    try:
        k = triton.compile(ASTSource(fn, sig, constexprs=cx), target=GPUTarget("cuda", cap, 32),
                           options=opts)
        sh, err = k.metadata.shared, None
    except Exception as e:  # noqa: BLE001
        sh, err = None, f"{type(e).__name__}: {str(e)[:200]}"
    rows.append(dict(kernel=name, cap=cap, tag=tag, stages=stages, shared=sh, limit=LIMIT[cap],
                     fits=sh is not None and sh <= LIMIT[cap], err=err))


def stages(st):
    return sorted({st, 1})


def dense(cap, R, dt):
    t = K.tiles(R)
    tag = f"{dt} R={R}"
    bm, bn, w = t.out_fwd
    comp("_lora_out_fwd", K._lora_out_fwd, dict(base=P(dt), a=P("fp32"), b=P("fp32"),
         out=P("fp32"), res=P("fp32"), M="i32", N="i32", s="fp32"),
         dict(R=R, BLOCK_M=bm, BLOCK_N=bn, RESIDUAL=True), w, 3, cap, tag)
    bm, bn, w, st = t.out_bwd
    for s in stages(st):
        comp("_lora_out_bwd", K._lora_out_bwd, dict(g=P("fp32"), a=P("fp32"), b=P("fp32"),
             gb=P(dt), da_part=P("fp32"), db_part=P("fp32"), M="i32", N="i32",
             rows_per_split="i32"), dict(R=R, BLOCK_M=bm, BLOCK_N=bn), w, s, cap, tag)
    bm, bn, w = t.swiglu_fwd
    for wh, da in ((True, False), (False, True)):
        comp("_swiglu_fwd", K._swiglu_fwd, dict(bg=P(dt), bu=P(dt), ag=P("fp32"), au=P("fp32"),
             wbg=P("fp32"), wbu=P("fp32"), hb=P(dt), hf=P("fp32" if wh else dt),
             ad_part=P("fp32"), wad=P("fp32"), M="i32", N="i32", s="fp32"),
             dict(R=R, BLOCK_M=bm, BLOCK_N=bn, WRITE_HF=wh, DOWN_A=da), w, 3, cap,
             tag + f" hf={wh}")
    bm, bn, w, st = t.swiglu_bwd
    for hd in (False, True):
        for s in stages(st):
            comp("_swiglu_bwd", K._swiglu_bwd, dict(bg=P(dt), bu=P(dt), ag=P("fp32"),
                 au=P("fp32"), wbg=P("fp32"), wbu=P("fp32"), dzb=P(dt), dad=P("fp32"),
                 wad=P("fp32"), dropm=P("i1" if hd else dt), dgb=P(dt), dub=P(dt),
                 dag_part=P("fp32"), dau_part=P("fp32"), dbg_part=P("fp32"), dbu_part=P("fp32"),
                 dwad_part=P("fp32"), M="i32", N="i32", s="fp32", pscale="fp32",
                 rows_per_split="i32"), dict(R=R, BLOCK_M=bm, BLOCK_N=bn, HAS_DROP=hd), w, s,
                 cap, tag + f" drop={hd}")
    for D in (64, 128):
        bm, w = t.rope_fwd
        comp("_lora_rope_fwd", K._lora_rope_fwd, dict(base=P(dt), a=P("fp32"), b=P("fp32"),
             out=P("fp32"), cos=P("fp32"), sin=P("fp32"), M="i32", N="i32", S="i32", BC="i32",
             s="fp32"), dict(R=R, BLOCK_M=bm, HALF=D // 2), w, 3, cap, tag + f" D={D}")
        bm, w, st = t.rope_bwd
        for s in stages(st):
            comp("_lora_rope_bwd", K._lora_rope_bwd, dict(g=P("fp32"), a=P("fp32"),
                 b=P("fp32"), gb=P(dt), da_part=P("fp32"), db_part=P("fp32"), cos=P("fp32"),
                 sin=P("fp32"), M="i32", N="i32", S="i32", BC="i32", rows_per_split="i32"),
                 dict(R=R, BLOCK_M=bm, HALF=D // 2), w, s, cap, tag + f" D={D}")


def moe(cap, RP):
    bf, f32, i32 = "*bf16", "*fp32", "*i32"
    bn, wbn, ww = MK.tiles(RP)
    common = dict(t_r0=i32, t_r1=i32, t_w=i32, t_e=i32, n_rows="i32", H="i32", NI="i32",
                  r="i32", s="fp32")
    t = f"RP={RP}"
    for st in stages(MK.GEMM_STAGES):
        comp("_gate_up_fwd", MK._gate_up_fwd, dict(x=bf, row_tok=i32, w=bf, a1=f32, b=bf, hb=bf,
             **common, limit="fp32"), dict(RP=RP, BM=MK.GEMM_BM, BN=bn, BK=MK.GEMM_BK,
             LIMIT=False, LORA=True), MK.GEMM_WARPS, st, cap, t)
        comp("_down_fwd", MK._down_fwd, dict(hb=bf, w=bf, a2=f32, b=bf, wr=f32, dest=i32, buf=f32,
             **common), dict(RP=RP, BM=MK.GEMM_BM, BN=wbn, BK=MK.GEMM_BK, LORA=True), ww, st,
             cap, t)
        for drop in (False, True):
            comp("_mid_bwd", MK._mid_bwd, dict(x=bf, gy=bf, row_tok=i32, wgu=bf, wd=bf, a1=f32,
                 c2=f32, bgu=bf, ad=bf, wr=f32, hb=bf, dgu=bf, dw_part=f32,
                 keep_h="*i1" if drop else bf, pscale="fp32", **common, limit="fp32"),
                 dict(RP=RP, BM=MK.GEMM_BM, BN=bn, BK=MK.GEMM_BK, LIMIT=False, LORA=True,
                      DROP=drop), MK.GEMM_WARPS, st, cap, t + f" drop={drop}")
            comp("_dx_rows", MK._dx_rows, dict(dgu=bf, wgu=bf, c1=f32, agu=bf, dest=i32, buf=f32,
                 keep_x="*i1" if drop else bf, pscale="fp32", **common),
                 dict(RP=RP, BM=MK.GEMM_BM, BN=wbn, BK=MK.GEMM_BK, LORA=True, DROP=drop), ww, st,
                 cap, t + f" drop={drop}")
    for g in (True, False):
        for drop in (False, True):
            comp("_lora_rows", MK._lora_rows, dict(x=bf, row_tok=i32, m=bf, out=f32,
                 keep="*i1" if drop else bf, pscale="fp32", t_r0=i32, t_r1=i32, t_e=i32,
                 n_rows="i32", K="i32", r="i32", m_se="i32", m_sr="i32", m_sk="i32"),
                 dict(RP=RP, BM=MK.GEMM_BM, BK=MK.GEMM_BK, GATHER=g, DROP=drop), 4, 3, cap,
                 f"{t} gather={g} drop={drop}")
            comp("_wgrad", MK._wgrad, dict(x=bf, row_tok=i32, wr=f32, y=f32, out=bf,
                 keep="*i1" if drop else bf, pscale="fp32", g_r0=i32, g_r1=i32, g_off="*i64",
                 N="i32", r="i32", s="fp32", o_sn="i32", o_sj="i32"),
                 dict(RP=RP, BM=MK.WG_BM, BN=MK.WG_BN, GATHER=g, SCALE=True, DROP=drop), 4, 3,
                 cap, f"{t} gather={g} drop={drop}")


caps = [int(c) for c in args.caps.split(",")]
ranks = [int(r) for r in args.ranks.split(",")]
pads = sorted({max(16, 1 << (r - 1).bit_length()) for r in ranks})
for cap in caps:
    for R in pads:
        dense(cap, R, "bf16")
        if cap < 80:
            dense(cap, R, "fp16")
        if cap >= 80:
            moe(cap, R)
    print("done", cap, flush=True)


def fits(cap, R, prefix, dt=None, head_dim=None):
    """Every kernel of the group fits at some stage count: per (kernel, tag), the min shared."""
    best = {}
    for r in rows:
        if r["cap"] != cap or not r["tag"].startswith(prefix):
            continue
        if dt and not r["tag"].startswith(dt):
            continue
        if head_dim is None and " D=" in r["tag"]:
            continue
        if head_dim is not None and " D=" in r["tag"] and f" D={head_dim}" not in r["tag"]:
            continue
        k = (r["kernel"], r["tag"])
        best[k] = best.get(k, False) or r["fits"]
    return bool(best) and all(best.values())


summary = []
for cap in caps:
    for r in ranks:
        R = max(16, 1 << (r - 1).bit_length())
        row = dict(cap=cap, rank=r, padded=R)
        for dt in ("bf16", "fp16") if cap < 80 else ("bf16",):
            row[f"dense_{dt}"] = fits(cap, R, f"{dt} R={R}", None, None)
            row[f"dense_{dt}_rope_hd64"] = fits(cap, R, f"{dt} R={R}", None, 64)
            row[f"dense_{dt}_rope_hd128"] = fits(cap, R, f"{dt} R={R}", None, 128)
        if cap >= 80:
            row["moe_bf16"] = fits(cap, R, f"RP={R}")
        summary.append(row)
json.dump(dict(triton=triton.__version__, rows=rows, summary=summary), open(args.out, "w"),
          indent=0)
print(len(rows), "compiled;", sum(1 for r in rows if r["err"]), "errors")
for s in summary:
    print(s)
