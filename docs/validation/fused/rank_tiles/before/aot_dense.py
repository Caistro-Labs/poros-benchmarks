"""AOT-compile every dense fused Triton kernel at each target arch; print shared bytes."""
import json, sys, triton
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource
sys.path.insert(0, sys.argv[1])
from poros.core.fused import kernels as K
LIMIT = {70: 98304, 75: 65536, 80: 166912, 86: 101376, 89: 101376, 90: 232448, 100: 232448, 120: 101376}
def P(t): return "*" + t
rows = []
def comp(name, fn, sig, cx, warps, stages, cap, tag):
    sig = dict(sig); [sig.__setitem__(k, "constexpr") for k in cx]
    try:
        k = triton.compile(ASTSource(fn, sig, constexprs=cx), target=GPUTarget("cuda", cap, 32),
                           options={"num_warps": warps, "num_stages": stages})
        sh = k.metadata.shared; ok = sh <= LIMIT[cap]; err = None
    except Exception as e:
        sh, ok, err = None, False, f"{type(e).__name__}: {str(e)[:160]}"
    rows.append(dict(kernel=name, cap=cap, tag=tag, stages=stages, shared=sh, limit=LIMIT[cap], fits=ok, err=err))
caps = [int(c) for c in sys.argv[2].split(",")]
ranks = [int(r) for r in sys.argv[3].split(",")]
for cap in caps:
  for dt in sys.argv[5].split(","):
    for R in ranks:
        t = f"{dt} R={R}"
        comp("_lora_out_fwd", K._lora_out_fwd, dict(base=P(dt), a=P("fp32"), b=P("fp32"), out=P("fp32"), res=P("fp32"), M="i32", N="i32", s="fp32"),
             dict(R=R, BLOCK_M=K.FWD_BM, BLOCK_N=K.FWD_BN, RESIDUAL=True), K.FWD_WARPS, 3, cap, t)
        for st in sorted({K.BWD_STAGES, 1}):
            comp("_lora_out_bwd", K._lora_out_bwd, dict(g=P("fp32"), a=P("fp32"), b=P("fp32"), gb=P(dt), da_part=P("fp32"), db_part=P("fp32"), M="i32", N="i32", rows_per_split="i32"),
                 dict(R=R, BLOCK_M=K.BWD_BM, BLOCK_N=K.BWD_BN), K.BWD_WARPS, st, cap, t)
        for wh, da in ((True, False), (False, True)):
            comp("_swiglu_fwd", K._swiglu_fwd, dict(bg=P(dt), bu=P(dt), ag=P("fp32"), au=P("fp32"), wbg=P("fp32"), wbu=P("fp32"), hb=P(dt), hf=P("fp32" if wh else dt), ad_part=P("fp32"), wad=P("fp32"), M="i32", N="i32", s="fp32"),
                 dict(R=R, BLOCK_M=K.FWD_BM, BLOCK_N=K.FWD_BN, WRITE_HF=wh, DOWN_A=da), K.FWD_WARPS, 3, cap, t + f" hf={wh}")
        for hd in (False, True):
          for st in sorted({K.SWIGLU_BWD_STAGES, 1}):
            comp("_swiglu_bwd", K._swiglu_bwd, dict(bg=P(dt), bu=P(dt), ag=P("fp32"), au=P("fp32"), wbg=P("fp32"), wbu=P("fp32"), dzb=P(dt), dad=P("fp32"), wad=P("fp32"), dropm=P("i1" if hd else dt),
                 dgb=P(dt), dub=P(dt), dag_part=P("fp32"), dau_part=P("fp32"), dbg_part=P("fp32"), dbu_part=P("fp32"), dwad_part=P("fp32"), M="i32", N="i32", s="fp32", pscale="fp32", rows_per_split="i32"),
                 dict(R=R, BLOCK_M=K.SWIGLU_BWD_BM, BLOCK_N=K.SWIGLU_BWD_BN, HAS_DROP=hd), K.SWIGLU_BWD_WARPS, st, cap, t + f" drop={hd}")
        for D in (64, 128):
            comp("_lora_rope_fwd", K._lora_rope_fwd, dict(base=P(dt), a=P("fp32"), b=P("fp32"), out=P("fp32"), cos=P("fp32"), sin=P("fp32"), M="i32", N="i32", S="i32", BC="i32", s="fp32"),
                 dict(R=R, BLOCK_M=K.FWD_BM, HALF=D // 2), K.FWD_WARPS, 3, cap, t + f" D={D}")
            for st in sorted({K.BWD_STAGES, 1}):
                comp("_lora_rope_bwd", K._lora_rope_bwd, dict(g=P("fp32"), a=P("fp32"), b=P("fp32"), gb=P(dt), da_part=P("fp32"), db_part=P("fp32"), cos=P("fp32"), sin=P("fp32"), M="i32", N="i32", S="i32", BC="i32", rows_per_split="i32"),
                     dict(R=R, BLOCK_M=K.BWD_BM, HALF=D // 2), K.BWD_WARPS, st, cap, t + f" D={D}")
    for dt in ("bf16", "fp16", "fp32"):
        comp("_add_f32", K._add_f32, dict(out=P("fp32"), x0=P(dt), x1=P(dt), x2=P(dt), M="i32", K="i32"), dict(NT=3, BLOCK_M=32, BLOCK_K=256), 4, 3, cap, dt)
    for H in (896, 3584, 8192):
        b = triton.next_power_of_2(H)
        comp("_rmsnorm_fwd", K._rmsnorm_fwd, dict(x=P("fp32"), w=P("fp32"), y=P("fp32"), rstd=P("fp32"), H="i32", eps="fp32"), dict(BLOCK=b), 8 if b >= 4096 else 4, 3, cap, f"H={H}")
        comp("_rmsnorm_bwd", K._rmsnorm_bwd, dict(dy=P("fp32"), x=P("fp32"), w=P("fp32"), rstd=P("fp32"), dx=P("fp32"), H="i32"), dict(BLOCK=b), 8 if b >= 4096 else 4, 3, cap, f"H={H}")
json.dump(rows, open(sys.argv[4], "w"), indent=0)
bad = [r for r in rows if not r["fits"]]
print(len(rows), "compiled;", len(bad), "do not fit / fail")
for r in bad: print(r)
