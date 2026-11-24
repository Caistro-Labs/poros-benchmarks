"""AOT-compile the MoE fused Triton kernels (bf16) at each target arch; print shared bytes."""
import json, sys, triton
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource
sys.path.insert(0, sys.argv[1])
from poros.moe import kernels as K
LIMIT = {70: 98304, 75: 65536, 80: 166912, 86: 101376, 89: 101376, 90: 232448, 100: 232448, 120: 101376}
rows = []
def comp(name, fn, sig, cx, warps, stages, cap, tag, extra=None):
    sig = {**sig, **{k: "constexpr" for k in cx}}
    opts = {"num_warps": warps, "num_stages": stages, **(extra or {})}
    try:
        k = triton.compile(ASTSource(fn, sig, constexprs=cx), target=GPUTarget("cuda", cap, 32), options=opts)
        sh = k.metadata.shared; ok = sh <= LIMIT[cap]; err = None
    except Exception as e:
        sh, ok, err = None, False, f"{type(e).__name__}: {str(e)[:200]}"
    rows.append(dict(kernel=name, cap=cap, tag=tag, stages=stages, shared=sh, limit=LIMIT[cap], fits=ok, err=err))
bf, f32, i32 = "*bf16", "*fp32", "*i32"
caps = [int(c) for c in sys.argv[2].split(",")]
for cap in caps:
  comp("_nf4_dequant", K._nf4_dequant, dict(slab_ptrs="*i64", out=bf, code8=f32, code4=f32, start="i64", pa="i64", a2="i64", off="i64", n_blocks="i32", out_stride="i32"), dict(BLOCKS=K.DQ_BLOCKS), 4, 3, cap, "bf16", {"enable_fp_fusion": False})
  comp("_combine", K._combine, dict(buf=f32, out=bf, T="i32", H="i32"), dict(KTOP=8, BT=K.COMB_BT, BH=K.COMB_BH), 4, 3, cap, "")
  comp("_sum_parts", K._sum_parts, dict(part=f32, out=f32, n="i32"), dict(P=12, BLOCK=1024), 4, 3, cap, "")
  for RP in (16, 32, 64, 128):
    for lora in (True, False):
      t = f"RP={RP} lora={lora}"
      for st in sorted({K.GEMM_STAGES, 1}):
        common = dict(t_r0=i32, t_r1=i32, t_w=i32, t_e=i32, n_rows="i32", H="i32", NI="i32", r="i32", s="fp32")
        comp("_gate_up_fwd", K._gate_up_fwd, dict(x=bf, row_tok=i32, w=bf, a1=f32, b=bf, hb=bf, **common, limit="fp32"),
             dict(RP=RP, BM=K.GEMM_BM, BN=K.GEMM_BN, BK=K.GEMM_BK, LIMIT=False, LORA=lora), K.GEMM_WARPS, st, cap, t)
        comp("_down_fwd", K._down_fwd, dict(hb=bf, w=bf, a2=f32, b=bf, wr=f32, dest=i32, buf=f32, **common),
             dict(RP=RP, BM=K.GEMM_BM, BN=K.WIDE_BN, BK=K.GEMM_BK, LORA=lora), K.WIDE_WARPS, st, cap, t)
        for drop in (False, True):
          comp("_mid_bwd", K._mid_bwd, dict(x=bf, gy=bf, row_tok=i32, wgu=bf, wd=bf, a1=f32, c2=f32, bgu=bf, ad=bf, wr=f32, hb=bf, dgu=bf, dw_part=f32, keep_h="*i1" if drop else bf, pscale="fp32", **common, limit="fp32"),
               dict(RP=RP, BM=K.GEMM_BM, BN=K.GEMM_BN, BK=K.GEMM_BK, LIMIT=False, LORA=lora, DROP=drop), K.GEMM_WARPS, st, cap, t + f" drop={drop}")
          comp("_dx_rows", K._dx_rows, dict(dgu=bf, wgu=bf, c1=f32, agu=bf, dest=i32, buf=f32, keep_x="*i1" if drop else bf, pscale="fp32", **common),
               dict(RP=RP, BM=K.GEMM_BM, BN=K.WIDE_BN, BK=K.GEMM_BK, LORA=lora, DROP=drop), K.WIDE_WARPS, st, cap, t + f" drop={drop}")
    for g in (True, False):
      for drop in (False, True):
        comp("_lora_rows", K._lora_rows, dict(x=bf, row_tok=i32, m=bf, out=f32, keep="*i1" if drop else bf, pscale="fp32", t_r0=i32, t_r1=i32, t_e=i32, n_rows="i32", K="i32", r="i32", m_se="i32", m_sr="i32", m_sk="i32"),
             dict(RP=RP, BM=K.GEMM_BM, BK=K.GEMM_BK, GATHER=g, DROP=drop), 4, 3, cap, f"RP={RP} gather={g} drop={drop}")
        comp("_wgrad", K._wgrad, dict(x=bf, row_tok=i32, wr=f32, y=f32, out=bf, keep="*i1" if drop else bf, pscale="fp32", g_r0=i32, g_r1=i32, g_off="*i64", N="i32", r="i32", s="fp32", o_sn="i32", o_sj="i32"),
             dict(RP=RP, BM=K.WG_BM, BN=K.WG_BN, GATHER=g, SCALE=True, DROP=drop), 4, 3, cap, f"RP={RP} gather={g} drop={drop}")
json.dump(rows, open(sys.argv[3], "w"), indent=0)
bad = [r for r in rows if not r["fits"]]
print(len(rows), "compiled;", len(bad), "do not fit / fail")
for r in bad: print(r)
