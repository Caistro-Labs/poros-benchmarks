import sys, triton, json
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource
sys.path.insert(0, sys.argv[1])
from poros.core.fused import kernels as K
P=lambda t:"*"+t
out=[]
for cap in (80, 89, 100):
  for R in (64,128):
    for wh,da in ((True,False),(False,True)):
      sig=dict(bg=P("bf16"), bu=P("bf16"), ag=P("fp32"), au=P("fp32"), wbg=P("fp32"), wbu=P("fp32"), hb=P("bf16"), hf=P("fp32" if wh else "bf16"), ad_part=P("fp32"), wad=P("fp32"), M="i32", N="i32", s="fp32", R="constexpr", BLOCK_M="constexpr", BLOCK_N="constexpr", WRITE_HF="constexpr", DOWN_A="constexpr")
      k=triton.compile(ASTSource(K._swiglu_fwd, sig, constexprs=dict(R=R, BLOCK_M=K.FWD_BM, BLOCK_N=K.FWD_BN, WRITE_HF=wh, DOWN_A=da)), target=GPUTarget("cuda",cap,32), options={"num_warps":4,"num_stages":3})
      out.append(dict(kernel="_swiglu_fwd",cap=cap,R=R,hf=wh,shared=k.metadata.shared)); print(out[-1], flush=True)
json.dump(out, open("swiglu_fwd_80_89_100.json","w"), indent=0)
