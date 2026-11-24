"""dense_r8.py OUT.pt -- the dense fused LoRA layer at rank 8 (padded to 16) from the tree on PYTHONPATH, through
tests/gpu/test_fused_compute.py's own layer, inputs and run (fused.install on), at LoRA dropout 0 and 0.1: the output,
the input grad and every LoRA grad, saved for a bitwise comparison of two trees; plus which grads own their storage."""
import sys
import tempfile

import torch
import transformers

import tests.gpu.test_fused_compute as T
from poros.core import fused

out = sys.argv[1]
d = tempfile.mkdtemp()
torch.manual_seed(0)  # the test module's checkpoint fixture, as written there
cfg = transformers.Qwen2Config(vocab_size=256, hidden_size=256, intermediate_size=512, num_hidden_layers=3,
                               num_attention_heads=8, num_key_value_heads=2)
model = transformers.AutoModelForCausalLM.from_config(cfg)
with torch.no_grad():
    for p in model.parameters():
        p.normal_(0, 0.05)
model.to(torch.bfloat16).save_pretrained(d)
res = {}
for drop in (0.0, 0.1):
    m, base, layer = T._layer(d, dropout=drop, rank=8)
    x, pos, pe, gy = T._inputs(base)
    assert fused.install(m)["layers"] == 3
    try:
        o, g, _ = T._run(layer, x, pos, pe, gy)
    finally:
        fused.uninstall(m)
    names = [n for n, p in layer.named_parameters() if p.requires_grad]
    res[f"drop{drop}"] = {"out": o.cpu(), "dx": g[0].cpu(), **{n: t.cpu().clone() for n, t in zip(names, g[1:])}}
    res[f"drop{drop}_owned"] = sum(t.untyped_storage().nbytes() == t.numel() * t.element_size() for t in g[1:])
    res[f"drop{drop}_n"] = len(names)
torch.save(res, out)
print("DENSE_R8", out, {k: (v if isinstance(v, int) else len(v)) for k, v in res.items()})
