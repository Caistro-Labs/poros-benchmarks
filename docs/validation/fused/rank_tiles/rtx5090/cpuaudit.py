#!/usr/bin/env python3
"""cpuaudit.py CONFIG.yaml OUT -- a real Poros training run (PorosTrainer, the config's path) under a TorchDispatchMode
that counts every aten op by (device of its float inputs, dtype, inside autograd's backward or not), from the first
update to the last. Question: does a CUDA run ever take a bf16 / fp16 backward on the CPU (what oneDNN refuses on
avx2_vnni_2 CPUs)? Prints CPU_HALF_BWD <n> (ops in the backward with a CPU bf16/fp16 tensor) and the totals."""
import collections, json, sys
import torch, yaml
from torch.utils._python_dispatch import TorchDispatchMode
from poros.core.config import PorosConfig
from poros.train.trainer import PorosTrainer
cfgp, out = sys.argv[1:3]
raw = yaml.safe_load(open(cfgp)); raw.update(output_dir=out + ".scratch", progress=False, save_steps=0)
cfg = PorosConfig(**raw)
tr = PorosTrainer.from_pretrained(cfg.model_name_or_path, config=cfg)
C = collections.Counter(); EX = {}
class Audit(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        bwd = torch._C._current_autograd_node() is not None
        ts = [a for a in torch.utils._pytree.tree_leaves((args, kwargs or {})) if isinstance(a, torch.Tensor) and a.is_floating_point()]
        for dev, dt in {(t.device.type, t.dtype) for t in ts}:
            k = (dev, str(dt).replace("torch.", ""), "bwd" if bwd else "fwd"); C[k] += 1
            if dev == "cpu" and dt in (torch.bfloat16, torch.float16):
                EX.setdefault(k, collections.Counter())[str(func)] += 1
        return func(*args, **(kwargs or {}))
with Audit():
    tr.train(cfg.dataset, steps=cfg.max_steps, seq_len=cfg.max_seq_length)
half_bwd = sum(v for (d, t, p), v in C.items() if d == "cpu" and t in ("bfloat16", "float16") and p == "bwd")
cuda_bwd = sum(v for (d, t, p), v in C.items() if d == "cuda" and p == "bwd")
rec = {"config": cfgp, "ran_compute": cfg.ran_compute, "counts": {"|".join(k): v for k, v in sorted(C.items())},
       "cpu_half_ops": {"|".join(k): dict(v.most_common(12)) for k, v in EX.items()}}
json.dump(rec, open(out + ".json", "w"), indent=1)
print("CPUAUDIT", cfgp.split("/")[-1], "CPU_HALF_BWD", half_bwd, "cuda_bwd_ops", cuda_bwd, "ran", json.dumps(cfg.ran_compute),
      "cpu_half", {k: sum(v.values()) for k, v in rec["cpu_half_ops"].items()})
