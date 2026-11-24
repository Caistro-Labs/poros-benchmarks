"""cmp_r8.py A.pt B.pt -- every tensor of two dense_r8.py outputs, bit for bit (raw bytes, so -0.0 != +0.0)."""
import sys

import torch

a, b = (torch.load(p, weights_only=True) for p in sys.argv[1:3])
bad, n = [], 0
for k in ("drop0.0", "drop0.1"):
    assert a[k].keys() == b[k].keys(), k
    for name in a[k]:
        n += 1
        x, y = a[k][name], b[k][name]
        if x.shape != y.shape or x.dtype != y.dtype or x.contiguous().view(torch.uint8).ne(y.contiguous().view(torch.uint8)).any():
            bad.append(f"{k}:{name}")
print("CMP_R8", "PASS" if not bad else "FAIL", f"{n} tensors", f"unequal={bad[:5]}",
      {k: (a[k], b[k]) for k in a if k.endswith(("_owned", "_n"))})
