"""Imported before poros (beside fed_rows.py): when H2H_VRAM_GIB is set, cap this process's CUDA
caching allocator at that many GiB with torch.cuda.set_per_process_memory_fraction, as a smaller
card would. It allocates nothing, so poros's own allocator settings still apply."""
import os

_gib = os.environ.get("H2H_VRAM_GIB")
if _gib:
    import torch

    _total = torch.cuda.get_device_properties(0).total_memory
    _frac = float(_gib) * 2**30 / _total
    torch.cuda.set_per_process_memory_fraction(_frac, 0)
    print(f"MEMFRAC {_frac:.4f} of {_total / 2**30:.2f} GiB: {float(_gib):.2f} GiB for this process",
          flush=True)
