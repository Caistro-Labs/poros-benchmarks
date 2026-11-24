# The dense fused compute: the kernels at every LoRA rank

Since 0.3.0 a dense NF4 run with PEFT LoRA adapters on a CUDA GPU computes its decoder layers
through `poros.core.fused` by default.

| Folder | What it is |
|---|---|
| [`rank_tiles/`](rank_tiles/README.md) | each rank's own kernel configuration: the ahead-of-time compile table for every supported card (`aot.json`, which `poros/core/fused/fit.py` and `poros/core/card.py` read), every kernel against fp64, and the four paths bit for bit at ranks 64 and 128 on one RTX 4090 and one RTX 5090 |
| `scripts/` | the scripts those runs used: the compile table's builder (`aot_rank_tiles.py`), the run and comparison helpers |
