# RTX 4090: earlier checks the package cites

The release's own checks on the RTX 4090 are in [`release-1a8d7e912/`](release-1a8d7e912/README.md), [`optimizer_selftest/`](optimizer_selftest/README.md)
and [`display_fix/`](display_fix/README.md). The folders below are earlier
development checks (the family gate runs Poros's checks on one model of every supported family
and records PASS or FAIL for each), kept only because a file of the package or a test names them. No figure in
the README, the docs or the paper comes from them.

| Path | What it is | Code | Cited by |
|---|---|---|---|
| `round8/fg14_RESULT.txt`, `round8/recheck/fg15_RESULT.txt` | the family gate's rows the model catalog cites | 5b7521d9 | `poros/catalog.toml` |
| `round3/` | the family gate on every supported family (`fg7_RESULT.txt`) | f163c28d | `poros/core/arch_registry.py`, `poros/moe/__init__.py` |
| `logs/fg_whole_*_RESULT.txt` | the family gate's checks on every layer of Qwen3.8-27B and Qwen3.6-35B-A3B | b6e03fc8 | `poros/core/arch_registry.py`, `poros/catalog.toml` |
| `round5_moegrad/` | the MoE gradient-view check at one row a micro-batch, before and after | 269d3bbc | a test |
