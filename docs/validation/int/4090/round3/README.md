# RTX 4090, round 3: the family gate and the release checks at f163c28d

This folder holds one job's receipts on the RTX 4090 at commit f163c28d (its script was `int7.sh`): path digests, the SSD store's compares, the family gate on every supported family and the GPU tests.

The job scripts this page names (`.sh`) are not in the public tree: they ran on the build machines, delete folders under their home and name those machines' layout. Each ran Poros on the commit and settings the page gives and wrote the receipts here, which ship.

`int7.sh rintf163c28d r8bbf44f0 u18 rfB` (tree f163c28d2b627b1af96e1d34232a2458be76ff99), allocator
env unset (`int7_env.txt`: PYTORCH_ALLOC_CONF and PYTORCH_CUDA_ALLOC_CONF unset, HF_HUB_OFFLINE
unset, so the Hub was reachable), driver 580.173.02, torch 2.14.1+cu130, transformers 5.22.2.
Stage outputs are `.txt` copies of the job's `.out` and `.log` files. `int7_status.txt` is the
job's own timeline.

- Digests (`int7_dig.txt`): rank 64 dense `1c9db53e3f04166d` and MoE `a8b11340ef35d67c`, rank 128
  `4f697a2763f1e0e9` and `076fbf7778e950d2`, each on sequential / fast / efficient / fast_resident.
  Dense `c90b03c78f6d2542` on all four paths.
- Interleave vs 8bbf44f0, 3 repeats (`int7_abx.txt`): Qwen3-30B-A3B fast == efficient
  `fb1fb8fff0254c1d` in all 12 runs.
  - fast: 1.404 / 1.400 / 1.408 s (f163c28d) vs 1.328 / 1.401 / 1.414 s (8bbf44f0).
  - efficient: 2.530 / 2.524 / 2.522 s vs 2.519 / 2.519 / 2.513 s.
  - Peaks are identical: fast 19.326 / 20.579 GiB, efficient 7.941 / 8.647 GiB. The one 8bbf44f0
    efficient run at 8.626 reserved is the exception.
  - The q72v8 rows wrote no receipt. The rebuilt model had no tokenizer, so every run refused with
    "0 usable rows", on both trees (`int7_q72v8.txt`). `int7c.sh` (rebuild plus Qwen2.5's
    tokenizer) was cancelled before it ran: the gates moved to a later commit, 9c2b9400.
- SSD store (`int7_sss.txt`), 8 compares, all pass, adapters equal:
  - fast + 4 GiB cache vs RAM: `fb1fb8ff` at 64 rows mb8 and `9c627963` at mb1 x 8.
  - 128 rows, 6 interleaved rounds f163c28d vs 17eb22e9: `e0d484fc` in every round, equal to RAM.
    s/update: 1.452 / 1.449 / 1.450 vs 1.462 / 1.460 / 1.467.
- G2's MoE row, Qwen1.5-MoE-A2.7B (`int7_g2moe.txt`), rfB vs f163c28d: losses, grad norms and the
  96 adapter tensors are equal.
- Card check (`int7_xcard_log.txt`, `xcard_RESULT.txt`): 18 PASS and 1 FAIL.
  - The FAIL is gpu-dirs: 2 failed, 706 passed, both in `tests/sss/test_arena_release.py`.
  - Cause: tests/gpu, tests/moe and tests/sss run in one process. An earlier test's head arena,
    held in a reference cycle, was collected by this test's `gc.collect()` and gave its 1 GiB
    grant back into the test's patched ledger.
  - 6790acf1 collects garbage before the ledger counts. A CPU repro (such a cycle made at session
    start) fails before that commit and passes after.
  - The lanes at the code head (`int7d.sh`) were cancelled with the rest.
- Family gate, every families.tsv row and the any-model ladder (`fg7_RESULT.txt`): 167 PASS,
  0 FAIL, 1 SKIP. The SKIP is gemma4_moe refcheck, as before. The status line's "1 FAIL" is the
  job's grep counting the `FAILS=0` line.
- SSD family gate (`fgssd7_RESULT.txt`): 50 PASS, 0 FAIL, 1 SKIP (gemma4_moe's reference row, as before).
- GPU pytest, tests/gpu and tests/sss in one process (`int7_pytest.txt`): 241 passed, 5 skipped, 1 failed:
  `tests/sss/test_ssd.py::test_the_host_cache_serves_what_it_still_holds[1024]` at line 305, the
  count of disk reads with a host cache that holds the whole model (each slab read from disk once,
  every other read a cache hit). Every byte read was right (the loop's `torch.equal` asserts passed
  first); what differed is how many reads went to disk. It passed in every CPU suite, and in 60 of
  60 runs of it alone on the CPU host after this. It is timing-dependent (SSD store concurrency)
  and was left open in this round.
