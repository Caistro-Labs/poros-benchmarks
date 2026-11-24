# The fused kernels at every LoRA rank, on every card

The job scripts this page names (`.sh`) are not in the public tree: they ran on the build machines, delete folders under their home and name those machines' layout. Each ran Poros on the commit and settings the page gives and wrote the receipts here, which ship.

Each padded LoRA rank R (a power of two, at least 16) has one launch configuration of every fused
Triton kernel, on every card: `poros.core.fused_tiles.tiles(R)` (dense) and `moe_tiles(rp)` (MoE
experts). A rank computes the same sums in the same order wherever it runs; the card never picks
a tile. Ranks up to 32 (dense) and 64 (MoE) keep the configuration every earlier receipt ran, so
their bits are unchanged. Above, an adapter tile is `[BLOCK, R]` fp32 and its shared memory grows
with R, so the blocks shrink until every kernel fits the smallest per-block shared memory the
supported card runs (101,376 B a block on sm_86 / sm_89 / sm_120; Poros supports bf16 cards
only: Ampere, Ada, Hopper, Blackwell); a call's rows go in chunks so the per-column-tile partial buffers stay
bounded whatever the call's rows (`partial_elems`: 64 MiB a buffer a launch at rank 64, 256 MiB
from 128).

| Kernel | R ≤ 32 | R = 64 | R ≥ 128 |
|---|---|---|---|
| `_lora_out_fwd` (BLOCK_M, BLOCK_N, warps) | 64, 128, 4 | 64, 128, 4 | 32, 64, **8** |
| `_lora_out_bwd` (BM, BN, warps, stages) | 32, 128, 8, 2 | 32, 128, 8, 1 | **16**, 64, **8, 2** |
| `_swiglu_fwd` | 64, 128, 4 | 32, 64, 4 | **32**, 32, 4 |
| `_swiglu_bwd` | 32, 64, 4, 1 | 32, 32, 4, 1 | 16, 32, 4, 1 |
| `_lora_rope_fwd` (BM, warps) | 64, 4 | 64, 4 | 16, **8** |
| `_lora_rope_bwd` (BM, warps, stages) | 32, 8, 2 | 32, 8, 1 | 16, **8**, 1 |
| rows in chunks | no | yes | yes |
| MoE `_gate_up_fwd` / `_mid_bwd` BN; `_down_fwd` / `_dx_rows` BN, warps | 64; 256, 8 (to RP 64) | | 32; 64, 4 (RP ≥ 128) |

In bold, what changed for R ≥ 128 at 2ce61d09 (the final rank-128 launch): at 4 warps the plain
linear's forward and backward and the rotary pair spilled registers, and rank 128 was slower than
PEFT's modules at Qwen2.5-72B's widths (below, *rank 128 at 72B widths: where the time went*).

## Ahead-of-time compiles: what fits each architecture

[`aot.json`](aot.json) (log: [`aot.log`](aot.log)) is every kernel the fused compute launches,
compiled by Triton 3.9.0 on a CPU-only host for each architecture (`triton.compile` with
`GPUTarget("cuda", cap, 32)`; nothing runs) at each rank's own tiles, dense in bf16 everywhere and
in fp16 on sm_70 / sm_75, the MoE experts in bf16 on sm_80 and newer, each loop kernel at its
configured stages and at one: 1,180 compiles, 0 errors (recompiled for the R ≥ 128 tiles above;
the fit of every rank on every architecture is the same as before)
([`scripts/aot_rank_tiles.py`](../scripts/aot_rank_tiles.py)). A rank fits an architecture when
every kernel fits its per-block shared memory at one stage or more. The largest rank that fits:

| Arch | Cards | Limit (B) | Dense, before | Dense, now | Dense, Qwen2 rotary at head dim 128, now | MoE experts, before | MoE experts, now |
|---|---|---|---|---|---|---|---|
| sm_70 (not supported: no bf16) | V100 | 98,304 | 32 | **128** | **128** | (bf16 only) | (bf16 only) |
| sm_75 (not supported: no bf16) | T4, RTX 20xx | 65,536 | 32 | **64** | **64** | (bf16 only) | (bf16 only) |
| sm_80 | A100, A30 | 166,912 | 64 (a) | **256** | **256** | 128 (b) | **256** |
| sm_86 | RTX 3090, A10, A40 | 101,376 | 32 | **128** | **128** | 64 | **128** |
| sm_89 | RTX 4090, L4, L40 | 101,376 | 32 (a) | **128** | **128** | 64 (b) | **128** |
| sm_90 | H100, H200 | 232,448 | 128 | **256** | **256** | 128 | **256** |
| sm_100 | B200, B300 | 232,448 | 128 (a) | **256** | **256** | 128 (b) | **256** |
| sm_120 | RTX 5090, RTX PRO 6000 | 101,376 | 64 | **128** | **128** | 64 (b) | **128** |

"Before" is the release-candidate tree before these tiles, where every rank took the R <= 32 tiles
(a rank past the column kept PEFT's modules there; earlier still it failed with Triton
`OutOfResources` at the first step), compiled the same way: [`before/`](before/) (its
[`env.txt`](before/env.txt); dense: [`dense_a.json`](before/dense_a.json) for sm_75 / 86 / 90,
[`dense_b.json`](before/dense_b.json) for sm_70 / 120, [`dense_c.json`](before/dense_c.json) sm_75
in fp16; the MoE experts: [`moe_a.json`](before/moe_a.json) for sm_75 / 86 / 90; the scripts and
logs beside them). (a) From the kernel that decides the column on those architectures, the
SwiGLU forward, compiled alone ([`swiglu_fwd_80_89_100.json`](before/swiglu_fwd_80_89_100.json):
at R = 64 and 128 it takes 106,496 / 204,800 B on sm_80 and sm_89, 100,352 / 198,656 B on
sm_100). (b) The MoE kernels were not compiled for that architecture: at RP 64 and 128 the largest of
them took 81,920 and 163,840 B on each of sm_75, sm_86 and sm_90, so the column is the largest of
those ranks within the architecture's limit. Ranks above 128 were not compiled before: 128 is the
most these columns can say (sm_80, sm_90, sm_100).
The most shared memory one kernel takes, now (bytes, at its best stage count; sm_86, the same on
every arch from sm_75 within 4 KB):

| R | Dense, no rotary | Dense rotary, head dim 64 / 128 | MoE experts |
|---|---|---|---|
| 64 | 61,440 (`_swiglu_bwd`; 65,536 on sm_70) | 36,864 / 57,344 | 81,920 (unchanged) |
| 128 | 79,872 (`_swiglu_bwd`) | 45,056 / 81,920 | 65,536 |
| 256 | 153,600 | 86,016 / 155,648 | 131,072 |

So ranks 64 and 128 run fused on every supported card: their largest kernel takes 81,920 B
(the Qwen2 rotary pair at head dim 128; the SwiGLU backward 79,872), under the smallest
supported limit, 101,376 B. The
pre-Ampere rows are informational only (those cards have no bf16 and are refused before the
download); no effort went into fitting sm_75's 64 KB, which R = 128 exceeds.
Rank 256 fits the
164-227 KB cards (sm_80, sm_90, sm_100) and keeps PEFT's modules (dense) or the reference experts
(MoE) on the 99 KB ones. Where the table and a card disagree, the card's own launch decides
(`poros.core.fused.fit.reason`, `poros.moe.fused.fit_reason`: each kernel launched once before
training), so a newer Triton that grew a kernel falls back with its reason instead of failing.

## Correctness

- **fp64, no GPU** ([`interp_cpu.log.txt`](interp_cpu.log.txt)): every dense kernel at R = 16, 32,
  64 and 128 under Triton's interpreter on the CPU, against the same math in fp64 at shapes with
  partial row and column tiles (`tests/gpu/test_rank_tiles.py`'s checks, run by
  `tests/core/test_rank_tiles_interp.py`): fp32 outputs within 2e-5 of fp64 (relative to the
  largest), the bf16 ones within 1e-2, and with the rows forced into dozens of chunks the same.
- **On an RTX 4090** ([`rtx4090/`](rtx4090/); 24 GB, the RTX 4090 host of
  [../README.md](../README.md), torch 2.14.1+cu130, Triton 3.8.0; the trees `rk1`, `rk2`, `rkF`
  and `rkG` are the release-candidate tree at four points, named in
  [`rtx4090/env.txt`](rtx4090/env.txt); `rkG`'s kernels and launches are the 0.3.0 tree's, and
  `rk1`'s are too at every rank but 128):
  - *the paths, bit for bit, at 64 and 128* ([`k1_rank.out.txt`](rtx4090/k1_rank.out.txt),
    [`rank_digest.py`](rtx4090/rank_digest.py), `rank_digest_*.json`): LoRA dropout 0.05, 2 updates
    of 2 micro-batches x 4 padded rows of 512 tokens (2,048 rows a call, so the row chunks run
    more than once). Every arm ran the fused compute (`ran_compute`):

    | Model | Rank | sequential = fast = efficient = fast resident | digest | PEFT's modules (sequential) |
    |---|---|---|---|---|
    | random Qwen2, head dim 128, MLP 5,632, 3 layers | 64 | PASS | `1c9db53e3f04166d` | `f101d855eadfc13b` |
    | | 128 | PASS | `4f697a2763f1e0e9` (`c3a4099b2ca277c4` with rkG's rank-128 launch, `1cd4dd004c5c3dc4` with rk1's) | `198b510d28fae7c1` |
    | random Qwen3-MoE, 16 experts of 256, 6 layers | 64 | PASS | `a8b11340ef35d67c` | reference: `a52435d626893e98` |
    | | 128 | PASS | `076fbf7778e950d2` | reference: `be1231b0a936dcd1` |

  - *the fused compute against PEFT's modules* ([`k2_refcheck.out.txt`](rtx4090/k2_refcheck.out.txt),
    `rk_r*_refcheck.json`, each run's receipt, events, config and log): Qwen2.5-0.5B, the family
    protocol of [../README.md](../README.md) (100 updates on oasst1, 8 rows of up to 512 tokens,
    lr 1e-4, dropout 0.05, path fast, block 4), alpha 2 x rank; the rule, declared in the job
    before it ran: mean and max |Δloss| of fused vs PEFT at micro-batch 1 at or under those of
    PEFT at micro-batch 2 vs 1.

    | Rank | fused vs PEFT, mean / max | PEFT mb2 vs mb1, mean / max | | s/update, fused / PEFT (mb1) | peak allocated / reserved GB, fused / PEFT |
    |---|---|---|---|---|---|
    | 64 | 0.00054 / 0.00158 | 0.02110 / 0.10457 | **PASS** | 2.556 / 2.564 | 2.562 / 3.666 against 2.130 / 4.677 |
    | 128 | 0.00063 / 0.00229 | 0.02164 / 0.10754 | **PASS** | 2.107 / 2.600 | 2.992 / 4.073 against 2.550 / 5.054 |

    Rank 128's rows are from the final rank-128 launch (`rT`: commit 2ce61d09,
    the tiles above; [`t3_proofs.out.txt`](rtx4090/t3_proofs.out.txt), `rT_r128_*`); PEFT's arms
    are k2's (PEFT's modules run none of these kernels). With the earlier launches it passed too:
    `rkG` 0.00057 / 0.00249 ([`m4_r128.out.txt`](rtx4090/m4_r128.out.txt)), `rk1` 0.00051 /
    0.00180.
  - *speed and VRAM* ([`t2_speed.out.txt`](rtx4090/t2_speed.out.txt), the tree before:
    [`t2_before.out.txt`](rtx4090/t2_before.out.txt); the job script `t2_speed3.sh` (not in the public tree),
    [`speed_table.py`](rtx4090/speed_table.py)): s/update, 3 repeats each, interleaved, the
    median of the repeats' medians. At 72B widths: q72v8 (8 random layers at Qwen2.5-72B's dims
    and vocab), path efficient, block 1, 8 rows of 512 tokens a micro-batch (4,096 rows a call),
    4 updates, each run's median of updates 2-4. At 0.5B: Qwen2.5-0.5B at the refcheck's
    settings (oasst1 rows of at least 1,200 characters, 1 x 8, dropout 0.05, path fast, block
    4), 8 updates, each run's median of updates 2-8. "Before" is the release-candidate tree at
    383d1294 (rank 128's launch `rkG`); PEFT's modules run none of these kernels, and rank 64's
    tiles did not change, so those columns are the new tree's alone:

    | Width | Rank | fused, before | fused, now | PEFT's modules | now vs PEFT | peak allocated / reserved GB, fused / PEFT |
    |---|---|---|---|---|---|---|
    | 72B | 128 | 2.659 | **2.096** | 2.492 | **15.9% faster** | 19.24 / 23.88 against 20.48 / 24.53 |
    | 72B | 64 | (unchanged) | 1.817 | 2.448 | 25.8% faster | 18.91 / 23.26 against 19.48 / 23.68 |
    | 0.5B | 128 | 2.070 | **2.162** | 2.625 | **17.6% faster** | 2.98 / 4.07 against 2.54 / 5.05 |
    | 0.5B | 64 | (unchanged) | 2.069 | 2.603 | 20.5% faster | 2.55 / 3.66 against 2.12 / 4.66 |

    Rank 128 at 72B widths was 6.7% slower than PEFT's modules (2.657 against 2.490,
    [`m4_r128.out.txt`](rtx4090/m4_r128.out.txt); 2.841 with the first rank-128 launch,
    [`m1_speed.out.txt`](rtx4090/m1_speed.out.txt)); it is now 15.9% faster, and 1.2 GB lower in
    allocated VRAM, as before. At 0.5B the calls are small (512 rows) and an update's time is
    mostly elsewhere: every kernel call is faster there too (`t1_kb_*`, 0.5B columns), but the
    update does not move beyond the run-to-run spread. Its steps take about 2.06 or 2.60 s,
    and how many of a run's take the slower one varies run to run: 4 of 21 before, 9 of 21 now,
    hence the medians; within each mode now is faster (2.053 against 2.070 s, 2.595 against
    2.610), and the 100-update refcheck arms take 2.282 s an update before, 2.287 now. Fused
    stays faster than PEFT's modules (2.625, every step in the slower mode).
    Rank 16 there: 1.543 against 2.439 ([`m1_speed.out.txt`](rtx4090/m1_speed.out.txt)).

  - *rank 128 at 72B widths: where the time went* ([`t0_profile.out.txt`](rtx4090/t0_profile.out.txt),
    [`prof128.py`](rtx4090/prof128.py), `t0_*_ks.json`; after: [`t0b_profile.out.txt`](rtx4090/t0b_profile.out.txt)):
    torch.profiler over one whole update of q72v8 (update 3 of 3), every kernel's device time.
    Rank 128's fused update took 3.05 s of device time against 2.21 at rank 64; the difference
    was in five kernels. Now 2.49 s (the update 2.10 s of wall time, against 2.66):

    | Kernel (calls) | rank 64 | rank 128, before | rank 128, now |
    |---|---|---|---|
    | `_lora_out_fwd` (40) | 17.5 ms | 322.2 | 35.7 |
    | `_swiglu_fwd` (128) | 95.8 | 281.4 | 231.6 |
    | `_swiglu_bwd` | 82.8 | 200.6 | 201.3 |
    | `_lora_rope_fwd` (32) | 99.0 | 162.1 | 37.3 |
    | `_lora_out_bwd` (24) | 9.8 | 109.9 | 11.5 |
    | the partials' sums (`reduce_kernel`) | 37.6 | 85.4 | 85.7 |

    Every launch candidate of each kernel at R = 128 was then compiled and timed on its own at
    72B and 0.5B widths, through the same helpers training calls (partial sums included), with
    its compiled shared memory, registers and spills ([`kbench.py`](rtx4090/kbench.py),
    `t1_kb_*`): at 4 warps `_lora_out_fwd` (32 x 64) spilled 1,184 registers, `_lora_out_bwd`
    (32 x 64) 788 and `_lora_rope_fwd` (16 rows, head dim 64) 56; at 8 warps (`_lora_out_bwd`
    16 x 64) 0, 0 and 22, and a 72B call
    went from 12.3 to 1.34 ms (o_proj with the residual), 6.7 to 1.0 (its backward) and 8.7 to
    2.0 (q with its rotary). `_swiglu_fwd` at 32 x 32 is the fastest that fits (20.5 -> 17.1 ms,
    14.9 -> 8.4 with the down dropout); for `_swiglu_bwd` and the 256 MiB partials nothing that
    fits was faster. The SwiGLU kernels stay the largest part: their fp32 LoRA products (exact
    IEEE dots, no TF32) at R = 128 are 2x rank 64's work.

  - *rank 16, unchanged*: the dense digest of the Qwen2 test model at rank 16 (paths sequential,
    fast, efficient and the offloaded head) is `c90b03c78f6d2542`, as before
    ([`k1_rank.out.txt`](rtx4090/k1_rank.out.txt)); the real Qwen3-30B-A3B's adapter digest is
    `fb1fb8fff0254c1d` at the 9 GiB cap and at the default cache, as before
    ([`k3_q30.out.txt`](rtx4090/k3_q30.out.txt)); and the committed family receipts of
    [../README.md](../README.md) (their own tree, named there) rerun on `rk1` at rank 16 -- each family's
    100-update fused refcheck arm and its four path arms -- give the same bits for all five
    families (Qwen2.5-0.5B, Qwen3-0.6B, Qwen3.5-0.8B, Gemma 3 270M, Gemma 4 31B's first 6 layers):
    every step's loss and grad norm `==` the committed events', every adapter digest the
    committed one ([`k4_fam.out.txt`](rtx4090/k4_fam.out.txt), the job script `k4_fam.sh` (not in the public tree)).
    On the final rank-128 launch (`rT`, [`t3_proofs.out.txt`](rtx4090/t3_proofs.out.txt)): the
    rank-16 dense digest is `c90b03c78f6d2542` and the real Qwen3-30B-A3B's `fb1fb8fff0254c1d`
    (9 GiB cap and default cache, the fed rows checked), as before; ranks 64 (dense
    `1c9db53e3f04166d`, MoE `a8b11340ef35d67c`) and the MoE at 128 (`076fbf7778e950d2`) keep
    their digests; `tests/gpu/test_rank_tiles.py` and `test_xcard_kernels.py` 57 passed; the
    `-m gpu` lane in one process: [`t4_gpu_lane_rT.txt`](rtx4090/t4_gpu_lane_rT.txt) (433 passed, 39 skipped, 0 failed; holds `t4_lockyield_rT.log`).

- **On the rented RTX 5090** ([`rtx5090/`](rtx5090/); sm_120, 32 GB, WSL2 on a Core Ultra 9
  285K with 31 GB of RAM, driver 621.14, torch 2.14.1+cu130, Triton 3.8.0; tree `rT`;
  the job script `r5.sh` (not in the public tree) / [`r5.out.txt`](rtx5090/r5.out.txt), the job script `r5b.sh` (not in the public tree) /
  [`r5b.out.txt`](rtx5090/r5b.out.txt)). q72v8 built there with the same
  [`mkmodel.py`](rtx5090/mkmodel.py) (the Qwen2 tokenizer beside it); the same settings as the
  4090's speed table (q72v8 4 updates, 0.5B 8 updates; `r5.sh`), 3 repeats:

    | Width | Rank | fused, before (383d1294) | fused, now | PEFT's modules | now vs PEFT | peak allocated GB, fused / PEFT |
    |---|---|---|---|---|---|---|
    | 72B | 128 | 1.749 | **1.486** | 1.797 | **17.3% faster** | 19.24 against 20.48 |
    | 72B | 64 | (unchanged) | 1.374 | 1.757 | 21.8% faster | 18.90 against 19.47 |
    | 0.5B | 128 | 0.955 | 0.985 | 1.199 | 17.8% faster | 2.95 against 2.51 |
    | 0.5B | 64 | (unchanged) | 0.955 | 1.218 | 21.6% faster | 2.53 against 2.09 |

  On this card rank 128 was already ahead of PEFT's modules at 72B widths (1.749 against 1.797);
  the retune takes 15% off it. The four paths bit for bit at ranks 64 and 128 with dropout 0.05,
  every arm fused: dense 64 `57cc69b19cfd0310`, dense 128 `bab7d32bce0dfda8`, MoE 64
  `27a716ff4d18c108`, MoE 128 `4b5c9e218657bfc7` (`rank_digest_*.json`; a card computes its own
  bits, the 4090's differ); the rank-16 dense digest `80eacec2eacde813`, the same as this card's
  on the tree before. `tests/gpu tests/moe tests/sss tests/test_run_block_major_cuda.py` in one
  process 677 passed, 0 failed ([`gpu_dirs.txt`](rtx5090/gpu_dirs.txt): the six that failed on
  this host before pass, below); the `-m gpu` lane 433 passed, 0 failed
  ([`gpu_lane.txt`](rtx5090/gpu_lane.txt)).

## The six tests that failed on the WSL2 RTX 5090 host

`gpu-dirs` of `scripts/xcard_check.sh` on tree a5525df8 failed six tests there; each was a
test-host assumption, and one showed a record a run lacked:

- `tests/moe/test_families.py::test_streamed_trains_bitwise_equal_to_resident[qwen3_5_moe]`,
  `tests/moe/test_fused_families.py::test_streamed_trains_bitwise_equal_to_resident[qwen]` and
  both tests of `tests/moe/test_prepare_wrapper.py` train a Gated DeltaNet MoE in bf16 **on the
  CPU**; this CPU's oneDNN refuses the bf16 backward of its depthwise conv1d ("DNNL does not
  support bf16/f16 backward on the platform with avx2_vnni_2"; a bf16 matmul's runs). A Poros run
  does not compute on the CPU: under a dispatch mode counting every aten op, real `poros train`
  runs on this host (Qwen2.5-0.5B on paths efficient and fast, a tiny Qwen3-MoE, a tiny Qwen3.6
  MoE with its Gated DeltaNet layers on paths efficient and sequential) ran no bf16 or fp16
  compute on the CPU, forward or backward: the only CPU bf16 ops are `detach`, `_to_copy`,
  `view` and `copy_`, the host copies of streamed weights
  ([`cpuaudit.py`](rtx5090/cpuaudit.py), `audit_*.json`). So the tests take the fixture
  `cpu_bf16_backward` (tests/conftest.py): it probes that conv1d once, and where oneDNN refuses,
  the test runs on PyTorch's native CPU kernels with oneDNN off and records why
  (`onednn_off`); every assertion is unchanged ([`six_host_tests.xml`](rtx5090/six_host_tests.xml):
  13 passed, `onednn_off` recorded on the 10 that train on the CPU).
- `tests/sss/test_ssd.py::test_pinned_budget_finds_a_ceiling_and_keeps_headroom` fakes a host
  that registers page-locked memory up to a ceiling, but under WSL2 `reset()` took the real WSL2
  child probe's budget instead: it now fakes a non-WSL host too (the WSL2 branch has its own
  tests).
- `tests/sss/test_ssd.py::test_cuda_reads_on_a_side_stream_match` asserted a pinned ring; that
  run's child probe could not pin 4 GiB (this host's page-locked ceiling varies: in `r5.sh`'s
  runs the probe pinned 4 GiB). It now asks this host's budget first: where it grants pins the
  ring must be pinned, as before; where it is 0 the ring must have fallen back to pageable
  memory with `pinning.staging()` saying unpinned, budget 0 and its refusals, and the side-stream
  reads are checked either way (with `POROS_WSL_PIN_PROBE=0`, no budget: 1 passed, ring
  unpinned).
- The product side: that fallback works (a dense run pinned and the same run with no page-locked
  budget: the same losses and grad norms over 4 updates, `PINFALLBACK ... True` in
  [`r5b.out.txt`](rtx5090/r5b.out.txt)), but a RAM-store run recorded it nowhere. A CUDA run's
  `receipt.json` now carries `host_staging` (mode pinned / partial / unpinned, the budget and
  its basis, the refusals: [`pin_off_receipt.json`](rtx5090/pin_off_receipt.json) says
  `unpinned`, "WSL2 (child probe failed or skipped)", 334 refusals), and a failed child probe
  is logged.
