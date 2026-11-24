# RTX 4090: the MoE 1x8 grad-view fix (round 5)

The job scripts this page names (`.sh`) are not in the public tree: they ran on the build machines, delete folders under their home and name those machines' layout. Each ran Poros on the commit and settings the page gives and wrote the receipts here, which ship.

The GPU runs used tree 269d3bbcd4cff50829ff3a12aa2787a7713bb1af: 919f5875 with the fix and its tests. Commit
e7f51534 holds the same files on top of 7e4595d4. That commit was made while these runs went on, and
it only changes the Mac agents gate (`scripts/metal/agents_gate.py` and its test). `git diff 269d3bbc e7f51534` shows
just those two files.

The PRE tree is 919f5875 with the new tests and no fix (tree f00be09afc3b14df1ec8dc4027415bb8df8f5712).

All GPU work ran under `~/gpu.lock`:
- `h19main.sh`, one hold, 06:38:07 to 07:45:07 UTC on 2026-11-17 (`status.txt`);
- `h19extra.sh`, one short hold for a same-session pair, 07:56:35 to 08:03:42 UTC.

Run conditions:
- Every run unset PYTORCH_ALLOC_CONF and PYTORCH_CUDA_ALLOC_CONF, and recorded both as unset (`rec/*_env.txt`).
- Caches, temp, POROS_HOME, the SSD reader library and the downloads all sat in the run's own folder.
- Software: driver 580.173.02, torch 2.14.1+cu130, transformers 5.22.2 (`rec/h19_env_start.txt`).

"Before" is h18: the profiling session that found the bug, on 9c2b9400, earlier the same day (04:31 to 05:27 UTC). It
used the same machine, the same workload and three runs per 1x8 arm (`before/h18_runs.log`). In `poros/`, 9c2b9400 and
919f5875 differ only in the SSD reader library's folder, checkpoints, run creation, export and preflight. None of those
is on the MoE grad path.

## The bug and the fix

`poros.moe.fused.accumulate_adapter_grads` stored the first micro-batch's expert adapter grads as `param.grad`. Under
block-major batching (`poros/moe/batching.py:94`), those grads are views into `experts_backward`'s `(segments, *shape)`
buffers (`poros/moe/fused.py:540` at 919f5875). So each view kept all eight segments alive until the grads were
cleared.

h18 recorded the CUDA memory history of one 1x8 update (`before/h18_PROF_fc4_1x8_memsnap.txt`):
- At the peak, 9.375 GiB is held in 192 tensors from `fused.py:540`, via `batching.py:94` < `block_major.py:409`.
- The real adapter grads are 1.17 GiB.

The fixed tree's snapshot is the same run shape, 3 updates (`rec/MEM_fc4_1x8_h18_mem_u2_memsnap.txt`):
- The `fused.py:540` site is gone.
- The peak's largest holder is now the fast path's GPU expert cache (`residency.py:199`, 10.8 GiB). The fast path
  sizes that cache to the room left, so it got about 11 GiB instead of 2.8.

The fix:
- **When it copies.** It clones the first segment when that segment's buffer holds more than it. A one-segment call
  (8x1, or no batching) is stored as before.
- **Why the bits stay.** The later segments are added in the same order, so the result has the same bits.
- **Why not zeros.** It is a copy, not zeros plus the grad, because `0 + -0.0` is `+0.0`.
  `tests/moe/test_adapter_grad_storage.py` checks this, and rejects the zeros variant.

The dense fused LoRA backward (`poros/core/fused/lora.py`) had the same pattern at padded ranks (r8 runs as r16):
- Each LoRA A grad was a row view of the padded stack.
- Those rows are now copied when rp > r.
- Ranks 16, 32, 64 and 128 take the unchanged code path.

The other `param.grad =` sites take their grads from `autograd.grad`:
- `train/block_major.py` and `core/blockwise.py`: plain linears, plus the fused LoRA above.
- `moe/grouped.py` and `moe/modules.py`: `p[experts]`.

Those grads own their memory.

## Digests: expected vs the fixed tree

| Check | Expected | 269d3bbc |
|---|---|---|
| Qwen3-30B-A3B 1x8, ram_efficient | 9c627963dd5fe0df | 9c627963dd5fe0df |
| 1x8 ssd_efficient == ram | 9c627963dd5fe0df | 9c627963dd5fe0df, pass |
| 1x8 ssd_fast == ram | 9c627963dd5fe0df | 9c627963dd5fe0df, pass |
| 1x8 ssd_fastcache (fast + 4G) == ram | 9c627963dd5fe0df | 9c627963dd5fe0df, pass |
| 8x1 efficient (RAM) | fb1fb8fff0254c1d | fb1fb8fff0254c1d |
| 8x1 fast + 4G == efficient | fb1fb8fff0254c1d | fb1fb8fff0254c1d, pass |
| SSD store, 128 rows, fast + 4G == RAM | e0d484fcc3939cbf | e0d484fcc3939cbf, pass |
| G2: Qwen1.5-MoE-A2.7B 1x8, adapter | 92f6272cd2482deb | 92f6272cd2482deb |
| G2: losses and grad norms | 18c48e947c96ba28 | 18c48e947c96ba28 |
| Family gate paths, qwen3_moe | dd694a6f7f6150ef | dd694a6f7f6150ef |
| olmoe | d396c71fd5422bbc | d396c71fd5422bbc |
| qwen2_moe | e16cfdfa3dc51fcf | e16cfdfa3dc51fcf |
| deepseek_v2 | c0f23524a1c843da | c0f23524a1c843da |
| qwen3_5_moe | 4c9f5c4f940bf98c | 4c9f5c4f940bf98c |
| qwen3_5_moe_text | 4c9f5c4f940bf98c | 4c9f5c4f940bf98c |
| gemma4_moe | 9cb0d6fcbc9bca7a | 9cb0d6fcbc9bca7a |
| Dense LoRA layer at rank 8, PRE vs fixed | bitwise | 32 of 32 tensors bitwise |

Where the comparisons come from:
- **"pass" rows.** Every step's loss and grad norm are equal, and all adapter tensors are equal (192 at 30B, 96 for
  G2). See `rec/cmp_*.json`.
- **Expected values.** These are round 4's (`../round4/`) and h18's.
- **Family rows.** `fg_RESULT.txt` holds `FG_CHECKS=paths` on the seven MoE families: 37 PASS, 0 FAIL, 0 SKIP. Each
  row is seq == eff == fast == resident on 8 updates of mb2 x 4.
- **Dense rows.** `rec/dense_r8.txt` covers the output, the input grad and the 14 LoRA grads, at dropout 0 and 0.1,
  compared as raw bytes.

G2's peaks also equal round 4's: 4.418 / 4.651 GB. So this row's peak does not include segment buffers. Whether its
layers batch was not examined.

## 1x8 memory and speed, Qwen3-30B-A3B

Workload:
- oasst1, the first 64 rows at 512 tokens (`rows_real64.pt`, sha d1240490; `rec/rows_check.txt`).
- 8 updates of 1 x 8, r16, alpha 32, dropout 0, lr 2e-5, seed 42, deterministic.
- SSD arms read `~/sss/store`.

Columns:
- **s/update:** the median of updates 2..N and 3..N.
- **alloc / reserved:** the receipt's peaks, in GiB.
- **host:** peak RSS.
- **cache:** the GPU expert cache the run chose (`ran_compute`).
- **disk:** the SSD reader threads' bytes per update (sssprof, median of 2..N).

Runs:
- Before: three runs each.
- After: one run each, in the order D, A, C, 8x1 efficient, B, Unsloth, 8x1 fast + 4G, with 60 s idle before every
  SSD arm.

| 1x8 path | run | s/update 2..N | 3..N | alloc / reserved GiB | host GiB | GPU expert cache GiB | disk GB/update |
|---|---|---|---|---|---|---|---|
| ram_efficient | before r1 | 5.146 | 5.145 | 15.600 / 16.200 | 21.880 | 0.290 | 0.000 |
| ram_efficient | before r2 | 5.136 | 5.131 | 15.600 / 16.200 | 21.880 | 0.290 | 0.000 |
| ram_efficient | before r3 | 5.171 | 5.157 | 15.600 / 16.200 | 22.150 | 0.290 | 0.000 |
| ram_efficient | after | 5.248 | 5.240 | 7.400 / 8.030 | 22.070 | 0.290 | 0.000 |
| ssd_efficient | before r1 | 14.064 | 15.178 | 15.600 / 15.900 | 8.390 | 0.290 | 29.370 |
| ssd_efficient | before r2 | 13.465 | 13.459 | 15.600 / 15.900 | 8.380 | 0.290 | 29.370 |
| ssd_efficient | before r3 | 21.752 | 21.808 | 15.600 / 15.900 | 8.400 | 0.290 | 29.370 |
| ssd_efficient | after | 15.115 | 16.870 | 7.400 / 7.720 | 8.390 | 0.290 | 29.370 |
| ssd_fast | before r1 | 15.467 | 15.416 | 18.130 / 18.400 | 7.190 | 2.824 | 24.040 |
| ssd_fast | before r2 | 11.371 | 11.293 | 18.130 / 18.420 | 7.190 | 2.824 | 24.040 |
| ssd_fast | before r3 | 11.186 | 11.171 | 18.130 / 18.420 | 7.190 | 2.824 | 24.040 |
| ssd_fast | after | 5.995 | 5.993 | 18.130 / 18.400 | 7.180 | 11.008 | 6.360 |
| ssd_fastcache (fast + 4G) | before r1 | 10.632 | 10.303 | 17.940 / 19.930 | 11.650 | 2.798 | 16.540 |
| ssd_fastcache (fast + 4G) | before r2 | 10.660 | 10.650 | 17.940 / 19.670 | 11.650 | 2.818 | 16.540 |
| ssd_fastcache (fast + 4G) | before r3 | 9.717 | 9.717 | 17.940 / 19.640 | 11.640 | 2.818 | 16.540 |
| ssd_fastcache (fast + 4G) | after | 4.487 | 4.474 | 18.100 / 19.850 | 11.660 | 10.982 | 0.010 |
| Unsloth (anchor) | before r1 | 5.909 | 5.907 | 19.735 / 19.910 | 58.164 | - | - |
| Unsloth (anchor) | before r2 | 5.821 | 5.821 | 19.735 / 19.910 | 58.429 | - | - |
| Unsloth (anchor) | before r3 | 5.817 | 5.816 | 19.735 / 19.910 | 57.954 | - | - |
| Unsloth (anchor) | this session | 5.913 | 5.909 | 19.735 / 19.910 | 58.302 | - | - |

What changed:
- **The efficient paths.** They now peak at the 8x1 efficient run's 7.941 GB: 15.6 down to 7.4 GiB allocated.
- **The fast paths.** They spend the freed room on the expert cache, 2.8 up to 11.0 GiB. SSD reads per update fall
  from 16.54 to 0.01 GB (fast + 4G) and from 24.04 to 6.36 GB (fast).
- **Speed.** s/update falls from 9.7-10.7 to 4.49 (fast + 4G), and from 11.2-15.5 to 6.00 (fast).
- **ssd_efficient's speed is disk-bound and unchanged.** It reads 29.37 GB an update with a one-layer cache. Its
  updates slow late in the run, from 13.4 s to 23.0 s by update 8, as in h18's r1 and r3. The median of updates 2..4
  is 13.394 after, against 13.358 in h18's r1 and r2 (r3: 21.752).

Unsloth anchor, same session, 1x8:
- Unsloth: 5.913 s/update at 19.735 / 19.910 GiB, with 58.3 GiB host.
- Poros ssd_fastcache: 4.487 s/update at 18.10 / 19.85 GiB, with 11.66 GiB host.
- Poros ram_efficient: 5.248 s/update at 7.40 / 8.03 GiB, with 22.07 GiB host.

### ram_efficient's speed: a same-session pair (`h19extra.sh`)

ram_efficient's after run (5.248) was 1.5-2.2% above h18's 5.136-5.171, so `h19extra.sh` ran PRE and then the fixed
tree again, back to back. Both runs give 9c627963dd5fe0df.

| ram_efficient 1x8 | tree | s/update 2..N | 3..N | mean of updates 3..8 | alloc / reserved GiB |
|---|---|---|---|---|---|
| A_ram_1x8_pre (07:56) | PRE f00be09a | 5.139 | 5.114 | 5.087 | 15.60 / 16.18 |
| A_ram_1x8_fix2 (08:00) | fixed 269d3bbc | 5.171 | 5.188 | 5.191 | 7.40 / 8.03 |
| A_ram_1x8 (06:40, above) | fixed 269d3bbc | 5.248 | 5.240 | 5.250 | 7.40 / 8.03 |

On these runs the fixed tree is 0.6-2.5% slower on ram_efficient: 2..N medians 5.171 and 5.248 against 5.139, and
3..8 means 5.191 and 5.250 against 5.087. For comparison:
- h18's three PRE-equivalent runs spread 5.136 to 5.171.
- The copy is 192 clones and 1.17 GiB per update.

These runs do not settle whether that gap is real. The SSD fast paths gain 2.4x (fast + 4G) and 1.9x (fast).

## 8x1 is unchanged

The fix never copies at one segment (`test_one_segment_is_stored_as_is`).

| 8x1 | before | after |
|---|---|---|
| ssd_fastcache (fast + 4G), s/update 2..N / 3..N | 1.465 / 1.463 (h18) | 1.486 / 1.468 |
| ssd_fastcache, alloc / reserved GiB, host GiB | 18.00 / 20.36, 11.65 (h18) | 18.00 / 20.06, 11.64 |
| ssd_fastcache, GPU expert cache GiB | 10.923 (h18) | 10.943 |
| efficient (RAM), s/update 2..N | 2.514 (round 4, 9c2b9400) | 2.535 (3..N 2.527) |
| efficient (RAM), peak alloc / reserved GB | 7.941 / 8.647 (round 4) | 7.941 / 8.647 |
| 128 rows, RAM, s/update 2..N, peaks GiB | 2.503, 7.4 / 8.13 (round 4) | 2.510, 7.4 / 8.13 |
| 128 rows, fast + 4G, s/update 2..N, peaks GiB | 1.445-1.464, 18.18-18.22 / 20.22-20.26 (round 4's three 9c2b9400 runs) | 1.455, 18.19 / 20.24 |

## Tests

CPU (<host>; `cpu_suites.txt`), on 919f5875 with the fix:
- `tests/moe/test_adapter_grad_storage.py` fails on 919f5875 (1 failed, 1 passed) and passes with the fix.
- tests/moe, tests/sss, tests/metal and the block-major files: 1749 passed, 0 failed.
- Full torch suite: 4515 passed, 0 failed. That is the 4513 baseline plus the 2 new tests.
- Torch-free suite: 2657 passed, 0 failed.
- Ruff is clean.
- `tests/metal/test_any_mac.py` is byte-identical to 919f5875's, which holds the 774a46a97a51bfc0 reference, and it
  passes.

GPU (this machine), from `rec/pytest_pre.txt` and `rec/pytest_fix.txt`:
- **PRE.** The three storage tests fail: the CPU 8-segment test, the trainer's real 1x8 fused run
  (`test_one_by_eight_expert_grads_own_their_memory`) and the dense rank-8 test. The one-segment test passes.
- **Fixed tree, 77 passed.** That covers tests/moe/test_adapter_grad_storage.py,
  tests/gpu/test_moe_fused_block_major.py, test_moe_fused.py, test_moe_fused_families.py, test_fused_compute.py,
  test_rank_tiles.py and tests/moe/test_block_major_moe.py.
- **Dense rank 8.** Owned grads go from 7/14 to 14/14 at dropout 0, and from 13/14 to 14/14 at dropout 0.1. The bits
  are equal.

## Commands

On the GPU machine, in `~/rf/h19_269d3bbc/`:
- `src/` is `git archive 269d3bbc` and `pre/` is `git archive f00be09a`.
- The kit is in `kit/`.
- The triton, inductor and unsloth caches are copies.
- Run `setsid nohup flock ~/gpu.lock bash h19main.sh`, then `h19extra.sh`, which takes the lock itself after
  `H19 DONE`.

The scripts:
- `scripts/h19run.sh NAME TIMEOUT key=val...`: one `poros train` of the workload above. key=val overrides the config.
- `scripts/h19uns.sh NAME MB GA`: Unsloth through h18's `h2h_u6.py`.
- `scripts/h19summ.py`: the RUN lines in `rec/runs.log`.
- `scripts/table.py`: the table above.

The kit is h18's (`fed_rows`, `sssprof`, `memprof`, `h18prof`, `memsnap`, `check_rows`, `h2h_u6`, `compare_runs`),
plus `dense_r8.py` and `cmp_r8.py`. The family rows are this tree's own `scripts/family_gate.sh`, run with
`FG_FAMILIES=<the 7 MoE keys> FG_CHECKS=paths FG_ANYMODEL=0 FG_PRUNE=1`.

After their digests were taken, the adapters, optimizer states, memory snapshots, the G2 model and the family models
were deleted.

## Files

- `status.txt`: the hold's log, with every run's summary line and every digest comparison.
- `rec/`: per run, the config, env, log, events, receipt, resolved config, fed rows, sssprof / memprof and a 1 s GPU
  sampler. Also `cmp_*.json`, `runs.log`, `rows_check.txt`, the memory snapshot summary, the pytest outputs,
  `dense_r8.txt` and G2's download and digest lines.
- `fg_RESULT.txt`: the family gate's rows.
- `before/`: h18's RUN lines, environment and memory snapshot summaries (1x8 and 8x1).
- `cpu_suites.txt`: the <host> suites.
- `scripts/`: everything that ran.
