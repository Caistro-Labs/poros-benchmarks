# Qwen3-30B-A3B on one RTX 4090: the fused MoE compute and the head-to-head

Real Qwen3-30B-A3B (revision `ad44e777bcd18fa416d9da3bd8f70d33ebb85d39`), NF4 experts in pinned host
RAM, LoRA r16 on every routed expert, one RTX 4090 (24 GB) with 60 GiB of host RAM. Scripts in
[`scripts/`](scripts/); every number below is read from the files next to it.

## Why the fused compute is the MoE default

Acceptance rule for a fast path, fixed before these runs: no quality loss, meaning its refcheck is
no worse than the stock compute within noise. That rule is one-sided. The refcheck below is also
scored two-sided (fused within the noise of the reference in either direction), and that reading
fails, because fused trains to a lower loss. So one more check was fixed, before the default
changed and binding on it: on identical inputs, fused must be at least as close as the reference
to an fp64 computation of the same math. It is, on every layer measured.

### Precision against fp64 ([`precision/`](precision/), [`scripts/prec64.py`](scripts/prec64.py))

Per layer, one set of inputs for all three arms: the layer's block input from a forward of 8 real
rows (512 tokens each), the router's top-8 on it, the layer's LoRA (B seeded N(0, 1e-3)), and a
seeded output grad. Arms: fused (`poros.moe.fused`), reference (`poros.moe.modules`, pow2 groups),
and fp64 (transformers' `Qwen3MoeExperts` math over the same NF4-decoded weights, every operand
upcast, autograd in fp64). Rule, written in the script before it ran: fused's error is at most the
reference's, for the output and for the layer's four adapter grads together, in max-abs and in
relative L2, on every layer.

| layer | output rel-L2 fused / ref | output max-abs fused / ref | adapter grads rel-L2 fused / ref | adapter grads max-abs fused / ref |
|---|---|---|---|---|
| 6-layer checkpoint, 0 | 2.34e-3 / 5.47e-3 | 2.47e-3 / 5.44e-3 | 2.34e-3 / 5.17e-3 | 8.94e-5 / 1.93e-4 |
| 1 | 2.34e-3 / 5.44e-3 | 2.15e-3 / 4.04e-3 | 2.36e-3 / 5.19e-3 | 1.67e-4 / 3.76e-4 |
| 2 | 2.35e-3 / 5.50e-3 | 1.82e-3 / 4.51e-3 | 2.33e-3 / 5.11e-3 | 2.51e-4 / 4.58e-4 |
| 3 | 2.34e-3 / 5.46e-3 | 1.92e-3 / 4.22e-3 | 2.36e-3 / 5.25e-3 | 2.00e-4 / 5.99e-4 |
| 4 | 2.35e-3 / 5.47e-3 | 1.59e-3 / 3.76e-3 | 2.35e-3 / 5.17e-3 | 2.17e-4 / 3.51e-4 |
| 5 | 2.35e-3 / 5.46e-3 | 1.60e-3 / 3.98e-3 | 2.35e-3 / 5.19e-3 | 1.99e-4 / 3.73e-4 |
| real 30B, 24 | 2.25e-3 / 4.93e-3 | 2.07e-2 / 2.07e-2 (equal) | 2.34e-3 / 5.06e-3 | 1.27e-4 / 1.46e-4 |

Pass on all 28 comparisons. The one tie is layer 24's output max-abs: both arms miss the same
element by the same amount. The 6-layer checkpoint has the real model's dimensions and random
weights.

### Refcheck on real data ([`refcheck/`](refcheck/), [`scripts/refcheck_moe.py`](scripts/refcheck_moe.py))

30 updates on full-length oasst1 rows (no padding, so 8 x 1 and 4 x 2 are the same update), lr 1e-4,
dropout 0, path fast. The held-out loss is on 200 oasst1 rows at the end. Arms: fused mb8 x 1 and
mb4 x 2, reference mb8 x 1 and mb4 x 2, and stock (transformers' expert loop, eager attention,
transformers' norms and loss). The reference with single grouping ran out of memory at mb8 after
one update; its one update is in the file and is left out of the spreads.

- The reference family agrees with itself. Reference mb8, reference mb4 and stock differ pairwise
  by a mean |loss difference| of 0.0010 to 0.0011 per update, with no sign bias (mean signed
  difference within ±0.0002).
- Fused trains lower. Against every reference-family arm, fused's per-update loss is lower by
  0.0012 to 0.0016 on average (fused mb8: 0.00164 below reference mb8, 0.00160 below stock).
  Fused mb8 against fused mb4: 0.00098.
- Held-out loss after 30 updates: fused mb8 1.15133, fused mb4 1.15209; reference mb8 1.15243,
  reference mb4 1.15190, stock 1.15231.
- Two-sided reading: FAIL. Fused against the reference is 0.00187, more than any same-update
  noise pair (reference mb8 against mb4: 0.00100).
- One-sided reading (the acceptance rule): PASS. Fused is never higher than the reference family by
  more than its spread, and its held-out loss is within that family's spread or below it.

The precision check explains the gap. The reference rounds each expert's intermediates to bf16
before the next product (`x A^T`, then `@ B^T`, then the add). Fused keeps them in fp32 and rounds
once, so its error against the exact math is about half the reference's. Early Adam steps are
close to sign(g) per element, so the elements whose gradient sits near zero follow the sign of the
rounding. The arm with less rounding error moves more of them the right way. The same pattern
(fused lower by about 3e-4, the family's own spread about 2e-4) showed on the 6-layer checkpoint.
There, a gradient against an fp32 run of the same model agreed in sign on 97.6% of elements for
fused, against 97.4% for the reference and 97.5% for stock
([`precision/gradcheck_fp32_6layer.json`](precision/gradcheck_fp32_6layer.json), [`scripts/gradcheck.py`](scripts/gradcheck.py)).

### Determinism and bit-exactness under fused

- Run to run: three runs of the head-to-head config give the same adapter (digest
  `fb1fb8fff0254c1d`).
- sequential == fast == efficient at mb8 x 1, and sequential == fast at mb1 x 8, all on the real
  model.
- Streamed == resident: on the 6-layer checkpoint (`poros.moe.gates`' G-TRAIN loop, 4 steps).

## Head-to-head with Unsloth ([`receipts/`](receipts/), [`summary.json`](summary.json))

The same 64 oasst1 rows of 512 tokens, 8 per update, 8 updates, lr 2e-5. Poros: `poros train` with
moe_expert_compute unset (the fused default), path fast, expert cache cap 9 GiB. Unsloth:
`h2h_u6.py` (Unsloth 2026.11.4, transformers 5.17.0), mb8 x 1, offload auto.

How the numbers are read:
- load is from launch to the first training event.
- T(N) is from launch to the end of update N.
- T(100)* = T(1) + 99 x the median update. It is a projection, not a measurement.
- Peaks are over training.

Same inputs: each Poros run hashes the ids it fed (sha256 `d124049063f1b5a1...`). That equals
rows_real64.pt's hash and the hash each Unsloth run took of the rows it feeds in order
(`*_rows_check.txt`).

Workload difference: Unsloth also trains LoRA on attention: 642.5M trainable parameters (its
receipt) against Poros's 629.1M on the experts alone (computed from the adapter shapes).

**Cold loads.** Before each run, `scripts/evict.py` dropped the checkpoint from the page cache.

| run | load s | update 1 s | median 2-8 s | T(1) s | T(8) s | T(100)* s | alloc / reserved GiB |
|---|---|---|---|---|---|---|---|
| Poros, cold, round 1 | 120.7 | 5.98 | 1.441 | 126.6 | 137.1 | 269.3 | 16.88 / 18.15 |
| Poros, cold, round 2 | 121.2 | 6.12 | 1.460 | 127.3 | 137.9 | 271.8 | 16.88 / 18.15 |
| Poros, cold, loader as shipped | 120.8 | 5.98 | 1.441 | 126.8 | 137.3 | 269.5 | 16.88 / 18.15 |
| Unsloth, cold, round 1 | 131.8 | 16.91 | 1.699 | 148.7 | 160.6 | 316.9 | 19.80 / 20.36 |
| Unsloth, cold, round 2 | 129.6 | 16.80 | 1.697 | 146.4 | 158.2 | 314.4 | 19.80 / 20.36 |

Poros rounds 1 and 2 ran with a shard read-ahead hint, `POSIX_FADV_WILLNEED`, which Poros does
not ship. It measured no effect: the third row runs the loader without it, 120.8 s against
120.7 s.

**Warm loads.** Three interleaved rounds where whatever the previous run left in the page cache
stays; loads vary with it.

| run | load s | update 1 s | median 2-8 s | T(1) s | T(8) s | T(100)* s | alloc / reserved GiB |
|---|---|---|---|---|---|---|---|
| Poros, round 1 | 127.4 | 6.18 | 1.444 | 133.6 | 144.1 | 276.6 | 16.88 / 18.15 |
| Poros, round 2 | 115.5 | 6.08 | 1.451 | 121.6 | 132.0 | 265.2 | 16.88 / 18.44 |
| Poros, round 3 | 86.7 | 5.85 | 1.465 | 92.5 | 103.1 | 237.5 | 16.88 / 18.15 |
| Unsloth, round 1 | 138.4 | 16.93 | 1.699 | 155.3 | 167.2 | 323.5 | 19.80 / 20.36 |
| Unsloth, round 2 | 101.8 | 16.87 | 1.691 | 118.7 | 130.5 | 286.1 | 19.80 / 20.36 |
| Unsloth, round 3 | 114.5 | 16.81 | 1.701 | 131.3 | 143.2 | 299.7 | 19.80 / 20.36 |

Summary:
- Poros is faster per update (1.44-1.47 s against 1.69-1.70 s).
- Poros uses less VRAM (16.88 / 18.15-18.44 GiB against 19.80 / 20.36 GiB).
- Poros's update 1 is faster (5.9-6.2 s against 16.8-16.9 s).
- Cold, Poros is ahead from update 1: T(1) 126.6-127.3 s against 146.4-148.7 s.
- Warm, round 2 is the exception. Unsloth's load came out of a cache Poros had just warmed, and
  it led at T(1) and T(8) by 2.9 s and 1.5 s. Poros leads by T(100)*.

The first load quantizes the experts to NF4 on the GPU (`quant/`: every expert tensor
byte-identical to the CPU path's). With the quantization on the CPU, the same load took 229.1 to
230.1 s (`receipts/q30_h2h_r1` to `r3`, an earlier tree, warm rounds).

## Defaults: efficient and fast

moe_expert_compute and moe_expert_cache_gib unset, mb8 x 1, on the release tree (`receipts/q30k_*`).
Path efficient, the lowest-VRAM path, keeps a one-layer expert cache. Path fast sizes its cache
before update 2 to what fits the GPU: the VRAM the process could hold at setup, minus the reserved
peak through update 1 and its optimizer step, minus a margin of max(3 GiB, 0.2 x that VRAM). Each
run's `config.resolved.json` records the size and that sum (`ran_compute.moe_expert_cache_gib`,
`moe_expert_cache_reason`). For the capped rows, [`scripts/memfrac.py`](scripts/memfrac.py),
imported first, capped the process with `torch.cuda.set_per_process_memory_fraction` at 16 GiB
and at 16 GB (14.90 GiB), as on a smaller card.

| run | expert cache GiB | load s | update 1 s | median 2-8 s | T(1) s | T(8) s | T(100)* s | alloc / reserved GiB |
|---|---|---|---|---|---|---|---|---|
| efficient, default | 0.29 | 130.6 | 6.63 | 2.498 | 137.2 | 154.6 | 384.5 | 7.49 / 8.94 |
| fast, default | 10.43 | 131.7 | 6.16 | 1.423 | 137.8 | 148.2 | 278.7 | 17.57 / 19.56 |
| fast, cache cap 9 GiB | 9.00 | 129.5 | 6.14 | 1.470 | 135.7 | 146.3 | 281.2 | 16.33 / 17.55 |
| fast, default, process capped at 16 GiB | 4.87 | 131.6 | 6.07 | 1.612 | 137.7 | 149.3 | 297.3 | 12.15 / 14.26 |
| fast, default, process capped at 14.90 GiB | 3.97 | 130.9 | 6.16 | 1.681 | 137.1 | 149.0 | 303.5 | 11.28 / 13.41 |

- Fast at its default is ahead of Unsloth on both axes: 1.423 s against 1.69-1.70 s per update,
  19.56 GiB reserved against 20.36.
- Capped at 16 GiB it fits its cache to the cap and trains without running out of memory, at
  1.612 s per update.
- Efficient at its default is the low-VRAM point: 7.49 / 8.94 GiB at 2.50 s per update.
- The 9 GiB-cap arm (the head-to-head config) holds 16.33 / 17.55 GiB here, against 16.88 / 18.15
  in the head-to-head rounds above (an earlier tree), at the same speed within noise and with the
  same digest.
- All five give the head-to-head digest `fb1fb8fff0254c1d`.

How the margin was chosen (`receipts/q30h_*`, `q30i_*`, earlier trees, same digest):
- With the peak taken before update 1's optimizer step (which makes Adam's state, 2.2 GiB here) and
  a 0.3 margin, the cache came to 10.32 GiB: 1.427 s at 17.49 / 19.27 GiB (`q30h_def_fast`).
- With the peak through the step and 0.3, it came to 8.13 GiB: 1.486 s at 15.45 / 17.55
  (`q30i_def_fast`).
- The release takes the peak through the step and 0.2, which leaves about what the first left free.
- The `_b16_fast` rows ran beside another process holding 7.1 GiB of the card.

Before the fitted cache, path fast grew it to a quarter of the expert bytes (3.48 GiB here):
1.675 s at 10.80 / 13.04 GiB (`q30f_def_fast`).

## Bit-exactness on the real model

All of these give the same adapter, digest `fb1fb8fff0254c1d`:
- sequential == fast == efficient at mb8 x 1 (cache cap 9 GiB);
- both defaults, cached and uncached;
- on the release tree, both defaults, the 9 GiB cap, and fast under the 16 GiB and 14.90 GiB
  per-process caps (`receipts/cmp_q30k_*`);
- three head-to-head rounds;
- two cold rounds.

sequential == fast also holds at mb1 x 8 (digest `9c627963dd5fe0df`).

| run | load s | update 1 s | median 2-8 s | T(1) s | T(8) s | T(100)* s | alloc / reserved GiB |
|---|---|---|---|---|---|---|---|
| mb8 x 1, sequential | 130.3 | 4.93 | 2.521 | 135.2 | 152.8 | 384.8 | 8.48 / 10.60 |
| mb1 x 8, sequential | 129.7 | 15.20 | 12.743 | 144.9 | 234.1 | 1406.5 | 7.55 / 9.25 |
| mb1 x 8, fast | 129.4 | 8.44 | 4.393 | 137.8 | 168.5 | 572.7 | 17.12 / 19.89 |

Streamed == resident on the real 30B ([`precision/gate30_streamed_vs_resident.json`](precision/gate30_streamed_vs_resident.json),
[`scripts/gate30.py`](scripts/gate30.py)):
- Both arms load the experts from the quantized .sss store, so there is no bf16 copy in host RAM.
- Resident ran out of memory at mb4: 22.77 GiB peak allocated on the 23.52 GiB card.
- It fits at mb2: 22.10 / 22.53 GiB. Streamed at mb2 holds 5.46 / 6.37 GiB.
- Over two SGD steps the losses (1.40094, 1.37926), the grad norms and all 192 adapter tensors
  are bitwise equal.

## Reproducing

All scripts are in [`scripts/`](scripts/).

**Inputs**
- `make_data.py OUT Qwen/Qwen3-30B-A3B 64 512` builds the rows: `oasst_full64.jsonl` (what
  `poros train` reads) and `rows_real64.pt` (the same ids in the order Poros feeds them, which the
  Unsloth harness reads).
- `mk30.py OUT --layers 6` builds the 6-layer random-weight checkpoint.
- `full_rows.py` keeps the full-length oasst1 rows of the refcheck.

**Runs**
- `poros train`: the configs are `receipts/*.yaml`. `fed_rows.py`, imported first, records the ids
  each run fed, and `check_rows.py` compares them with the rows and with Unsloth's hash.
  `memfrac.py`, imported first with `H2H_VRAM_GIB=16` (or `14.9`), caps the process for the
  capped rows.
- `h2h_u6.py --model Qwen/Qwen3-30B-A3B --rows rows_real64.pt --deterministic --local-snapshot
  --offload-layers auto --mb 8 --ga 1 --updates 8`: Unsloth.
- `evict.py SNAPSHOT_DIR` before a run makes it a cold load.

**Checks**
- `refcheck_moe.py`, with `refcheck_hooks.py` imported for the stock arm and the held-out loss.
- `prec64.py` for the fp64 precision table.
- `gradcheck.py` for the fp32 gradient check.
- `gate30.py` for streamed == resident on the real model.
- `quantcheck.py` for the GPU quantization's bytes.

**Tables**
- `summarize.py receipts summary.json` writes the head-to-head tables.
