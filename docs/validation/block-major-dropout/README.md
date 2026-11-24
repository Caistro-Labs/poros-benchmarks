# Block-major accumulation and its loops: receipts

Block-major accumulation (`path: efficient` or `fast`) streams each frozen block to the GPU once
a step for all of the step's micro-batches instead of once per micro-batch. These receipts check,
on the release tree, that its update is the sequential loop's (`path: sequential`), bit for bit,
at `lora_dropout` 0.05 (`PorosConfig`'s default; `poros train`'s SFT and DPO recipes, for chat
and preference-pair rows, train at 0 unless set, so their configs here set 0.05), and measure
what its two paths cost.

## How block-major draws the sequential loop's masks

The sequential loop runs one micro-batch through every block, then the next micro-batch. Every
LoRA dropout in a block draws its mask from the generator as the block runs, and the backward's
recompute restores the state that block's forward started from (INV-1,
`poros/core/blockwise.py`), so a block's masks are fixed by the state it starts from.
Block-major runs the same forwards in another order: block 0 for every micro-batch, then block
1, and so on.

`poros/core/sequential_rng.py` walks the sequential order without the model. As each
micro-batch is fetched it records, block by block, the generator state there and advances the
generator as that block's dropouts would: one draw per LoRA-wrapped linear, over rows × length
× `in_features` elements, drawn on a scratch buffer of that size. On CUDA a draw of n elements
moves the Philox offset by an amount set by n and the device
([`philox_step.json`](receipts/philox_step.json)), so each size is drawn once and later draws of
it add that measured step. Every (micro-batch, block) forward and recompute then starts from its
recorded state, and after each forward (and the
embedding, and the head) Poros checks that the generator moved exactly as the walk did; a draw
the walk does not know about (the model's own dropout, say) stops the update before anything is
applied, and that update and the rest of the run train sequentially, with a warning. When the
update ends the generators hold what the sequential loop leaves, so checkpoints, resumes and the
next update are the same on either loop.

A MoE model's masks cannot be walked this way: each expert's adapter draws over the rows routed
to it, so the step depends on routing that block-major order has not computed when a later
micro-batch starts. On the RTX 4090 (128 SMs) a draw of up to 786,432 elements moves the offset
by 4, and past that by more: 8 at 1,048,576, 24 at 4,194,304, 88 at 16,777,216
([`philox_step.json`](receipts/philox_step.json)). An expert group's adapter input is its padded
routed rows × hidden size: on Qwen3-30B-A3B (hidden 2048), more than 384 rows move it by more
than 4. A MoE model's masks are keyed instead ([below](#moe-models-keyed-masks)).

## MoE models: keyed masks

A MoE layer's expert adapters draw their dropout over the rows routed to each expert, so how far
a draw moves the generator is known only once the layer has routed (above), and block-major
order cannot replay the sequential loop's stream past it. A MoE model's masks are keyed instead
(`KeyedRng`, `poros/core/sequential_rng.py`): a forward pre-hook puts the CPU and CUDA generators
at a pure function of (run seed, update, micro-batch, layer) before every decoder layer's
forward, in the sequential loop, in block-major order and in every recompute (the block's, the
expert function's), and the same function sets them before each micro-batch is fetched and when
an update's passes end. No draw moves another layer's masks, so the two loops, every block size
and the model fully resident draw the same ones, and a checkpoint resumes the same on either
loop. The keys fail closed: an update in which some micro-batch's forward never keyed some layer
stops before it is applied.

## Host and tree

One RTX 4090 (24 GB, PCIe 4.0 x16, 22-24 GB/s pinned host-to-device), 59 GiB of host RAM; the
release tree in two Python environments ([`receipts/rel_env.txt`](receipts/rel_env.txt)): torch
2.14.1+cu130, peft 0.20.0, bitsandbytes 0.50.2, with transformers 5.22.2, and with 5.20.2 (the
`[moe]` extra's range) for the MoE tests. The host's environment sets
`PYTORCH_ALLOC_CONF=expandable_segments:False`, which Poros leaves alone, so every run used the
default CUDA allocator. The logs are the record of the run: every stage, and each run's order
and result, is in [`receipts/queue.log.txt`](receipts/queue.log.txt) (the `rel_` lines), its
config in `receipts/<run>.yaml`, and its `config.resolved.json`, `receipt.json` and
`events.jsonl` beside it. In the log's first `rel_resume` pass the three resumes, and in its
first `rel_bits7` and `rel_base` passes the 7B runs, stopped before training (rc 2: the resumes'
run folders had been emptied, and the 7B weights were not yet on the host); every stage ran
again in full after them, and those runs are the receipts. `rel_one` and the last `rel_tests`
and `rel_keyed520` passes ran the release's final code; the stages before them ran its code
before an update of one micro-batch ran block-major too, which differs from it only in such an
update (every `poros train` run before them has four or eight micro-batches a step). [`scripts/compare_runs.py`](scripts/compare_runs.py)
compares two `poros train` output folders: every step's loss and grad norm (`events.jsonl`) and every tensor of
the final adapter, by `torch.equal` (`receipts/cmp_<a>__<b>.json`). The rows come from
[`scripts/make_data.py`](scripts/make_data.py) (OpenAssistant/oasst1, Apache-2.0).

## Block-major is the sequential loop's update

Each config trains three times on the release tree, with `path: sequential`, `path: efficient`
and `path: fast` (`receipts/rel_<config>_<path>.yaml`), and each block-major run is compared
with the sequential one: equal means every step's loss and grad norm and every
adapter tensor (`receipts/cmp_rel_<config>_seq__rel_<config>_<path>.json`). Qwen2.5 NF4 unless
said, 512 tokens, `lora_dropout` 0.05 unless said, block 4. Each run's `config.resolved.json`
records the loop its steps took (`ran_path`).

| Config | What it covers | `efficient` | `fast` |
|---|---|---|---|
| `q05_def` | 0.5B, text rows of uneven length, one row × 8 | equal (5 steps, 336 tensors) | equal |
| `q05_pad` | two rows × 4, each micro-batch padded to its longest | equal | equal |
| `q05_sft` | chats (SFT, `poros.Run`), `group_by_length`, two rows × 4 padded | equal (4 steps) | equal |
| `q05_dpo` | preference pairs (DPO, `poros.Run`), one pair × 4 | equal (3 steps) | equal |
| `q7_sft` | Qwen2.5-7B, chats (SFT), `group_by_length`, two rows × 4 padded | equal (4 steps, 392 tensors) | equal |
| `q7_pad` | Qwen2.5-7B, text rows, two rows × 4 padded | equal (4 steps) | equal |
| `q05_d0` (against `q05_def`'s `efficient` run) | the control: the same run at `lora_dropout` 0 | differs (grad norms from the first step, losses from the third, the adapter) | |

### Every block size

`q05_pad` and `q7_pad` at block sizes 1, 2 and 8 on each loop, against the sequential run at
block 4 (`cmp_rel_q05_pad_seq__rel_q05_pad_<path>_b<n>.json`; the 0.5B runs also against the
sequential run at their own block size):

| | block 1 | block 2 | block 4 | block 8 |
|---|---|---|---|---|
| Qwen2.5-0.5B, `sequential` | equal | equal | (the reference) | equal |
| Qwen2.5-0.5B, `efficient` | equal | equal | equal | equal |
| Qwen2.5-0.5B, `fast` | equal | equal | equal | equal |
| Qwen2.5-7B, `efficient` | equal | equal | equal | equal |
| Qwen2.5-7B, `fast` | equal | equal | equal | equal |

### Resumes

`q05_res` (two rows × 4 padded, 6 steps, `save_steps: 3`): the sequential run, stopped at its
`checkpoint-3` and resumed with `--resume-from latest` on `efficient`; the `efficient` run's
`checkpoint-3` resumed on `sequential` and on `fast`. Each resumed run's steps (4 and 5 in
`events.jsonl`) and final adapter equal the uninterrupted sequential run's, as does the
uninterrupted `efficient` run (`cmp_rel_q05_res_seq__rel_q05_res_<from>_then_<to>.json`).

### One micro-batch a step

Block-major runs at one micro-batch a step too: it streams each block once, as the sequential loop
does, but skips HF's per-layer recompute inside a block. `q05_one` (0.5B, text rows, two rows a
step), `m7_one` and `m7_one8` (7B, `synthetic`, one and eight rows a step), all at
`gradient_accumulation_steps: 1`, 512 tokens, 10 steps (the `rel_one` stage): every block-major
run equals the sequential one (`cmp_rel_<config>_seq__rel_<config>_<path>.json`; 10 steps, 336
and 392 tensors), and its `config.resolved.json` says it ran block-major (`ran_path`). Mean
s/step and peaks from each run's `receipt.json`:

| Run | sequential | `efficient` | `fast` |
|---|---|---|---|
| 0.5B, 2 rows: s/step | 0.61 | 0.60 | 0.51 |
| allocated / reserved (GB) | 2.65 / 4.33 | 3.25 / 5.50 | 3.27 / 5.59 |
| 7B, 1 row: s/step | 0.65 | 0.64 | 0.60 |
| allocated / reserved (GB) | 6.65 / 11.81 | 7.45 / 11.84 | 9.22 / 11.82 |
| 7B, 8 rows: s/step | 3.81 | 3.05 | 3.09 |
| allocated / reserved (GB) | 13.44 / 21.78 | 16.11 / 24.65 | 18.08 / 24.67 |

No slower anywhere, faster at eight rows a step, at a higher peak (the block's layers'
activations for the micro-batch, which the sequential loop's per-layer checkpointing does not
keep): `path: sequential`, or `block_size: 1`, where that peak does not fit.

### MoE models

Masks keyed per layer ([above](#moe-models-keyed-masks)). `tests/moe/test_block_major_families.py`,
per family (`qwen3_moe`, `olmoe`, `qwen2_moe`, `qwen3_5_moe`, `gemma4_moe`, `deepseek_v2`), tiny
models: block-major against the sequential loop at dropout 0 and 0.05 × experts in RAM or on
SSD × `pow2` or `single` grouping × block inputs on the GPU, pinned or in ordinary RAM (24
cases); at 0.05 on a nine-layer model with padded, shuffled micro-batches of two rows, the
sequential loop at dense blocks of 4 and 1, block-major at 1, 2, 4 and 8, and the model fully
resident (experts too, nothing streamed): every step's loss and grad norm, every adapter tensor
and the CUDA generator after, bit for bit; and a run stopped after its first of three updates and
resumed on block-major, sequential then block-major, and block-major then sequential, against the
uninterrupted run (a new epoch's shuffle falls inside the third update).

| Tests | transformers | Result |
|---|---|---|
| the family tests, six families | 5.22.2 | 28 passed a family, 168 in all ([`rel_fam_<family>.log.txt`](receipts/)) |
| the family tests, six families | 5.20.2 | 28 passed a family, 168 in all ([`rel_fam520_<family>.log.txt`](receipts/)) |
| `tests/moe/test_keyed_rng.py`, `tests/moe/test_block_major_moe.py`, `tests/test_run_block_major_cuda.py` | 5.20.2 | 29 passed, 7 skipped, as on 5.22.2 ([`rel_keyed520.log.txt`](receipts/rel_keyed520.log.txt)) |

OLMoE-1B-7B-0924 (`allenai/OLMoE-1B-7B-0924`, `olmoe`, 64 experts, 16 layers), `poros train`, 40
updates of one row × 8 micro-batches at 256 tokens over 320 English OpenAssistant rows
([`configs/moe_ref.yaml`](configs/moe_ref.yaml), `data/text320.jsonl`), `lora_dropout` 0.05:
`efficient` equals `sequential` (all 40 updates' losses and grad norms, all 64 adapter tensors:
[`cmp_rel_moe_ref_seq__rel_moe_ref_efficient.json`](receipts/cmp_rel_moe_ref_seq__rel_moe_ref_efficient.json)),
at 3.90 s an update against 6.78 s, and a peak of 4.44 GB allocated against 3.98 GB (each run's
`receipt.json`).

### Gemma 3 4B, vision tower included

`unsloth/gemma-3-4b-it` (an ungated copy of Gemma 3 4B IT, loaded as
`Gemma3ForConditionalGeneration` with its vision tower), `poros train` at the defaults:
[`configs/gemma3_ref.yaml`](configs/gemma3_ref.yaml) names no `path`, so the run decided
`efficient`, and the default all-linear LoRA puts adapters on the vision tower too (324 of the
800 adapter tensors). 10 updates of one row × 8 at 256 tokens, `lora_dropout` 0.05, against the
same config with `path: sequential`: equal (every update's loss and grad norm, all 800 tensors:
[`cmp_rel_gemma3_seq__rel_gemma3_default.json`](receipts/cmp_rel_gemma3_seq__rel_gemma3_default.json)).
A text-only forward never reaches the tower, so its adapters get no gradient on either loop
(every vision `lora_B` is still zero in the saved adapter) and the run checks that after its
first update.

### CUDA tests

`tests/test_run_block_major_cuda.py`, `tests/gpu/test_block_major_16bit.py`,
`tests/test_block_major_dropout.py`, `tests/moe/test_keyed_rng.py`,
`tests/moe/test_block_major_moe.py`, `tests/gpu/test_host_ram_nf4.py` and
`tests/test_gemma4_variants.py` on the release tree: 75 passed, 7 skipped (a MoE model's block
sizes, which are the family tests'; dense-only cases on the MoE model; a CPU-only case)
([`rel_cuda.log.txt`](receipts/rel_cuda.log.txt)). They cover dropout 0 and 0.1 on tiny Qwen2 and
`qwen3_moe`: `poros.Run` SFT with and without `group_by_length`, DPO pairs; both paths at blocks
of 1, 2 and 3 layers with a hook on every
adapter dropout (each mask a recompute draws is one its forward drew, and block-major's masks
are the sequential loop's); resumes at an epoch start and mid-epoch, from either loop to either;
a draw the walk cannot follow, which trains that update and the rest sequentially; `poros.Run`
at one and four micro-batches a step, block-major against the sequential loop, nothing
page-locked at load; tiny Gemma 3 and Gemma 4 VLMs with LoRA on the vision tower (block-major
the sequential update; a tower adapter given a gradient stops the run on either loop); the keyed
MoE runs, at four and one micro-batch a step, against fully resident, and resumed; the CUDA
generator after each run.

## What the paths cost on the RTX 4090

Every step time and peak in this README was measured with `fused_compute` off (PEFT's modules;
`moe_expert_compute: reference` on the MoE model), so its path-cost tables are that compute's.

`poros train` on `synthetic` rows at `lora_dropout` 0.05, one row × 8, 10 steps (6 at 2048
tokens), block 4 unless said, on a development tree whose training step is the release's (it
page-locked `efficient`'s host slabs at load; the release makes them at the first update).
The release's own runs of the 7B, 512-token rows (`rel_m7_<path>`, the `rel_measure7` stage, each
block-major run equal to `rel_m7_sequential`, which equals `m7_seq` bit for bit) took 4.81, 3.97,
5.51 and 3.34 s a step on
`sequential`, `efficient`, `efficient` at block 1 and `fast`, at the same peaks within 0.01 GB,
and 16.2 GB of host RAM at block 1 against 17.0. Mean s/step and peak VRAM and host RAM as each
run's `receipt.json` gives them (GB = 10^9 bytes; [`scripts/paths_table.py`](scripts/paths_table.py)
prints this table from the records in [`receipts/`](receipts/)). Every block-major run's losses,
grad norms and adapter tensors equal the same model's sequential run (`receipts/cmp_*.json`).

| Run | model, tokens | loop | block | s/step | allocated / reserved (GB) | host RAM (GB) |
|---|---|---|---|---|---|---|
| `m7_seq` | Qwen2.5-7B, 512 | sequential | 4 | 4.75 | 6.71 / 11.81 | 16.2 |
| `m7_eff` | Qwen2.5-7B, 512 | efficient | 4 | 3.98 | 7.63 / 11.98 | 16.2 |
| `m7_eff_b1` | Qwen2.5-7B, 512 | efficient | 1 | 5.43 | 6.53 / 10.88 | 17.0 |
| `m7_fast` | Qwen2.5-7B, 512 | fast | 4 | 3.34 | 9.56 / 11.81 | 16.2 |
| `m7_seq_2k` | Qwen2.5-7B, 2048 | sequential | 4 | 18.42 | 9.65 / 14.31 | 16.2 |
| `m7_eff_2k` | Qwen2.5-7B, 2048 | efficient | 4 | 14.47 | 13.52 / 16.05 | 17.0 |
| `m7_eff_b1_2k` | Qwen2.5-7B, 2048 | efficient | 1 | 15.36 | 10.65 / 15.01 | 19.4 |
| `m7_fast_2k` | Qwen2.5-7B, 2048 | fast | 4 | 13.65 | 16.40 / 18.48 | 16.2 |
| `m32_seq` | Qwen2.5-32B, 512 | sequential | 4 | 18.43 | 11.94 / 13.84 | 23.0 |
| `m32_eff` | Qwen2.5-32B, 512 | efficient | 4 | 12.18 | 13.18 / 14.99 | 23.6 |
| `m32_eff_b1` | Qwen2.5-32B, 512 | efficient | 1 | 14.22 | 10.47 / 11.61 | 25.7 |
| `m32_fast` | Qwen2.5-32B, 512 | fast | 4 | 9.96 | 17.50 / 19.52 | 23.0 |

`efficient` is block-major's least VRAM at a given block size: its block inputs wait in host RAM
and nothing is kept dequantized. `fast` is the fastest: inputs and one block's dequantized weights
stay on the GPU. At block 4 both are faster than the sequential loop everywhere measured. On
either path a block's backward holds the activations of the block's layers for one micro-batch
(the sequential loop's per-layer checkpointing holds one layer's) and every micro-batch's gradient
between blocks, so `efficient` peaks above the sequential loop at 7B (0.9 GB at 512 tokens, 3.9 GB
at 2048) and 32B (1.2 GB at 512), and below it at 72B with the tables in host RAM.
`block_size: 1` holds one layer's activations: the lowest peak, below the sequential loop's at 7B
and 512 tokens, 32B and 72B, for four times the block inputs in host RAM and a slower step than
block 4 on average (still faster than the sequential loop except at 7B and 512 tokens; at 72B
its first update took 39 s and the rest about block 4's).

Keeping HF's per-layer checkpointing inside a block's backward on `efficient` was measured too
(the `*_ckpt` runs, against the `*_nockpt` runs; the same values throughout). It lowered the peak
less than `block_size: 1` does and ran 26-30% slower, so neither path keeps it.

| Run | s/step | allocated / reserved (GB) |
|---|---|---|
| 7B, 512 tokens, `efficient` | 3.91 | 7.63 / 11.98 |
| the same, per-layer checkpointing kept | 4.92 | 6.88 / 11.98 |
| 7B, 2048 tokens, `efficient` | 14.47 | 13.52 / 16.05 |
| the same, per-layer checkpointing kept | 18.53 | 11.00 / 16.05 |
| the same, `block_size: 1` | 15.36 | 10.65 / 15.01 |
| 32B, 512 tokens, `efficient` | 12.16 | 13.18 / 14.99 |
| the same, per-layer checkpointing kept | 15.84 | 11.99 / 13.83 |

Block size stays the user's knob, with 4 as its default on both paths: `block_size: 1` was the
lowest peak wherever both were measured, but it holds four times the block inputs in host RAM
(72B at 512 tokens: 53.9 against 46.0 GB peak host RAM; at 2048 tokens, by the arithmetic, 43 GB
of block inputs instead of 11, beside the 45 GB the 72B run takes at 512), which a 64 GB host
does not have.

`poros plan` sizes each path from the sequential loop's receipt at the same settings plus what
the path holds on top (`poros.plan.path_extra`); `tests/test_plan_paths.py` checks that the
estimate is at least every peak measured here.

## Qwen2.5-72B on the same RTX 4090 ([`rtx4090_72b/`](rtx4090_72b/))

Measured on a development tree with `lora_dropout: 0`, 512 tokens of OpenAssistant rows, one row
× 8, `offload_embeddings_and_head: true` (the tables in host RAM), 8 updates; the mean s/update,
peak VRAM and peak host RAM from each run's `receipt.json` (GB = 10^9 bytes), as for the other
models. Every arm's 8 losses, grad norms and 1,120 adapter tensors equal the sequential loop's
(`cmp_*.json`).

| Run | s/update | allocated / reserved (GB) | host RAM (GB) |
|---|---|---|---|
| sequential, block 4 | 38.15 | 12.65 / 17.49 | 44.7 |
| sequential, block 1 | 37.89 | 13.34 / 17.64 | 44.7 |
| block inputs on the GPU, dequant cache on: `fast` | 20.14 | 18.46 / 23.92 | 44.7 |
| block inputs in host RAM, dequant cache on (no release path) | 20.76 | 16.84 / 21.29 | 46.2 |
| block inputs in host RAM, dequant cache off, block 4: `efficient` | 24.47 | 10.80 / 15.11 | 46.0 |
| the same, block 1 | 25.42 | 10.23 / 10.76 | 53.9 |

The same config with the tables on the GPU and the dequant cache on (`q72_eff_b4_def`) ran out of
memory at update 1 on this 24 GB card. In these files each command file's host line is dropped,
the configs read `oasst_full64.jsonl` (`data_64.json` says how it was made) from the folder they
run in, `machine_check.json` keeps only the GPU and bandwidth facts, the queue log is
`queue.log.txt`, and the tree is named by a label; nothing else.

## Reproduce

On one CUDA GPU, from a work folder holding the release tree as `rel/` and a Python environment
with Poros's dependencies, one pair at a time (each run's config is its `receipts/<run>.yaml`):

```bash
D=rel/docs/validation/block-major-dropout
python $D/scripts/make_data.py data          # text40, chats48, pairs48, text320
poros train $D/configs/q05_def.yaml --output-dir runs/a          # the default path
poros train $D/receipts/rel_q05_def_seq.yaml --output-dir runs/b    # the same with path: sequential
python $D/scripts/compare_runs.py runs/b runs/a out.json
python $D/scripts/paths_table.py $D/receipts rel_m7_sequential rel_m7_efficient rel_m7_fast
PYTHONPATH=rel python $D/scripts/philox_step.py philox_step.json   # this GPU's Philox steps
```

The comparison prints `"pass": true` when every step's loss and grad norm and every adapter
tensor are equal. The jobs wrote their logs as `.log`; here they are `.log.txt` (the repository
ignores `*.log`).

## Earlier trees

- **The release candidate before block-major accumulation was the default** (`base`, whose CUDA
  default was the sequential loop at `lora_dropout` 0.05): its runs of `q05_def`, `q05_pad` and
  `q7_pad` (`base_<config>`, the `rel_base` stage) equal the release's runs of the same configs
  on `efficient`, its default, bit for bit (`cmp_base_<config>__rel_<config>_efficient.json`; 5, 5 and 4
  steps, 336, 336 and 392 tensors), so a dense run at the defaults trains what it trained there.
- **Development trees.** The files without the `rel_` or `base_` prefix are runs of the trees
  between that candidate and the release; their configs name the loop the way those trees did
  (`block_major_accumulation`), which the release refuses. Block-major against that candidate's sequential loop on every config above, at block
  sizes 1, 2, 4 and 8 and across resumes (`<config>_base` against `<config>_new` and the
  `q05_pad_*`, `q7_pad_*` and `q05_res_*` runs): equal throughout. The six MoE families' tests
  (`famd_<family>`, 28 passed a family; `fam_<family>`, at dropout 0 before the keys, 12 a
  family) and the CUDA tests (`moe_cuda`, `dropout_cuda`, `core_cuda`, `runbm_pin0`, `expseg`,
  `bmmoe`, `gpu_parity`, `hostram_gate`): passed. OLMoE-1B-7B over 40 updates at 0.05
  (`moe_ref_*`, [`moe_refcheck.json`](receipts/moe_refcheck.json)): the per-update losses of
  the keyed masks and of the candidate's stream differ by a mean 0.012, about what either
  differs from no dropout (0.011 for the candidate's, 0.006 for the keyed one).
- **Two faults those trees had, which the release does not.** `*.hooks_on_proxies.log.txt`: the
  family tests at 0.05 with the keys' hooks on the collector proxies a streamed model's layer
  list yields, so that no layer was keyed. In the release the keys go on the layers the engine
  runs (`real_layers()`), and an update that leaves a layer unkeyed stops.
  `fam_deepseek_v2.hung1.log.txt`: the stack of a family run that hung. The collector, run
  during a reader thread's start-up while it held a lock of `threading`'s own, finalized an
  earlier case's SSD source, whose finalizer joined that source's readers, which takes the same
  lock. In the release the finalizer leaves idle readers to exit on their own
  (`poros/sss/ssd.py`, `tests/sss/test_ssd.py`).
