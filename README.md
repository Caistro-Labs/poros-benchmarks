# Poros benchmark logs

The raw logs behind the numbers in the [Poros](https://github.com/Caistro-Labs/poros) README, its
docs and its paper: each run's command, settings, environment, per-step events and results, and
the release's own checks. Poros itself, its tests and the paper are in
[Caistro-Labs/poros](https://github.com/Caistro-Labs/poros).

Every file is under `docs/validation/`, at the same path the Poros package, its docs and the paper
cite: a path such as `docs/validation/h2h/4090/runs.log` names the same file here. Each folder's
README says what its files are. Per-step training events are in each run's `events.jsonl`;
`poros schema events` prints their schema.

**Redactions in this public copy.** Absolute paths, host names and device ids are replaced by `<path>`, `<host>` and `<id>`; nothing else changes. Logs no public page cites are left out.

**Commit ids.** The commit ids in these logs are those of Caistro Labs' development repository.
The public history of Poros starts at the 1.0.0 export, so they do not resolve there. The package
released as v1.0.0 is the tree of commit 1a8d7e912b166c373fcfec862ad7297221d33a71 with six display
files changed, so that the run card shows from the start of a run in the session
(`poros/cli/train_cmd.py` and five under `poros/cli/ui/`), and the generated
`poros/recipes/certified.json` and `poros/plan/cells.json`. `scripts/certify/sources.toml` pins
the commit and names each display file with why it changes no run, and
`python scripts/certify/generate.py --release` refuses a package that differs from the pin in any
other file. The certification runs, the release's own checks and the Qwen3-30B-A3B comparison with
Unsloth (November 24, 2026) were measured at that commit; the Qwen2.5-72B comparison rows are
November 18's, at 5d75e31a. The `display_fix/` folders under `int/4090/`, `int/4060l/` and
`int/mac/` show runs with the six files changed ending at the certified adapters, and
`int/display_fix/resolve/` shows every built-in recipe resolving to the same run at the pin and
at the release. The earlier runs below say when they were measured.

## By machine

### RTX 4090 (24 GB)

| Folder | What it holds |
|---|---|
| [`recipes/rtx4090/`](docs/validation/recipes/rtx4090/) | the certification runs of the built-in recipes on the RTX 4090, one folder a recipe |
| [`int/4090/release-1a8d7e912/`](docs/validation/int/4090/release-1a8d7e912/README.md) | the release's checks on the RTX 4090: every supported family's paths and resume, the dropout-on checks, the GPU optimizer tests and the OLMoE optimizer study, `poros prove` on Qwen2.5-7B, the recipes' runs and their pre-run checks |
| [`int/4090/optimizer_selftest/`](docs/validation/int/4090/optimizer_selftest/README.md) | the expert optimizer's self-test on the RTX 4090 |
| [`h2h/4090-r2/`](docs/validation/h2h/4090-r2/README.md) | Poros and Unsloth 2026.11.10 on the RTX 4090, November 24, 2026: three interleaved runs an arm, with what differs from November 18 on Unsloth's side |

### RTX 4060 Laptop GPU (8 GB)

| Folder | What it holds |
|---|---|
| [`recipes/rtx4060-laptop/`](docs/validation/recipes/rtx4060-laptop/) | the certification run of `first-run` on the laptop, and two measured runs of `moe-30b-8gb` (certified on the RTX 4090), each folder with a README |
| [`int/4060l/optimizer_selftest/`](docs/validation/int/4060l/optimizer_selftest/README.md) | the expert optimizer's self-test on the laptop |

### Mac with an M4 and 16 GB

| Folder | What it holds |
|---|---|
| [`recipes/m4-16gb/`](docs/validation/recipes/m4-16gb/) | the certification runs of the built-in Mac recipes, one folder a recipe, each with the drive it read its model from |
| [`int/mac/launch-144-int/`](docs/validation/int/mac/launch-144-int/README.md) | Qwen2.5-72B-Instruct for 144 steps on the Mac, read from its internal SSD |
| [`int/mac/optimizer_selftest/`](docs/validation/int/mac/optimizer_selftest/README.md) | the expert optimizer's self-test on the Mac |

### Any machine

| Folder | What it holds |
|---|---|
| [`int/cpu/expert_adam_precision/`](docs/validation/int/cpu/expert_adam_precision/README.md) | the expert optimizer studied on the CPU, and the Mac kernel's arithmetic checked there |

### Earlier runs the package cites

Earlier development runs, kept only because a file of the package (or one of its tests) names
them. No figure in the Poros README, its docs or its paper comes from them, but
for the runs of November 17 to 20 named below, which the Benchmarks page gives as reference runs. Each folder's page says what is in it.

| Folder | What it is | Commit | Cited by |
|---|---|---|---|
| [`h2h/4090/`](docs/validation/h2h/4090/) | the comparison with Unsloth 2026.11.4 on November 18, 2026: with the earlier Unsloth runs of November 17 in `prefreeze/` | 5d75e31a | the Benchmarks page (More reference runs) and the paper |
| [`h2h/4060l/`](docs/validation/h2h/4060l/) | the RTX 4060 Laptop GPU's run of November 18, 2026, and its `poros prove` records of November 19 and 20 | 5d75e31a; `exact/` at 748099f5 and 03843905 | the Benchmarks page (More reference runs), `poros/catalog.toml` |
| [`block-major-dropout/`](docs/validation/block-major-dropout/README.md) | the block-major loops against the sequential one, and their memory and time at 7B, 32B and 72B | before the freeze | `poros/plan/estimate.py`, `poros/plan/cells.json` |
| [`fused/`](docs/validation/fused/README.md) | the fused kernels at every LoRA rank and their compile table | before the freeze | `poros/plan/make.py`, `poros/core/fused/fit.py`, `poros/core/fused_tiles.py` |
| `moe/` | MoE runs on an RTX 3090, an RTX A5000, an L40S and one RTX 4090; of the first three only the files the package names are in this copy | before the freeze | `poros/plan/cells.json`, `poros/plan/estimate.py`, `poros/core/config.py` |
| [`sss/4090/`](docs/validation/sss/4090/README.md) | the SSD expert store's runs | 65a51917 | `poros/plan/estimate.py` |
| [`metal/m4_16gb/`](docs/validation/metal/m4_16gb/README.md) | the measured figures the Mac's pre-run check reads on that class of Mac | before the freeze | `poros/metal/plan.py`, `poros/catalog.toml` |
| [`int/4090/`](docs/validation/int/4090/README.md), [`int/mac/`](docs/validation/int/mac/README.md) | the family rows and Mac rows the model catalog cites, and two test inputs | before the freeze | `poros/catalog.toml`, `poros/core/arch_registry.py`, `tests/test_recipe_certification.py` |

### Earlier runs behind the pre-run check's cells

These are earlier development runs, kept here only because the package's own tables cite them:
the check `poros train` makes before a run decides from measured runs, and the package's table of
them (`poros/plan/cells.json`) and its MoE host-RAM bound (`poros/plan/estimate.py`) name the file
behind each figure. No figure in the Poros README, its docs or its paper comes
from them. They were measured before the release's freeze; of their folders only the files the
package names are here. All are NF4 runs. The date and the code are given where the file, or the
table's own note of the run, records them.

| Card | Model | Measured | Code | Files |
|---|---|---|---|---|
| RTX 4090 | Qwen/Qwen2.5-32B | 2026-11-12 | not recorded | `docs/validation/rtx4090-rc/receipts/s32rb_eff_b1_mb4_receipt.json`, `docs/validation/rtx4090-rc/receipts/s32rb_eff_b1_mb8_receipt.json`, `docs/validation/rtx4090-rc/receipts/s32rb_seq_b4_mb4_receipt.json`, `docs/validation/rtx4090-rc/receipts/s32rb_seq_b4_mb8_receipt.json` |
| RTX 4090 | Qwen/Qwen2.5-72B | 2026-11-13 | b5e2a277 | `docs/validation/rtx4090-rc/runs.log` |
| RTX 4090 | Qwen/Qwen2.5-7B | 2026-11-13 | b5e2a277 | `docs/validation/rtx4090-rc/runs.log` |
| RTX 4090 | Qwen/Qwen3-30B-A3B | 2026-11-13 | not recorded | `docs/validation/rtx4090-rc/receipts/q30_c12_receipt.json`, `docs/validation/rtx4090-rc/receipts/q30g_c9_receipt.json`, `docs/validation/rtx4090-rc/receipts/q30g_def_eff_receipt.json`, `docs/validation/rtx4090-rc/receipts/q30rc_fz_mb4_receipt.json`, `docs/validation/rtx4090-rc/receipts/q30rc_fz_mb8_receipt.json`, `docs/validation/rtx4090-rc/receipts/q30rc_ref_mb4_receipt.json`, `docs/validation/rtx4090-rc/receipts/q30rc_ref_mb8_receipt.json` |
| RTX 3090 | Qwen/Qwen2.5-32B | 2026-11-04 | not recorded | `docs/validation/launch_3090/node6/receipts/q32_a/receipt.json` |
| RTX 3090 | Qwen/Qwen2.5-72B | 2026-11-03 | not recorded | `docs/validation/launch_3090/node5/receipts/x72_a/receipt.json` |
| RTX 3090 | Qwen/Qwen2.5-7B | not recorded | not recorded | `docs/validation/audits/audit_7b_3090_rerun/summary.json` |
| RTX 3090 | Qwen/Qwen3-30B-A3B-Instruct-2507 | not recorded | not recorded | `docs/validation/moe/3090/table6_qwen3_30b_a3b.json` |
| RTX 3090 | Qwen/Qwen3-30B-A3B-Instruct-2507 | committed 2026-10-08, at e390e19bd | not recorded | `docs/validation/moe/moe_streamed_load_3090.json` |
| RTX 3090 | Qwen/Qwen3-Coder-30B-A3B-Instruct | 2026-10-09 | not recorded | `docs/validation/moe/3090/qwen3_coder_30b_a3b_e2e.json` |
| RTX 3090 | Qwen/Qwen3.8-27B | 2026-11-04 | not recorded | `docs/validation/launch_3090/node7/receipts/q38_a/receipt.json` |
| RTX 3090 Ti | deepseek-ai/DeepSeek-V2-Lite | not recorded | not recorded | `docs/validation/moe/3090/deepseek_v2_lite_oasst1_200.json`, `docs/validation/moe/3090/deepseek_v2_lite_synthetic_200.json` |
| RTX 3090 Ti | allenai/OLMoE-1B-7B-0924 | not recorded | not recorded | `docs/validation/moe/3090/olmoe_oasst1_200.json`, `docs/validation/moe/3090/olmoe_synthetic_200.json` |
| RTX 3090 Ti | Qwen/Qwen1.5-MoE-A2.7B | not recorded | not recorded | `docs/validation/moe/3090/qwen15_moe_oasst1_200.json`, `docs/validation/moe/3090/qwen15_moe_synthetic_200.json` |
| RTX 5090 | Qwen/Qwen2.5-0.5B | not recorded | not recorded | `docs/validation/streamed-load/smoke_0p5b/summary.json` |
| RTX 5090 | Qwen/Qwen2.5-32B | 2026-06-10 | not recorded | `docs/validation/consumer-5090/nf4_5090_syn/events.jsonl`, `docs/validation/consumer-5090/nf4_5090_syn/summary.json` |
| RTX 5090 | Qwen/Qwen2.5-32B | not recorded | not recorded | `docs/validation/consumer-5090/nf4_5090_oasst1_masked/summary.json`, `docs/validation/streamed-load/sweep/v32b_rss_refix/summary.json` |
| RTX 5090 | Qwen/Qwen2.5-72B | not recorded | not recorded | `docs/validation/streamed-load/sweep/v72b_rss_refix/summary.json` |
| RTX 5090 | Qwen/Qwen2.5-7B | not recorded | not recorded | `docs/validation/rss/rss_7b_5090_summary.json`, `docs/validation/streamed-load/sweep/oob_7b/summary.json` |
| RTX A5000 | deepseek-ai/DeepSeek-V2-Lite | not recorded | not recorded | `docs/validation/moe/a5000/e2e_deepseek_v2_lite.json` |
| RTX A5000 | allenai/OLMoE-1B-7B-0924 | not recorded | not recorded | `docs/validation/moe/a5000/e2e_olmoe_1b_7b_0924.json` |
| RTX A5000 | Qwen/Qwen1.5-MoE-A2.7B | not recorded | not recorded | `docs/validation/moe/a5000/e2e_qwen1_5_moe_a2_7b.json` |
| RTX A5000 | Qwen/Qwen3-30B-A3B-Instruct-2507 | not recorded | not recorded | `docs/validation/moe/a5000/t6_poros_streamed_d0.05.json`, `docs/validation/moe/a5000/t6_poros_streamed_d0.json` |
| RTX PRO 6000 Blackwell | Qwen/Qwen2.5-32B | 2026-06-11 | not recorded | `docs/validation/downstream-eval/train_events.jsonl`, `docs/validation/downstream-eval/train_summary.json` |
| RTX PRO 6000 Blackwell | Qwen/Qwen2.5-32B | not recorded | not recorded | `docs/validation/audits/audit_32b_pro6000_rerun/summary.json` |
| RTX PRO 6000 Blackwell | Qwen/Qwen2.5-72B | not recorded | not recorded | `docs/validation/rss/rss_72b_pro6000_summary.json` |
| L40S | Qwen/Qwen3-30B-A3B-Instruct-2507 | not recorded | not recorded | `docs/validation/moe/moe_oasst1_quality_l40s_200.json`, `docs/validation/moe/moe_synthetic_quality_l40s_200.json` |
| A100 80GB | Qwen/Qwen2.5-32B | 2026-06-10 | not recorded | `docs/validation/bf16/blog/nf4_qwen32b_syn/events.jsonl`, `docs/validation/bf16/blog/nf4_qwen32b_syn/summary.json` |

## Using these logs with Poros

Some of the Poros tests and the paper's scripts read these files. They find them through
`POROS_BENCHMARKS`, the root of a checkout of this repository (unset, they look in the Poros
checkout itself, which in the development tree holds `docs/validation/`):

```bash
git clone https://github.com/Caistro-Labs/poros
git clone https://github.com/Caistro-Labs/poros-benchmarks
cd poros
POROS_BENCHMARKS=$PWD/../poros-benchmarks pytest -m "not gpu"     # without it, those tests skip
make -C paper POROS_BENCHMARKS=$PWD/../poros-benchmarks           # the paper
```

The paper's README (`paper/README.md` in the Poros repository) says what else its scripts read.

## License

Apache-2.0, the same license as Poros ([LICENSE](LICENSE)). Two scripts here,
`docs/validation/moe/4090/scripts/refcheck_hooks.py` and `docs/validation/moe/4090/scripts/prec64.py`,
are modified from transformers' code (its Qwen3MoeExperts loop), under the same license; the
Poros repository's
[NOTICE](https://github.com/Caistro-Labs/poros/blob/main/NOTICE) lists them.
