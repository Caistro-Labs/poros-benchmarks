# Release commit 1a8d7e912 on the RTX 4090, block (b): the 7B prove and the recipes up to the MoE trio

**When and where.** 2026-11-24, 08:28 to 10:01 UTC, on the RTX 4090 machine.

**What ran.** The certification driver (`scripts/certify/run_recipe.py`, called by a job script) ran from a real git clone of
1a8d7e912 with git on PATH. Every receipt records commit 1a8d7e912 and `tree_clean: true`.

**Conditions.**
- These are published times, so nothing else of ours ran on the CPU or GPU: the full suite and block (a) had ended first.
- Jobs ran under `env -i`.
- One model was on disk at a time, downloaded at the Hub's main (the revision is in each step's log) and deleted when its
  recipes ended.
- Each recipe's `--check-fit` ran right after it, with its weights still on disk (`step6/`). The MoE no-recipe run is in
  `step7/`.

| Item | Result | Files |
|---|---|---|
| `poros prove Qwen/Qwen2.5-7B --steps 200 --seq-len 512 --json --force` | identical | `prove/prove.json`, `prove/prove-*.json` (its record), `release_commit.txt` |
| first-run | PASS | `step1/first-run/` |
| support, voice, prefer | PASS | `step2/` |
| big-32b | PASS | `step3/big-32b/` |
| moe-30b, moe-30b-ssd, moe-30b-8gb | PASS | `step4/` |
| MoE no-recipe run (16 updates) | PASS, 16 of 16, ran `efficient` | `step7/` |

## Against the certified receipts of 67bd13cc

Compared with `docs/validation/recipes/rtx4090/<recipe>/certification.json` as committed at 1a8d7e912:
- **Unchanged:** eval_loss is equal on every dense recipe and within 0.2% on the MoE ones.
- **Moved more than 10%:**
  - voice: s_per_step 2.5942 → 2.8861 (+11.3%); memory unchanged.
  - moe-30b: GPU memory used 8.100 → 8.959 GiB (+10.6%), reserved peak +10.7%, allocated peak 7.396 → 8.261 GiB (+11.7%).
    That is the expert optimizer's new state, within the optimizer study's expected growth of at most 1.2 GiB at rank 16.
  - moe-30b-8gb: allocated peak 4.488 → 5.074 GiB (+13.1%; the optimizer study expects at most +0.6 GiB at rank 8); s_per_step
    19.13 → 15.80 (−17.4%).
- **Everything else:** within ±6.6%.

`status.txt` and `blockB_status.txt` are the drivers' logs.
