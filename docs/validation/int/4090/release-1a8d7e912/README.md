# Release commit 1a8d7e912 on the RTX 4090: the release checks

**What this is.** Checks of the release commit 1a8d7e912b166c373fcfec862ad7297221d33a71 on the
RTX 4090 machine, made 2026-11-24 (UTC).

**How it ran.**
- The tree is a real git clone from the release bundle: `git fsck` clean, `status --porcelain` empty. A user-space git 2.43.0
  is on PATH, so the drivers read `rev-parse HEAD` and `status --porcelain` themselves.
- It runs on PYTHONPATH of the machine's CUDA venv: torch 2.14.1+cu130, transformers 5.22.2, unchanged.
- Every job was launched under `env -i`, so nothing from the machine's shell profile reaches a log.
- Every GPU step held `~/gpu.lock` and started only with no other compute process on the card.

**Folders.**
- `blockA/`: the untimed checks.
- `blockB/`: `poros prove` on Qwen2.5-7B and the recipe runs and their pre-run checks (`step1/` to `step7/`), with `blockB_status.txt`.
- `blockB_voice_rerun/`: a diagnostic re-run of the `voice` recipe; its README says why.
- `big72b/`: the `big-72b` and `big-72b-fast` recipes' runs; its README compares them with the earlier receipts.
- The comparison with Unsloth is in `h2h/4090-r2/`.

## Block (a): untimed checks

| Check | Result | Files |
|---|---|---|
| `scripts/moe_precision/device_selftest.py --device cuda --time` | rc 0, every check line `ok`. One step over 128M bf16 elements: ExpertAdamW 0.0072 s, torch AdamW (bf16) 0.0060 s | `blockA/out/selftest_cuda_time.txt` |
| `pytest -m gpu tests/gpu/test_expert_adamw_cuda.py tests/moe/test_block_major_moe.py` | `11 passed in 35.19s` | `blockA/out/gputests.txt` |
| Family gate, `scripts/family_gate.sh` with `FG_CHECKS=paths,moeref,resume FG_RESUME=qwen2,olmoe FG_ANYMODEL=0` | **68 checks pass** on the 15 supported families' rows (besides 16 model downloads and the environment and data lines). The 16th row, `inkling2_loop`, is refused as intended: the family is not validated in 1.0.0, so its runs are refused as an unvalidated MoE (rc 4) and its 7 checks fail as designed | `blockA/family/RESULT.txt`, `cmp/`, `logs/` |
| Block-major against the sequential loop, `lora_dropout` 0.05, 8 micro-batches: 7B (efficient, fast), OLMoE-1B-7B (efficient), Qwen2.5-32B (efficient, fast) | **equal**, all five pairs: every step's loss and grad norm, every adapter tensor | `blockA/status.txt` (`dropout` lines), `blockA/dropout*/` |
| ExpertAdamW quality on OLMoE-1B-7B, the optimizer study's scripts (`scripts/moe_precision/`, with its control arm), its RTX 4090 stage for OLMoE: 300 updates at 2e-5; arms new s42, fp32 s42, fp32 s43, control s42 (torch AdamW in bf16); every arm at 1a8d7e912 | **PASS** (`verdicts/cuda_olmoe.json`): the new rule is within the fp32 seed band at every held-out point; update 1 is equal; v tracks the exact average; all arms name 1a8d7e912 (check 4) | `blockA/olmoe_study/` |

The OLMoE study's figures, from each run's events, sample and receipt:

| Arm | Held-out at 299 | Median s/update | Peak allocated | Weights unchanged by an update (mean) |
|---|---|---|---|---|
| new (ExpertAdamW, fused kernel) | 1.634163 | 0.508 | 2.34 GB | 0.45% |
| fp32 s42 | 1.634395 | 0.736 | 3.50 GB | 0.12% |
| fp32 s43 | 1.628388 | 0.733 | 3.50 GB | 0.02% |
| control (torch AdamW, bf16) | 1.646271 | 0.495 | 2.10 GB | 47.1% |

- The new run's v/exact stays between 1.00003 and 1.00015 from update 20 on. The control run ends at 1.107.
- The new rule's step is 2.6% slower than the control's, at accumulation 1.
- The new run's `optimizer.json` records the fused kernel, with "the kernel stores the reference's bits".

Not in this folder: the full test suite logs from this machine. They print the machine's environment and are not published.
