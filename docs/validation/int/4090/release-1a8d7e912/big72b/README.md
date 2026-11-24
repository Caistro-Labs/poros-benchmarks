# Release commit 1a8d7e912 on the RTX 4090 machine: the two 72B recipes

**When and where.** 2026-11-24 (UTC), on the RTX 4090 machine, through the release-check driver (`scripts/certify/run_recipe.py`)
from a real git clone of 1a8d7e912 with git on PATH. Each receipt records commit 1a8d7e912 and `tree_clean: true`. Jobs ran
under `env -i`; nothing else of ours ran on the CPU or GPU during a recipe. The model is Qwen/Qwen2.5-72B-Instruct at the
Hub's main, 495f39366efef23836d0cfae4fbe635880d2be31 (`status.txt`).

| Recipe | Result | Seconds per step | GPU memory used | Reserved peak | Allocated peak | Host RAM peak | Held-out loss (base → after) | Updates |
|---|---|---|---|---|---|---|---|---|
| big-72b | PASS, 14:02 to 14:42 | 19.504 | 18.295 GiB | 18.156 GiB | 11.845 GiB | 45.113 GiB | 6.4862 → 5.4462 | 16; resume stopped after 9 and continued to the same adapter |
| big-72b-fast | PASS, 14:44 to 15:22 | 15.768 | 21.279 GiB | 21.672 GiB | 18.278 GiB | 42.864 GiB | 6.4748 → 5.4460 | 16; resume stopped after 9 and continued to the same adapter |

**Against the certified receipts of 67bd13cc** (`docs/validation/recipes/rtx4090/<recipe>/certification.json`):
- big-72b: the adapter sha256 959fdd15e489e324… and the held-out loss are equal; seconds per step +0.5%; GPU memory and
  reserved peak +0.4%; allocated peak and host RAM unchanged.
- big-72b-fast: the adapter sha256 b4dd6a078a1088ab… and the held-out loss are equal; seconds per step +0.5%; GPU memory,
  reserved and allocated peaks unchanged; host RAM −0.1%.

**Both recipes' check-fit** ran right after big-72b, with the weights on disk: both fit (reserved peak 18.078 and 21.672
GiB of 23.51), in `status.txt`.

**big-72b-fast started twice.** The driver first skipped it at 14:42 for disk (25 GB free against 20 + the 10 GB floor),
because `cert_keep.sh` had kept big-72b's run folder on disk for a later recording. That folder was removed (big-72b's
receipt was already copied) and the driver ran again at 14:44 (`run72f.sh`'s lines in `reorder_status.txt`); big-72b was
already passing and was not run again.

`big-72b/` and `big-72b-fast/` are the receipt folders (certification.json, the main run's receipt, events and resolved config, the logs).
`status.txt` and `reorder_status.txt` are the drivers' logs; `drivers/cert_keep.sh` is the driver as it ran: `cert.sh` with
two lines changed so the model and the recipe's folder stay on disk after a pass (`CERT_KEEP72=1`); its room check and
every pass condition are `cert.sh`'s own.
