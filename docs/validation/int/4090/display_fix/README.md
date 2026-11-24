# The display fix on the RTX 4090: first-run, in the session and through the driver

**When and what.** 2026-11-24, 11:06 to 13:59 UTC. A clone of 40c2e9e348811cc608bc0fa5c2aad53eacbf2554 (the frozen engine
1a8d7e912 plus six files under `poros/cli`, `res/diff_stat.txt`), clean (`status --porcelain` empty), on the RTX 4090
machine's CUDA venv. Jobs ran under `env -i`; GPU steps held `~/gpu.lock` with no other compute process on the card.

| Check | Result | Files |
|---|---|---|
| first-run through the release-check driver (`scripts/certify/run_recipe.py first-run`, 16 updates) | PASS; adapter sha256 ccdcb923ed250836… equal to the certified 1a8d7e912 first-run; every loss and both held-out scores equal (base 7.34639, after 5.95255) | `out/step1/first-run/`, `res/driver_first_run.out`, `res/compare.txt` |
| the same run typed in the Poros session (`res/session_line.txt`), recorded with `session_cast.py` (100 × 32) | finished in 127 s; adapter ccdcb923ed250836… equal; the 16 losses equal; held-out at 0 and 15 equal | `res/first-run-session.cast`, `res/session_cast.json`, `pack/session_run/` |
| `cast_check.py` on that recording | the finished card is drawn ("╭─ Trained"); 1238 synchronized frames, each whole in one event, none split | `res/cast_check.txt` |
| `pytest -m gpu tests/gpu/test_expert_adamw_cuda.py tests/moe/test_block_major_moe.py` | 11 passed in 34.05 s | `res/gputests.txt` |

- **The session's rows.** The session run reads the first-run recipe's 143 sample rows. They were written again with
  `run_recipe.py`'s own `sample_rows` and `write_rows` (kind, rows, length and recipe key from the driver's
  `certification.json`); their sha256 a997f1ba3e73fa78… equals the driver's (`res/rows_b.txt`).
- **A first attempt that is not a result.** The first in-session attempt typed `--data` with an empty path, because the
  driver had emptied its work folder after the recipe. It was stopped by hand; its recording is kept as
  `res/first-run-session_stopped-empty-data.cast`. `drivers/livecard1b.sh` is the script of the run above.
- **Edits to the recordings.** Home paths read `<home>`. In `first-run-session.cast` the machine's user name, where the
  screen cut a path short, is replaced by a six-letter stand-in, so the frame widths are unchanged.

`status.txt` is the drivers' log. `drivers/` holds the scripts as they ran.
