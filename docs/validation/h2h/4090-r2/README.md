# Head-to-head on the RTX 4090, November 24, 2026: the release commit 1a8d7e912 against unsloth 2026.11.10

**What this is.** The November 18 head-to-head (`../4090/`) repeated on 2026-11-24 (UTC) on the same RTX 4090 machine. The
harness is unchanged: `conf.sh`, `lib.sh`, `prun.sh`, `urun.sh`, `setup.sh` and `kit/` (`h2h_u7.py`, `hostsamp.py`, `summ.py`).
The rows are the same pinned files (oasst_full128.jsonl, sha256 f9e914aa…, and rows_real128.pt, 2a9b186e…; `data_sha256.txt`), and so are the settings.
The arms are the two default arms only.

**Versions.**
- Poros is the release commit 1a8d7e912 (`versions.txt`).
- Unsloth is the newest on PyPI that day: unsloth 2026.11.10 and unsloth_zoo 2026.11.10, in their own venv
  (`unsloth_versions_date.txt`).
- The call to Unsloth needed no change.

**How it ran.** Each run's command, environment, launch and end times, host-memory samples, fed rows and record are beside
it. The three rounds were interleaved in November's relative order of the two arms: ram, uns / ram, uns / uns, ram.
`runs.log` lists the runs in order.

## Qwen3-30B-A3B @ad44e777, batch 1 × accumulation 8, 16 updates

| | Poros (defaults: RAM store, path efficient) | Unsloth (offload_layers auto) |
|---|---|---|
| s/update, median (range) | 4.6056 (4.5882–4.6067) | 8.8737 (8.8291–8.8747) |
| VRAM reserved, GiB | 9.107 (9.088–9.127) | 19.721 (all three) |
| host RAM, GiB, summed over the run's process tree | 19.850 (19.834–19.866), 2 processes | 22.899 (22.891–23.162), 35 processes at the peak; 11.19 (11.19–11.24) with 3 or fewer |
| first update, s | 122.4 (105.1–137.1) | 139.0 (133.5–155.3) |

**Poros's median against Unsloth's best run**, the rule of the November 18 comparison:
- Poros is 1.9× faster per update (1.92 unrounded).
- Poros uses 2.1× less VRAM (2.17 unrounded).
- Poros reaches its first update 9% sooner (1.09 unrounded).
- Host RAM: no ratio. Unsloth's process-tree peak sums 35 processes, which exist from about 120 s
  to 215 s of each run; its main process peaked at 10.82 to 10.87 GiB (`NOTE_unsloth_versions.md`,
  fact 5). Poros keeps the whole 4-bit base in host RAM.

Ratios are rounded to one decimal (a whole percent under 1.1×) against Poros: down where Poros is
ahead.

**Unsloth 2026.11.4 on November 18** (`../4090/arms/`): 5.764 s/update (best), 19.76 GiB VRAM, 18.07 GiB host RAM
(best). What differs on Unsloth's side between the two dates is in `NOTE_unsloth_versions.md`.

`rows_check_30b_ga8.json` checks that both tools were fed the same rows.
