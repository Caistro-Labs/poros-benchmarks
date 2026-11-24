# Note: Unsloth on 2026-11-18 and on 2026-11-24

These are facts read from the two sessions' own files. Neither session's logs state why Unsloth's step time changed.

**Step time.** On the 30B arm (Qwen3-30B-A3B @ad44e777, batch 1 × accumulation 8, 16 updates), Unsloth's s/update was:
- 2026-11-18, with unsloth 2026.11.4: median 6.181, best 5.764.
- 2026-11-24, with unsloth 2026.11.10: median 8.874, best 8.829.

## 1. The Unsloth stack on each date

From `../4090/versions.txt` (2026-11-18) and `versions.txt` (2026-11-24). The 2026-11-24 venv freeze also lists
cut-cross-entropy 25.1.1, torchao 0.18.0, torchvision 0.29.2 and trl 1.15.0; the 2026-11-18 versions file does not list
them. No xformers is installed on 2026-11-24.

| Package | 2026-11-18 | 2026-11-24 |
|---|---|---|
| unsloth | 2026.11.4 | 2026.11.10 |
| unsloth_zoo | 2026.11.4 | 2026.11.10 |
| peft | 0.21.2 | 0.22.0 |
| accelerate | 1.15.0 | 1.16.0 |
| torch | 2.14.1 | 2.14.1 (same) |
| transformers | 5.17.0 | 5.17.0 (same) |
| bitsandbytes | 0.50.2 | 0.50.2 (same) |
| triton | 3.8.0 | 3.8.0 (same) |

## 2. Where the model was placed: Unsloth's own lines, the same in all six runs

Every run, on both dates, logs the same two lines:

```
Unsloth: offload_layers = 'auto': the input embedding (0.58 GiB) and 28 of 48 decoder layers (8.13 GiB) stay in host RAM so 13.32 GiB stays free for training
Unsloth: Offloading embeddings to RAM to save 0.58 GB.
```

The harness's record of the placement (`unsloth` in each run's JSON) is identical on both dates:
- `offload_layers` auto, `swapped_layers` 28 of 48 decoder layers
- `pinned_bytes` = `host_bytes` = 8734896128
- `input_embedding_device` cpu, `gc` unsloth

No log line on either date mentions a fallback, a disk placement, or any other device placement.

**Host memory.** Host RAM is the anonymous + shared peak of the process tree. Run 1: anonymous memory at the peak
17.999 GiB (2026-11-18) against 23.001 GiB (2026-11-24); shared 0.066 against 0.161 GiB. VRAM reserved is 19.76 against
19.72 GiB.

**Timing details, run 1.** Load 132.18 s against 122.64 s; first update 158.65 s against 155.32 s. The 2026-11-24 first
update lists are in each Unsloth run's `a30_ga8_uns_r<n>_raw.json` (`update_s`).

## 3. Harness, rows and settings

- **The command line** (`a30_ga8_uns_r1_cmd.txt`) is identical on both dates, apart from the venv's and the run folder's
  paths: `h2h_u7.py --model Qwen/Qwen3-30B-A3B --rows …/rows_real128.pt --deterministic --out … --local-snapshot
  --offload-layers auto --mb 1 --ga 8 --updates 16 --time-limit-s 2400 --targets o_proj,gate_proj,up_proj,down_proj
  --freeze-targets o_proj`.
- **The rows** are the same files on both dates: oasst_full128.jsonl (sha256 f9e914aa78035bec…), rows_real128.pt
  2a9b186e4977f8d7…, and the fed-rows sha256 13b5db2ab23e7073….
- **The harness files** are byte-identical: the 2026-11-24 copies on the RTX 4090 machine have the same sha256 as the
  copies kept from 2026-11-18. The sha256 prefixes are:

| File | sha256 prefix |
|---|---|
| kit/h2h_u7.py | 186b353d82192419 |
| kit/hostsamp.py | 0714fb71191d8a17 |
| kit/summ.py | e39e7d44b6fea150 |
| urun.sh | 149dcbb3d340d5fe |
| prun.sh | 4cb3b6e20fc4efda |
| conf.sh | 79f7dbc7040ef022 |
| lib.sh | c842d3d8c5c63170 |

**The adapter.** The trained adapter's digest differs between the two dates (32856c23… against c83ac9b2…), with the same
rows and seed. So the two Unsloth versions do not compute the same bits.

## 4. The transformers version: each tool ran on the version its own install resolves to

Unsloth ran on transformers 5.17.0 and Poros on 5.22.2, the same on both dates. Neither was pinned by us: Unsloth's venv
was made with `pip install --no-cache-dir unsloth unsloth_zoo`, with no version given for any package.

- **unsloth caps transformers at 5.17.0**, in 2026.11.4 and 2026.11.10 alike. The `Requires-Dist` line of
  `unsloth-2026.11.10.dist-info/METADATA` in the installed venv, and the same line on PyPI for 2026.11.4, reads:
  ```
  Requires-Dist: transformers!=4.53.0,!=4.54.0,!=4.55.0,!=4.55.1,!=4.57.0,!=4.57.4,!=4.57.5,!=5.0.0,!=5.1.0,<=5.17.0,>=4.52.4
  ```
- **unsloth_zoo** caps it at `<=5.17.0` in 2026.11.4 and at `<=5.18.0` in 2026.11.10 (its `Requires-Dist` line on PyPI,
  and in the installed venv for 2026.11.10). unsloth's own cap is the tighter one, so pip resolves to 5.17.0 on both dates.
- **Poros 1.0.0** requires `transformers>=5.20,<5.23` on Linux (`pyproject.toml` at 1a8d7e912), which resolves to 5.22.2.

## 5. Host memory: the process count at the peak

Host RAM in these pages is RssAnon + RssShmem summed over the Unsloth run's process tree, sampled every 0.25 s by
the harness's host sampler, hostsamp.py (`a30_ga8_uns_r<n>_host.csv`: time, process count, the sums). The sampler records how many processes there
are, not their names or command lines.

| Run | Peak (GiB) | Processes at the peak | Seconds into the run | Peak while 3 or fewer processes (GiB) | When there were more than 3 |
|---|---|---|---|---|---|
| 2026-11-18 run 1 | 18.065 | 2 | 121.7 | 18.065 | never (2 or 3 throughout) |
| 2026-11-18 run 2 | 18.091 | 2 | 124.7 | 18.091 | never (2 throughout) |
| 2026-11-18 run 3 | 18.089 | 2 | 132.5 | 18.089 | never (2 throughout) |
| 2026-11-24 run 1 | 23.162 | 35 | 155.5 | 11.237 | 125.9 to 217.9 s, 29 to 36 processes (and 4 processes for 0.5 s at 5 s) |
| 2026-11-24 run 2 | 22.899 | 35 | 164.8 | 11.191 | 124.4 to 215.1 s, 9 to 35 processes |
| 2026-11-24 run 3 | 22.891 | 35 | 140.5 | 11.189 | 118.8 to 208.9 s, 4 to 35 processes |

- **When the children appear.** On 2026-11-24 the extra processes start a few seconds after the model has loaded
  (`load_s` 122.57, 119.14 and 113.73 s) and end about 60 s after the first update has finished (the first update took 32.6 s
  in run 1).
- **The main process alone.** The harness also records its own process's peak (`host_peak_anon_shmem_kb` in
  `a30_ga8_uns_r<n>_raw.json`). On 2026-11-18 it is 18.08, 17.88 and 18.16 GiB, reached while the model loads. On 2026-11-24 it is 10.87,
  10.82 and 10.82 GiB.
- **What the children are.** No line in Unsloth's output names them. On both dates the log has the same lines around that
  time, including `Unsloth: compiling each transformer block as one region (Qwen3MoeDecoderLayer).` No line in either
  log mentions a worker count, `num_proc`, a DataLoader worker or a compile worker.
- **Worker counts from the harness.** `h2h_u7.py` passes none: it has no `num_proc`, `num_workers`, DataLoader or compile
  thread setting. The file is byte-identical on both dates (sha256 186b353d82192419…, section 3). The run's environment
  (`a30_ga8_uns_r<n>_env.txt`) sets no TORCHINDUCTOR or worker variable on either date.
