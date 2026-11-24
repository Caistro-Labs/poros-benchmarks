# The SSD expert store on one RTX 4090: the runs behind two figures

Qwen/Qwen3-30B-A3B (NF4 experts in an SSD store), 512 tokens, 8 updates a run, on the release
tree before the freeze. The files of seven runs are in `rec/`, each as `<run>.yaml` (the config
`poros train` ran) and `<run>_events.jsonl`, `<run>_receipt.json`, `<run>_config.resolved.json`,
`<run>_fed_rows.json`, `<run>_sssprof.json` (the store reader's timers and byte counts).

## Path fast with a 4 GiB host cache against path fast

`poros/cli/verbs/shared.py` states 1.46 s an update against 3.2. Eight rows in one micro-batch;
s/update is the median of updates 3 to 8.

| Run | Path | s/update | Adapter |
|---|---|---|---|
| `m01_u14_fast_c4_a` | fast, `store_options: {cache: 4G}` | 1.461 | `fb1fb8fff0254c1d` |
| `m01_u14_fast_c4_b` | fast, `store_options: {cache: 4G}` | 1.454 | `fb1fb8fff0254c1d` |
| `m01_u14_fast_a` | fast | 3.180 | `fb1fb8fff0254c1d` |
| `m01_u14_fast_b` | fast | 3.190 | `fb1fb8fff0254c1d` |
| `m01_ram_ref` | efficient, experts in RAM (the reference) | 2.506 | `fb1fb8fff0254c1d` |

Each SSD run's adapter equals the RAM reference's on all 192 tensors
(`rec/cmp_m01_ram_ref__m01_u14_fast_a.json` and its three siblings).

## Bytes read an update on path efficient

`poros/plan/estimate.py` states that a block-major path reads about 1.95 times the experts an
update, at either batch shape. The store's experts are 14.97 GB.

| Run | Batch shape | GB read an update (median of updates 3 to 8) |
|---|---|---|
| `k04_u13_eff` | 8 rows x 1 micro-batch | 29.23 |
| `k04_u13_ssd_1x8` | 1 row x 8 micro-batches | 29.22 |

The count is the `bytes_read` of each step event's `sss` block, a running total.
