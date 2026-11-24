# One Mac: M4, 10 GPU cores, 16 GB

What `poros fit` reads on this class of Mac, and what `poros/catalog.toml`'s Mac rows cite.

- `receipts/<model>-s<length>[-r<rank>]/`: `receipt.json` and `events.jsonl`, a whole model's real
  `poros train` run. `gate-<gate>.json`, where a row has one, is that gate's verdict.
- `plan_sources.yaml`: pointers into those files, which the planner's figures for this class of
  Mac are built from.
- `probe/`: probes, not receipts.

All but `probe/` is written from the Mac's run folders, anew for each set of runs: nothing of
the last set is kept. Paths and machine names are out of every file, and numbers are as measured.

Of the runs at a setting, the one with the most steps counts, then the latest. A run's peak is the
process's own, and a first run packs the weights in that process, so it peaks higher than these. A
figure from few steps says so, and its mean includes the slower first step.
