#!/usr/bin/env python3
"""summarize.py RUNS_DIR OUT.json -- one row per run of the 4090 Qwen3-30B-A3B receipts, the table the README shows
and poros plan's cells read (docs/validation/plan_sources.yaml).

Poros runs (RUNS_DIR/<name>_events.jsonl, _receipt.json, and the "start <name>" line of RUNS_DIR/runs.log, written
when the run was launched): load = launch to the setup event; update 1; the median of updates 2..N; T(1) and T(N) =
launch to the end of update 1 / N; peak allocated and reserved GiB over the steps (the events' vram_peak_mb and
vram_reserved_mb); peak host RAM GiB (the receipt's peak_host_ram_gb). Unsloth runs (RUNS_DIR/<name>.json from
h2h_u6.py): the same from its own clock (epoch_start, epoch_train_start, update_end_epoch) and its peaks. T(100)* =
T(1) + 99 x median is a projection, kept apart as t100_projected_s."""
import datetime, glob, json, os, statistics, sys

D, OUT = sys.argv[1], sys.argv[2]
G = 2**30
runs = open(os.path.join(D, "runs.log")).read().splitlines()


def launch(n):
    line = [x for x in runs if f" start {n} " in x][-1]
    return datetime.datetime.strptime(line[:20], "%Y-%m-%dT%H:%M:%SZ").replace(
        tzinfo=datetime.timezone.utc).timestamp()


rows = {}
for ev_path in sorted(glob.glob(os.path.join(D, "*_events.jsonl"))):
    n = os.path.basename(ev_path)[: -len("_events.jsonl")]
    ev = [json.loads(line) for line in open(ev_path)]
    st = [e for e in ev if e["event_type"] == "step"]
    t0 = launch(n)
    s = [e["data"]["s_per_step"] for e in st]
    rec = json.load(open(os.path.join(D, f"{n}_receipt.json")))
    rows[n] = {"tool": "poros", "updates": len(s), "load_s": round(next(
        e["timestamp"] for e in ev if e["event_type"] == "setup") - t0, 1),
        "update1_s": round(s[0], 3), "median_s": round(statistics.median(s[1:]), 3),
        "t1_s": round(st[0]["timestamp"] - t0, 1), "tn_s": round(st[-1]["timestamp"] - t0, 1),
        "peak_alloc_gib": round(max(e["data"]["vram_peak_mb"] for e in st) * 1e6 / G, 2),
        "peak_reserved_gib": round(max(e["data"]["vram_reserved_mb"] for e in st) * 1e6 / G, 2),
        "peak_host_ram_gib": round(rec["peak_host_ram_gb"] * 1e9 / G, 2)}
for p in sorted(glob.glob(os.path.join(D, "u30*_h2h_r*.json"))):
    d = json.load(open(p))
    s, t0 = d["update_s"], d["epoch_start"]
    rows[os.path.basename(p)[:-5]] = {
        "tool": "unsloth", "updates": len(s), "load_s": round(d["epoch_train_start"] - t0, 1),
        "update1_s": round(s[0], 3), "median_s": round(statistics.median(s[1:]), 3),
        "t1_s": round(d["update_end_epoch"][0] - t0, 1), "tn_s": round(d["update_end_epoch"][-1] - t0, 1),
        "peak_alloc_gib": d["peak_allocated_gib_train"], "peak_reserved_gib": d["peak_reserved_gib_train"]}
for r in rows.values():
    r["t100_projected_s"] = round(r["t1_s"] + 99 * r["median_s"], 1)
json.dump(rows, open(OUT, "w"), indent=1)
for n, r in rows.items():
    print(f"{n:24s} load {r['load_s']:6.1f}  u1 {r['update1_s']:6.2f}  med {r['median_s']:6.3f}  T(1) {r['t1_s']:6.1f}"
          f"  T({r['updates']}) {r['tn_s']:6.1f}  T(100)* {r['t100_projected_s']:6.1f}  alloc {r['peak_alloc_gib']:5.2f}"
          f"  reserved {r['peak_reserved_gib']:5.2f}")
