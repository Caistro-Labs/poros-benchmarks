"""h18summ.py REC NAME RC poros|unsloth -- one JSON line per run (RUN NAME {...}): s/update per update, the medians of
updates 2..N, 3..N and 2..4; load (launch to the setup event / to Unsloth's load end) and T(1) (launch to the end of
update 1) from the launch time NAME.t0; peaks in GiB (Poros: the receipt's GB / 1.073741824, whole run; Unsloth:
max of its load and train peaks, GiB); host RAM peak GiB (Poros VmHWM, Unsloth ru_maxrss); disk GB an update
(sssprof reader bytes, median of 2..N); the expert-cache sizing Poros chose; the fed rows' sha256."""
import json
import os
import statistics
import sys

R, n, rc, tool = sys.argv[1:5]
p = lambda s: os.path.join(R, f"{n}_{s}")
G = 1e9 / 2**30


def med(xs, lo, hi=None):
    xs = xs[lo:hi]
    return round(statistics.median(xs), 3) if xs else None


def j(path):
    try:
        return json.load(open(path))
    except Exception:
        return None


t0 = float(open(os.path.join(R, f"{n}.t0")).read())
out = {"rc": int(rc), "tool": tool}
if tool == "unsloth":
    d = j(os.path.join(R, f"{n}.json")) or {}
    s = d.get("update_s", [])
    a = d.get("args", {})
    ue = d.get("update_end_epoch", [])
    es = d.get("epoch_start")
    out.update(shape=f"{a.get('mb')}x{a.get('ga')}", stage=d.get("stage"), error=d.get("error"), steps=len(s),
               s=[round(x, 3) for x in s], s_med_2_N=med(s, 1), s_med_3_N=med(s, 2), s_med_2_4=med(s, 1, 4),
               load_s=round(es - t0 + d["load_s"], 2) if es and d.get("load_s") else None,
               T1_s=round(ue[0] - t0, 2) if ue else None,
               alloc_GiB=max(d.get("peak_allocated_gib_train") or 0, d.get("load_peak_allocated_gib") or 0) or None,
               reserved_GiB=max(d.get("peak_reserved_gib_train") or 0, d.get("load_peak_reserved_gib") or 0) or None,
               train_alloc_res_GiB=[d.get("peak_allocated_gib_train"), d.get("peak_reserved_gib_train")],
               host_GiB=d.get("host_maxrss_gib"), digest=(d.get("adapter_digest_fp32") or "")[:16],
               rows=(d.get("rows_sha256") or "")[:16],
               loss=[round(sum(x) / len(x), 5) for x in d.get("loss", []) if x], grad_norm=d.get("grad_norm"))
else:
    try:
        ev = [json.loads(l) for l in open(p("events.jsonl"))]
    except FileNotFoundError:
        ev = []
    st = [e for e in ev if e.get("event_type") == "step"]
    setup = [e for e in ev if e.get("event_type") == "setup"]
    s = [e["data"].get("s_per_step") for e in st]
    r = j(p("receipt.json")) or {}
    c = j(p("config.resolved.json")) or {}
    fr = j(p("fed_rows.json")) or {}
    g = lambda k: None if r.get(k) is None else round(r[k] * G, 2)
    rcmp = c.get("ran_compute") or {}
    prof = j(p("sssprof.json")) or j(p("h18prof.json"))
    disk = ensure = None
    if prof and prof.get("rows"):
        rows = [x for x in prof["rows"] if x["update"] >= 2 and "trace" not in x]
        if rows:
            disk = round(statistics.median(x["pread_bytes"] for x in rows) / 1e9, 2)
            ens = [x.get("ensure_s") if "ensure_s" in x else (x.get("host", {}).get("ensure") or [0, 0])[1] for x in rows]
            ensure = round(statistics.median(ens), 2)
    last = st[-1]["data"].get("sss") if st else None
    out.update(shape=f"{c.get('per_device_batch_size')}x{c.get('gradient_accumulation_steps')}", steps=len(st),
               s=[round(x, 3) for x in s], s_med_2_N=med(s, 1), s_med_3_N=med(s, 2), s_med_2_4=med(s, 1, 4),
               load_s=round(setup[0]["timestamp"] - t0, 2) if setup else None,
               T1_s=round(st[0]["timestamp"] - t0, 2) if st else None,
               alloc_GiB=g("peak_vram_allocated_gb"), reserved_GiB=g("peak_vram_reserved_gb"), host_GiB=g("peak_host_ram_gb"),
               store=c.get("store"), path=c.get("path"), ran_path=c.get("ran_path"), store_options=c.get("store_options"),
               expandable=c.get("cuda_expandable_segments"), expert_cache_gib=rcmp.get("moe_expert_cache_gib"),
               expert_cache_reason=rcmp.get("moe_expert_cache_reason"), compute=rcmp.get("moe_expert_compute"),
               disk_GB_per_update=disk, ensure_s_med=ensure, rows=(fr.get("sha256_all") or "")[:16],
               loss=[e["data"].get("loss") for e in st], grad_norm=[e["data"].get("grad_norm") for e in st],
               sss_last=last)
print(f"RUN {n} " + json.dumps(out, default=str))
