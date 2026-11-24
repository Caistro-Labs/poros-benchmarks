"""table.py BEFORE_runs.log AFTER_runs.log -- the before (h18, 9c2b9400) / after (h19, 269d3bbc) table in Markdown,
from h19summ.py's RUN lines: s/update (median of updates 2..N and 3..N), peak allocated / reserved and host GiB,
the GPU expert cache the fast path chose, disk GB an update (the SSD reader threads' bytes, median of 2..N)."""
import json
import sys


def runs(path):
    out = {}
    for line in open(path):
        if line.startswith("RUN "):
            _, name, js = line.split(" ", 2)
            out[name] = json.loads(js)
    return out


b, a = runs(sys.argv[1]), runs(sys.argv[2])
f = lambda v: "-" if v is None else (f"{v:.3f}" if isinstance(v, float) else str(v))
rows = [("ram_efficient", "A_ram_1x8_r{}", "A_ram_1x8"), ("ssd_efficient", "B_ssd_eff_1x8_r{}", "B_ssd_eff_1x8"),
        ("ssd_fast", "C_ssd_fast_1x8_r{}", "C_ssd_fast_1x8"), ("ssd_fastcache (fast + 4G)", "D_ssd_fc4_1x8_r{}", "D_ssd_fc4_1x8")]
print("| 1x8 path | run | s/update 2..N | 3..N | alloc / reserved GiB | host GiB | GPU expert cache GiB | disk GB/update |")
print("|---|---|---|---|---|---|---|---|")
for label, bn, an in rows:
    for tag, d in [(f"before r{i}", b.get(bn.format(i))) for i in (1, 2, 3)] + [("after", a.get(an))]:
        if d is None:
            continue
        print(f"| {label} | {tag} | {f(d['s_med_2_N'])} | {f(d['s_med_3_N'])} | {f(d['alloc_GiB'])} / {f(d['reserved_GiB'])} | "
              f"{f(d['host_GiB'])} | {f(d.get('expert_cache_gib'))} | {f(d.get('disk_GB_per_update'))} |")
print()
print("| 8x1 / anchor | run | s/update 2..N | 3..N | alloc / reserved GiB | host GiB | GPU expert cache GiB | disk GB/update |")
print("|---|---|---|---|---|---|---|---|")
for label, names in [("ssd_fastcache 8x1", [("before", b.get("P8_fc4_8x1")), ("after", a.get("P8_fc4_8x1"))]),
                     ("efficient (RAM) 8x1", [("after", a.get("P8_eff_8x1"))]),
                     ("Unsloth 1x8", [(f"before r{i}", b.get(f"E_uns_1x8_r{i}")) for i in (1, 2, 3)] + [("after session", a.get("E_uns_1x8"))])]:
    for tag, d in names:
        if d is None:
            continue
        print(f"| {label} | {tag} | {f(d['s_med_2_N'])} | {f(d['s_med_3_N'])} | {f(d['alloc_GiB'])} / {f(d['reserved_GiB'])} | "
              f"{f(d['host_GiB'])} | {f(d.get('expert_cache_gib'))} | {f(d.get('disk_GB_per_update'))} |")
