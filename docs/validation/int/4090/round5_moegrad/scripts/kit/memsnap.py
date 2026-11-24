"""memsnap.py SNAPSHOT.pickle [TOP] -- what was live on the GPU at the recorded update's allocated peak.

Replays the snapshot's device trace (alloc +, free_completed -) from the allocations already live when recording
began (the snapshot's segments' blocks that have no alloc event in the trace are counted as a baseline), finds the
peak, and groups the allocations live at the peak (made during the recording) by their first poros frame
(file:line function), largest first. Prints JSON lines."""
import collections
import json
import pickle
import sys

snap = pickle.load(open(sys.argv[1], "rb"))
top = int(sys.argv[2]) if len(sys.argv) > 2 else 25
traces = snap.get("device_traces") or [[]]
tr = max(traces, key=len)
G = 2 ** 30


def site(frames):
    frames = frames or []
    pf = [f for f in frames if "/poros/" in f.get("filename", "")]
    f = pf[0] if pf else (frames[0] if frames else {"filename": "?", "line": 0, "name": "?"})
    path = f.get("filename", "?")
    path = path.split("/poros/", 1)[-1] if "/poros/" in path else path.rsplit("/", 1)[-1]
    chain = " < ".join(f"{x.get('filename', '?').rsplit('/', 1)[-1]}:{x.get('line')}:{x.get('name')}" for x in pf[:4])
    return f"{path}:{f.get('line')}:{f.get('name')}", chain


live = {}
cur = 0
peak, peak_i, peak_live = 0, -1, None
n_alloc = 0
for i, e in enumerate(tr):
    a = e.get("action")
    if a == "alloc":
        live[e["addr"]] = (e["size"], e.get("frames"))
        cur += e["size"]
        n_alloc += 1
        if cur > peak:
            peak, peak_i = cur, i
            peak_live = None  # rebuilt lazily below
    elif a == "free_completed":
        v = live.pop(e["addr"], None)
        if v is not None:
            cur -= v[0]
# replay again to the peak index to get the live set there
live2 = {}
for i, e in enumerate(tr[:peak_i + 1]):
    a = e.get("action")
    if a == "alloc":
        live2[e["addr"]] = (e["size"], e.get("frames"))
    elif a == "free_completed":
        live2.pop(e["addr"], None)
groups = collections.defaultdict(lambda: [0, 0, ""])
for size, frames in live2.values():
    s, chain = site(frames)
    g = groups[s]
    g[0] += size
    g[1] += 1
    g[2] = g[2] or chain
seg_alloc = sum(b["size"] for seg in snap.get("segments", []) for b in seg.get("blocks", []) if b.get("state") == "active_allocated")
print("MEMSNAP " + json.dumps({"events": len(tr), "allocs": n_alloc, "peak_over_start_gib": round(peak / G, 3),
                               "peak_event": peak_i, "live_at_peak_n": len(live2),
                               "allocated_at_dump_gib": round(seg_alloc / G, 3)}))
for s, (b, n, chain) in sorted(groups.items(), key=lambda kv: -kv[1][0])[:top]:
    print("SITE " + json.dumps({"gib": round(b / G, 3), "n": n, "site": s, "chain": chain}))
