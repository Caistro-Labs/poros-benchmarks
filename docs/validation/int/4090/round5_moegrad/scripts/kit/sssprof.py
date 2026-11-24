"""sssprof -- import before `poros train` (python -c "import sssprof; from poros.cli.main import cli; cli()" train ...):
where an update's time goes on the expert stores. Per update: wall; disk (SsdSource reader threads: summed pread
seconds and bytes, and the wall seconds at least one read was in flight); the main thread's seconds blocked in
SsdSource.read_into and in ExpertBlockManager.ensure / prefetch; and for update SSSPROF_UPDATE (default 4) a
torch.profiler CUDA trace: GPU busy (union of every device interval), kernels only, H2D copies only (seconds and
bytes), and the top kernels. Writes {output_dir}/sssprof.json after every update. Changes no value: every wrapper
times a call and passes it through."""
import json
import os
import threading
import time

import torch

from poros.moe import residency as _res
from poros.sss import ssd as _ssd
from poros.train import trainer as _T

PROF_AT = int(os.environ.get("SSSPROF_UPDATE", "4"))
C = {"pread_s": 0.0, "pread_bytes": 0, "pread_n": 0, "disk_busy_s": 0.0, "read_into_s": 0.0, "read_into_n": 0,
     "ensure_s": 0.0, "ensure_n": 0, "prefetch_s": 0.0, "prefetch_n": 0}
_lock = threading.Lock()
_active = [0, 0.0]  # reads in flight, when the first began


def _timed(cls, name, key):
    orig = getattr(cls, name)

    def wrapper(self, *a, **k):
        t = time.perf_counter()
        try:
            return orig(self, *a, **k)
        finally:
            C[key + "_s"] += time.perf_counter() - t
            C[key + "_n"] += 1
    setattr(cls, name, wrapper)


_orig_chunk = _ssd.SsdSource._read_chunk


def _read_chunk(self, e, start, buf):
    t = time.perf_counter()
    with _lock:
        if _active[0] == 0:
            _active[1] = t
        _active[0] += 1
    try:
        return _orig_chunk(self, e, start, buf)
    finally:
        t1 = time.perf_counter()
        with _lock:
            C["pread_s"] += t1 - t
            C["pread_bytes"] += len(buf)
            C["pread_n"] += 1
            _active[0] -= 1
            if _active[0] == 0:
                C["disk_busy_s"] += t1 - _active[1]


_ssd.SsdSource._read_chunk = _read_chunk
_timed(_ssd.SsdSource, "read_into", "read_into")
_timed(_res.ExpertBlockManager, "ensure", "ensure")
_timed(_res.ExpertBlockManager, "prefetch", "prefetch")

_orig_step = _T.PorosTrainer._adapter_optimizer_step
_state = {"t": None, "c": dict(C), "n": 0, "rows": [], "prof": None, "trace": None}


def _union(iv):
    tot, end = 0.0, None
    for s, e in sorted(iv):
        if end is None or s > end:
            tot += e - s
            end = e
        elif e > end:
            tot += e - end
            end = e
    return tot


def _trace(prof, wall):
    from torch.autograd import DeviceType
    allv, kern, h2d, d2h = [], [], [], []
    h2d_bytes = 0
    top = {}
    for ev in prof.events():
        if ev.device_type != DeviceType.CUDA:
            continue
        s, e = ev.time_range.start / 1e6, ev.time_range.end / 1e6
        allv.append((s, e))
        name = ev.name
        if "Memcpy HtoD" in name or "HtoD" in name:
            h2d.append((s, e))
        elif "Memcpy DtoH" in name or "DtoH" in name:
            d2h.append((s, e))
        elif "Memcpy" in name or "Memset" in name:
            pass
        else:
            kern.append((s, e))
            top[name[:80]] = top.get(name[:80], 0.0) + (e - s)
    return {"wall_s": wall, "gpu_busy_s": _union(allv), "kernel_busy_s": _union(kern), "h2d_busy_s": _union(h2d),
            "d2h_busy_s": _union(d2h), "kernel_sum_s": sum(e - s for s, e in kern), "h2d_n": len(h2d),
            "top_kernels": sorted(((round(v, 4), k) for k, v in top.items()), reverse=True)[:12]}


def _step(self):
    out = _orig_step(self)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    now = time.perf_counter()
    st = _state
    st["n"] += 1
    if st["t"] is not None:
        row = {"update": st["n"], "wall_s": round(now - st["t"], 4)}
        row.update({k: round(C[k] - st["c"][k], 4) for k in C})
        src = getattr(self, "_sss_source", None)
        if src is not None:
            s = src.stats
            row["sss"] = {k: s.get(k) for k in ("reads", "prefetch_hits", "sync_reads", "prefetch_dropped", "bytes_read",
                                                "blocked_s", "hinted", "phase_ahead", "depth")}
        if st["prof"] is not None:
            st["prof"].__exit__(None, None, None)
            row["trace"] = _trace(st["prof"], now - st["t"])
            st["prof"] = None
        st["rows"].append(row)
    if st["n"] == PROF_AT - 1 and torch.cuda.is_available():
        from torch.profiler import ProfilerActivity, profile
        st["prof"] = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=False)
        st["prof"].__enter__()
        torch.cuda.synchronize()
        now = time.perf_counter()
    st["t"], st["c"] = now, dict(C)
    try:
        json.dump({"rows": st["rows"], "prof_update": PROF_AT}, open(os.path.join(self.config.output_dir, "sssprof.json"), "w"),
                  indent=1, default=str)
    except OSError:
        pass
    return out


_T.PorosTrainer._adapter_optimizer_step = _step
_orig_train = _T.PorosTrainer.train


def _train(self, *a, **k):  # update 1 is timed from here (its row includes the loop's own setup)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    _state["t"], _state["c"] = time.perf_counter(), dict(C)
    return _orig_train(self, *a, **k)


_T.PorosTrainer.train = _train
