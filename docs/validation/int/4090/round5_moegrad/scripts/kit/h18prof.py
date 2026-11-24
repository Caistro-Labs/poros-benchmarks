"""h18prof -- import before `poros train` (instead of sssprof): sssprof's per-update counters (SsdSource reader
threads: pread seconds/bytes, disk busy; the training thread's seconds in SsdSource.read_into and
ExpertBlockManager.ensure / prefetch) plus host-side exclusive timers on the training thread, a torch.profiler
trace of one update (H18_TRACE_UPDATE) with its key_averages tables and a slim gzip chrome trace, and a CUDA
memory-history snapshot of one update (H18_MEM_UPDATE). Changes no value: every wrapper times a call and passes
it through. Writes {output_dir}/h18prof.json after every update."""
import gzip
import json
import os
import threading
import time

import torch

from poros.moe import batching as _bat
from poros.moe import residency as _res
from poros.sss import ssd as _ssd
from poros.train import trainer as _T

TRACE_AT = int(os.environ.get("H18_TRACE_UPDATE", "999"))
MEM_AT = int(os.environ.get("H18_MEM_UPDATE", "999"))
C = {"pread_s": 0.0, "pread_bytes": 0, "pread_n": 0, "disk_busy_s": 0.0}
_lock = threading.Lock()
_active = [0, 0.0]
_main = [threading.main_thread()]  # the thread that runs train() (set there)
# exclusive / inclusive host seconds and counts per timed call, training (main) thread only
X = {}
_tl = threading.local()


def _timed(owner, name, key):
    orig = getattr(owner, name)

    def wrapper(*a, **k):
        if threading.current_thread() is not _main[0]:
            return orig(*a, **k)
        st = getattr(_tl, "stack", None)
        if st is None:
            st = _tl.stack = []
        st.append(0.0)
        t = time.perf_counter()
        try:
            return orig(*a, **k)
        finally:
            el = time.perf_counter() - t
            child = st.pop()
            r = X.setdefault(key, [0.0, 0.0, 0])
            r[0] += el - child
            r[1] += el
            r[2] += 1
            if st:
                st[-1] += el
    wrapper.__wrapped__ = orig
    setattr(owner, name, wrapper)


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
_timed(_T.PorosTrainer, "_forward_block_layers", "fwd_block")
_timed(torch.autograd, "grad", "autograd_grad")
_timed(_bat.ForwardVisit, "finish", "visit_finish")
_timed(_bat.BatchState, "backward", "batch_backward")
_timed(torch.cuda.Event, "synchronize", "event_sync")
_timed(torch.cuda, "synchronize", "cuda_sync")
_timed(torch.cuda.Stream, "synchronize", "stream_sync")
_timed(torch.Tensor, "item", "item")
_timed(torch.Tensor, "tolist", "tolist")
_timed(torch.Tensor, "cpu", "to_cpu")

_orig_step = _T.PorosTrainer._adapter_optimizer_step
_state = {"t": None, "c": dict(C), "x": {}, "n": 0, "rows": [], "prof": None, "mem": False}


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


def _trace(prof, wall, out_dir, n):
    from torch.autograd import DeviceType
    allv, kern, h2d, d2h, other = [], [], [], [], []
    h2d_bytes_n = 0
    top = {}
    for ev in prof.events():
        if ev.device_type != DeviceType.CUDA:
            continue
        s, e = ev.time_range.start / 1e6, ev.time_range.end / 1e6
        allv.append((s, e))
        name = ev.name
        if "HtoD" in name:
            h2d.append((s, e)); h2d_bytes_n += 1
        elif "DtoH" in name:
            d2h.append((s, e))
        elif "Memcpy" in name or "Memset" in name:
            other.append((s, e))
        else:
            kern.append((s, e))
            top[name[:90]] = top.get(name[:90], 0.0) + (e - s)
    res = {"wall_s": wall, "gpu_busy_s": _union(allv), "kernel_busy_s": _union(kern), "h2d_busy_s": _union(h2d),
           "d2h_busy_s": _union(d2h), "other_copy_busy_s": _union(other), "kernel_sum_s": sum(e - s for s, e in kern),
           "kernel_n": len(kern), "h2d_n": len(h2d), "d2h_n": len(d2h),
           "top_kernels": sorted(((round(v, 4), k) for k, v in top.items()), reverse=True)[:25]}
    ka = prof.key_averages()
    try:
        open(os.path.join(out_dir, f"h18_u{n}_kavg_cuda.txt"), "w").write(
            ka.table(sort_by="self_cuda_time_total", row_limit=60, max_name_column_width=70))
        open(os.path.join(out_dir, f"h18_u{n}_kavg_cpu.txt"), "w").write(
            ka.table(sort_by="self_cpu_time_total", row_limit=60, max_name_column_width=70))
    except Exception as exc:  # noqa: BLE001
        res["kavg_error"] = repr(exc)[:300]
    # host-side CUDA runtime calls (all threads, and the main thread alone): launches, syncs, copies
    rt = {}
    try:
        p = os.path.join(out_dir, f"h18_u{n}.trace.json")
        prof.export_chrome_trace(p)
        ev = json.load(open(p))
        ev = ev["traceEvents"] if isinstance(ev, dict) else ev
        tids = {}
        for e in ev:
            if e.get("ph") == "X" and e.get("cat") == "cuda_runtime":
                tids[e.get("tid")] = tids.get(e.get("tid"), 0) + 1
        main_tid = max(tids, key=tids.get) if tids else None
        for e in ev:
            if e.get("ph") == "X" and e.get("cat") in ("cuda_runtime", "cuda_driver"):
                k = ("main " if e.get("tid") == main_tid else "other ") + e["name"]
                r = rt.setdefault(k, [0.0, 0])
                r[0] += e.get("dur", 0) / 1e6
                r[1] += 1
        KEEP = {"kernel", "gpu_memcpy", "gpu_memset", "cuda_runtime", "cuda_driver", "user_annotation",
                "gpu_user_annotation", "cpu_op"}
        slim = [{k: e[k] for k in ("ph", "cat", "name", "pid", "tid", "ts", "dur", "args")
                 if k in e and (k != "args" or e.get("cat") == "gpu_memcpy")}
                for e in ev if e.get("ph") == "X" and e.get("cat") in KEEP and "dur" in e]
        del ev
        with gzip.open(os.path.join(out_dir, f"h18_u{n}.slim.json.gz"), "wt", compresslevel=6) as g:
            json.dump({"traceEvents": slim}, g)
        res["slim_events"] = len(slim)
        res["main_tid"] = main_tid
        os.remove(p)
    except Exception as exc:  # noqa: BLE001
        res["chrome_error"] = repr(exc)[:300]
    res["runtime"] = sorted(([round(v[0], 4), v[1], k] for k, v in rt.items()), reverse=True)[:30]
    return res


def _mem_dump(self, n):
    try:
        p = os.path.join(self.config.output_dir, f"h18_mem_u{n}.pickle")
        torch.cuda.memory._dump_snapshot(p)
        return p
    except Exception as exc:  # noqa: BLE001
        return f"dump error {exc!r}"[:300]
    finally:
        torch.cuda.memory._record_memory_history(enabled=None)


def _step(self):
    t_opt = time.perf_counter()
    out = _orig_step(self)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    now = time.perf_counter()
    st = _state
    st["n"] += 1
    if st["t"] is not None:
        row = {"update": st["n"], "wall_s": round(now - st["t"], 4), "opt_step_s": round(now - t_opt, 4)}
        row.update({k: round(C[k] - st["c"][k], 4) for k in C})
        row["host"] = {k: [round(v[0] - st["x"].get(k, [0, 0, 0])[0], 4), round(v[1] - st["x"].get(k, [0, 0, 0])[1], 4),
                           v[2] - st["x"].get(k, [0, 0, 0])[2]] for k, v in X.items()}
        src = getattr(self, "_sss_source", None)
        if src is not None:
            s = src.stats
            row["sss"] = {k: s.get(k) for k in ("reads", "prefetch_hits", "sync_reads", "prefetch_dropped", "bytes_read",
                                                "blocked_s", "hinted", "phase_ahead", "depth", "cache_hits", "cache_bytes")}
        if torch.cuda.is_available():
            ms = torch.cuda.memory_stats()
            row["mem"] = {"alloc_peak_gib": round(ms.get("allocated_bytes.all.peak", 0) / 2**30, 3),
                          "reserved_peak_gib": round(ms.get("reserved_bytes.all.peak", 0) / 2**30, 3),
                          "alloc_gib": round(ms.get("allocated_bytes.all.current", 0) / 2**30, 3)}
        if st["prof"] is not None:
            st["prof"].__exit__(None, None, None)
            row["trace"] = _trace(st["prof"], now - st["t"], self.config.output_dir, st["n"])
            st["prof"] = None
        if st["mem"]:
            row["mem_snapshot"] = _mem_dump(self, st["n"])
            st["mem"] = False
        st["rows"].append(row)
    nxt = st["n"] + 1
    if nxt == MEM_AT and torch.cuda.is_available():
        try:
            torch.cuda.memory._record_memory_history(enabled="all", context="all", stacks="python",
                                                     max_entries=400000)
            st["mem"] = True
        except Exception:  # noqa: BLE001
            st["mem"] = False
    if nxt == TRACE_AT and torch.cuda.is_available():
        from torch.profiler import ProfilerActivity, profile
        st["prof"] = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=False)
        st["prof"].__enter__()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    now = time.perf_counter()
    st["t"], st["c"], st["x"] = now, dict(C), {k: list(v) for k, v in X.items()}
    try:
        json.dump({"rows": st["rows"], "trace_update": TRACE_AT, "mem_update": MEM_AT},
                  open(os.path.join(self.config.output_dir, "h18prof.json"), "w"), indent=1, default=str)
    except OSError:
        pass
    return out


_T.PorosTrainer._adapter_optimizer_step = _step
_orig_train = _T.PorosTrainer.train


def _train(self, *a, **k):  # update 1 is timed from here (its row includes the loop's own setup)
    _main[0] = threading.current_thread()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    _state["t"], _state["c"], _state["x"] = time.perf_counter(), dict(C), {k2: list(v) for k2, v in X.items()}
    if MEM_AT == 1 and torch.cuda.is_available():
        torch.cuda.memory._record_memory_history(enabled="all", context="all", stacks="python", max_entries=400000)
        _state["mem"] = True
    return _orig_train(self, *a, **k)


_T.PorosTrainer.train = _train
