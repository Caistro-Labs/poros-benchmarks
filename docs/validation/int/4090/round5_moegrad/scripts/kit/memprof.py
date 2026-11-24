"""memprof -- import before `poros train`: after each update, torch's allocator counters (current and peak allocated,
reserved, inactive split bytes; segments; alloc retries) and the MoE residency's capacity and resident bytes, into
{output_dir}/memprof.json. Resets nothing (the run's own peaks stay the run's); changes no value."""
import json
import os

import torch

from poros.moe import residency as _res
from poros.train import trainer as _T

_orig_step = _T.PorosTrainer._adapter_optimizer_step
_rows = []
_mgrs = []
_orig_init = _res.ExpertBlockManager.__init__


def _init(self, *a, **k):
    _orig_init(self, *a, **k)
    _mgrs.append(self)


_res.ExpertBlockManager.__init__ = _init
G = 2 ** 30


def _step(self):
    out = _orig_step(self)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        s = torch.cuda.memory_stats()
        g = lambda k: round(s.get(k, 0) / G, 3)
        m = _mgrs[0] if _mgrs else None
        _rows.append({"update": len(_rows) + 1,
                      "alloc": g("allocated_bytes.all.current"), "alloc_peak": g("allocated_bytes.all.peak"),
                      "reserved": g("reserved_bytes.all.current"), "reserved_peak": g("reserved_bytes.all.peak"),
                      "inactive_split": g("inactive_split_bytes.all.current"),
                      "inactive_split_peak": g("inactive_split_bytes.all.peak"),
                      "segments": s.get("segment.all.current"), "retries": s.get("num_alloc_retries"),
                      "capacity": round(m._capacity / G, 3) if m is not None else None,
                      "resident": round(m._resident_bytes / G, 3) if m is not None else None,
                      "runs": len(set(m._run_of.values())) if m is not None else None})
        try:
            json.dump(_rows, open(os.path.join(self.config.output_dir, "memprof.json"), "w"), indent=1)
        except OSError:
            pass
    return out


_T.PorosTrainer._adapter_optimizer_step = _step
