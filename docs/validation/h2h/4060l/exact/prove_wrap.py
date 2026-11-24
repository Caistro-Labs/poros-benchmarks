"""prove_wrap.py OUT_JSON -- poros ARGS...: run the poros CLI unchanged; also save training_gate.compare's record
(both arms' per-step losses, the equality flags and counts) to OUT_JSON, with the largest |loss difference|."""
import json, sys
out = sys.argv[1]; sys.argv = ["poros"] + sys.argv[3:]
from poros.validate import training_gate as gate
_orig = gate.compare
def compare(a, b):
    rec = _orig(a, b)
    keep = {k: v for k, v in rec.items() if isinstance(v, (bool, int, float, str, list, type(None)))}
    ra, rb = rec.get("resident_losses") or [], rec.get("blockwise_losses") or []
    keep["max_abs_loss_diff"] = max((abs(float(x) - float(y)) for x, y in zip(ra, rb)), default=None)
    keep["loss_lengths"] = [len(ra), len(rb)]
    json.dump(keep, open(out, "w"), indent=1, default=str)
    return rec
gate.compare = compare
from poros.cli.main import cli
cli()
