"""Every built-in recipe on the simulated machine it is certified for (the shipped book at the
head): the run configuration the train verb resolves for its printed command (config_dict, what
config.resolved.json holds before the trainer starts), its steps and length, and on CUDA the
preflight's decisions (gate_of: path, tables, warning, refusals). Nothing loads or trains."""
import json, os, sys
from pathlib import Path
T, BOOK, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
sys.path[:0] = [T, f"{T}/scripts", f"{T}/scripts/certify"]
os.environ["POROS_DEV_SIMULATE"] = "1"
import decisions as d, run_recipe as run
from poros.cli.context import Context
from poros.cli.verbs import mac as mac_verb, train as tv
from poros.cli.verbs.train_run import config_dict
from poros.cli.words.grammar import parse
from poros.metal import table
from poros.recipes import presets
# the recipes' sample rows and the simulated store: a folder given, or one beside the output file
DATA = Path(sys.argv[4]) if len(sys.argv) > 4 else Path(OUT).resolve().parent / "resolve-data"
DATA.mkdir(parents=True, exist_ok=True)
mac_verb.mac_store = lambda *a, **k: str(DATA); mac_verb.run_room = lambda *a, **k: ""
book = json.loads(Path(BOOK).read_text())["recipes"]
out = {}
for r in presets.DEFINED:
    m = book[r.key][0]["machine"]
    if m["platform"] == "mac":
        ctx = Context(cwd=DATA, fixed_gpu=table.plan_gpu(m["mac_class"]), fixed_interactive=False)
    else:
        ctx = d.machine(f"NVIDIA GeForce {m['gpu']}:{m['vram_gib']}", m["host_ram_gib"],
                        m["host_ram_gib"] - 4, DATA)
    data = DATA / f"{r.key}.jsonl"
    run.write_rows(data, run.sample_rows(run.sample_kind(r), 64, int(r.settings["max_seq_length"]),
                                         key=r.key))
    a = tv.inspect(tv.resolve(parse(r.argv(str(data))), ctx).action, ctx)
    got = {"machine": m.get("gpu") or m.get("mac_class"), "steps": a.steps, "seq_len": a.seq_len,
           "config": config_dict(a, "<out>")}
    if m["platform"] != "mac":
        from poros.core.config import PorosConfig
        got["gate"] = run.gate_of(r, ctx, str(data), PorosConfig(**got["config"]))
    out[r.key] = got
Path(OUT).write_text(json.dumps(out, indent=1, sort_keys=True, default=str))
print(len(out), "recipes")
