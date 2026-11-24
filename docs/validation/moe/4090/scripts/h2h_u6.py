#!/usr/bin/env python3
"""h2h_u6.py -- Unsloth on the shared head-to-head workload (4090 receipts, derived from the 3090 h2h_plain_v5.py).
Same token rows and order as `poros train` (--rows: a .pt from make_data.py), 512-token rows, 8 rows per optimizer
update (--mb x --ga), LoRA r16 alpha 32 dropout 0 on q/k/v/o/gate/up/down, AdamW lr 2e-5 betas 0.9/0.999 eps 1e-8
wd 0.01, constant LR, grad-norm clip 1.0, seed 42, strict determinism (CUBLAS_WORKSPACE_CONFIG=:4096:8 +
torch.use_deterministic_algorithms). Unsloth's own offload (--offload-layers auto|N|none) and gradient
checkpointing (--gc unsloth). Each update is timed between torch.cuda.synchronize() calls. --sweep-mb 8,4,2,1:
after one load, two updates at each micro-batch (OOM caught) to choose the config; not a timed receipt.
Every outcome is a result, a crash included. Memory: GiB = 2^30 bytes."""
import argparse, gc, hashlib, json, os, resource, subprocess, sys, time, traceback

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen2.5-72B")
ap.add_argument("--rows", required=True)
ap.add_argument("--updates", type=int, default=8)
ap.add_argument("--mb", type=int, default=1)
ap.add_argument("--ga", type=int, default=8)
ap.add_argument("--seq-len", type=int, default=512)
ap.add_argument("--offload-layers", default="auto")
ap.add_argument("--gc", default="unsloth")
ap.add_argument("--deterministic", action="store_true")
ap.add_argument("--lr", type=float, default=2e-5)
ap.add_argument("--wd", type=float, default=0.01)
ap.add_argument("--clip", type=float, default=1.0)
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--sweep-mb", default="")
ap.add_argument("--local-snapshot", action="store_true", help="load the local HF-cache snapshot of --model (no Hub mapping/download)")
ap.add_argument("--profile-update", type=int, default=-1)
ap.add_argument("--profile-out", default="")
ap.add_argument("--targets", default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")
ap.add_argument("--time-limit-s", type=int, default=5400)
ap.add_argument("--out", required=True)
a = ap.parse_args()
if a.deterministic:
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
R = {"harness": "h2h_u6", "tool": "unsloth", "args": vars(a), "stage": "start", "epoch_start": time.time(),
     "update_s": [], "microstep_s": [], "loss": [], "grad_norm": [], "tokens": [], "update_end_epoch": []}
T0 = time.time()
GIB = 2**30

def smi():
    try:
        return int(subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True).split()[0])
    except Exception:
        return None

def dump():
    json.dump(R, open(a.out, "w"), indent=1)

def finish(code=0):
    import torch
    R.update(wall_s=round(time.time() - T0, 2), host_maxrss_gib=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 3))
    if torch.cuda.is_available():
        R.update(peak_allocated_gib_last_window=round(torch.cuda.max_memory_allocated() / GIB, 3),
                 peak_reserved_gib_last_window=round(torch.cuda.max_memory_reserved() / GIB, 3),
                 card_gib=round(torch.cuda.get_device_properties(0).total_memory / GIB, 3))
    R["nvidia_smi_used_mib_end"] = smi()
    dump(); print("H2H_RESULT", json.dumps({k: v for k, v in R.items() if k not in ("microstep_s", "loss", "traceback_tail", "args", "update_end_epoch")}))
    sys.exit(code)

try:
    import unsloth  # noqa: F401  must load before transformers
    from unsloth import FastLanguageModel
    import torch
    if a.deterministic:
        torch.use_deterministic_algorithms(True); torch.backends.cudnn.benchmark = False
    torch.manual_seed(a.seed)
    import importlib.metadata as md
    R["versions"] = {k: md.version(k) for k in ("unsloth", "unsloth_zoo", "torch", "transformers", "peft", "bitsandbytes", "accelerate")}
    R["versions"]["torch_cuda"] = torch.version.cuda
    R["stage"] = "load"
    kw = {}
    if a.offload_layers != "none":
        kw["offload_layers"] = int(a.offload_layers) if a.offload_layers.isdigit() else a.offload_layers
    src = a.model
    if a.local_snapshot:
        import glob as _g
        snaps = sorted(_g.glob(os.path.join(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "hub",
                                            "models--" + a.model.replace("/", "--"), "snapshots", "*")))
        assert len(snaps) == 1, snaps
        src = snaps[0]; R["local_snapshot_of"] = a.model; R["local_snapshot_revision"] = os.path.basename(src)
    R["load_source"] = src
    model, tok = FastLanguageModel.from_pretrained(src, max_seq_length=a.seq_len, load_in_4bit=True, dtype=None, **kw)
    R["load_model_s"] = round(time.time() - T0, 2)
    model = FastLanguageModel.get_peft_model(model, r=16, lora_alpha=32, lora_dropout=0, bias="none",
                                             target_modules=a.targets.split(","),
                                             use_gradient_checkpointing=(True if a.gc == "true" else a.gc), random_state=a.seed)
    R["load_s"] = round(time.time() - T0, 2)
    R["after_load_allocated_gib"] = round(torch.cuda.memory_allocated() / GIB, 3)
    R["after_load_reserved_gib"] = round(torch.cuda.memory_reserved() / GIB, 3)
    R["load_peak_allocated_gib"] = round(torch.cuda.max_memory_allocated() / GIB, 3)
    R["load_peak_reserved_gib"] = round(torch.cuda.max_memory_reserved() / GIB, 3)
    R["after_load_smi_mib"] = smi()
    R["model_source"] = getattr(model.config, "_name_or_path", None)
    rows = torch.load(a.rows)["input_ids"]
    R["rows_sha256"] = hashlib.sha256(rows.contiguous().numpy().tobytes()).hexdigest()
    params = [p for p in model.parameters() if p.requires_grad]
    R["trainable_params"] = sum(p.numel() for p in params)
    R["trainable_dtypes"] = sorted({str(p.dtype) for p in params})
    opt = torch.optim.AdamW(params, lr=a.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=a.wd)
    model.train()
    dev = params[0].device
    dump()

    def update(u, mb, ga):
        torch.cuda.synchronize(); t = time.perf_counter()
        ms, ls = [], []
        for j in range(ga):
            t1 = time.perf_counter()
            lo = u * mb * ga + j * mb
            ids = rows[lo:lo + mb].to(dev)
            out = model(input_ids=ids, labels=ids)
            (out.loss / ga).backward()
            ls.append(float(out.loss.detach())); ms.append(round(time.perf_counter() - t1, 4))
            del out
        gn = torch.nn.utils.clip_grad_norm_(params, a.clip) if a.clip > 0 else None
        opt.step(); opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        return time.perf_counter() - t, ms, ls, (float(gn) if gn is not None else None)

    if a.sweep_mb:
        R["stage"] = "sweep"; R["sweep"] = []
        for mb in [int(x) for x in a.sweep_mb.split(",")]:
            ga = 8 // mb; rec = {"mb": mb, "ga": ga}
            torch.cuda.reset_peak_memory_stats()
            try:
                for u in range(2):
                    s, ms, ls, gn = update(u, mb, ga)
                    rec.setdefault("update_s", []).append(round(s, 3)); rec.setdefault("loss", []).append(ls)
                rec["ok"] = True
            except torch.OutOfMemoryError as e:
                rec["ok"] = False; rec["error"] = f"OutOfMemoryError: {str(e)[:300]}"
            rec["peak_allocated_gib"] = round(torch.cuda.max_memory_allocated() / GIB, 3)
            rec["peak_reserved_gib"] = round(torch.cuda.max_memory_reserved() / GIB, 3)
            opt.zero_grad(set_to_none=True); gc.collect(); torch.cuda.empty_cache()
            R["sweep"].append(rec); dump(); print("SWEEP", json.dumps(rec), flush=True)
        R["stage"] = "done"; finish(0)

    R["stage"] = "train"; R["epoch_train_start"] = time.time()
    torch.cuda.reset_peak_memory_stats()
    for u in range(a.updates):
        if time.time() - T0 > a.time_limit_s:
            R["stage"] = "time_limit"; finish(0)
        if u == 1:
            R["peak_allocated_gib_update1"] = round(torch.cuda.max_memory_allocated() / GIB, 3)
            R["peak_reserved_gib_update1"] = round(torch.cuda.max_memory_reserved() / GIB, 3)
            torch.cuda.reset_peak_memory_stats()
        prof = None
        if u == a.profile_update:
            from torch.profiler import ProfilerActivity, profile
            prof = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=False, with_stack=False); prof.__enter__()
        s, ms, ls, gn = update(u, a.mb, a.ga)
        if prof is not None:
            prof.__exit__(None, None, None); R["profiled_update"] = u; R["profiled_wall_s"] = round(s, 3)
            ka = prof.key_averages(); po = a.profile_out
            open(po + "_kavg.txt", "w").write(ka.table(sort_by="self_cuda_time_total", row_limit=80))
            ops = [{"op": e.key, "count": e.count, "self_device_us": getattr(e, "self_device_time_total", getattr(e, "self_cuda_time_total", 0)),
                    "device_us": getattr(e, "device_time_total", getattr(e, "cuda_time_total", 0)), "cpu_us": e.cpu_time_total} for e in ka]
            ops.sort(key=lambda r: -r["self_device_us"]); json.dump(ops, open(po + "_ops.json", "w"), indent=0)
            prof.export_chrome_trace(po + ".trace.json"); del prof, ka
            import gzip
            KEEP = {"kernel", "gpu_memcpy", "gpu_memset", "cuda_runtime", "cuda_driver", "user_annotation", "gpu_user_annotation"}
            ev = json.load(open(po + ".trace.json")); ev = ev["traceEvents"] if isinstance(ev, dict) else ev
            slim = [{k: e[k] for k in ("ph", "cat", "name", "pid", "tid", "ts", "dur", "args") if k in e and (k != "args" or e.get("cat") == "gpu_memcpy")}
                    for e in ev if e.get("ph") == "X" and e.get("cat") in KEEP and "dur" in e]
            del ev
            with gzip.open(po + ".slim.json.gz", "wt", compresslevel=6) as g: json.dump({"traceEvents": slim}, g)
            os.remove(po + ".trace.json"); R["profile_slim_events"] = len(slim)
        R["update_s"].append(round(s, 4)); R["microstep_s"].append(ms); R["loss"].append(ls); R["grad_norm"].append(gn)
        R["tokens"].append(a.mb * a.ga * rows.shape[1]); R["update_end_epoch"].append(time.time())
        print(f"update {u} {s:.2f}s loss {sum(ls)/len(ls):.5f} gn {gn}", flush=True)
        dump()
    R["peak_allocated_gib_updates2plus"] = round(torch.cuda.max_memory_allocated() / GIB, 3)
    R["peak_reserved_gib_updates2plus"] = round(torch.cuda.max_memory_reserved() / GIB, 3)
    R["peak_allocated_gib_train"] = max(R.get("peak_allocated_gib_update1", 0), R["peak_allocated_gib_updates2plus"])
    R["peak_reserved_gib_train"] = max(R.get("peak_reserved_gib_update1", 0), R["peak_reserved_gib_updates2plus"])
    h = hashlib.sha256()
    for n, p in sorted((n, p) for n, p in model.named_parameters() if p.requires_grad):
        h.update(n.encode()); h.update(p.detach().float().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    R["adapter_digest_fp32"] = h.hexdigest()
    R["stage"] = "done"
    finish(0)
except SystemExit:
    raise
except BaseException as e:
    R["error"] = f"{type(e).__name__}: {str(e)[:600]}"
    R["traceback_tail"] = traceback.format_exc()[-2500:]
    finish(1)
