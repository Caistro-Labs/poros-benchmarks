"""rank_digest.py OUT_DIR RANK {dense,moe} -- the paths at one LoRA rank, bit for bit, dropout 0.05.

dense: a random Qwen2 (seeded; hidden 512, 4 heads of 128, MLP 5632, 3 layers), NF4, PEFT LoRA on
the seven projections, alpha 2 x rank; 2 updates of 2 micro-batches x 4 padded rows of 512 tokens
(2,048 rows a call: the rank-64/128 row chunks run more than once). moe: the small random
Qwen3-MoE of scripts/xcard/mk_tiny_moe.py at OUT_DIR/../tiny_moe (16 experts of 256, 6 layers).
Arms sequential / fast / efficient / fast fully resident, the default compute (fused), then the
same sequential with the fused compute off. Prints one sha256 over every trained tensor per arm,
the losses and grad norms, the compute that ran and why."""
import hashlib
import json
import os
import sys

import torch
import transformers

from poros.train.trainer import PorosTrainer

out, rank, kind = sys.argv[1], int(sys.argv[2]), sys.argv[3]
os.makedirs(out, exist_ok=True)
if kind == "dense":
    torch.manual_seed(0)
    cfg = transformers.Qwen2Config(vocab_size=256, hidden_size=512, intermediate_size=5632,
                                   num_hidden_layers=3, num_attention_heads=4,
                                   num_key_value_heads=2)
    model = transformers.AutoModelForCausalLM.from_config(cfg)
    with torch.no_grad():
        for p in model.parameters():
            p.normal_(0, 0.05)
    ckpt = f"{out}/ckpt"
    model.to(torch.bfloat16).save_pretrained(ckpt)
    del model
    vocab, extra = 256, dict(lora_target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                                  "gate_proj", "up_proj", "down_proj"],
                             block_size=1, blockwise_arch_check="off")
    arms = [("sequential", dict(path="sequential")), ("fast", dict(path="fast")),
            ("efficient", dict(path="efficient")),
            ("fast_resident", dict(path="fast", resident_blocks=999)),
            ("sequential_peft", dict(path="sequential", fused_compute=False))]
else:
    ckpt = os.path.join(os.path.dirname(out.rstrip("/")), "tiny_moe")
    vocab = transformers.AutoConfig.from_pretrained(ckpt).vocab_size
    extra = {}
    arms = [("sequential", dict(path="sequential")), ("fast", dict(path="fast")),
            ("efficient", dict(path="efficient")),
            ("fast_resident", dict(path="fast", moe_expert_cache_gib=8.0)),
            ("sequential_reference", dict(path="sequential", moe_expert_compute="reference"))]
g = torch.Generator().manual_seed(5)
rows = []
for n in (512, 300, 512, 77, 512, 431, 256, 512, 389, 512, 140, 512, 512, 23, 480, 512):
    ids = torch.randint(3, min(vocab, 250), (512,), generator=g)
    mask = torch.zeros(512, dtype=torch.long)
    mask[:n] = 1
    rows.append({"input_ids": ids * mask, "attention_mask": mask,
                 "labels": torch.where(mask.bool(), ids, torch.full_like(ids, -100))})
res = {}
for name, knobs in arms:
    tr = PorosTrainer.from_pretrained(
        ckpt, quantization="nf4", lora_rank=rank, lora_alpha=2 * rank, lora_dropout=0.05,
        output_dir=f"{out}/{name}", max_steps=2, save_steps=0, progress=False,
        gradient_accumulation_steps=2, per_device_batch_size=4, learning_rate=1e-3,
        warmup_steps=0, max_seq_length=512, **extra, **knobs)
    tr.train(torch.utils.data.DataLoader(rows, batch_size=4, shuffle=False), steps=2,
             seq_len=512)
    h = hashlib.sha256()
    n_t = 0
    for pn, p in sorted(tr.model.named_parameters()):
        if p.requires_grad:
            h.update(pn.encode())
            h.update(p.detach().float().cpu().numpy().tobytes())
            n_t += 1
    ran = getattr(tr.config, "ran_compute", None)
    res[name] = dict(digest=h.hexdigest(), losses=list(tr._losses),
                     grad_norms=list(tr._grad_norms), tensors=n_t, ran_compute=ran,
                     compute_reason=getattr(tr.config, "compute_reason", None),
                     peak_alloc_gib=round(torch.cuda.max_memory_allocated() / 2**30, 3))
    print("RANK", kind, rank, name, h.hexdigest()[:16], [round(x, 6) for x in tr._losses],
          json.dumps(ran), res[name]["compute_reason"], flush=True)
    del tr
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
four = ["sequential", "fast", "efficient", "fast_resident"]
same = all(res[a]["digest"] == res["sequential"]["digest"] and res[a]["losses"] ==
           res["sequential"]["losses"] and res[a]["grad_norms"] == res["sequential"]["grad_norms"]
           for a in four)
res["four_paths_bitwise"] = same
json.dump(res, open(f"{out}/rank_digest_{kind}_{rank}.json", "w"), indent=1)
print("RANKPATHS", kind, rank, "PASS" if same else "FAIL", res["sequential"]["digest"][:16])
