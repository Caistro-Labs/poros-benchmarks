#!/usr/bin/env python3
"""make_data.py OUT_DIR MODEL N SEQ
Real rows: the first N OpenAssistant/oasst1 (train split, dataset order) texts whose tokenized length with
MODEL's tokenizer is >= SEQ, so every row trains at full length (truncated to SEQ). Writes
oasst_full{N}.jsonl ({"text": ...}) and the source indices.
Then the exact rows and order `poros train` feeds: Poros's own loader (_load_hf_dataset_as_dataloader, the
function PorosTrainer._prepare_dataloader calls on CUDA, seed 42, batch 1) iterated once, saved as
rows_real{N}.pt (input_ids/labels/attention_mask, [N, SEQ]). Synthetic: Poros's 'synthetic:N' rows
(torch.randint(100, 30000, (N, SEQ)) from a generator seeded 42, unshuffled) as rows_syn{N}.pt.
The Unsloth harness reads these .pt files, so both tools see the same token ids in the same order."""
import hashlib, json, os, sys
import torch
from datasets import load_dataset
from transformers import AutoTokenizer
out, model, n, seq = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
os.makedirs(out, exist_ok=True)
tok = AutoTokenizer.from_pretrained(model)
d = load_dataset("OpenAssistant/oasst1", split="train")
rows, idx, seen = [], [], 0
for i, x in enumerate(d):
    t = x.get("text")
    if not t: continue
    seen += 1
    if len(tok(t)["input_ids"]) >= seq:
        rows.append(t); idx.append(i)
        if len(rows) == n: break
assert len(rows) == n, len(rows)
jl = os.path.join(out, f"oasst_full{n}.jsonl")
open(jl, "w").write("".join(json.dumps({"text": t}) + "\n" for t in rows))
from poros.train.trainer import _load_hf_dataset_as_dataloader
dl = _load_hf_dataset_as_dataloader(dataset_name=jl, model_name_or_path=model, max_seq_length=seq, batch_size=1, seed=42, pad_to_longest=True)
b = [x for x in dl]
ids = torch.cat([x["input_ids"] for x in b]); lab = torch.cat([x["labels"] for x in b]); am = torch.cat([x["attention_mask"] for x in b])
assert ids.shape == (n, seq), ids.shape
assert bool((am == 1).all()) and bool((lab == ids).all()), "a row is shorter than SEQ"
torch.save({"input_ids": ids, "labels": lab, "attention_mask": am}, os.path.join(out, f"rows_real{n}.pt"))
g = torch.Generator().manual_seed(42)
syn = torch.randint(100, 30000, (n, seq), generator=g)
torch.save({"input_ids": syn, "labels": syn.clone(), "attention_mask": torch.ones_like(syn)}, os.path.join(out, f"rows_syn{n}.pt"))
sha = lambda t: hashlib.sha256(t.contiguous().numpy().tobytes()).hexdigest()
rec = {"model_tokenizer": model, "n": n, "seq": seq, "oasst1_train_indices": idx, "oasst1_texts_scanned": seen,
       "jsonl_sha256": hashlib.sha256(open(jl, "rb").read()).hexdigest(), "real_ids_sha256": sha(ids), "syn_ids_sha256": sha(syn),
       "real_first_tokens": ids[:, :4].tolist()[:4], "poros_order_note": "rows_real is in the order poros train iterates (seed 42)"}
json.dump(rec, open(os.path.join(out, f"data_{n}.json"), "w"), indent=1)
print(json.dumps({k: v for k, v in rec.items() if k != "oasst1_train_indices"}))
