#!/usr/bin/env python3
"""full_rows.py MODEL IN.jsonl OUT.jsonl SEQ -- the rows of IN ({"text": ...}) whose tokenization with MODEL's
tokenizer (as Poros's loader calls it: truncation at SEQ) fills SEQ tokens, in IN's order: every batch of them is
unpadded, so micro-batch shapes (8 x 1 vs 4 x 2) give the same update and differ by rounding alone."""
import json, sys
from transformers import AutoTokenizer
model, src, dst, seq = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
tok = AutoTokenizer.from_pretrained(model)
rows = [json.loads(l) for l in open(src)]
keep = [r for r in rows if len(tok(r["text"], truncation=True, max_length=seq)["input_ids"]) == seq]
open(dst, "w").write("".join(json.dumps({"text": r["text"]}) + "\n" for r in keep))
print(json.dumps({"in": len(rows), "kept": len(keep), "seq": seq, "out": dst}))
