#!/usr/bin/env python3
"""make_rows.py OUT.jsonl -- the refcheck's rows: the first 2,400 OpenAssistant/oasst1 (Apache-2.0)
train-split texts of at least 1,200 characters, in dataset order, one {"text": ...} per line.

Rebuilds oasst_long2400.jsonl byte for byte (sha256 6c8609d5be3172802c5ebb0c8499e8e794ad1a448648e7110e45138a68539fb5,
checked against the file the runs read). Each run tokenizes with its own model's tokenizer and truncates
to max_seq_length 512; eval_split 0.05 holds rows out of training."""
import hashlib
import json
import sys

from datasets import load_dataset

out = sys.argv[1]
d = load_dataset("OpenAssistant/oasst1", split="train")
rows = [x["text"] for x in d if x.get("text") and len(x["text"]) >= 1200][:2400]
assert len(rows) == 2400, len(rows)
text = "".join(json.dumps({"text": t}) + "\n" for t in rows)
open(out, "w").write(text)
print(out, hashlib.sha256(text.encode()).hexdigest())
