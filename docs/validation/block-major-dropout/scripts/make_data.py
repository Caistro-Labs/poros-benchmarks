"""make_data.py [OUT_DIR] -- the runs' rows, from OpenAssistant/oasst1 (Apache-2.0), train split:

- text40.jsonl: the first 40 `text` rows (the classic loop's: lengths vary row to row);
- chats48.jsonl: the first 48 English first turns, prompt and its rank-0 reply (SFT);
- pairs48.jsonl: the first 48 English first prompts with two ranked replies: rank 0 chosen, the
  lowest-ranked rejected (DPO);
- text320.jsonl: the first 320 English `text` rows (the MoE loss check: one pass of 40 updates).
"""
import hashlib
import json
import os
import sys

from datasets import load_dataset

out = sys.argv[1] if len(sys.argv) > 1 else "."
os.makedirs(out, exist_ok=True)
rows = load_dataset("OpenAssistant/oasst1", split="train")
replies = {}
for r in rows:
    if r["role"] == "assistant" and r["parent_id"] is not None and r["rank"] is not None:
        replies.setdefault(r["parent_id"], []).append((r["rank"], r["text"]))
firsts = [r for r in rows if r["role"] == "prompter" and r["parent_id"] is None
          and r["lang"] == "en" and len(replies.get(r["message_id"], [])) >= 2]
sets = {
    "text40.jsonl": [{"text": r["text"]} for r in rows if r["text"]][:40],
    "chats48.jsonl": [{"messages": [{"role": "user", "content": p["text"]},
                                    {"role": "assistant",
                                     "content": min(replies[p["message_id"]])[1]}]}
                      for p in firsts[:48]],
    "pairs48.jsonl": [{"prompt": p["text"], "chosen": min(replies[p["message_id"]])[1],
                       "rejected": max(replies[p["message_id"]])[1]} for p in firsts[:48]],
    "text320.jsonl": [{"text": r["text"]} for r in rows if r["text"] and r["lang"] == "en"][:320],
}
for name, items in sets.items():
    path = os.path.join(out, name)
    with open(path, "w") as f:
        for item in items:
            f.write(json.dumps(item) + "\n")
    print(name, len(items), hashlib.sha256(open(path, "rb").read()).hexdigest()[:16])
