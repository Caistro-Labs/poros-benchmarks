#!/usr/bin/env python3
"""check_rows.py ROWS_PT FED_ROWS_JSON [UNSLOTH_JSON ...] -- the same token rows for both tools, or exit 1.

ROWS_PT: the rows the Unsloth harness reads (rows_real64.pt). FED_ROWS_JSON: what `poros train` fed (fed_rows.py).
Passes when Poros fed the first N rows of ROWS_PT in order (sha256 of the int64 ids, the same bytes), every row
full length with labels == ids, and every UNSLOTH_JSON's rows_sha256 (h2h_u6's hash of ROWS_PT) is ROWS_PT's.
Prints one JSON line ROWS_CHECK {...} to keep with both receipts."""
import hashlib, json, sys
import torch
sha = lambda t: hashlib.sha256(t.contiguous().numpy().tobytes()).hexdigest()
rows = torch.load(sys.argv[1])["input_ids"].to(torch.int64)
fed = json.load(open(sys.argv[2]))
n = fed["rows"]
rec = {"rows_pt": sys.argv[1], "rows_pt_sha256": sha(rows), "poros_fed_rows": n, "poros_fed_seq": fed["seq"],
       "poros_fed_sha256": fed["sha256_all"], "rows_pt_first_n_sha256": sha(rows[:n]),
       "poros_full_rows": fed["full_length_and_labels_are_ids"], "unsloth": {}}
ok = fed["sha256_all"] == rec["rows_pt_first_n_sha256"] and fed["seq"] == rows.shape[1] and fed["full_length_and_labels_are_ids"]
for p in sys.argv[3:]:
    u = json.load(open(p))
    rec["unsloth"][p] = u.get("rows_sha256")
    ok &= u.get("rows_sha256") == rec["rows_pt_sha256"]
rec["pass"] = bool(ok)
print("ROWS_CHECK " + json.dumps(rec))
sys.exit(0 if ok else 1)
