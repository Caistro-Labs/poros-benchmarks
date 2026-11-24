"""fed_rows -- import before `poros train` runs (python -c "import fed_rows; from poros.cli.main import cli; cli()"
train ...): records the token rows the trainer actually feeds, in the order it feeds them.

PorosTrainer._prepare_dataloader is wrapped so every batch the training loop draws is hashed as it is drawn;
{output_dir}/fed_rows.json is rewritten after each batch with: rows, seq (None when the batches' lengths
differ), each batch's shape and sha256, the sha256 of all fed input_ids bytes in feed order (int64; for rows of
one length it is the hash make_data.py and the Unsloth harness take of rows_real64.pt), and whether every row is
full length with labels == input_ids. Nothing the
trainer computes changes (the batches pass through untouched)."""
import hashlib
import json
import os

import torch

from poros.train import trainer as _T

_orig = _T.PorosTrainer._prepare_dataloader


def _sha(t):
    return hashlib.sha256(t.contiguous().numpy().tobytes()).hexdigest()


def _prepare_dataloader(self, dataset, max_seq_length):
    dl = _orig(self, dataset, max_seq_length)
    if getattr(self, "_fed_rows_recorded", False):  # the training rows only (not an eval set)
        return dl
    self._fed_rows_recorded = True
    out = os.path.join(self.config.output_dir, "fed_rows.json")
    state = {"h": hashlib.sha256(), "rows": 0, "shapes": [], "per_batch": [], "full": True, "first": None}
    base = type(dl)

    def __iter__(it_self):
        for batch in base.__iter__(it_self):
            ids = batch["input_ids"].detach().cpu().to(torch.int64).contiguous()
            state["h"].update(ids.numpy().tobytes())  # the bytes of the rows, in feed order
            state["rows"] += int(ids.shape[0])
            state["shapes"].append(list(ids.shape))
            state["per_batch"].append(_sha(ids))
            if state["first"] is None:
                state["first"] = ids[:4, :4].tolist()
            am, lab = batch.get("attention_mask"), batch.get("labels")
            state["full"] &= bool(am is None or bool((am == 1).all())) and bool(
                lab is None or torch.equal(lab.detach().cpu().to(torch.int64), ids))
            os.makedirs(os.path.dirname(out), exist_ok=True)
            seqs = {s[1] for s in state["shapes"]}
            json.dump({"rows": state["rows"], "seq": seqs.pop() if len(seqs) == 1 else None,
                       "batch_shapes": state["shapes"], "sha256_all": state["h"].hexdigest(),
                       "sha256_per_batch": state["per_batch"],
                       "full_length_and_labels_are_ids": state["full"],
                       "first_tokens": state["first"]}, open(out, "w"), indent=1)
            yield batch

    dl.__class__ = type("FedRows" + base.__name__, (base,), {"__iter__": __iter__})
    return dl


_T.PorosTrainer._prepare_dataloader = _prepare_dataloader
