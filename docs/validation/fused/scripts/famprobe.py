"""famprobe.py MODEL OUT.json -- what the fused compute does with one model: its config's activation, its decoder layer
and MLP classes, and what poros.core.fused routes (counts, refusals), as poros train sets it up (NF4, PEFT LoRA r16 on
the seven projections). No training."""
import json, sys

import torch
from poros.core import fused
from poros.train.trainer import PorosTrainer

model, out = sys.argv[1], sys.argv[2]
tr = PorosTrainer.from_pretrained(model, quantization="nf4", lora_rank=16, lora_alpha=32, lora_dropout=0.05,
                                  max_steps=1, progress=False, output_dir="/tmp/famprobe_out")
cfg = tr.model.config
text = cfg.get_text_config() if hasattr(cfg, "get_text_config") else cfg
layers = [m for m in tr.model.modules() if type(m).__name__.endswith("DecoderLayer")]
mlps = sorted({type(getattr(l, "mlp", None)).__name__ for l in layers})
acts = sorted({type(getattr(getattr(l, "mlp", None), "act_fn", None)).__name__ for l in layers})
rec = {"model": model, "model_type": text.model_type,
       "activation": getattr(text, "hidden_act", None) or getattr(text, "hidden_activation", None),
       "act_fn_classes": acts, "decoder_layer_classes": sorted({type(l).__name__ for l in layers}),
       "mlp_classes": mlps, "attn_implementation": getattr(text, "_attn_implementation", None),
       "layer_types": sorted(set(getattr(text, "layer_types", None) or [])),
       "fused_compute": tr.config.fused_compute,
       "fused_installed": dict(getattr(tr.model, "_poros_fused", {}) or {}),
       "fused_refusals": fused.refusals(tr.model),
       "torch": torch.__version__}
json.dump(rec, open(out, "w"), indent=1)
print(json.dumps(rec))
