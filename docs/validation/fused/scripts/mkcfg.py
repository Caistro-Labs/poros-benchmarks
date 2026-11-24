#!/usr/bin/env python3
"""mkcfg.py OUT.yaml MODEL DATASET key=value ... -- the shared head-to-head workload for poros train.
Shared with the Unsloth harness: 512-token rows, 8 rows per optimizer update (micro-batch 1 x accumulation 8),
LoRA r16 alpha 32 dropout 0 on the 7 linear projections, AdamW lr 2e-5 (betas 0.9/0.999, eps 1e-8, wd 0.01),
constant LR, no warmup, grad-norm clip 1.0, seed 42, strict determinism."""
import sys, yaml
out, model, data = sys.argv[1:4]
c = dict(model_name_or_path=model, dataset=data, quantization="nf4", max_seq_length=512,
         lora_rank=16, lora_alpha=32, lora_dropout=0.0,
         lora_target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
         learning_rate=2e-5, weight_decay=0.01, max_grad_norm=1.0, warmup_steps=0, lr_scheduler="constant",
         gradient_accumulation_steps=8, per_device_batch_size=1, max_steps=8, save_steps=0, logging_steps=1,
         eval_split=0.0, seed=42, deterministic=True, progress=False)
for kv in sys.argv[4:]:
    k, v = kv.split("=", 1); c[k] = yaml.safe_load(v)
yaml.safe_dump(c, open(out, "w"), sort_keys=False)
print(out, c)
