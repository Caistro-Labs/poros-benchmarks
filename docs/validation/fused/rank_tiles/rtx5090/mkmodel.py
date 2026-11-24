#!/usr/bin/env python3
"""mkmodel.py OUT_DIR [--layers L] [--preset q72|q32|q7] -- a random-weight Qwen2 checkpoint with a real model's layer
dimensions (bf16 safetensors) and a small vocab, for per-layer timing without the real checkpoint. Values are
Qwen2's own init (normal 0.02); timing does not depend on them."""
import argparse, json, os, time

P = {
    "q72": dict(hidden_size=8192, intermediate_size=29568, num_attention_heads=64, num_key_value_heads=8),
    "q32": dict(hidden_size=5120, intermediate_size=27648, num_attention_heads=40, num_key_value_heads=8),
    "q7": dict(hidden_size=3584, intermediate_size=18944, num_attention_heads=28, num_key_value_heads=4),
}
ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("--layers", type=int, default=4)
ap.add_argument("--preset", default="q72"); ap.add_argument("--vocab", type=int, default=32768)
a = ap.parse_args()
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

if os.path.exists(os.path.join(a.out, "config.json")):
    print("exists", a.out); raise SystemExit(0)
cfg = Qwen2Config(**P[a.preset], num_hidden_layers=a.layers, vocab_size=a.vocab, max_position_embeddings=32768,
                  rope_theta=1000000.0, rms_norm_eps=1e-6, tie_word_embeddings=False, use_sliding_window=False,
                  bos_token_id=0, eos_token_id=1, pad_token_id=1)
t = time.time()
torch.manual_seed(0)
torch.set_default_dtype(torch.bfloat16)
m = Qwen2ForCausalLM(cfg)
m.save_pretrained(a.out, safe_serialization=True)
print("saved", a.out, "params", sum(p.numel() for p in m.parameters()), "in", round(time.time() - t, 1), "s")
json.dump({"preset": a.preset, "layers": a.layers, "vocab": a.vocab}, open(os.path.join(a.out, "mkmodel.json"), "w"))
