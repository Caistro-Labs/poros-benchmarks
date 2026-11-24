#!/usr/bin/env python3
"""mk30.py OUT_DIR --layers L -- a random-weight checkpoint with Qwen3-30B-A3B's exact per-layer config (hidden 2048,
128 experts top-8, moe_intermediate 768, 32 heads / 4 KV, head_dim 128, its vocab and rope), L decoder layers, bf16
safetensors, tied embeddings (disk), and Qwen2.5's tokenizer files copied in (from a downloaded Qwen/Qwen2.5-0.5B
under HF_HOME). Values: normal(0, 0.02), norms 1. The 6-layer checkpoint of the precision table: --layers 6."""
import argparse, glob, json, os, shutil, time

ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("--layers", type=int, default=1)
a = ap.parse_args()
import torch
from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM

if os.path.exists(os.path.join(a.out, "config.json")):
    print("exists", a.out); raise SystemExit(0)
cfg = Qwen3MoeConfig(vocab_size=151936, hidden_size=2048, intermediate_size=6144, moe_intermediate_size=768,
                     num_hidden_layers=a.layers, num_attention_heads=32, num_key_value_heads=4, head_dim=128,
                     num_experts=128, num_experts_per_tok=8, norm_topk_prob=True, decoder_sparse_step=1,
                     mlp_only_layers=[], rms_norm_eps=1e-6, rope_theta=1e6, max_position_embeddings=40960,
                     tie_word_embeddings=True, bos_token_id=151643, eos_token_id=151643, torch_dtype="bfloat16")
t = time.time()
torch.manual_seed(0)
torch.set_default_dtype(torch.bfloat16)
m = Qwen3MoeForCausalLM(cfg)
with torch.no_grad():
    for n, p in m.named_parameters():
        if "norm" in n:
            p.fill_(1.0)
        else:
            p.normal_(0.0, 0.02)
m.save_pretrained(a.out, safe_serialization=True)
hub = os.path.join(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "hub")
snap = sorted(glob.glob(os.path.join(hub, "models--Qwen--Qwen2.5-0.5B/snapshots/*")))[0]  # its tokenizer
for f in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "generation_config.json"):
    if os.path.exists(os.path.join(snap, f)):
        shutil.copy(os.path.realpath(os.path.join(snap, f)), os.path.join(a.out, f))
json.dump({"layers": a.layers, "source": "mk30.py random init (Qwen3-30B-A3B per-layer config)"},
          open(os.path.join(a.out, "mk30.json"), "w"))
print("saved", a.out, "params", sum(p.numel() for p in m.parameters()), "in", round(time.time() - t, 1), "s")
