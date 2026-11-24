"""trunc_gemma4.py OUT LAYERS -- a text-only checkpoint of google/gemma-4-31B's first LAYERS decoder layers (real
weights), fetched with HTTP range requests (only the tensors kept cross the network): the embedding (tied head), the
final norm and layers 0..LAYERS-1, renamed from Gemma4ForConditionalGeneration's model.language_model.* to
Gemma4ForCausalLM's model.*; the text config with num_hidden_layers=LAYERS and its first LAYERS layer types; the
tokenizer. Records the revision and every source tensor's shard and byte range."""
import json, os, struct, sys, urllib.request

import torch
from safetensors.torch import save_file

REPO, REV = "google/gemma-4-31B", "5bbc2fb1c1b2c611d06e3d9f23c170ba21659d89"
out, layers = sys.argv[1], int(sys.argv[2])
os.makedirs(out, exist_ok=True)
URL = f"https://huggingface.co/{REPO}/resolve/{REV}/{{}}"
DT = {"BF16": torch.bfloat16, "F32": torch.float32, "F16": torch.float16}


def get(name, start=None, end=None):
    req = urllib.request.Request(URL.format(name), headers={"User-Agent": "trunc-script"})
    if start is not None:
        req.add_header("Range", f"bytes={start}-{end - 1}")
    with urllib.request.urlopen(req, timeout=600) as r:
        return r.read()


cfg = json.loads(get("config.json"))
index = json.loads(get("model.safetensors.index.json"))["weight_map"]
keep = {k: v for k, v in index.items() if k.startswith("model.language_model.") and (
    ".layers." not in k or int(k.split(".layers.")[1].split(".")[0]) < layers)}
headers = {}
for shard in sorted(set(keep.values())):
    n = struct.unpack("<Q", get(shard, 0, 8))[0]
    headers[shard] = (8 + n, json.loads(get(shard, 8, 8 + n)))
tensors, sources = {}, {}
for k, shard in sorted(keep.items()):
    base, h = headers[shard]
    meta = h[k]
    s, e = meta["data_offsets"]
    raw = bytearray(get(shard, base + s, base + e))
    t = torch.frombuffer(raw, dtype=DT[meta["dtype"]]).reshape(meta["shape"]).clone()
    name = "model." + k[len("model.language_model."):]
    tensors[name] = t
    sources[name] = {"shard": shard, "bytes": [base + s, base + e], "dtype": meta["dtype"], "shape": meta["shape"]}
    print(name, meta["shape"], flush=True)
save_file(tensors, os.path.join(out, "model.safetensors"), metadata={"format": "pt"})
text = dict(cfg["text_config"])
text.update(num_hidden_layers=layers, layer_types=text["layer_types"][:layers],
            architectures=["Gemma4ForCausalLM"], tie_word_embeddings=cfg.get("tie_word_embeddings", True))
json.dump(text, open(os.path.join(out, "config.json"), "w"), indent=1)
for f in ("tokenizer.json", "tokenizer_config.json", "generation_config.json"):
    open(os.path.join(out, f), "wb").write(get(f))
json.dump({"source": REPO, "revision": REV, "layers": layers, "tensors": sources},
          open(os.path.join(out, "truncation.json"), "w"), indent=1)
print("wrote", out, sum(t.numel() for t in tensors.values()), "params")
