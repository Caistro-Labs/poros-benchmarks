"""refcheck_hooks -- import before `poros train` for the refcheck's stock arm (REFCHECK_STOCK=1): every MoE layer's
experts computed as transformers computes them -- Qwen3MoeExperts.forward (modeling_qwen3_moe.py: one_hot mask,
per-expert torch.where, F.linear, the routing weight, index_add_) over the store's NF4 weights, decoded per expert
(quant.dequantize, the weights every arm trains on), with the expert adapter added as NativeLoraLinear does
(base + (dropout(x) A^T B^T).to(base dtype) * scaling). Pair it with moe_expert_compute: reference (the custom
Function this replaces the compute of), fused_compute: false and attn_implementation: eager, so attention, norms
and loss are transformers' own too. Without REFCHECK_STOCK=1 importing this does nothing."""
import json
import os

import torch
import torch.nn.functional as F

if os.environ.get("REFCHECK_EVAL"):
    # every arm: the trained model's mean loss on the held-out rows when train() returns (the trainer's own
    # evaluate(): the model's forward, its loss), into {output_dir}/refcheck_eval.json
    from poros.train import trainer as _T

    _train = _T.PorosTrainer.train

    def _train_then_eval(self, *args, **kwargs):
        out = _train(self, *args, **kwargs)
        m = self.evaluate(os.environ["REFCHECK_EVAL"])
        json.dump({"eval_dataset": os.environ["REFCHECK_EVAL"], **{k: float(v) for k, v in m.items()}},
                  open(os.path.join(self.config.output_dir, "refcheck_eval.json"), "w"), indent=1)
        return out

    _T.PorosTrainer.train = _train_then_eval

if os.environ.get("REFCHECK_SKIP"):
    # localizing a difference: the fused path without some of its extras (comma list of attention, norm, head)
    from poros.core.fused import attention as _attn, head as _head, norm as _norm

    _skip = set(os.environ["REFCHECK_SKIP"].split(","))
    for _name, _mod, _fn in (("attention", _attn, "install_sdpa"), ("norm", _norm, "install_rms_norm"),
                             ("head", _head, "install_loss")):
        if _name in _skip:
            setattr(_mod, _fn, lambda model: None)
    from poros.train import compute as _C

    _inst = _C.RunCompute.install_moe

    def _install(self):
        _inst(self)
        if "head" in _skip:
            self._t.head.moe_fused = False

    _C.RunCompute.install_moe = _install

if os.environ.get("REFCHECK_STOCK") == "1":
    import poros.moe.modules as _m

    def _hf_eager_experts(hidden_states, top_k_index, top_k_weights, block):
        ad = block.adapter

        def lora(x, a, b):
            update = F.linear(F.linear(ad.dropout(x).to(a.dtype), a), b)
            return update.to(x.dtype) * ad.scaling

        final = torch.zeros_like(hidden_states)
        with torch.no_grad():
            mask = F.one_hot(top_k_index, num_classes=block.num_experts).permute(2, 1, 0)
            hit = torch.greater(mask.sum(dim=(-1, -2)), 0).nonzero()
        keys = [(block.layer_idx, int(e[0])) for e in hit]
        block.residency.ensure(keys)
        for key in keys:
            e = key[1]
            top_k_pos, token_idx = torch.where(mask[e])
            x = hidden_states[token_idx]
            slab = block.residency.resident_qbytes(key)
            gate_up = F.linear(x, block.store.dequantize_tensor(key, "gate_up_proj", slab))
            gate_up = gate_up + lora(x, ad.lora_A_gate_up[e], ad.lora_B_gate_up[e])
            gate, up = gate_up.chunk(2, dim=-1)
            h = block.act_fn(gate) * up
            down = F.linear(h, block.store.dequantize_tensor(key, "down_proj", slab))
            down = down + lora(h, ad.lora_A_down[e], ad.lora_B_down[e])
            down = down * top_k_weights[token_idx, top_k_pos, None]
            final.index_add_(0, token_idx, down.to(final.dtype))
        return final

    _m._experts_compute = _hf_eager_experts
