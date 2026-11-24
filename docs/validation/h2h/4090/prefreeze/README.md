# Earlier-session runs of Unsloth from its own default (pre-quantized 4-bit) sources

These runs are from the pre-freeze sessions on the same RTX 4090, before the code was frozen at 5d75e31a. Each is a single run.
Unsloth was the same version as in the formal session: unsloth 2026.11.4 and unsloth_zoo 2026.11.4 (`versions_<session>.txt`).
No comparison key is built from them; their keys carry gate "none", and each key's note begins "Earlier session, single run:".

| Record | Session (Poros commit) | Model, source | Setting | Result |
|---|---|---|---|---|
| p72_uns_auto | 2026-11-17, Poros at 0398b814 | Qwen/Qwen2.5-72B from unsloth/Qwen2.5-72B-bnb-4bit (revision 88a44567) | offload auto | trained 8 updates: 21.054 s/update (median of updates 2..8), 21.025 GiB reserved VRAM, 42.944 GiB host RAM (not file-backed), load 140.18 s, first update at 176.71 s |
| p72_uns_min | 2026-11-17, Poros at 0398b814 | the same copy | all 80 layers and the embedding offloaded | killed by the kernel while loading at 58.041 GiB of host RAM on a 59.6 GiB machine (rc 137) |
| p72_uns_mid | 2026-11-17, Poros at 0398b814 | the same copy | 70 of 80 layers and the embedding offloaded | killed by the kernel while loading at 58.288 GiB of host RAM on a 59.6 GiB machine (rc 137) |
| m_uns_mirror | 2026-11-17, Poros at a2531867 | Qwen/Qwen3-30B-A3B from unsloth/Qwen3-30B-A3B-bnb-4bit | offload auto | did not train: failed at the first training step, its error line: `TorchRuntimeError: RuntimeError when making fake tensor call` (a packed 4-bit linear weight of shape (131072, 1) where the computation expects an unpacked one) |
