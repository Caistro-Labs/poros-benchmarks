# The release's own checks

What was run on the release's code, commit 1a8d7e912, on one RTX 4090, on one Mac (M4, 16 GB)
and on one 8 GB laptop card, and on a CPU. The display fix is the release's change to six display
files, so that the run card shows from the start of a run in the session; each `display_fix/`
folder shows runs at that change ending at the certified adapter.

| Folder | What it is |
|---|---|
| [`4090/release-1a8d7e912/`](4090/release-1a8d7e912/README.md) | the release checks on the RTX 4090: every supported family's paths and resume, the dropout-on checks, the GPU optimizer tests and the OLMoE optimizer study, `poros prove` on Qwen2.5-7B, and the recipes' runs |
| [`4090/optimizer_selftest/`](4090/optimizer_selftest/README.md) | the expert optimizer's self-test on the RTX 4090 |
| [`4090/display_fix/`](4090/display_fix/README.md) | the display fix on the RTX 4090: first-run ends at its 1a8d7e912 adapter, through the driver and in the session |
| [`4060l/optimizer_selftest/`](4060l/optimizer_selftest/README.md) | the same self-test on the RTX 4060 Laptop GPU |
| [`4060l/display_fix/`](4060l/display_fix/README.md) | the display fix on the laptop: first-run and moe-30b-8gb end at their 1a8d7e912 adapters |
| [`mac/optimizer_selftest/`](mac/optimizer_selftest/README.md) | the same self-test on the Mac |
| [`mac/display_fix/`](mac/display_fix/README.md) | the display fix on the Mac: mac-first-run and mac-moe end at their 1a8d7e912 adapters |
| [`mac/launch-144-int/`](mac/launch-144-int/README.md) | Qwen2.5-72B-Instruct for 144 steps on the Mac, from its internal SSD |
| [`cpu/expert_adam_precision/`](cpu/expert_adam_precision/README.md) | the expert optimizer studied on the CPU, and the Mac kernel's arithmetic checked there |
| [`display_fix/resolve/`](display_fix/resolve/README.md) | the pin and the release resolve every built-in recipe to the same run, byte for byte |
| [`resolve_check/`](resolve_check/README.md) | `resolve.py`, which resolves every built-in recipe's run settings without loading a model, as the train command would before a run |

The family rows and Mac rows the model catalog cites from earlier commits are in
[`4090/`](4090/README.md) and [`mac/`](mac/README.md), listed with the other earlier runs on the
front page.
