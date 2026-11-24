Laptop release checks, 2026-11-19 (UTC), RTX 4060 Laptop GPU, Poros at 748099f5922f01372aa62482f21b8b891aa6fbce
(a copy of the tree with its manifest, poros-748099f5....sha256, installed as a wheel into a Python in the run folder).

Step 1, poros prove MODEL --steps 100 --seq-len 512 --download --json (the resident reference: every layer on the GPU
under ordinary autograd, against streaming in blocks of 4, both in one command on this card):
  Qwen/Qwen2.5-0.5B  identical; nothing differs (losses at 100 steps, 336 adapter gradients, 336 adapter tensors,
                     1008 optimizer tensors). Run twice: the first, plain run is kept in laptop/rec/run1_plain/;
                     the second ran through prove_wrap.py, which calls the same CLI and also saves the comparison
                     record (both arms' per-step losses) to prove_Qwen2.5_0.5B_record.json. Both runs' digests agree:
                     bd5fb4106e0ebedd03496e68001cb31695808198096b26f36ed4ca676012e5f0.
  Qwen/Qwen2.5-1.5B  refused before any download: E_USAGE "prove runs a model a parity receipt covers; ... is not one"
  Qwen/Qwen2.5-3B    the same refusal (prove takes validated dense catalog models only)
  Qwen/Qwen2.5-7B    first asked its documented question (resident_wont_fit; prove_Qwen2.5_7B_asked.json), then with
                     --force refused before loading: E_OOM_PREDICTED "the resident reference holds every layer of
                     Qwen/Qwen2.5-7B on the GPU: at least 7.80 GiB, and 6.70 GiB is free" (prove_Qwen2.5_7B.json)
Step 2, poros machine (pty capture machine.raw, rendered machine_screen.txt; --json machine.json) and the start box
  (startbox_screen.txt; poros with no arguments, stopped by a 20 s timeout as it waits for input). The start box shown
  here was captured again on 2026-11-20 at the certification commit 90570998 (poros 1.0.0), so it shows the release's
  line (startbox_screen.NOTE.txt says when and where); every measurement in this folder is at 748099f5, as above.
Step 3, FG_CHECKS=parity FG_ANYMODEL=0 bash scripts/family_gate.sh, FG_RESIDENT_EVIDENCE unset: laptop/fg_out/RESULT.txt.
  PASS (resident == streamed, 8 steps): qwen2 (Qwen2.5-0.5B), qwen3 (Qwen3-0.6B), qwen3_5 (Qwen3.5-0.8B),
  gemma3_text (gemma-3-270m). FAIL, the resident arm out of GPU memory, no comparison made: qwen3_8 (8 of 64 layers of
  Qwen3.8-27B), gemma3 (6 of 34 layers of gemma-3-4b-it), gemma4 and gemma4_text (6 of 60 layers of gemma-4-31B).
Not run (not part of steps 1-3 or waiting): the recipe certification (step 4), the 30B (step 5).
