# voice: a diagnostic re-run (not the certified run)

**Why it was run.** The `voice` recipe's certified run at 1a8d7e912 (`../blockB/step2/voice`) took 2.8861 s/step, 11.3% above
the 2.5942 certified at 67bd13cc. Its eval_loss and memory were unchanged.

**What was run.** The same driver ran `voice` once more, alone on a quiet machine, into its own folder: 2026-11-24, 10:07 to
10:12 UTC.

**Result.** 2.9342 s/step, eval_loss 5.525723184858049 (equal), allocated peak 6.386 GiB (equal), commit 1a8d7e912,
tree_clean true.

**What counts.** The certified run stays the first one, at 2.8861 s/step. This one is kept as evidence of the run-to-run
spread on this recipe: on 2026-11-24, 2.886 and 2.934 s/step.
