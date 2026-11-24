# The resolve check: every built-in recipe's run settings, without loading a model

`resolve.py` runs, for each built-in recipe on the simulated machine the certified data names for
it, the train command on the recipe's printed command line with nothing loaded or trained. It
records the steps and the row length, the run configuration the command hands the trainer (what a
run's resolved configuration file starts from), and on CUDA the decisions of the check before a run:
the path, the tables' place, the warning and the refusals. Two trees that give byte-identical
output resolve every recipe to the same run.

    python resolve.py <tree> <poros/recipes/certified.json> <out.json> [<data folder>]

The data folder (sample rows and the simulated store) is the fourth argument, or `resolve-data`
beside the output file. The release's own result, the pin against the release, is in
`docs/validation/int/display_fix/resolve/`.
