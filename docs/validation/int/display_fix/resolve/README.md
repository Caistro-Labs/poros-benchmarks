# The resolve check: the pin and the release resolve every recipe to the same run

`pin.json` is `resolve.py` (`../../resolve_check/`) run on the tree of commit 1a8d7e912, the
certification commit. `release.json` is the same run on the release's tree, which differs from it
in six display files and the generated `poros/recipes/certified.json` and `poros/plan/cells.json`.
Both read the release's certified data. The two files are byte-identical: every built-in recipe
gets the same steps, row length and run configuration, and on CUDA the same decisions before the
run. Paths under the run's home read `<home>`.

To check it again:

    python docs/validation/int/resolve_check/resolve.py <tree at 1a8d7e912> poros/recipes/certified.json pin.json
    python docs/validation/int/resolve_check/resolve.py . poros/recipes/certified.json release.json
    cmp pin.json release.json
