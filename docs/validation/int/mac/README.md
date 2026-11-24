# The Mac: earlier checks the package cites

The release's own Mac runs are the certification runs in `docs/validation/recipes/m4-16gb/`,
[`launch-144-int/`](launch-144-int/README.md) and [`optimizer_selftest/`](optimizer_selftest/README.md).
The folders below are earlier development checks, kept only because a file of the package or a
test names them. No figure in the README, the docs or the paper comes from them.

| Path | What it is | Code | Cited by |
|---|---|---|---|
| `round4/receipts/` | the Mac table rows the model catalog cites, and their packs' sizes | 5d75e31a | `poros/catalog.toml`, `tests/test_recipe_certification.py` |
| `round5/cert-first/` | the first certification runs, which two recipe tests read | 748099f5 | `tests/test_recipe_certification.py` |
