# The display fix on the RTX 4060 Laptop GPU: the same adapters as at the pin

At commit 40c2e9e348811cc608bc0fa5c2aad53eacbf2554 (the release, with the run card's display fix), `first-run` ran through the certification driver (`driver/first-run/`) and inside the interactive session (`session/first-run/`), and `moe-30b-8gb` ran inside the session (`session/moe-30b-8gb/`).
`adapter_digests.txt` lists each run's adapter sha256 beside the run of the same recipe at 1a8d7e912 (`docs/validation/recipes/rtx4060-laptop/`); every pair is equal, and each `*_compare.json` holds that comparison for one run.
In `session/`, the home folder's path in the runs' records reads `~` and the user name `<user>`; nothing else is changed.
