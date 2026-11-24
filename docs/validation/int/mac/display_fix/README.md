# The display fix on the Mac: the same adapters as at the pin

At commit 40c2e9e348811cc608bc0fa5c2aad53eacbf2554 (the release, with the run card's display fix), `mac-first-run` ran through the certification driver (`driver/mac-first-run/`) and inside the interactive session (`session/mac-first-run/`), and `mac-moe` ran inside the session (`session/mac-moe/`); `pytest-metal.txt` is the Mac tests at that commit.
`results.jsonl` and each run's `adapter.sha256` (the driver run's in its `certification.json`) hold the adapter digests, each equal to the certified run's at 1a8d7e912 in `docs/validation/recipes/m4-16gb/`.
