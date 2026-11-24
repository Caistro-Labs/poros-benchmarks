# A measured run of moe-30b-8gb on the RTX 4060 Laptop GPU, not a certification

The recipe leaves its path to the run, and on this 8 GB card the check turned down `efficient` and the run took `fast` (`main/config.resolved.json`, `path_reason`). The generator certifies such a recipe only where the run decides `efficient` or `sequential`, so `moe-30b-8gb` is certified on the RTX 4090 (`../../rtx4090/moe-30b-8gb/`) and measured here.
It ran through the certification driver at commit 1a8d7e912b166c373fcfec862ad7297221d33a71 (`commit.txt`), which wrote `certification.json`; a second run at the same commit (`../moe-30b-8gb-repeat/`) ended at the same adapter, byte for byte.
