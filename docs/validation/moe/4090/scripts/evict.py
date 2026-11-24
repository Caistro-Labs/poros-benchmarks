#!/usr/bin/env python3
"""evict.py SNAPSHOT_DIR -- drop the checkpoint's files from the page cache (POSIX_FADV_DONTNEED on each, through
its symlink to the blob), so the next load reads them from disk: a cold load for whichever tool runs next."""
import os, sys
n = 0
for root, _, files in os.walk(sys.argv[1]):
    for name in files:
        path = os.path.realpath(os.path.join(root, name))
        fd = os.open(path, os.O_RDONLY)
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            n += 1
        finally:
            os.close(fd)
print(f"EVICTED {n} files")
