#!/usr/bin/env python3
"""philox_step.py OUT.json -- how far one dropout draw moves the CUDA generator, by element count.

A block's dropout masks are fixed by the generator state it starts from, and each draw moves
the Philox offset by an amount set by the element count and the device. A MoE expert's adapter
draws over the rows routed to it: where the step depends on the count, the state a later
micro-batch starts from depends on routing that block-major order has not computed yet."""
import json
import sys

import torch
import torch.nn.functional as F

torch.cuda.init()
gen = torch.cuda.default_generators[torch.cuda.current_device()]
props = torch.cuda.get_device_properties(0)
rows = []
for n in (1, 1024, 65536, 262144, 786432, 786433, 1048576, 2097152, 4194304, 16777216):
    x = torch.empty(n, device="cuda")
    before = gen.get_offset()
    F.dropout(x, 0.05, training=True)
    rows.append({"elements": n, "offset_step": gen.get_offset() - before})
out = {"gpu": props.name, "multiprocessors": props.multi_processor_count,
       "max_threads_per_multiprocessor": props.max_threads_per_multi_processor,
       "torch": torch.__version__, "draws": rows}
json.dump(out, open(sys.argv[1], "w"), indent=1)
print(json.dumps(out))
