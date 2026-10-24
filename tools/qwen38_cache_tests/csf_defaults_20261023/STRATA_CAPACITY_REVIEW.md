# Strata / SGLang expert capacity comparison (2026-10-23)

This is a source and existing-artifact audit, not a new Strata GPU measurement.
The requested `/data/workspace/strata` does not exist on this host. The inspected
checkout is `/home/user/workspace/Strata`. The exact command/log behind the user's
8,000+ observation has not yet been supplied.

## Findings

The Strata checkout documents approximately 8,350 NVFP4 slots in
`docs/NVFP4.md:50`: RTX 5090, 64K context, INT8 KV, 1,500 MiB reserve.
That section reports the fork author's OrcaRouter/ModelOpt run, not a matched
FP4-Local QAD run. Its FP4-Local discussion is a separate section at line 227.

Our current native control is `root-paired-native-graph/command.json` and
`default.log`: FP4-Local QAD, context 220000, KV stages 80000/160000/220032,
target FP8 KV, resident NVFP4 MTP, BF16 draft KV, original batched MTP,
8,192-token prefill chunks, and fixed cache budget 17,661,593,504 bytes.

| Property | Strata source/documentation | Our artifacts |
| --- | --- | --- |
| Native expert bytes | 2,764,816 before slot alignment | 2,764,808 |
| Slot allocation alignment | 256 bytes; 2,764,992-byte stride without extra guard | VMM allocation regions; not 256-byte padding per expert |
| Native stage capacities | Documented ~8,350 at 64K | 7,006 / 6,643 / 6,388 |
| CSF stage 0 / final | Documented native pack expands CSF scales | 7,296 / 6,652 |
| Expert budget policy | Free memory after weights/session, less reserves | Explicit byte budget, plus loans from unused target KV |
| Draft expert format | Q2_0, 1,382,400 bytes/expert | NVFP4 plus scales, approximately twice as large |
| Output head | Q8_0 conversion policy | BF16 |

Expert geometry is H=2560, FF=640. Native bytes are
`3 * H * FF * (1/2 + 1/16)`: 2,764,800, plus 16 bytes of Strata
projection scales or 8 bytes of our packed alphas. Expert packing therefore
does not explain a thousand-slot deficit; Strata's aligned native slot is
slightly larger.

## Quantified format differences

The local FP4-Local safetensors headers contain 1,415,583,744 bytes of MTP
expert codes/scales. Strata's Q2_0 representation of the same geometry is
707,788,800 bytes. Difference: 675.006 MiB, approximately 256 native slots.
This is an additional lossy quantization, not a lossless packing improvement.

The checkpoint output head has shape [248320,2560]. Its BF16 size is
1,271,398,400 bytes; Q8_0 is 675,430,400 bytes. Difference: 568.359 MiB,
approximately 215.6 native slots. The existing GPU ownership capture
`../memory_audit_20261019/clean/short-first/owners.json.gz` confirms our head
was resident BF16. This capture is older than the current control; the current
loader retains the checkpoint's mixed-precision policy.

Together these two format choices represent approximately 472 native slots.
They are geometric calculations, not measured allocator savings or a measured
capacity improvement. Other dense tensors and draft buffers can also differ.
Our dense path retains mixed precision; it must not be described as all BF16.

## Budget and context differences

The 8,350 vs 7,006 native gap corresponds to 3.461 GiB of native payload.
Comparing 8,350 with the final-stage 6,388 instead would mix context stages.

At our 80K stage only the unused target KV tail is loaned to experts. Changing
the initial stage from 80,000 to 65,536 would recover only approximately 64
native slots, estimated from the observed 618-slot loan across 140,032 tokens.
It would not account for a thousand slots by itself. The draft KV remains
separately allocated in this control.

Our current log reports approximately 2.24 GiB free after graph capture,
Mamba state allocation for 10 cache entries (including intermediate verify
state), a separate BF16 draft KV allocation of approximately 0.42 GiB, and
approximately 0.44 GiB consumed during graph capture. These are audit targets,
not demonstrated waste. Startup free memory does not establish prefill peak
headroom; prior 8,192-token prefill investigations recorded workspace OOMs.

The explicit byte budget means 7,006/7,296 is a configured test capacity, not
a search result proving the maximum usable capacity of the latest code.
`mem_fraction_static=0.9` also affects pool sizing; its remaining fraction
must not be treated as wholly disposable memory.

## Remaining evidence needed

Obtain the exact Strata startup command, model/pack identity, final resident
slot count and GPU memory breakdown for the reported 8,000+ run. Compare both
at equal context, draft format, reserve and prefill chunk, including long
prefill peak allocations. Until then, the remaining capacity gap cannot be
assigned precisely to runtime overhead, reserve or additional format changes.
No production configuration was changed and no service was restarted.
