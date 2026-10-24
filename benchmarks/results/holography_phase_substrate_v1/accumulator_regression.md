# Accumulator allocation regression and correction

The first integrated task-serialized sweep exposed a material regression at
resolution 16/dimension 512: eight-vector bundling took approximately 22.5 µs
versus the original 5.7 µs, and 32-vector bundling took 81.8 µs versus 13.7 µs.
These slow measurements were internally consistent across two fresh processes;
a third process completed the new 32-vector bundle in 9.6–11.1 µs. Similarity,
binding, and unbinding stayed stable. This was a reproducible process-dependent
regression, not a result to dismiss as ordinary scheduler noise.

The failed/intermediate sweep is preserved in
[`pre_accumulator_serial.json`](pre_accumulator_serial.json). Its exact production
source is [`pre_accumulator_phase_hvec.rs`](pre_accumulator_phase_hvec.rs), SHA-256
`6b95b18702ced57c2ef0ecff327a155b1a6df726453748183c87a0e82ad4d8d5`.
The archived source hash was checked against the measurement metadata.

The general bundle accumulator now uses one `Vec<[i64; 2]>`, with real and
imaginary sums adjacent, instead of two separately allocated `Vec<i64>` buffers.
Integer sums, vector iteration order, phase conversion, total accumulator
payload, and public output representation remain identical. There is one fewer
temporary allocation. The generated ARM code changes from two scalar
load/add/store streams to one vector load/add/store stream (`add.2d`);
[`accumulator_codegen.txt`](accumulator_codegen.txt) preserves the compared
instruction sequences and source hashes. The specific underlying cache or
memory-dependency mechanism has not been established with hardware counters.

The bounded verification used nine fresh processes for each dimension 511, 512,
and 513 at resolution 16, with three alternating 5 ms samples per operation:

```sh
uv run --no-project python benchmarks/research/phase_substrate/run.py \
  --label serial-validation \
  --case 16:511 --case 16:512 --case 16:513 \
  --sample-ms 5 --repeats 3 --cold-trials 9 \
  --output benchmarks/results/holography_phase_substrate_v1/accumulator_serial.json
```

[`accumulator_serial.json`](accumulator_serial.json) retains every sample. Across
the nine dimension-512 processes, corrected per-process medians were:

| Bundle size | Minimum median | Maximum median |
| ---: | ---: | ---: |
| 2 | 2.818 µs | 2.862 µs |
| 8 | 3.966 µs | 4.153 µs |
| 32 | 7.640 µs | 7.869 µs |

The original baseline itself reproduced the slow mode in two of these processes:
32-vector bundling took 82.641/83.178 µs while the corrected implementation took
7.772/7.828 µs. That result supports allocation-layout sensitivity in the original
two-array design. It also explains why a ratio measured against one favorable
allocation arrangement should not be treated as a universal speedup. Corrected
bundles at the neighboring dimensions improved as well.

All 27 fresh-process runs passed 27,675 bitwise similarity checks and 940,032
exact phase checks. The focused suite now has 16 passing tests, including direct
fixed-point reference comparisons at dimensions 511/512/513, cached and uncached
resolutions, and bundle sizes 2/8/32. Clippy passes with warnings denied; commands
and output are in [`accumulator_validation.txt`](accumulator_validation.txt).

Measurement was serialized within this task; unrelated external host workloads
were still possible. The parent agent runs the final integrated sweep against
the corrected production source, retaining this failed/intermediate evidence.
