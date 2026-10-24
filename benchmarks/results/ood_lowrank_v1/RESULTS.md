# Reject the low-rank OOD prototype

The preregistered yi-128-ip test failed: the selected low-rank index achieved
0.27118, 0.28953 and 0.28465 times the graph's throughput in three alternating
pairs. Both selected configurations met recall@100 >=0.95. The original stop rule
therefore ends this experiment; coco-100k is unmeasured. No VIBE winning dataset
or standing is established.

Development measurement: Intel Core Ultra 9 185H, Linux 6.18.33.2/WSL2, CPU 0 for
queries, two build threads, load1 samples 1.00000–1.13721, three tuning rounds and
three selected-point confirmation rounds. Source eb32e65267b40419d13d6f75eee5465e937ca95d,
parent ac235724967cf6c7046ba3c02882a6046d37d075, x86-64-v3 with runtime kernel
dispatch. Raw values and dataset/module/harness hashes are in yi-development.json.

| Selected configuration | Learn validation recall | Confirmation recall | Median confirmation QPS |
|---|---:|---:|---:|
| Graph R64, ef160, rerank400 | 0.95119 | 0.95016 | 7020.6374 |
| Low-rank C1024, rank32, nprobe256, rerank400 | 0.95236 | 0.95120 | 1998.4033 |

All six candidate builds are retained below. These tuning results are not
independent confirmations. Every one of the309 query cells and 927 tuning batches
is preserved in the raw JSON; verify.py independently reconstructs the complete
grid, eligibility, selected indices and the paired verdict.

| Low-rank build | Best validation recall in grid | Fastest eligible nprobe/rerank | Median tuning QPS |
|---|---:|---|---:|
| C256/r16 | 0.94204 | none | — |
| C256/r32 | 0.98346 | 96/800 | 1958.6694 |
| C256/r64 | 0.99908 | 96/400 | 1705.3525 |
| C1024/r16 | 0.94325 | none | — |
| C1024/r32 | 0.96449 | 256/400 | 2052.5293 |
| C1024/r64 | 0.96817 | 192/400 | 1259.3202 |

Training used 32,768 seeded official learn rows, tuning the next1,000, with exact
float 64 reference neighbors. Official test queries were opened only after
selection was written, then only the two selected points were evaluated. They
must not be reused for tuning. This is a bounded native development comparison,
not a VIBE harness run, leaderboard submission, Modal run, or paid run.

The prototype and Python registration are excluded from production. The exact
measured code is preserved in prototype.patch, applicable to the stated parent;
reproduce.sh builds it in an isolated checkout and executes the frozen protocol.
The abandoned input source was also preserved separately in coordination storage.
No saved format changed. Strict lint found a boolean simplification in the
prototype; its equivalent correction passed lint, but the timed source remains
the recorded SHA and the production code was rejected.

Unmeasured: coco, per-stage query counters, per-index peak memory, common-hardware
competitor arms and all live standings. The loss is large enough that integration
is unjustified; a different hypothesis requires its own preregistration.
