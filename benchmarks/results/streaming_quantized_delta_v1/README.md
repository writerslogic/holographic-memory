# Quantized delta search: rejected development result

The selected ordinary i8/ef256 traversal fails its reserved-query recall check.
Keep exact delta search. The optional native implementation is preserved only in
`production.patch`; this result does not add or enable that implementation.

| Prefix | State | Exact hits / 1280 | i8 hits / 1280 | Recall change | T1 time ratio | T8 time ratio |
|---|---|---:|---:|---:|---:|---:|
| 1M | live | 949 | 943 | -0.004688 | 0.6311 | 0.6634 |
| 1M | deleted | 926 | 922 | -0.003125 | 0.5845 | 0.6589 |
| 3M | live | 886 | 878 | -0.006250 | 0.4678 | 0.4418 |
| 3M | deleted | 884 | 877 | -0.005469 | 0.4673 | 0.4222 |

Every state exceeds the preregistered maximum recall loss of 0.0025. All query
pairs were faster, but speed does not satisfy the recall condition. Query IDs
384..511 were opened once at each size after the six-candidate tuning grid
selected ordinary i8 at ef256; no replacement setting was selected afterward.
The failed rotated and binary variants remain in `tune-summary.json`.

The final production verification on tuning IDs256..383 matched the prototype's
full result IDs and score bits in both states. Its live/deleted median time
ratios were T1 0.6260/0.6212 and T8 0.6673/0.6699. Appending 100k rows took a
median 1.0226x the exact path, with identical graph checksums across all six arms.
The cache used 116 MB for 1M 100D rows and 130.5 MB after append; original float rows
remain allocated. These numbers do not demonstrate a total-memory saving.

Measurements used Intel Core Ultra 9 185H, CPUs 0–7, T1/T8, three alternating
pairs and five batches per arm under an exclusive timing lock. Each bounded
process used a no-swap 3 GiB or 6 GiB cgroup. Exact per-run load averages and
oracle-inclusive process peaks are in `summary.json`. They are isolated-delta
development measurements, not mixed-index throughput, standing, or proof that
the 30M runbook completes within one hour and 8 GiB.

`summary.json` records source/binary hashes, all 36 production pairs, checks and
artifact SHA256. Input manifests identify the normalized prefix and official queries. Raw logs and the deterministic parser are included. Native
production source is commit b7713612; final binding-test source f6c1fd7f.
Default fmt/clippy/tests, Python feature compilation, 42 Linux update tests and
two Python binding metric tests passed before measurement. No durable bytes
changed; correctness checks do not override the failed recall criterion.

To reproduce the experimental source independently, apply `production.patch`
to parent ac235724, build `cargo test --locked --release --lib --no-run`, and use
the resulting test binary as `grid-test`. `selected.sh` documents the exact
selected-arm environment and limits; its only allowed query offsets are 256
(tuning) and 384(reserved confirmation). The original six-mode source patch
against b228cae0 and its raw tuning log are retained separately. Recompute each
log with `uv run --no-project python summarize.py LOG`.

The full official command remains frozen. After an explicit freeze lift, run the
exact default path from the pinned harness, with a separately validated final
operating point:

```sh
flock ~/hms-timing.lock systemd-run --user --scope -q \
  -p MemoryMax=8G -p MemorySwapMax=0 taskset -c 0-7 \
  /usr/bin/time -v timeout 3600 uv run python run.py --nodocker \
  --neurips23track streaming --algorithm hms --dataset msturing-30M-clustered \
  --runbook_path neurips23/runbooks/final_runbook.yaml
```

No result here establishes readiness or a leaderboard win.
