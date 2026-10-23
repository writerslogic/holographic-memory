# VIBE

The target is the CPU overview on VIBE's modern embeddings, using its published
default dataset selection and recall@100. SIFT and GloVe results do not establish
that HMS is competitive here. Report the complete recall/throughput curve,
per-dataset results and target coverage alongside the overview score.

## Published comparison, 2026-10-21

The [saved published inputs](../../benchmarks/results/vibe_v1/leaderboard_inputs/)
and [recomputed ranking](../../benchmarks/results/vibe_v1/leaderboard.json)
reproduce the [live overview](https://vector-index-bench.github.io/overview.html).
The score averages each method's QPS divided by the fastest eligible CPU QPS on
each dataset. An unreached recall target contributes zero; it is not dropped
from the denominator. The snapshot contains k=100 results only.

| Minimum recall | First | Second | Third |
|---|---|---|---|
| 0.73 | NGT-QG | SymphonyQG | Glass |
| 0.77 | SymphonyQG | Glass | NGT-QG |
| 0.90 | Glass | SymphonyQG | PAG |
| 0.94 | Glass | PAG | SymphonyQG |
| 0.95 | PAG | Glass | SymphonyQG |
| 0.99 | PAG | Glass | SymphonyQG |
| 1.00 | PyNNDescent | NGT-QG | PAG |

These are measured points, not inferred crossover boundaries. At 1.00,
PyNNDescent reaches the target on 10/11 datasets, NGT-QG on 6/11, PAG on 4/11,
and Glass on 3/11. A first-place average does not mean first place everywhere.
The result JSON retains the per-dataset comparisons.

VIBE's [exporter](https://github.com/vector-index-bench/vibe/blob/main/export_results.py)
counts returned distances within 0.001 of the exact kth-neighbor distance.
Consequently, its 1.00 recall is not a guarantee of exact neighbor-ID recovery.
Local diagnostics record both metrics, without rounding a near-1 value up to 1.

Large raw predictions, full timing reports and the deployed Parquet snapshot use
Git LFS. Fetch them before replaying or checking the saved results:

```sh
git lfs pull --include='benchmarks/results/vibe_v1/**'
```

| Dataset | Dimensions | Query distribution |
|---|---:|---|
| imagenet-clip | 512 | In distribution |
| arxiv-nomic | 768 | In distribution |
| landmark-dino | 768 | In distribution |
| msmarco-qwen | 1024 | In distribution |
| gooaq-distilroberta | 768 | In distribution |
| yahoo-minilm | 384 | In distribution |
| agnews-mxbai | 1024 | In distribution |
| yandex | 200 | Out of distribution |
| hotpotqa-harrier | 640 | Out of distribution |
| imagenet-align | 640 | Out of distribution |
| laion-clip | 512 | Out of distribution |

This is the published circle's selection, not every dataset in the upstream
repository. The additional 2048-dimensional iNaturalist set is a separate
dimension stress test. Larger-dimensional additional sets must not silently
replace the default circle.

## Measured risk

The [checked diagnostic](../../benchmarks/results/vibe_v1/summary.json) confirms
a dataset-dependent loss with the current 4-bit configuration. These results
use 100,000 indexed rows and 256 original queries, not the full datasets.
The follow-up below identifies a graph connectivity problem and tests an
encoding change; the initial deficit is not an established 4-bit ceiling.

| LAION adapter configuration | Maximum VIBE recall@100 | Reaches 0.95 |
|---|---:|---|
| HMS, current 4-bit query grid | 0.931680 | No |
| Glass, SQ4U search / FP32 refinement | 0.366328 | No |
| Glass, SQ8U search / FP32 refinement | 0.999102 | Yes |
| SymphonyQG, raw refinement | 0.956055 | Yes |
| RaBitQ-HNSW, 5 bits | 0.962656 | Yes |
| hnswlib | 0.999727 | Yes |

The two Glass rows demonstrate why one arbitrary quantization configuration
cannot represent that method. These are fixed-build diagnostic sweeps, not the
complete author parameter grids. All six completed adapter results were checked
against their saved predictions and timings, with independent float64 oracle
spot checks against the original vectors.

A native ablation then holds the graph, queries and search configurations
constant while changing traversal precision. It expands search up to ef=2000
and rerank=2000 and measures **strict neighbor-ID recall**, separately from the
tolerant VIBE metric above:

| Dataset | Dimensions | Maximum strict recall@100, 4-bit | 8-bit | 4-bit index bytes | 8-bit index bytes |
|---|---:|---:|---:|---:|---:|
| arxiv-nomic | 768 | 0.999609 | 0.999961 | 135,983,700 | 174,383,700 |
| laion-clip | 512 | 0.930195 | 0.978125 | 97,554,980 | 123,154,980 |
| inaturalist-resnet | 2048 | 0.998984 | 0.999961 | 327,960,732 | 430,360,732 |

Each pair has an identical recorded graph hash and identical search grids.
Both retain an 8-bit residual. The larger loss occurs in the tested
text-to-image workload, while the two higher-dimensional in-distribution sets
retain high recall. This does not establish behavior on the other VIBE datasets
or at their full scale. The wider native 4-bit sweep reports strict recall;
its tolerant VIBE recall was not measured.

LoRANN's trial exits with SIGILL after building; the cause is unestablished.
PyNNDescent's trial returns duplicate IDs and fails validation. Their
[failure logs](../../benchmarks/results/vibe_v1/logs/) are retained. Neither trial
supplies a valid score or invalidates its method's published results. NGT-QG's
native CLI is not installed. PAG remains blocked on an AVX-512 host.

No throughput winner is claimed from these initial diagnostic runs: they were not
interleaved and some builds overlapped. The native ablations also exposed an
existing Linux load-reporting bug: the driver tried macOS `sysctl`, leaving load
unknown. That bug is fixed for subsequent runs; the saved native results retain
unknown load and failed timing gates.

## Measured improvement

The [independently checked follow-up](../../benchmarks/results/vibe_v1/rotation/rotation-validation.json)
compares four encodings with the same graph, original queries and search grid.
An existing three-round signed Walsh-Hadamard transform, fixed seed 24301,
raises LAION's maximum VIBE recall from **0.933555 to 0.980469** with 4-bit
traversal. The transform adds 6,144 bytes at 512 dimensions. Query transformation
is included in timing. All configurations retain an 8-bit residual.

The shared 0.980469 ceiling in rotated 4-bit and both 8-bit arms led to a second
control. The cached directed graph has 63 strongly connected components, with
99,932 vertices in the largest one. Four small sink components have no outgoing
path. A deterministic, data-only control adds one edge from each sink to the
nearest vector in the largest component. It uses no queries or neighbor truth,
preserves every existing edge, and fits the existing degree limit and index
allocation. The [four edges and input hashes](../../benchmarks/results/vibe_v1/rotation/laion-escape-control.json)
are recorded; the original cache is preserved.

| LAION graph / encoding | Maximum VIBE recall@100 | Maximum strict ID recall@100 | Index bytes |
|---|---:|---:|---:|
| Original / plain 4-bit | 0.933555 | 0.929648 | 97,554,980 |
| Original / rotated 4-bit | 0.980469 | 0.977422 | 97,561,124 |
| Four escape edges / plain 4-bit | 0.999961 | 0.995625 | 97,554,980 |
| Four escape edges / rotated 4-bit | 1.000000 | 0.996953 | 97,561,124 |
| Four escape edges / plain 8-bit | 1.000000 | 0.997656 | 123,154,980 |
| Four escape edges / rotated 8-bit | 1.000000 | 0.997578 | 123,161,124 |

This separates two effects: quantized entry selection can lead into a graph
region that cannot be escaped, and rotation improves the accuracy of the compact
codes. The earlier 4/8-bit difference therefore does not establish a recall
ceiling intrinsic to 4-bit storage or out-of-distribution queries.

On the graph with escape edges, the fastest measured configurations reaching
0.99 VIBE recall are:

| Encoding | ef / rerank | VIBE recall | Median QPS |
|---|---|---:|---:|
| Plain 4-bit | 512 / 400 | 0.996406 | 1,488 |
| Rotated 4-bit | 256 / 400 | 0.993594 | 2,612 |
| Plain 8-bit | 256 / 1000 | 0.993984 | 2,374 |
| Rotated 8-bit | 256 / 100 | 0.992578 | 2,697 |

Rotated 4-bit is 1.76x plain 4-bit by these medians; the three paired-round ratios
are 1.64x, 1.77x and 1.72x. It uses 20.8% less index storage than rotated 8-bit.
At 0.95 recall, 8-bit is faster: plain 8-bit reaches 4,002 QPS, versus 2,612 for
rotated 4-bit. At 1.00 recall, rotated 4-bit reaches 549 QPS at ef=2000/rerank=400
with strict ID recall 0.996797. The faster curve varies with the target; rounding
strict recall to 1.00 would be incorrect.

The original-graph controls also show why rotation should remain a measured
choice. On arxiv-768 it pads to 1024 dimensions: 4-bit index storage grows from
135,983,700 to 174,397,012 bytes and QPS at 1.00 VIBE recall falls from 2,092 to
1,629. On the additional iNaturalist-2048 stress test, it adds only 24,576 bytes
and QPS at 1.00 VIBE recall rises from 389 to 1,147. These are the fastest
eligible points on the fixed grid, not extrapolated speedups or new public
leaderboard scores.

All four reports retain raw original IDs and per-query nanoseconds. The checker
recomputes every reported recall, timing and target selection and checks four
queries against all original corpus vectors. Missing neighbors receive zero
credit. Timings alternate the four encodings within each of three rounds on
one pinned CPU, with no concurrent benchmark builds. Runs on the two graph
versions are separate; their timing differences are not a paired graph-speed
comparison. The [source and binary receipt](../../benchmarks/results/vibe_v1/rotation/receipt.json)
records exact versions. The controls remain experimental: the VIBE adapter and
production construction defaults have not adopted either change, and full-scale
default-circle coverage remains unmeasured.
All tested graphs use pruning alpha=1.0; this does not establish the optimum
construction setting or a general connectivity repair for production.

## Measurement protocol

The first diagnostic uses pinned VIBE files at dataset revision
`07b387891a221b7b073b83d2f752b76462e5fa03`: 100,000 corpus rows and 256 original
test queries for arxiv-nomic-768, laion-clip-512 and inaturalist-resnet-2048.
The corpus is a prefix. Ground truth is recomputed over that prefix; the
full-corpus neighbor IDs are never reused. LAION queries retain the original
text-to-image distribution. This is a bounded diagnostic, not an official-scale
result or a held-out final evaluation.

Each method uses its VIBE adapter and one explicitly recorded build, with the
query grid fixed before evaluation. This is not the complete build-parameter
search used for the published leaderboard. Adapter builds and batch-one search
run on one pinned Linux CPU. Native arxiv/iNaturalist graph builds use four
threads, recorded in their reports; native query loops use one thread. Three
query timing repeats, raw adapter IDs and per-query timings,
source hashes, package versions, load, build time and process memory are saved.
Targets 0.73, 0.77, 0.90, 0.94, 0.95, 0.99 and 1.00 use the fastest measured
configuration reaching the target; a missing target stays missing. Timing
comparisons remain exploratory: methods are not interleaved and this machine
differs from VIBE's hardware. Do not compare these QPS directly with public QPS.

The accepted HMS implementation is commit
`28a0a5b1bc6e84037fed51fdba8c6b508f821502`: 4-bit mean-centered traversal codes,
8-bit residual reranking, aligned rows, BFS order, 3-byte IDs and quantized graph
construction. The default wrapper does **not** rotate traversal vectors; the
earlier wrapper description was incorrect. A rotation or 8-bit ablation is a
separate experiment and must be identified as such.

The benchmark host is an Intel Core Ultra 9 185H under WSL2 with AVX2 and no AVX-512.
PAG's pinned release `ee34ed78b7a28225b2db6210ad942152e1cd540d`
[requires its AVX-512 backend](https://github.com/KejingLu-810/PAG/blob/ee34ed78b7a28225b2db6210ad942152e1cd540d/cmake/PAGCompilerOptions.cmake).
An AVX-512 host is required for an honest direct PAG comparison. The HMS Python
adapter serves all four VIBE float distances: cosine and normalized by angle,
euclidean directly, and ip through the exact inner-product-to-Euclidean reduction
(one appended coordinate per row).

PAG requires AVX-512, which this host lacks, so it stays unmeasured here.

The completed local experiment uses encoding-only rotation on each cached graph,
using the existing three-round signed Walsh-Hadamard transform with fixed seed
24301. The `experimental` diagnostic compares plain/rotated 4-bit and 8-bit
encodings with an 8-bit residual, identical graph topology, ef
128/256/512/1000/2000 and rerank 100/400/1000. Query transformation is timed.
The arms alternate in each timing round, and original-query distances plus
strict IDs are both scored. Rotation pads to a power of two; the resulting
code bytes and transform state must be counted, particularly at 768 dimensions.
Production defaults remain unchanged. Its reproducible commands, after building
the pinned graph cache and preparing the native data, are:

```sh
cargo build --release --locked --features experimental --bin vibe-rotation
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 RAYON_NUM_THREADS=1 \
  taskset -c 0 target/release/vibe-rotation --data /path/to/native-laion \
  --graph-cache /path/to/laion.graph --out /path/to/rotation-laion.json
uv run --no-project --with numpy==2.2.6 --with scipy==1.17.1 --with h5py==3.15.1 \
  python benchmarks/vibe/connectivity.py /path/to/laion.graph \
  --data /path/to/native-laion --escape-cache /path/to/laion-escape.graph \
  --out /path/to/laion-escape-control.json
# Repeat vibe-rotation with the escape cache and a new output path, then check:
uv run --no-project --with numpy==2.2.6 --with h5py==3.15.1 \
  python benchmarks/vibe/check_rotation.py /path/to/rotation-laion.json \
  /path/to/rotation-laion-escape.json --data /path/to/hdf5-data \
  --out /path/to/rotation-validation.json
```

Use `to_native.py` to create a fresh native directory containing `distances.f32`
as well as vectors and IDs. The original diagnostic native directories predate
that additional file and remain unchanged. The follow-up adds 14 Python
integrity/connectivity regressions; the qgraph test run passes 36 tests and
retains its three explicitly ignored experiments.

## Contribution

The [HMS adapter](../../benchmarks/vibe/hms/)
follows VIBE's [new-method guide](https://github.com/vector-index-bench/vibe/blob/main/docs/NEW_METHOD.md):
one-thread builds, unique original row IDs, separate build/query parameters,
four build combinations and 33 query combinations (56 for ip), with a pinned implementation
revision in `image.def`. It is the quantized graph algorithm, not the memory
database service. It advertises cosine, normalized, euclidean and ip datasets;
the ip grid searches deeper (ef up to 8000) because the reduction places every
row on one sphere.

Reproduce the published ranking and focused regression checks:

```sh
uv run benchmarks/vibe/leaderboard.py benchmarks/results/vibe_v1/leaderboard_inputs \
  --out /path/to/leaderboard.json
uv run --no-project --with numpy==2.2.6 --with h5py==3.15.1 \
  --with faiss-cpu==1.14.3 --with fsspec --with aiohttp --with duckdb==2.0.0 \
  --with pytest python -m pytest -q benchmarks/vibe/test_measurement.py
```

For the diagnostic, install the pinned HMS wheel and comparator dependencies in
an isolated environment, copy `benchmarks/vibe/hms` to `vibe/algorithms/hms` in
VIBE revision `7ac76c1f8320a411ce7c9cb3791bdd7f2349b30c`, then run:

```sh
uv run --no-project python benchmarks/vibe/prepare.py laion-clip-512-normalized \
  --out /path/to/data
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 \
  uv run --no-project python benchmarks/vibe/run.py --vibe /path/to/vibe \
  --data /path/to/data/laion-clip-512-normalized-n100000.hdf5 \
  --method hms --out /path/to/laion-hms.json
```

An upstream contribution also needs the HMS row in VIBE's algorithm table and
a Linux image build through `./install.sh --algorithm hms`. The
[prepared upstream patch](../../benchmarks/vibe/upstream.patch) includes the
README row and all three adapter files; it passes the reverse-apply check on the
pinned checkout (VIBE 7ac76c1). The image has not been built: this host has no Apptainer, so the
same recipe was replayed in a micromamba copy of `base.sif` and `run.py --local` passed on a
smoke subset of llama-128-ip. The
[real adapter checks](../../benchmarks/results/vibe_v1/adapter_validation.json)
cover both advertised metrics, dimensions 768/2048, original IDs, input
immutability, result-count boundaries and VIBE's configuration loader. Local
native-wheel validation does not establish that the Apptainer image has built
successfully.

The native driver now supports `ann-qgraph --count 100` as well as its existing
default of 10 and emits the corresponding recall field. Seven executable
regressions cover corpus/truth boundaries, preservation of recall@10, empty
queries and Linux load reporting. Nine diagnostic regression tests pass;
`public-bench` also passes the scoped all-features check and Clippy with warnings
denied. Build/check logs and the source/host
[receipt](../../benchmarks/results/vibe_v1/receipt.json) are retained.
