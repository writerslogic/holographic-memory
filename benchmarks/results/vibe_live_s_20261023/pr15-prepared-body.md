Add HMS as an in-process Rust vector index for VIBE's float datasets. The adapter supports cosine, normalized, Euclidean and direct inner-product search, with one build/query thread, 8-bit traversal codes, exact reranking and unique original row IDs. OOD learning samples up to 32,768 official learn rows with fixed seed 24301, without learn neighbors or test queries for training. The four-build / 45-query grid is fixed by metric section.

`image.def` pins current-main HMS `12c80decb094d57ee13ba187a01bf1de8788ca2f`, using portable x86-64-v3 with runtime kernel dispatch. Module/grid match fork `60625ecd00c639ebe1b6ce19434e89dae7fedff0`; adapter head remains `c4df927808bd9d993d2f413456486dadecffb6fd`. [Source CI passed](https://github.com/writerslogic/holographic-memory/actions/runs/40070371238). Exact-source wheels ran in Modal; an Apptainer image from image.def remains unbuilt.

Twenty of 24 configured HMS grids completed 180/180 groups each before a Modal workspace interruption. Three full dataset pipelines completed, two retain terminal MLANN-RF failures alongside valid HMS/RoarGraph arms, and 19 pipelines are incomplete. Remaining native measurements are not reported as complete. All completed arms use k=100, 1,000 official test queries and five passes with best QPS. The 24-dataset scope includes 19 current and 5 deprecated datasets.

The completed-arm figures below use the unchanged official VIBE exporter at upstream `7ac76c1f8320a411ce7c9cb3791bdd7f2349b30c`. Local post-interruption export used copies of preserved raw files; 5,488 complete groups were checked against recorded expected counts, arrays and original completion markers. All five previously exported official point sets reproduce within 1e-9. Only the official truth-distance byte ranges were retrieved for remaining metric exports; no fits or queries were rerun. [Per-dataset standings, parquet, native SHA256 hashes, validation and truth receipts](https://github.com/writerslogic/holographic-memory/tree/bench/vibe-live-s/benchmarks/results/vibe_live_s_20261023/interrupted-complete-arm-exports) retain all parameters and metrics.

Each pair below has complete configured grids in one proved dataset container; different pairs use different containers. Values are QPS (actual recall), selected independently at each fixed recall floor. These are threshold comparisons, not equal-recall dominance claims.

| Dataset | System | Recall ≥0.90 | ≥0.95 | ≥0.99 |
|---|---|---:|---:|---:|
| celeba-resnet-2048-cosine (deprecated) | HMS | 1,685.24(0.98806) | 1,685.24(0.98806) | 1,515.15(0.99280) |
| celeba-resnet-2048-cosine (deprecated) | ngt-qg | 3,166.77(0.91825) | 2,989.86(0.97013) | 2,194.09(0.99432) |
| coco-nomic-768-normalized (deprecated) | HMS | 3,247.02(0.99779) | 3,247.02(0.99779) | 3,247.02(0.99779) |
| coco-nomic-768-normalized (deprecated) | glass | 3,593.69(0.97696) | 3,593.69(0.97696) | 2,936.60(0.99202) |
| coco-nomic-768-normalized (deprecated) | HMS | 3,247.02(0.99779) | 3,247.02(0.99779) | 3,247.02(0.99779) |
| coco-nomic-768-normalized (deprecated) | lorann | 7,339.08(0.91631) | 6,195.12(0.95027) | 3,312.75(0.99386) |
| imagenet-clip-512-normalized | HMS | 3,573.01(0.99161) | 3,573.01(0.99161) | 3,573.01(0.99161) |
| imagenet-clip-512-normalized | symphonyqg | 1,879.28(0.90116) | 781.37(0.95477) | no qualifying point |
| llama-128-ip | HMS | 6,293.07(0.90840) | 5,162.22(0.95740) | 2,585.31(0.99034) |
| llama-128-ip | roargraph | 8,555.53(0.90054) | 5,913.69(0.95223) | 1,305.51(0.99113) |
| yandex-200-cosine | HMS | 6,224.22(0.92197) | 4,189.08(0.96497) | 2,142.98(0.99261) |
| yandex-200-cosine | glass | 4,309.38(0.91515) | 3,022.03(0.96815) | 1,691.84(0.99056) |
| yi-128-ip | HMS | 6,197.53(0.92325) | 4,264.21(0.95335) | 1,517.23(0.99146) |
| yi-128-ip | roargraph | 5,319.62(0.90083) | 3,618.98(0.95125) | no qualifying point |

Losses are retained: HMS is slower at Llama floors 0.90/0.95, all Celeba floors versus NGT-QG, and all Coco floors versus Lorann. Against Coco Glass, HMS is slower at 0.90/0.95 and faster at 0.99. It exceeds the listed comparators at the Yi, Yandex and ImageNet-CLIP floors with qualifying comparator points, and at Llama 0.99. Missing comparator points do not establish dominance over all methods. ImageNet-CLIP Glass was interrupted, so its HMS/SymphonyQG comparison is limited to that complete pair.

Build/memory tradeoffs remain part of the result artifacts. For example, ImageNet-CLIP HMS's selected index took 2,625.118 s and 1,450,152 KiB USS delta; SymphonyQG selected builds took 4,483.492 s / 2,616.820 s with 11,266,200 / 12,493,044 KiB at floors 0.90/0.95. HMS has no blanket build-time win there. Yandex HMS took 1,567.430 s and 801,016 KiB versus Glass 571.471 / 725.882 / 725.882 s and 1,231,172 / 1,290,612 / 1,290,612 KiB.

Modal hardware differs from VIBE's published Xeon Gold 6230; published QPS is diagnostic only. Several Modal memory requests are below the 16 GiB-per-worker recommendation; timestamped cgroup samples establish neither peak RSS nor published-host compliance. VIBE index size is USS delta, not peak RSS. Arxiv recovery and migrated CCNews/Landmark/Yahoo groups are diagnostic against their mismatched competitors; no favorable subgroup was selected and no completed HMS grid was rerun for a claim.

MLANN-RF produced zero validated groups on Yi/Llama after three bounded attempts, with first-query SIGSEGV and an invalid-depth configuration retained. The workspace interruption's causal reason remains unknown. All 7,461 unique result/checkpoint/log/metadata files(5,590,265,460bytes,253 raw HDF5) are preserved locally with hashes and remain in the private Modal Volume; official 215 GB input data remain remote with inventory and verified source digests.

[Complete report data, interruption and rescue receipts](https://github.com/writerslogic/holographic-memory/tree/bench/vibe-live-s/benchmarks/results/vibe_live_s_20261023) distinguish full, partial and failed runs. This PR remains draft; no public standings submission is requested.
