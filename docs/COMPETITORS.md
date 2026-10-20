# Competitor pins (2026-10-09)

Published numbers are vendor-native results. They do not populate the matched
`Them` column until HMS and the competitor share the evaluation conditions.
HMS's repeatedly inspected 100-question S dev result and a vendor's full
500-question result are incomparable; their matched margin stays empty.

| Competitor | Frozen implementation/source | Vendor-native result | Matched result |
|---|---|---|---|
| Zep managed memory | [Research page](https://www.getzep.com/research/), retrieved 2026-10-09; service revision undisclosed | LongMemEval 451/500; reader `gpt-5.4`, medium reasoning; judge `gpt-5.4`, chain-of-thought grading; median context 4,408 tokens; multi-scope retrieval | |
| Mem0 managed v3, top-50 | [Benchmark README](https://github.com/mem0ai/memory-benchmarks/blob/4b61c5d31b9c668a12b4f5e78064248a02c82d2b/README.md); managed service revision undisclosed | S cleaned 474/500 at top-50; 472/500 at top-200; proprietary platform, distinct from OSS | |
| FAISS HNSW, IVF-PQ | Cached `faiss-cpu==1.15.1`; [v1.15.1](https://github.com/facebookresearch/faiss/tree/7ea7339886edf8b4d9e719f593a5dfccff158274), commit `7ea7339886edf8b4d9e719f593a5dfccff158274` | | |
| hnswlib | Cached `hnswlib==0.8.0`; [v0.8.0](https://github.com/nmslib/hnswlib/tree/3f3429661187e4c24a490a0f148fc6bc89042b3d), commit `3f3429661187e4c24a490a0f148fc6bc89042b3d` | | |
| SymphonyQG | [Author repository](https://github.com/gouyt13/SymphonyQG/tree/6124ddb34ee4d176edea1bd7ad38d1672343df28), commit `6124ddb34ee4d176edea1bd7ad38d1672343df28`; cached author library `rabitqlib==0.5.2` | | |
| RaBitQ-Library | [Author repository](https://github.com/vectordb-ntu/RaBitQ-Library/tree/9454d54939858299fdf7ac05e7dd115a7199d561), commit `9454d54939858299fdf7ac05e7dd115a7199d561`; cached `rabitqlib==0.5.2` | | |

Zep discloses an alias, not an exact snapshot or judge prompt. OpenAI documents
[`gpt-5.4-2026-03-05`](https://developers.openai.com/api/docs/models/gpt-5.4)
as a fixed snapshot. It supports `POST /v1/responses`; standard rates per million
tokens are $2.50 input, $0.25 cached input, and $15 output. A snapshot match to
Zep remains unverified. [Official tiktoken mapping](https://github.com/openai/tiktoken/blob/4e71bbe0c078468e00fefbf94b39849389f346e5/tiktoken/model.py)
maps the GPT-5 prefix to `o200k_base`; the
[input-token API](https://developers.openai.com/api/docs/guides/token-counting)
counts complete request inputs, including message framing.

The official LongMemEval answer judge is
[`src/evaluation/evaluate_qa.py`](https://github.com/xiaowu0162/LongMemEval/blob/9e0b455f4ef0e2ab8f2e582289761153549043fc/src/evaluation/evaluate_qa.py)
at commit `9e0b455f4ef0e2ab8f2e582289761153549043fc`, SHA256
`ecce9c4c79dc89d99534ac17b383a5cbb5b9f0c69ee98adaf0684742e3d95251`.
Its per-type prompts differ from retrieval `eval_utils.py`.
[Mem0's pinned runner](https://github.com/mem0ai/memory-benchmarks/blob/4b61c5d31b9c668a12b4f5e78064248a02c82d2b/benchmarks/longmemeval/run.py)
defaults reader/judge to `gpt-5`; its
[custom unified judge](https://github.com/mem0ai/memory-benchmarks/blob/4b61c5d31b9c668a12b4f5e78064248a02c82d2b/benchmarks/longmemeval/prompts.py)
differs from the official prompt. Defaults do not prove the published run's
model identifiers. Source hashes belong in the frozen reader protocol.

x86: no x86 run. CPU, microcode, kernel, compiler and target flags remain empty.
M4 measurements cannot populate the AVX-512 rows.
Provisioning probe 2026-10-09 20:49 UTC: no AWS CLI, shared credentials, SSO
cache, environment authentication or connected compute provider; no authenticated
AWS, Equinix Metal, Vultr or Hetzner Robot configuration in the private environment.
Blocking resource: dedicated bare-metal provider credentials. Provisioning spend $0.

Local native measurements use Apple M4, 10 CPUs, 32 GiB RAM, Darwin 27.0.0,
Rust 1.96.0 and `-C target-cpu=native`. The pinned
[ann-benchmarks recall calculation](https://github.com/erikbern/ann-benchmarks/blob/2e081ad32c1eccab72dcb739ad886c310b90f715/ann_benchmarks/plotting/metrics.py)
counts returned-vector distances within the tenth ground-truth distance plus 0.001;
[angular distance](https://github.com/erikbern/ann-benchmarks/blob/2e081ad32c1eccab72dcb739ad886c310b90f715/ann_benchmarks/distance.py)
is computed from the original vectors. ID overlap is reported separately.
