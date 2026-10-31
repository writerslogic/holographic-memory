# Python SDK per-call overhead

Development A/B, not a standings run. Source: `benchmarks/results/python_sdk_overhead.json`,
produced by `bindings/python/warm-sdk/benchmarks/client_overhead.py`.

Setup: Apple M-series, 10 threads, CPython 3.14.7, httpx 0.28.1. Local release `warm-server`
(`--features security,server`, default settings), 10,000 rows, dimension 384, `top_k` 10, sequential
calls on one connection. One process per SDK build per round, 3 rounds with the build order
rotated; within a process, raw `http.client`, the sync client and the async client alternate in
20 blocks with a rotating order, so the SDK-minus-raw difference is paired. 1-minute load average
was 2.7 to 8.0 during the runs (other jobs shared the machine). Writes fsync on the server (about
4 ms), so their SDK difference is within noise; the p99 of the `before` writes reflects load spikes
in its first round, not the client.

- `before`: SDK at 82b42f7b (httpx default transport, `json=` bodies; sniffio not installed).
- `after`: this change, stdlib `json`.
- `after-orjson`: this change with the `fast` extra (orjson).

p50 / p99 in microseconds; the last column is the query p50 minus raw's in the same process
(median of the 3 rounds). Raw encodes its body with stdlib `json.dumps`.

| build | client | query | add | replace | delete | query over raw |
|---|---|---|---|---|---|---|
| before | raw | 334 / 1009 | 4002 / 23892 | 4021 / 31092 | 4056 / 21941 | +0 |
| before | sync | 487 / 1439 | 4014 / 34464 | 4034 / 27018 | 4041 / 22657 | +153 |
| before | async | 745 / 2272 | 4896 / 28258 | 4989 / 27015 | 4654 / 20304 | +411 |
| after | raw | 325 / 577 | 3986 / 5444 | 3999 / 5063 | 3988 / 6194 | +0 |
| after | sync | 402 / 754 | 3993 / 5218 | 3997 / 6549 | 3991 / 6848 | +74 |
| after | async | 624 / 1183 | 4106 / 5866 | 4509 / 7273 | 4030 / 10471 | +294 |
| after-orjson | raw | 326 / 482 | 3981 / 4995 | 3984 / 4942 | 3982 / 4820 | +0 |
| after-orjson | sync | 307 / 556 | 3978 / 5030 | 3981 / 5066 | 3982 / 5059 | -19 |
| after-orjson | async | 535 / 976 | 3992 / 5663 | 3993 / 5242 | 3992 / 5873 | +208 |

Where the time went (cProfile and a timed transport comparison, 384-dim query):

- httpx's default transport (httpcore + h11 framing) cost about 70 us more per call than
  `http.client`; the sync client now uses a keep-alive `http.client` transport for plain-HTTP URLs
  with no proxy configured.
- httpcore imports `sniffio` on every async lock setup; without it installed, every async request
  repeated failed import lookups (about 20 path-finder searches). `sniffio` is now a dependency.
- Encoding 384 floats with stdlib `json` takes about 90 us, the same cost raw pays; orjson cuts it to
  under 10 us, which is why the `fast` build is below raw.
- The remaining async overhead is httpx/anyio scheduling (about 9 event-loop iterations per request).

The server has no batch-query or binary-vector route, so neither is in the SDK; both need a
versioned server endpoint.
