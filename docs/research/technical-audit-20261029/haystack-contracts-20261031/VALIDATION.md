# Haystack contract corrections

Base: `08776a83780e670056ec3881acbf98e1a95dd560` (PR #362). Nine tests
reproduce silent incorrect results or ignored policies with Haystack 2.0.0 and a
real local WARM server: zero after writes; empty filter results despite existing
documents; ignored FAIL/SKIP and invalid policies. The retained baseline patch
adds those original assertions to the base. No paid API, model, telemetry or
external inference is used.

The adapter obtains authenticated tenant counts from the stats endpoint. Sync
and async SDK clients expose stats with existing error handling. Unsupported
filtering and FAIL/SKIP now raise before writes rather than fabricate results or
overwrite data contrary to the caller's policy. NONE retains the existing default
overwrite behavior, explicitly documented; unknown policy types raise TypeError.
An explicit close method releases the owned client.

[Haystack's 2.27.0 DocumentStore protocol](https://github.com/deepset-ai/haystack/blob/v2.27.0/haystack/document_stores/types/protocol.py)
allows store-defined NONE behavior, defines FAIL/SKIP/OVERWRITE, and requires
actual count/filter results. Correct FAIL/SKIP still need atomic server operations;
client preflight would race. Full filtering/reconstruction requires a bounded
full-document endpoint. Serialization and complete DocumentStore interoperability
remain unimplemented; no full Haystack conformance or production-readiness claim.

The 12 final tests pass with Haystack 2.0.0 / Python 3.10 and 2.27.0 / Python
3.14, including actual DocumentWriter use, count changes on overwrite/deletion,
tenant isolation, and sync/async authentication errors. No dependency version or
supported-extra changes. Whole SDK run: 71 tests and 46 subtests pass; three skips are the separately
validated LangChain/LlamaIndex environments and optional orjson. Wheel/sdist,
actionlint and offline zizmor pass (15 existing suppressions, no new ones).
Exact-head CI remains required.
