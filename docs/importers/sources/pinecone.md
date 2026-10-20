# Source: pinecone

Verification: MOCK ONLY, but driven through the real `pinecone` client library (`Index(host=...)`) against a mock of the data-plane API. Pinecone Local speaks the same protocol (`--host http://localhost:5081`). The destination is a real `hms-server`.

```sh
export PINECONE_API_KEY=...
hms-import pinecone --host https://my-index-xxxx.svc.region.pinecone.io --hms-url http://127.0.0.1:8080
```

Calls (https://docs.pinecone.io/reference/api/2025-10/data-plane/): `POST /describe_index_stats` (namespaces and counts), `GET /vectors/list?namespace&limit&paginationToken&prefix` (ids in sorted order, `pagination.next`), `GET /vectors/fetch?ids=..&ids=..&namespace` (values, metadata, `sparseValues`). Header `Api-Key`. Serverless and Pinecone Local only: pod-based indexes have no list endpoint (export through your own id list, or re-ingest from source data). `--index-name` resolves the host through the control plane instead of `--host`.

Mapping: every namespace (or the ones given with `--namespace`) becomes `hms_namespace`; the default namespace (`""` or `__default__`) keeps bare ids, others get `namespace::id`, so the same id in two namespaces stays distinct. The text is a metadata key (`--text-field`, else `text`, `content`, `chunk_text`, ...) and is moved out of the metadata; `--timestamp-field`/`--updated-field` name metadata keys holding times. Sparse values are not dropped: they are kept as `pinecone_sparse: {indices, values}` in metadata (HMS has no sparse vector slot). An id that is listed but gone by fetch time is rejected with a reason.

Resume is seekable: the cursor is `[namespace, page token, offset in page]`.
