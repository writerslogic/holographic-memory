# Source: weaviate

Verification: REAL for the core read path: `test_weaviate_real_embedded_roundtrip` runs against a real embedded Weaviate 1.30.1 (multi-tenant class with two tenants, plain class, cursor paging, resume; opt in with `HMS_TEST_WEAVIATE_EMBEDDED=1`, downloads the binary on first use). Auth, the page cap, COLD tenants and named vectors are covered by a MOCK only. The mock implements the documented REST behavior, including a page cap below `limit`, tenant enforcement and bearer auth. The destination is a real `hms-server`.

```sh
export WEAVIATE_API_KEY=...
hms-import weaviate --url http://localhost:8080 --hms-url http://127.0.0.1:8080
```

Endpoints (REST, httpx), https://docs.weaviate.io/weaviate/api/rest:

- `GET /v1/schema`: classes, their properties and whether multi-tenancy is enabled.
- `GET /v1/schema/{class}/tenants`: tenants of multi-tenant classes. Tenants that are not HOT are reported as rejected records (`Class/tenant`), never skipped silently; activate them and re-run.
- `GET /v1/objects?class=C&limit=N&after=<uuid>&include=vector&tenant=T`: cursor listing in uuid order. The loop reads until an empty page because a server may cap pages below `limit`.

Mapping: object uuid is the id; tenant is the namespace (`hms_namespace`, ids become `tenant::uuid`; the same uuid in two tenants stays distinct); the text property is `--text-property`, else `text`/`content`/`body`/... among the class's `text` properties, else the first text property; the remaining properties plus `weaviate_class` are metadata; `creationTimeUnix` and `lastUpdateTimeUnix` become `hms_created_at`/`hms_updated_at`. Vectors: the object's `vector`, else the only named vector, else `--vector-name`.

Resume is seekable: the cursor is `[class, tenant, last uuid]`.

Not covered: cross-reference beacons are copied verbatim in metadata, not followed; multi-vector (ColBERT) and Weaviate-side modules are not exported.
