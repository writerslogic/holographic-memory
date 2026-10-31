# Source: vespa

Verification: MOCK ONLY. The mock implements the visit API of https://docs.vespa.ai/en/reference/document-v1-api-reference.html (continuation tokens, `wantedDocumentCount`, `format.tensors=short`, bearer auth). The destination is a real `warm-server`.

```sh
export VESPA_TOKEN=...
warm-import vespa --url https://app.example.vespa-app.cloud --vespa-namespace shop --doctype item --vector-field embedding --warm-url http://127.0.0.1:8080
```

Endpoint: `GET /document/v1/{namespace}/{doctype}/docid?wantedDocumentCount=N&continuation=...&format.tensors=short[&cluster=..][&selection=..]`. The response is `{"pathId", "documents": [{"id", "fields"}], "continuation"?, "documentCount"}`; the walk ends when `continuation` is absent.

Mapping: the user-specified part of `id:ns:type:key:user` is the id; the text field and timestamp fields are `--text-field`, `--timestamp-field`, `--updated-field`; all other fields are metadata, plus `vespa_namespace`, `vespa_doctype` and the `g=`/`n=` locality (`vespa_group`/`vespa_number`). `--namespace-from none|namespace|doctype|group` picks the WARM namespace.

Tensors: one-dimensional dense tensors in the `values` list, `cells` list (indexed addresses 0..n-1), a bare list, or the int8 hex string form. Mapped (sparse), multi-dimensional and mixed (`blocks`) tensors are not vectors: such records are rejected with the reason, never imported without their vector. Several tensor fields require `--vector-field`.

Auth: bearer token (`$VESPA_TOKEN`) or mTLS (`--cert`, `--key`, `--ca-cert`). Resume re-reads and hash-verifies the committed prefix (continuation tokens are per page and are not stable across runs).
