# Source: opensearch

Verification: MOCK ONLY (same mock harness as Elasticsearch, with basic-auth enforcement and the vector in `_source`; no live OpenSearch was available). The destination is a real `warm-server`.

```sh
export OPENSEARCH_PASSWORD=...
warm-import opensearch --url https://os.example:9200 --user admin --index 'docs-*' --warm-url http://127.0.0.1:8080
```

Same flow and options as [elasticsearch](elasticsearch.md): scroll API (`POST /{index}/_search?scroll=`, `POST /_search/scroll`, `DELETE /_search/scroll`), `GET /{index}/_mapping` to find the `knn_vector` field, `POST /{index}/_count`. Docs: https://docs.opensearch.org/latest/api-reference/search-apis/scroll/ and https://docs.opensearch.org/latest/field-types/supported-field-types/knn-vector/.

Credentials: `--user` or `$OPENSEARCH_USER`, `$OPENSEARCH_PASSWORD`, optionally `$OPENSEARCH_API_KEY`. AWS SigV4 (managed OpenSearch Service with IAM) is not supported; use a basic-auth user or put a signing proxy in front. `--insecure` skips TLS verification, `--ca-cert` pins a CA bundle.
