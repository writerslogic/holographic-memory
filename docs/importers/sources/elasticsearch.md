# Source: elasticsearch

Verification: MOCK ONLY. The tests (`tests/importers/test_cloud_b2.py`) run the importer against an in-process mock of the documented REST API; no live Elasticsearch was available. The destination in those tests is a real `warm-server`.

```sh
export ELASTIC_API_KEY=...            # or ELASTIC_USER + ELASTIC_PASSWORD, or ELASTIC_BEARER_TOKEN
export ELASTICSEARCH_URL=https://es.example:9200
warm-import elasticsearch --index 'docs' --warm-url http://127.0.0.1:8080
```

Endpoints used (REST, via httpx; no client library needed):

| Call | Purpose | Docs |
|---|---|---|
| `GET /{index}/_mapping` | detect the single `dense_vector` field (override with `--vector-field`; several candidates is an error) | https://www.elastic.co/docs/api/doc/elasticsearch/operation/operation-indices-get-mapping |
| `POST /{index}/_count` | progress total | https://www.elastic.co/docs/api/doc/elasticsearch/operation/operation-count |
| `POST /{index}/_search?scroll=5m` with `{"size", "query", "sort": ["_doc"], "fields": [vector]}` | first page | https://www.elastic.co/docs/api/doc/elasticsearch/operation/operation-search |
| `POST /_search/scroll` `{"scroll","scroll_id"}` | next pages until an empty page | https://www.elastic.co/docs/api/doc/elasticsearch/operation/operation-scroll |
| `DELETE /_search/scroll` | frees the context | same |

Mapping: `_id` is the id; the text field (`--text-field`, else `text`, `content`, `body`, ...) becomes the text; the vector comes from `_source` or, when the cluster excludes vectors from `_source`, from the `fields` response; `--timestamp-field` / `--updated-field` become `warm_created_at` / `warm_updated_at`; every other `_source` field is metadata. Index name is the namespace when `--index` is a pattern or a list (`--namespace-from index|none|auto`), so equal `_id`s in different indices stay distinct.

Limits: scroll contexts cannot be resumed, so a resume re-reads the committed prefix and verifies it against the checkpoint hash (see the main importer docs). `--query` takes a JSON query DSL to import a subset. Credentials only from the environment or arguments; the API key is never written to the report, checkpoint or logs. No AWS SigV4.
