# Source: milvus

`hms-import milvus --uri URI_OR_FILE --collection NAME [--vector-field F] [--text-field F] [--namespace-field F] [--partition P ...] [--created-at-field F] [--updated-at-field F] [--page-size N]`

Reads a collection through `pymilvus.MilvusClient`: a Milvus Lite database file, a Milvus server (`http://host:19530`) or Zilliz Cloud. Verified against real Milvus Lite (pymilvus 3.0, milvus-lite) including partitions, a VARCHAR primary key and dynamic fields. A Milvus server and Zilliz Cloud use the same client calls but were not run here (no server available); treat them as unverified.

## Mapping

- Id: the primary key, as a string.
- Vector: the single `FLOAT_VECTOR` field (or `--vector-field`). Binary, fp16, bf16 and sparse vectors are not supported; the run fails with the field list.
- Text: `--text-field`, else the first field named text, content, document, chunk, body or page_content.
- Metadata: every other field, including dynamic fields (returned flat by `output_fields=["*"]`). Nested JSON is preserved.
- Namespace: the partition name (`_default` becomes the empty namespace), or the value of `--namespace-field` when given. `--partition` restricts which partitions are read.
- Timestamps: `--created-at-field` / `--updated-at-field` (copied; the field also stays in metadata).

## Reading and resume

`query_iterator`, one partition at a time in primary-key order, `--page-size` rows per round trip (default 1000, max 16384). Seekable: the cursor is `[partition index, last primary key]` and resume uses a `pk > last` filter inside the iterator. Rows written during the import may or may not be seen, as with any live collection.

## Credentials

`$MILVUS_TOKEN` (or `--token`), `$MILVUS_USER` / `$MILVUS_PASSWORD`. A password inside the URI is also masked. None appear in reports, checkpoints or errors.
