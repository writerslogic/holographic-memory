# qdrant

`warm-import qdrant (--path DIR | --url URL) [--collection NAME ...] [--vector-name N] [--text-field F] [--namespace-field F]`

Scrolls every collection (or the named ones) with payloads and vectors through the real `qdrant-client`, in point-id order. Resume is a keyset: the cursor is `[collection, last point id]`.

| Qdrant | WARM |
|---|---|
| point id | `warm_source_id` (ints become strings) |
| payload text (`--text-field`, else first of `document`, `text`, `content`, `page_content`, `data`) | text, removed from metadata |
| rest of payload | metadata; a nested `metadata` object is flattened for LangChain-style `page_content` payloads (or `--metadata-field`) |
| collection | namespace, or the payload key named by `--namespace-field` (multi-tenant payload pattern) |
| vector | dense vector; with several named vectors `--vector-name` is required (the importer refuses to guess) |
| timestamps | `--created-at-field` / `--updated-at-field` payload keys (copied, also kept in metadata) |

Credentials: `$QDRANT_API_KEY` (header `api-key`), never logged.

Limits: sparse and multi-vectors are not carried; unchosen named vectors are dropped.

Verification: local mode against a real embedded Qdrant (round trip, named vectors, cursor, resume). Server mode runs the real `qdrant-client` against an HTTP mock of the documented REST API (`GET /collections`, `POST /collections/{c}/points/scroll`, `.../count`, `api-key` header). No live Qdrant server was available.
