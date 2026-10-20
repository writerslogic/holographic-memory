# chroma

`hms-import chroma (--path DIR | --host HOST [--port N --ssl]) [--collection NAME ...]`

Reads every collection (or the named ones) through the real `chromadb` client, paged with `get(limit, offset)`: ids, documents, metadata, stored embeddings and URIs.

| Chroma | HMS |
|---|---|
| id | `hms_source_id`; document id is `<collection>::<id>` (namespace prefix, see `--id-prefix-namespace`) |
| collection | namespace (`hms_namespace`) |
| document | text |
| embedding | vector (carried by default) |
| metadata | metadata, verbatim; a URI is kept as `chroma_uri` |
| timestamps | Chroma stores none; `--created-at-key` / `--updated-at-key` copy a metadata key into `hms_created_at` / `hms_updated_at` (the key also stays in metadata) |

Credentials: server token from `$CHROMA_API_KEY` (sent as `X-Chroma-Token`), never logged. `--chroma-tenant` / `--chroma-database` select the Chroma tenant and database.

Resume: not seekable (Chroma offsets follow insertion order but are not a keyset); the pipeline re-reads the committed prefix and checks its hash.

Verification: real local `PersistentClient` instance (round trip, namespaces, timestamps, resume). The HTTP server path uses the same client calls but was not run against a live Chroma server.
