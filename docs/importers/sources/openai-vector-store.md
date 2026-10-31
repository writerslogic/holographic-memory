# openai-vector-store

Reads OpenAI vector stores through the real `openai` SDK (`$OPENAI_API_KEY`, `--base-url`, `--organization`, `--vector-store` repeatable, default every store). Endpoints used: `GET /v1/vector_stores`, `.../{id}/files`, `.../{id}/files/{file_id}/content`, `GET /v1/files/{file_id}`.

- OpenAI does not expose embeddings, so records are the parsed text chunks, one per content part (`<file_id>#<n>`), and WARM encodes them.
- Namespace = vector store id (`--namespace-by name` for its name). Metadata: file `attributes` flat, `file_id`, `filename`, `vector_store_id`, `vector_store_name`, `chunk_index`, store metadata. `created_at` = the file's creation time.
- A file that is not `completed`, or whose content has more pages than the API lets a client follow, is rejected visibly.
- Verification: MOCK. The real SDK runs against a mock of the documented API (https://platform.openai.com/docs/api-reference/vector-stores-files); no live OpenAI call was made.
