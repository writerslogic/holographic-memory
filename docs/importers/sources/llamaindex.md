# llamaindex

Reads a `StorageContext.persist` directory: `docstore.json` and a SimpleVectorStore JSON (`--vector-store`, default `default__vector_store.json`; named stores use `<name>__vector_store.json`). Plain JSON, no pickle, no LlamaIndex import. Verified with indexes built and persisted by real `llama_index.core`.

- Each node becomes a record: id = node id, text, vector from `embedding_dict` (or the node's own embedding), node metadata, plus `ref_doc_id`, relationships (`llamaindex_source`, `_previous`, `_next`, `_parent`, `_child`), `llamaindex_node_type`, `start_char_idx`, `end_char_idx`, `mimetype`, and any excluded-metadata key lists. `creation_date` / `last_modified_date` metadata set the timestamps.
- Source `Document`s are records only when they have no chunks (`--documents auto|always|never`).
- Ids present only in the vector store (external-style layouts) are imported with their metadata, and text from `_node_content` when present.
- Limit: the format is whole-file JSON, and LlamaIndex loads it whole too, so memory is proportional to the persisted index.
