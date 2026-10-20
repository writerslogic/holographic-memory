# langchain

`--format auto` detects the layout. Verified with stores built by real `langchain_community` FAISS and `langchain_core`.

- `faiss`: a `FAISS.save_local` folder (`index.faiss` + `index.pkl`, `--index-name`). Vectors are reconstructed from FAISS (IVF needs a direct map; quantized indexes return their lossy reconstruction). `index.pkl` is a pickle: it is read with an allow-list unpickler that maps only LangChain's Document and InMemoryDocstore classes to inert stand-ins, so a hostile pickle cannot run code (tested) and LangChain need not be installed. Document ids come from the docstore mapping.
- `inmemory`: `InMemoryVectorStore.dump` JSON (current `{id: item}` and older `{"store": ...}` layouts).
- `jsonl`: one Document per line, either `dumpd` form or `{"page_content", "metadata", "id"}`.
- `filestore`: a `LocalFileStore` / `create_kv_docstore` directory (file name = id, content = serialized Document; no vectors).

Document metadata is preserved flat. LangChain stores no timestamps, so none are invented. Text-only layouts need an encoder.
