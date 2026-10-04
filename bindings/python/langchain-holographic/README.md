# LangChain Holographic Memory

A Zero-Trust Vector Store for LangChain using Holographic Memory System (HMS).

## Usage

```python
from langchain_holographic import HolographicVectorStore
from langchain_openai import OpenAIEmbeddings

# The embeddings are bound with the zero_trust_key before storage
# The database server never sees the plaintext embeddings
vectorstore = HolographicVectorStore(
    embedding_function=OpenAIEmbeddings(),
    persist_directory="./hms-store",
    zero_trust_key="super-secret-master-key"
)

vectorstore.add_texts(["Project Orion launches on Tuesday."])

# The query is encrypted locally; the server searches over noise.
docs = vectorstore.similarity_search("When does Orion launch?")
```
