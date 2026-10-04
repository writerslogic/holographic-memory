import { HolographicMemorySystem } from '../index.mjs'
import { DocumentMemory, createLocalEmbedder, createLocalReranker } from '../semantic.js'

const embedder = process.env.HMS_EMBEDDING_DIR ? await createLocalEmbedder({
  modelPath: process.env.HMS_EMBEDDING_DIR,
  modelId: process.env.HMS_EMBEDDING_MODEL ?? 'Xenova/all-MiniLM-L6-v2',
  revision: process.env.HMS_EMBEDDING_REVISION,
  dtype: 'q8',
}) : undefined
let rerank
try {
  if (process.env.HMS_RERANKER_DIR) rerank = await createLocalReranker({ modelPath: process.env.HMS_RERANKER_DIR, dtype: 'q8' })
  const config = embedder ? {
    embeddingModel: embedder.space.model,
    embeddingRevision: embedder.space.revision,
    embeddingDimensions: embedder.space.dimensions,
  } : undefined
  const hms = new HolographicMemorySystem(16384, process.argv[2] ?? './example-documents-v2', config)
  const memory = new DocumentMemory(hms, { embedder, rerank })
  await memory.ingest([
    { id: 'backups', text: 'Restore deleted documents from verified backups.', sourceUri: 'operations.md', metadata: { project: 'alpha' } },
    { id: 'indexing', text: 'Train search indices after large ingestion batches.', sourceUri: 'search.md', metadata: { project: 'alpha' } },
  ])
  const results = await memory.search('restore backups', { k: 3, filter: { project: 'alpha' } })
  console.log(JSON.stringify(results, null, 2))
  if (results[0]?.documentId !== 'backups') throw new Error('Expected the backup source to rank first')
  await memory.flush()
} finally {
  await rerank?.dispose?.()
  await embedder?.dispose?.()
}
