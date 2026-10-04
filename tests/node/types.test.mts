import { HolographicMemorySystem, type DocumentInput } from '../../index.js'
import { DocumentMemory, createLocalEmbedder, type Embedder } from '../../semantic.js'

const input: DocumentInput = {
  id: 'guide', text: 'Restore verified backups.', metadata: { project: 'alpha' },
}
const hms = new HolographicMemorySystem(16384, './types-only')
const memory = new DocumentMemory(hms, { maxPending: 16 })
await memory.memorize(input)
const results = await memory.search('backups', { filter: { project: 'alpha' }, k: 3 })
const documentId: string | undefined = results[0]?.documentId
void documentId
const embedder: Embedder = await createLocalEmbedder({ modelPath: './model', modelId: 'example', revision: '1' })
const values: ArrayLike<number> = await embedder.encodeQuery('query')
void values

// These operations must remain invalid at the public TypeScript boundary.
// @ts-expect-error document text must be a string
await memory.memorize({ id: 'bad', text: 42 })
// @ts-expect-error the candidate limit must be numeric
await memory.search('query', { candidateLimit: 'many' })
