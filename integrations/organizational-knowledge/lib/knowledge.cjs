'use strict'

const { DocumentMemory } = require('holographic-memory/semantic')
const { KnowledgeError } = require('./errors.cjs')
const { NAMESPACE_KEY, adaptDocument, documentId, isPlainObject, parseDocumentId, requestFailure } = require('./document.cjs')

const MAX_IMPORT_DOCUMENTS = 10_000
const MAX_FORGET_TARGETS = 10_000
// The engine rejects a document whose chunk embeddings total more than this many values.
const MAX_EMBEDDING_VALUES = 1_048_576

/**
 * Validates every entry against the adapter's rules and rejects duplicate identities. It
 * needs no engine, so a caller can reject bad input before opening (and thereby creating) a
 * store: a `HolographicMemorySystem` without a path persists to the working directory.
 */
function prepareImport(namespace, entries) {
  if (entries === null || typeof entries?.[Symbol.iterator] !== 'function') {
    throw new KnowledgeError('invalid_request', 'entries must be iterable')
  }
  const list = []
  for (const entry of entries) {
    list.push(entry)
    if (list.length > MAX_IMPORT_DOCUMENTS) break
  }
  if (list.length === 0 || list.length > MAX_IMPORT_DOCUMENTS) {
    throw new KnowledgeError('invalid_request', `entries must hold 1..=${MAX_IMPORT_DOCUMENTS} documents`)
  }
  const seen = new Map()
  const prepared = []
  list.forEach((entry, index) => {
    if (!isPlainObject(entry) || !('document' in entry)) {
      throw new KnowledgeError('invalid_request', `entries[${index}] must be { document, source? }`)
    }
    const source = entry.source === undefined ? `entries[${index}]` : String(entry.source)
    const item = adaptDocument(namespace, entry.document, source)
    const first = seen.get(item.documentId)
    if (first !== undefined) {
      throw new KnowledgeError('duplicate_identity', `${source} and ${first} have the same namespace, source_system and id`, { source })
    }
    seen.set(item.documentId, source)
    prepared.push(item)
  })
  return prepared
}

/**
 * `prepareImport` plus the engine's chunking limits (chunk count and size), checked through
 * `memory.chunkDocument` without writing anything. Pass `embeddingDimensions` when an embedder
 * will supply one vector per chunk, so the engine's per-document embedding bound is checked too.
 */
async function validateImport(namespace, entries, memory, { embeddingDimensions } = {}) {
  const prepared = prepareImport(namespace, entries)
  for (const item of prepared) {
    try {
      const chunks = await memory.chunkDocument(item.input)
      if (embeddingDimensions !== undefined && chunks.length * embeddingDimensions > MAX_EMBEDDING_VALUES) {
        throw new Error(`${chunks.length} chunks of ${embeddingDimensions}-dimension embeddings exceed ${MAX_EMBEDDING_VALUES} values per document`)
      }
    } catch (cause) {
      throw new KnowledgeError('invalid_document', `${item.source}: ${cause.message}`, { source: item.source }, { cause })
    }
  }
  return prepared
}

function checkTarget(target, index) {
  const field = `targets[${index}]`
  if (!isPlainObject(target) || Object.keys(target).some(key => key !== 'sourceSystem' && key !== 'id')) {
    requestFailure(field, 'must be exactly { sourceSystem, id }')
  }
  const fail = (name, message) => requestFailure(`${field}.${name}`, message)
  return { sourceSystem: target.sourceSystem, id: target.id, fail }
}

/**
 * Curated documents from one organization namespace in an HMS store. Identity is
 * `[namespace, source_system, id]`; the document's content identifier (SAID) is its version.
 * One writer per store: do not open the same store directory from two processes.
 */
class OrganizationalKnowledge {
  #memory
  #documents
  #namespace
  #embeddingDimensions
  #writes = Promise.resolve()

  /**
   * @param memory A `HolographicMemorySystem`.
   * @param options.namespace The organization namespace for every import, search, and forget.
   * @param options.embedder Optional `DocumentMemory` embedder; must match the store's embedding space.
   * @param options.rerank Optional `DocumentMemory` reranker.
   */
  constructor(memory, { namespace, embedder, rerank, maxPending } = {}) {
    if (!memory || typeof memory.chunkDocument !== 'function' || typeof memory.searchDocuments !== 'function') {
      throw new TypeError('memory must be a HolographicMemorySystem')
    }
    documentId(namespace, 'source', 'id') // validates the namespace
    this.#memory = memory
    this.#namespace = namespace
    this.#embeddingDimensions = embedder?.space?.dimensions
    this.#documents = new DocumentMemory(memory, { embedder, rerank, maxPending })
  }

  get namespace() {
    return this.#namespace
  }

  /**
   * Imports or replaces documents. Everything is validated before the first write. Each
   * document is then committed atomically, replacing any earlier version with the same
   * identity, but the import is not one transaction: on a write failure the thrown
   * `partial_failure` error lists `committed` document IDs, and re-running is idempotent.
   * Re-importing a document that was forgotten restores it.
   * @param entries `{ document, source? }` objects; `document` is a flat curated document.
   */
  async importDocuments(entries) {
    return this.#serialize(() => this.#import(entries))
  }

  /** Writes from overlapping calls run one after another, in call order. */
  #serialize(operation) {
    const run = this.#writes.then(operation)
    this.#writes = run.then(() => undefined, () => undefined)
    return run
  }

  async #import(entries) {
    const prepared = await validateImport(this.#namespace, entries, this.#memory, { embeddingDimensions: this.#embeddingDimensions })
    const documents = []
    let chunks = 0
    const partial = (item, cause) =>
      new KnowledgeError(
        'partial_failure',
        `${item.source}: write failed after ${documents.length} of ${prepared.length} document(s) committed: ${cause.message}`,
        { source: item.source, committed: documents.map(document => document.documentId) },
        { cause },
      )
    for (const item of prepared) {
      try {
        const count = await this.#documents.memorize(item.input)
        chunks += count
        documents.push({ ...item.identity, documentId: item.documentId, version: item.input.version, chunks: count })
      } catch (cause) {
        throw partial(item, cause)
      }
    }
    try {
      await this.#documents.flush()
    } catch (cause) {
      throw partial(prepared.at(-1), cause)
    }
    return { namespace: this.#namespace, documents, chunks }
  }

  /**
   * Searches this namespace only. `options` are `DocumentMemory.search` options; `filter` is an
   * exact top-level metadata match (an `organizational_unit` filter must equal the whole array).
   * Namespace scoping is a query filter, not access control.
   * Each hit carries the engine result plus `identity` (null if the ID is not one this adapter wrote).
   */
  async search(query, options = {}) {
    if (typeof query !== 'string') requestFailure('query', 'must be a string')
    if (!isPlainObject(options)) requestFailure('options', 'must be an object')
    const { filter, ...rest } = options
    if (filter !== undefined && !isPlainObject(filter)) requestFailure('filter', 'must be an object')
    if (filter && Object.hasOwn(filter, NAMESPACE_KEY)) requestFailure('filter', `${NAMESPACE_KEY} is set by the adapter`)
    const hits = await this.#documents.search(query, { ...rest, filter: { ...filter, [NAMESPACE_KEY]: this.#namespace } })
    return hits.map(hit => ({ ...hit, identity: parseDocumentId(hit.documentId) }))
  }

  /**
   * Removes the listed documents from HMS retrieval. Targets are explicit
   * `{ sourceSystem, id }` pairs inside this namespace; there is no wildcard or bulk form.
   * Without `apply: true` it only reports the IDs it would remove. Repeating is safe.
   * This removes HMS index entries; it does not erase upstream sources, backups, or
   * filesystem snapshots, and importing the same document again restores it.
   */
  async forget(targets, { apply = false } = {}) {
    return this.#serialize(() => this.#forget(targets, { apply }))
  }

  async #forget(targets, { apply }) {
    if (!Array.isArray(targets) || targets.length === 0 || targets.length > MAX_FORGET_TARGETS) {
      requestFailure('targets', `must hold 1..=${MAX_FORGET_TARGETS} entries`)
    }
    if (typeof apply !== 'boolean') requestFailure('apply', 'must be a boolean')
    const seen = new Set()
    const planned = targets.map((target, index) => {
      const { sourceSystem, id, fail } = checkTarget(target, index)
      const key = documentId(this.#namespace, sourceSystem, id, fail)
      if (seen.has(key)) requestFailure(`targets[${index}]`, 'duplicates an earlier target')
      seen.add(key)
      return { sourceSystem, id, documentId: key }
    })
    if (!apply) return { namespace: this.#namespace, applied: false, targets: planned }
    const removed = []
    const absent = []
    try {
      for (const target of planned) (await this.#documents.delete(target.documentId) ? removed : absent).push(target)
      await this.#documents.flush()
    } catch (cause) {
      throw new KnowledgeError(
        'partial_failure',
        `forget failed after ${removed.length + absent.length} of ${planned.length} target(s): ${cause.message}`,
        { removed, absent },
        { cause },
      )
    }
    return { namespace: this.#namespace, applied: true, targets: planned, removed, absent }
  }
}

module.exports = { OrganizationalKnowledge, prepareImport, validateImport }
