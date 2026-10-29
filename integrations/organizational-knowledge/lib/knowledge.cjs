'use strict'

const { DocumentMemory } = require('holographic-memory/semantic')
const { KnowledgeError } = require('./errors.cjs')
const { NAMESPACE_KEY, adaptDocument, documentId, isPlainObject, parseDocumentId, requestFailure } = require('./document.cjs')

const MAX_IMPORT_DOCUMENTS = 10_000
const MAX_FORGET_TARGETS = 10_000
const SCAN_PAGE = 1000
const MAX_SCOPE_LIMIT = 100_000
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

function checkTarget(target, index, list = 'targets') {
  const field = `${list}[${index}]`
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

  #plan(targets, list, { maxTargets = MAX_FORGET_TARGETS } = {}) {
    if (!Array.isArray(targets) || targets.length === 0 || targets.length > maxTargets) {
      requestFailure(list, `must hold 1..=${maxTargets} entries`)
    }
    const seen = new Set()
    return targets.map((target, index) => {
      const { sourceSystem, id, fail } = checkTarget(target, index, list)
      const key = documentId(this.#namespace, sourceSystem, id, fail)
      if (seen.has(key)) requestFailure(`${list}[${index}]`, 'duplicates an earlier target')
      seen.add(key)
      return { sourceSystem, id, documentId: key }
    })
  }

  async #forget(targets, { apply }) {
    const planned = this.#plan(targets, 'targets')
    if (typeof apply !== 'boolean') requestFailure('apply', 'must be a boolean')
    if (!apply) return { namespace: this.#namespace, applied: false, targets: planned }
    return { namespace: this.#namespace, applied: true, targets: planned, ...(await this.#remove(planned)) }
  }

  async #remove(planned) {
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
    return { removed, absent }
  }

  /**
   * Removes every document of a source system, or of the whole namespace, from HMS retrieval,
   * except an optional `keep` list. With `keep` set to the identities an upstream still holds,
   * it prunes what the upstream dropped. `scope` is `{ sourceSystem }` or `{ all: true }`, plus an
   * optional nonempty `keep` of `{ sourceSystem, id }` targets. Without `apply: true` it only
   * reports what it would remove. At most `limit` documents (default 10,000) are handled per
   * call; `truncated` says more match. With `apply: true`, repeat until it is false; a preview
   * always shows the same first `limit` documents. Only documents this
   * namespace's adapter wrote (identity and `hms_namespace` agree) are matched. Requires a binding
   * with `listDocuments`. Same limits as `forget`: not physical erasure, not an upstream delete,
   * and importing a removed document again restores it.
   */
  async forgetScope(scope, options = {}) {
    // An ignored misplaced `keep` would widen a prune to everything, so unknown options are errors.
    if (!isPlainObject(options) || Object.keys(options).some(key => key !== 'apply' && key !== 'limit')) {
      requestFailure('options', 'must be { apply, limit } only; put keep inside scope')
    }
    const { apply = false, limit = 10_000 } = options
    return this.#serialize(() => this.#forgetScope(scope, { apply, limit }))
  }

  async #forgetScope(scope, { apply, limit }) {
    const names = ['sourceSystem', 'all', 'keep']
    if (!isPlainObject(scope) || Object.keys(scope).some(key => !names.includes(key))) {
      requestFailure('scope', 'must be { sourceSystem } or { all: true }, with an optional keep list')
    }
    const { sourceSystem, all, keep } = scope
    if ((sourceSystem === undefined) === (all === undefined)) requestFailure('scope', 'set exactly one of sourceSystem or all')
    if (all !== undefined && all !== true) requestFailure('scope.all', 'must be true')
    if (sourceSystem !== undefined) {
      documentId(this.#namespace, sourceSystem, 'id', (field, message) => requestFailure(`scope.${field}`, message))
    }
    if (typeof apply !== 'boolean') requestFailure('apply', 'must be a boolean')
    if (!Number.isInteger(limit) || limit < 1 || limit > MAX_SCOPE_LIMIT) requestFailure('limit', `must be an integer in 1..=${MAX_SCOPE_LIMIT}`)
    const kept = new Set()
    if (keep !== undefined) {
      for (const target of this.#plan(keep, 'scope.keep')) {
        if (sourceSystem !== undefined && target.sourceSystem !== sourceSystem) {
          requestFailure('scope.keep', `${target.documentId} is outside the source system ${sourceSystem}`)
        }
        kept.add(target.documentId)
      }
    }
    if (typeof this.#memory.listDocuments !== 'function') {
      throw new KnowledgeError('unsupported_binding', 'this holographic-memory binding has no listDocuments; rebuild it from a source that exports it')
    }

    const { planned, truncated } = await this.#scan(sourceSystem, kept, limit)
    const result = { namespace: this.#namespace, applied: apply, scope: sourceSystem === undefined ? { all: true } : { sourceSystem }, targets: planned, truncated }
    return apply ? { ...result, ...(planned.length > 0 ? await this.#remove(planned) : { removed: [], absent: [] }) } : result
  }

  /** Walks the namespace's slice of the id-ordered document listing, keeping up to `limit` matches. */
  async #scan(sourceSystem, kept, limit) {
    // Every identity in the scope is a JSON array that starts with this prefix, so they are contiguous.
    const parts = sourceSystem === undefined ? [this.#namespace] : [this.#namespace, sourceSystem]
    const prefix = `${JSON.stringify(parts).slice(0, -1)},`
    const prefixBytes = Buffer.from(prefix)
    const planned = []
    // Starts after `["ns"`, which sorts before the prefix; ids that sort before it are skipped.
    let after = prefix.slice(0, -1)
    for (;;) {
      const page = await this.#memory.listDocuments(after, SCAN_PAGE)
      for (const record of page) {
        if (!record.id.startsWith(prefix)) {
          if (Buffer.compare(Buffer.from(record.id), prefixBytes) > 0) return { planned, truncated: false }
          continue
        }
        const identity = parseDocumentId(record.id)
        if (identity?.namespace !== this.#namespace || record.metadata?.[NAMESPACE_KEY] !== this.#namespace || kept.has(record.id)) continue
        if (planned.length === limit) return { planned, truncated: true }
        planned.push({ sourceSystem: identity.sourceSystem, id: identity.id, documentId: record.id })
      }
      if (page.length < SCAN_PAGE) return { planned, truncated: false }
      after = page.at(-1).id
    }
  }
}

module.exports = { OrganizationalKnowledge, prepareImport, validateImport }
