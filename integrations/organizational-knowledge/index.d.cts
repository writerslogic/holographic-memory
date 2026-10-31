import type { DocumentInput, DocumentResult, WarmMemorySystem, SearchOptions } from 'writerslogic-warm'
import type { Embedder, Reranker } from 'writerslogic-warm/semantic'

/** A flat curated document as written by the ToIP EGWG prototype's `src.curate`. */
export interface CuratedDocument {
  /** Stable, human-readable ID; with `source_system` and the namespace it is the document's identity. */
  id: string
  source_system: string
  /** Content identifier (SAID). Kept verbatim as the WARM version; not verified. */
  identifier: string
  content: string
  organizational_unit: string[]
  project?: string
  source_identifier?: string
  /** Empty, or an absolute http(s) URL without credentials. */
  source_url?: string
  date?: string
  asset_type?: string
  content_type?: string
  title?: string
  identifier_type?: string
  collected_at?: string
  /** Other JSON fields are preserved in metadata. `hms_namespace` is reserved. */
  [field: string]: unknown
}

export interface ImportEntry {
  /** A parsed JSON value, validated at runtime; see `CuratedDocument` for the accepted shape. */
  document: unknown
  /** Names the document in errors, such as its file path. Defaults to `entries[i]`. */
  source?: string
}

export interface Identity {
  namespace: string
  sourceSystem: string
  id: string
}

export interface ForgetTarget {
  sourceSystem: string
  id: string
}

export interface PlannedTarget extends ForgetTarget {
  documentId: string
}

export interface ImportedDocument extends Identity {
  documentId: string
  version: string
  chunks: number
}

export interface ImportResult {
  namespace: string
  documents: ImportedDocument[]
  chunks: number
}

export interface ForgetResult {
  namespace: string
  applied: boolean
  targets: PlannedTarget[]
  /** Present when `applied` is true: targets whose document existed and was removed. */
  removed?: PlannedTarget[]
  /** Present when `applied` is true: targets that were not stored. */
  absent?: PlannedTarget[]
}

export type ForgetScope = ({ sourceSystem: string; all?: never } | { all: true; sourceSystem?: never }) & {
  /** Documents to leave in place; must not be empty when given. */
  keep?: ForgetTarget[]
}

export interface ForgetScopeResult {
  namespace: string
  applied: boolean
  scope: { sourceSystem: string } | { all: true }
  /** The documents matched, at most `limit`. */
  targets: PlannedTarget[]
  /** More documents matched than `limit`. Applying and repeating continues; a preview repeats the same first `limit`. */
  truncated: boolean
  removed?: PlannedTarget[]
  absent?: PlannedTarget[]
}

export type KnowledgeErrorCode =
  | 'invalid_document'
  | 'invalid_request'
  | 'duplicate_identity'
  | 'unsupported_binding'
  | 'partial_failure'

export class KnowledgeError extends Error {
  code: KnowledgeErrorCode
  source?: string
  field?: string
  /** `partial_failure` from an import: document IDs already committed. */
  committed?: string[]
  /** `partial_failure` from forget: targets already processed. */
  removed?: PlannedTarget[]
  absent?: PlannedTarget[]
}

export type Hit = DocumentResult & { identity: Identity | null }

export interface OrganizationalKnowledgeOptions {
  namespace: string
  /** Must match the store's embedding space; see `DocumentMemory`. */
  embedder?: Embedder
  rerank?: Reranker
  maxPending?: number
}

export class OrganizationalKnowledge {
  constructor(memory: WarmMemorySystem, options: OrganizationalKnowledgeOptions)
  readonly namespace: string
  importDocuments(entries: Iterable<ImportEntry>): Promise<ImportResult>
  search(query: string, options?: Omit<SearchOptions, 'filter'> & { filter?: Record<string, unknown> }): Promise<Hit[]>
  forget(targets: ForgetTarget[], options?: { apply?: boolean }): Promise<ForgetResult>
  /** Needs a binding that exports `listDocuments`. Dry run unless `apply` is true; `limit` defaults to 10,000. */
  forgetScope(scope: ForgetScope, options?: { apply?: boolean; limit?: number }): Promise<ForgetScopeResult>
}

/** The WARM document ID for an identity. Throws `invalid_request` for unusable parts. */
export function documentId(namespace: string, sourceSystem: string, id: string): string
/** Reads one curated document per `.json` file, from files or directories (not recursive). */
export function readCuratedFiles(targets: string[]): Promise<Array<{ source: string; document: unknown }>>
export interface PreparedDocument {
  source: string
  documentId: string
  identity: Identity
  input: DocumentInput
}
/** Validates adapter rules and duplicate identities without an engine or any write. */
export function prepareImport(namespace: string, entries: Iterable<ImportEntry>): PreparedDocument[]
/** `prepareImport` plus the engine's chunking limits, checked through `memory.chunkDocument`. */
export function validateImport(
  namespace: string,
  entries: Iterable<ImportEntry>,
  memory: WarmMemorySystem,
  options?: { embeddingDimensions?: number },
): Promise<PreparedDocument[]>
