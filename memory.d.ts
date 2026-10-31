// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0
import type { WarmMemorySystem } from './index'

/** A Date, or an integer of Unix milliseconds. Results always carry milliseconds. */
export type Time = Date | number
export type FactStatus = 'current' | 'superseded' | 'expired' | 'not-yet-valid'
export type HistoryMode = 'current' | 'all' | 'historical'

export interface MemoryConfig {
  /** Results per recall when `k` is absent. Default 10. */
  defaultK?: number
  /** Which facts recall returns by default. Default `current`. */
  defaultHistory?: HistoryMode
  /** `explicit` (default): only `supersedes` ids. `latest-by-subject`: a subject names one slot and the latest valid fact wins. */
  supersession?: 'explicit' | 'latest-by-subject'
  /** Default `{ kind: 'none' }`: scores are untouched. */
  recency?: { kind: 'none' } | { kind: 'exponential'; halfLifeMs: number; basis?: 'event' | 'ingest'; floor?: number }
  /** Multiplier on non-current facts' scores when recall includes them. Default 1. */
  historicalWeight?: number
  /** Used by `applyForgetting`. Default `{ kind: 'keep-all' }`. */
  forgetting?:
    | { kind: 'keep-all' }
    | { kind: 'expired-for'; graceMs: number }
    | { kind: 'superseded-for'; ageMs: number }
    | { kind: 'max-facts'; limit: number }
  /** Used by `consolidate`. Default `none`. */
  consolidation?: 'none' | 'collapse-duplicates'
  /** An id that exists with different content: `reject` (default) or `replace`. */
  onIdConflict?: 'reject' | 'replace'
}

export interface RememberInput {
  /** Without an id, one derived from the content is used, so retries store one fact. */
  id?: string
  text: string
  /** When the described thing happened. Distinct from the time the store ingested it. */
  eventTime?: Time
  /** The fact holds from this time (inclusive) ... */
  validFrom?: Time
  /** ... until this time (exclusive). */
  validTo?: Time
  /** What the fact is about; the key supersession policies match on. */
  subject?: string
  /** Ids of facts this one replaces. They are kept as history. */
  supersedes?: string[]
  entities?: string[]
  /** Caller metadata; the key `_mem` is reserved. */
  metadata?: Record<string, unknown>
  embedding?: number[]
  sourceUri?: string
}

export interface Remembered {
  id: string
  /** False when the identical fact was already stored (an idempotent retry). */
  created: boolean
  supersedes: string[]
  supersededBy: string[]
  ingestTime: number
}

export interface RecallOptions {
  k?: number
  /** Evaluate validity and supersession at this time. Default now. */
  asOf?: Time
  /** Evaluate what the store knew at this ingest time. */
  knownAt?: Time
  history?: HistoryMode
  subject?: string
  entities?: string[]
  filter?: Record<string, unknown>
  embedding?: number[]
  lexicalWeight?: number
  semanticWeight?: number
  minSemanticScore?: number
  fusion?: 'blend' | 'rrf'
  /** Eligible candidates ranked before optional recency adjustment. */
  candidateLimit?: number
  /** Attach an `explanation` to every result. */
  explain?: boolean
}

export interface ChainLink {
  id: string
  relation: 'predecessor' | 'successor'
  effectiveFrom: number
  ingestTime: number
}

export interface Explanation {
  retrieved: boolean
  rank: number | null
  lexicalScore: number | null
  semanticScore: number | null
  retrievalScore: number | null
  lexicalWeight: number
  semanticWeight: number
  fusion: string
  matchedTerms: string[]
  recencyFactor: number
  historicalFactor: number
  finalScore: number | null
  status: FactStatus
  asOf: number
  knownAt: number | null
  validFrom: number | null
  validTo: number | null
  withinValidity: boolean
  chain: ChainLink[]
}

export interface Recalled {
  id: string
  text: string | null
  score: number
  status: FactStatus
  ingestTime: number | null
  eventTime: number | null
  validFrom: number | null
  validTo: number | null
  subject: string | null
  entities: string[]
  supersedes: string[]
  supersededBy: string[]
  metadata: Record<string, unknown>
  sourceUri: string | null
  explanation: Explanation | null
}

export interface FactRecord {
  id: string
  text: string | null
  status: FactStatus
  ingestTime: number | null
  eventTime: number | null
  validFrom: number | null
  validTo: number | null
  effectiveFrom: number
  subject: string | null
  entities: string[]
  supersedes: string[]
  supersededBy: string[]
  metadata: Record<string, unknown>
}

/** Exactly one scope. */
export type ForgetTarget =
  | { ids: Iterable<string> }
  | { subject: string }
  | { entity: string }
  | { ingestedBefore: Time }
  | { eventBefore: Time }
  | { matching: Record<string, unknown> }

export interface ForgetOptions {
  /** Also remove every version linked by supersession. */
  includeChain?: boolean
  /** Report without removing. */
  dryRun?: boolean
  /** Compact the store so deleted text leaves the journal; the text segment file is rebuilt at next open. */
  purge?: boolean
}

export interface ForgetReport {
  forgotten: string[]
  dryRun: boolean
  purged: boolean
}

export interface ConsolidateOptions {
  dryRun?: boolean
  /** Honor a policy's request to remove consolidated sources. */
  allowDestructive?: boolean
}

export interface ConsolidationReport {
  created: Remembered[]
  forgotten: string[]
  dryRun: boolean
}

/** Stable failure class shared with the HTTP memory API. */
export class MemoryError extends Error {
  code: 'invalid_request' | 'not_found' | 'conflict' | 'concurrent_mutation' | 'resource_limit' | 'mutation_committed' | 'internal' | (string & {})
}

export class Memory {
  /** Applies `config` to `system`'s memory API (replacing any earlier configuration). */
  constructor(system: WarmMemorySystem, config?: MemoryConfig)
  remember(input: RememberInput): Promise<Remembered>
  remember(text: string, options?: Omit<RememberInput, 'text'>): Promise<Remembered>
  /** One atomic transaction: every fact is stored or none is. */
  rememberMany(inputs: Iterable<RememberInput>): Promise<Remembered[]>
  recall(query: string, options?: RecallOptions): Promise<Recalled[]>
  capabilities(contractVersion?: number): Promise<{
    contractVersion: number
    profile: string
    operations: string[]
    temporalQueries: boolean
    exactMetadataFilters: boolean
    subjectEntityFilters: boolean
    documentPagination: boolean
    committedErrorCode: string
  }>
  explain(query: string, id: string, options?: Omit<RecallOptions, 'k' | 'history' | 'explain'>): Promise<Explanation>
  forget(target: ForgetTarget, options?: ForgetOptions): Promise<ForgetReport>
  get(id: string): Promise<FactRecord | null>
  /** Every version of the fact's supersession chain, in effect order. */
  history(id: string): Promise<FactRecord[]>
  applyForgetting(options?: ForgetOptions): Promise<ForgetReport>
  consolidate(options?: ConsolidateOptions): Promise<ConsolidationReport>
}
