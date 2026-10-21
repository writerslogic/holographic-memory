// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0
import type { HmsStore, Metadata } from '@holographic-memory/core'

const SEQ_DIGITS = 10
const BATCH = 100

export interface LogEntry {
  seq: number
  text: string
  metadata: Metadata
}

/**
 * An append-only, gap-free ordered log inside an `HmsStore`.
 *
 * The engine has no list-all and no ordering, so entry `n` of log `name` is the record
 * `name#<n zero padded>`. The length is found by galloping and binary search over `get`
 * (about 2 log2(n) lookups), every entry is fetched by exact id, and nothing depends on the
 * per-query result cap, so logs are unbounded. One writer per log: concurrent writers to the
 * same log from different processes can interleave or overwrite entries.
 */
export class OrderedLog {
  private length: number | undefined

  constructor(
    private readonly store: HmsStore,
    readonly name: string,
  ) {
    if (!name) throw new RangeError('log name must be non-empty')
  }

  private id(seq: number): string {
    return `${this.name}#${String(seq).padStart(SEQ_DIGITS, '0')}`
  }

  private async exists(seq: number): Promise<boolean> {
    const [rec] = await this.store.get([this.id(seq)])
    return rec !== undefined
  }

  /** Number of entries. Re-verified against the store on every call, so other writers are noticed. */
  async size(): Promise<number> {
    let lo = this.length ?? 0
    // Invariant: entries 0..lo-1 exist (checked below), entry `hi` does not.
    if (lo > 0 && !(await this.exists(lo - 1))) lo = 0
    let step = 1
    let hi = lo
    while (await this.exists(hi)) {
      lo = hi + 1
      hi = lo + step - 1
      step *= 2
    }
    while (lo < hi) {
      const mid = Math.floor((lo + hi) / 2)
      if (await this.exists(mid)) lo = mid + 1
      else hi = mid
    }
    this.length = lo
    return lo
  }

  async append(entries: { text: string; metadata?: Metadata }[]): Promise<void> {
    if (!entries.length) return
    const start = await this.size()
    await this.store.remember(
      entries.map((e, i) => ({
        id: this.id(start + i),
        text: e.text,
        metadata: { ...(e.metadata ?? {}), hms_log: this.name, hms_seq: start + i },
      })),
    )
    this.length = start + entries.length
  }

  /** Entries `[from, to)` in order. */
  async read(from: number, to: number): Promise<LogEntry[]> {
    const out: LogEntry[] = []
    for (let s = from; s < to; s += BATCH) {
      const seqs = Array.from({ length: Math.min(BATCH, to - s) }, (_, i) => s + i)
      const recs = await this.store.get(seqs.map((q) => this.id(q)))
      recs.forEach((r, i) => {
        if (r) out.push({ seq: seqs[i] as number, text: r.text, metadata: stripLog(r.metadata) })
      })
    }
    return out
  }

  /** The last `limit` entries (all when omitted) in chronological order. */
  async tail(limit?: number): Promise<LogEntry[]> {
    const n = await this.size()
    const from = limit === undefined ? 0 : Math.max(0, n - Math.max(0, limit))
    return this.read(from, n)
  }

  /** Remove and return the last entry. */
  async pop(): Promise<LogEntry | undefined> {
    const n = await this.size()
    if (n === 0) return undefined
    const [entry] = await this.read(n - 1, n)
    await this.store.forget([this.id(n - 1)])
    this.length = n - 1
    return entry
  }

  async clear(): Promise<void> {
    const n = await this.size()
    for (let s = 0; s < n; s += BATCH) {
      await this.store.forget(Array.from({ length: Math.min(BATCH, n - s) }, (_, i) => this.id(s + i)))
    }
    this.length = 0
  }
}

/** Metadata is capped at 64 KiB by the engine; payloads up to this size stay in metadata, larger ones move to the record text. */
const INLINE_PAYLOAD_BYTES = 32 * 1024

/**
 * Serialize `payload` for a log entry. Small payloads go in metadata (`searchText` stays the
 * readable record text). Larger ones, which would exceed the metadata cap, become the record
 * text itself, so they are still stored and returned byte-exactly, but lexical search then sees
 * the JSON.
 */
export function packPayload(searchText: string, payload: unknown, extra: Metadata = {}): { text: string; metadata: Metadata } {
  const json = JSON.stringify(payload)
  if (Buffer.byteLength(json, 'utf8') <= INLINE_PAYLOAD_BYTES) return { text: searchText, metadata: { ...extra, payload: json } }
  return { text: json, metadata: { ...extra, payload_in_text: true } }
}

export function unpackPayload<T>(entry: { text: string; metadata: Metadata }): T {
  return JSON.parse(entry.metadata.payload_in_text === true ? entry.text : (entry.metadata.payload as string)) as T
}

function stripLog(metadata: Metadata): Metadata {
  const { hms_log: _l, hms_seq: _s, ...rest } = metadata
  return rest
}
