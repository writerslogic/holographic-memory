// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0
import { HolographicMemorySystem } from '../../index.js'
import { Memory, MemoryError, type Recalled, type ForgetReport, type MemoryConfig } from '../../memory.js'

const config: MemoryConfig = {
  supersession: 'latest-by-subject',
  recency: { kind: 'exponential', halfLifeMs: 86_400_000, basis: 'event' },
  forgetting: { kind: 'expired-for', graceMs: 1000 },
}
const memory = new Memory(new HolographicMemorySystem(4096, './types-only'), config)
const stored = await memory.remember('Alice lives in Paris', { subject: 'alice/city', validFrom: new Date(), entities: ['Alice'] })
const id: string = stored.id
const created: boolean = stored.created
const found: Recalled[] = await memory.recall('Alice', { history: 'all', asOf: Date.now(), explain: true })
const status: 'current' | 'superseded' | 'expired' | 'not-yet-valid' | undefined = found[0]?.status
const terms: string[] | undefined = found[0]?.explanation?.matchedTerms
const report: ForgetReport = await memory.forget({ ids: [id] }, { includeChain: true, dryRun: true })
const fact = await memory.get(id)
const history = await memory.history(id)
void [created, status, terms, report, fact, history, MemoryError]

// These must remain invalid at the public TypeScript boundary.
// @ts-expect-error a fact needs text
await memory.remember({ id: 'x' })
// @ts-expect-error history takes a mode name
await memory.recall('q', { history: 'everything' })
// @ts-expect-error times are Dates or integer milliseconds
await memory.recall('q', { asOf: 'yesterday' })
// @ts-expect-error unknown configuration values are rejected
new Memory(new HolographicMemorySystem(4096), { supersession: 'magic' })
