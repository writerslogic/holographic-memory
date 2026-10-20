// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

// Five-minute quickstart: ingest a corpus, search it, compare with SQLite FTS5, encrypt it.
// From the repository root: `npm ci --omit=optional && npm run build && node examples/quickstart/node/quickstart.mjs`
// Needs Node 24 or newer for the built-in `node:sqlite`.
import assert from 'node:assert/strict'
import { execFileSync } from 'node:child_process'
import { mkdtempSync, readFileSync, readdirSync } from 'node:fs'
import { createRequire } from 'node:module'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { DatabaseSync } from 'node:sqlite'
import { fileURLToPath } from 'node:url'

const require = createRequire(import.meta.url)
const { HolographicMemorySystem } = require('../../../index.js')
const { DocumentMemory } = require('../../../semantic.js')

const here = fileURLToPath(new URL('..', import.meta.url))
const corpus = readFileSync(join(here, 'corpus.jsonl'), 'utf8').trim().split('\n').map(line => JSON.parse(line))
const queries = JSON.parse(readFileSync(join(here, 'queries.json'), 'utf8'))
const toInput = doc => ({ id: doc.id, text: doc.text, sourceUri: `udhr#${doc.title}` })
const passphrase = 'correct horse battery staple'
const sealedConfig = { encryptionEnabled: true, encryptionPassphrase: passphrase }

// A store is a directory and holds an exclusive lock while open, so the encryption step below
// writes each store from a short-lived child process of this same file.
if (process.argv[2] === '--ingest') {
  const [, , , dir, mode] = process.argv
  const store = new DocumentMemory(new HolographicMemorySystem(4096, dir, mode === 'encrypted' ? sealedConfig : undefined))
  await store.ingest(corpus.map(toInput))
  await store.flush()
  process.exit(0)
}

// --8<-- [start:ingest]
// 1. Ingest. A store is a directory; it persists there and reopens with the same arguments.
const memory = new DocumentMemory(new HolographicMemorySystem(4096, mkdtempSync(join(tmpdir(), 'hms-quickstart-'))))
const started = performance.now()
await memory.ingest(corpus.map(toInput))
console.log(`ingested ${corpus.length} documents in ${(performance.now() - started).toFixed(1)} ms`)
// --8<-- [end:ingest]

// --8<-- [start:compare]
// The same corpus in SQLite FTS5 (bm25 ranking), for comparison.
const db = new DatabaseSync(':memory:')
db.exec('CREATE VIRTUAL TABLE docs USING fts5(id UNINDEXED, text)')
const insert = db.prepare('INSERT INTO docs (id, text) VALUES (?, ?)')
for (const doc of corpus) insert.run(doc.id, doc.text)
// FTS5 treats a bare multi-word query as AND; OR of the words is the fair lexical baseline.
const fts5Query = query => query.split(/[^\p{L}\p{N}]+/u).filter(Boolean).map(w => `"${w}"`).join(' OR ')
const ftsTop = db.prepare('SELECT id FROM docs WHERE docs MATCH ? ORDER BY bm25(docs) LIMIT 1')

// 2. Search both engines with the same queries.
console.log(`\n${'query'.padEnd(58)} ${'HMS'.padEnd(9)} ${'FTS5'.padEnd(9)} expected`)
let hmsHits = 0
let ftsHits = 0
for (const { query, expect } of queries) {
  const [hit] = await memory.search(query, { k: 3 })
  const hms = hit?.documentId ?? '-'
  const fts = ftsTop.get(fts5Query(query))?.id ?? '-'
  hmsHits += hms === expect
  ftsHits += fts === expect
  console.log(`${query.padEnd(58)} ${hms.padEnd(9)} ${fts.padEnd(9)} ${expect}`)
}
console.log(`top-1 correct: HMS ${hmsHits}/${queries.length}, FTS5 ${ftsHits}/${queries.length}`)
assert.ok(hmsHits >= 5, `HMS found the expected article for only ${hmsHits} queries`)
// --8<-- [end:compare]

// --8<-- [start:hit]
// Hits carry the passage and its byte range in the source text.
const [hit] = await memory.search('asylum from persecution')
console.log(`\n${hit.documentId} bytes ${hit.startByte}..${hit.endByte} score ${hit.score.toFixed(3)}: ${hit.text}`)
// --8<-- [end:hit]

// --8<-- [start:encrypt]
// 3. Encrypt: the same corpus in a persistent store with AES-256-GCM (Argon2id key).
const needle = Buffer.from('periodic holidays with pay')
const filesHolding = dir => readdirSync(dir, { recursive: true, withFileTypes: true })
  .filter(entry => entry.isFile() && readFileSync(join(entry.parentPath, entry.name)).includes(needle))
  .map(entry => entry.name)
  .sort()

const sealedDir = mkdtempSync(join(tmpdir(), 'hms-sealed-'))
execFileSync(process.execPath, [fileURLToPath(import.meta.url), '--ingest', sealedDir, 'encrypted'], { stdio: 'inherit' })
const leaks = filesHolding(sealedDir)
console.log(`\nencrypted store, files holding the passage: ${JSON.stringify(leaks)}`)
// Pins the current behaviour: `documents.text`, the engine's file-backed passage cache, is
// written in plaintext even when encryption is on, so encryption does not yet keep passages
// off the disk. A new leak, or the engine fixing this, fails the assertion and forces the docs
// (docs/concepts/privacy-limits.md, README) to be updated.
assert.deepEqual(leaks, ['documents.text'])
console.log('note: passage text in documents.text is not yet encrypted at rest')

// A wrong passphrase must not open the store; the right one must.
const wrong = { ...sealedConfig, encryptionPassphrase: 'wrong passphrase' }
assert.throws(() => new HolographicMemorySystem(4096, sealedDir, wrong), error => {
  console.log(`opening with a wrong passphrase fails: ${error.message}`)
  return !/already open/.test(error.message)
})
const reopened = new HolographicMemorySystem(4096, sealedDir, sealedConfig)
assert.equal(reopened.securityStatus().encryptionActive, true)
const [again] = await new DocumentMemory(reopened).search('paid holidays')
console.log(`reopened with the passphrase, top hit: ${again.documentId}`)
assert.equal(again.documentId, 'udhr-24')
// --8<-- [end:encrypt]
