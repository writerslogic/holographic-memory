// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0
// node examples/vector.mjs  (offline: toy 4-dimensional embeddings)
import { mkdtempSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { HolographicVector } from '../dist/index.js'

const vector = new HolographicVector({ id: 'warm', dimension: 4, path: mkdtempSync(join(tmpdir(), 'warm-mastra-ex-')) })
await vector.createIndex({ indexName: 'notes', dimension: 4 })
await vector.upsert({
  indexName: 'notes',
  vectors: [[1, 0, 0, 0], [0, 1, 0, 0], [0.9, 0.1, 0, 0]],
  metadata: [{ text: 'cats', topic: 'pets', year: 2024 }, { text: 'rockets', topic: 'space', year: 2025 }, { text: 'kittens', topic: 'pets', year: 2025 }],
  ids: ['cats', 'rockets', 'kittens'],
})
const hits = await vector.query({
  indexName: 'notes',
  queryVector: [1, 0, 0, 0],
  topK: 2,
  filter: { topic: 'pets', year: { $gte: 2025 } },
})
console.log(hits)
console.log(await vector.describeIndex({ indexName: 'notes' }))
