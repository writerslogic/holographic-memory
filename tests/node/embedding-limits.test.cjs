// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

const assert = require('node:assert/strict')
const { test } = require('node:test')
const { join } = require('node:path')
const binding = require(process.env.HMS_EMBED_ADDON || '../../index.js')
const { LocalEmbedder } = binding
const metadata = require('../fixtures/tiny-embed-int8.json')
const model = join(__dirname, '../fixtures/tiny-embed-int8.hmsemb')
const options = { model, sha256: metadata.sha256, threads: 1, offline: true }

test('local embedding reports preserve vectors and expose both truncation stages', { skip: !LocalEmbedder }, async () => {
  const embedder = await LocalEmbedder.load({ ...options, maxLength: 8 })
  const texts = ['cat '.repeat(5), 'cat '.repeat(6), 'cat '.repeat(7), 'é'.repeat(127) + '🦀']
  for (const [reportMethod, vectorMethod] of [
    ['embedQueryWithReport', 'embedQuery'], ['embedPassagesWithReport', 'embedPassages'],
  ]) {
    const report = await embedder[reportMethod](texts)
    assert.deepEqual(report.embeddings, await embedder[vectorMethod](texts))
    assert.deepEqual(report.inputs.slice(0, 3).map(input => input.tokenTruncated), [false, false, true])
    assert.equal(report.inputs[3].byteTruncated, true)
    assert.equal(report.inputs[3].retainedBytes, 254)
  }
})

test('local embedding call limits reject excess work and retain usable admission', { skip: !LocalEmbedder }, async () => {
  const embedder = await LocalEmbedder.load({ ...options, limits: { maxTexts: 2 } })
  for (const count of [1, 2]) assert.equal((await embedder.embedQuery(Array(count).fill('cat'))).length, count)
  await assert.rejects(embedder.embedQuery(['cat', 'dog', 'mat']), /max_texts/)
  assert.equal((await embedder.embedPassages(['dog'])).length, 1)
  for (const name of ['maxTexts', 'maxInputBytes', 'maxTokens', 'maxOutputBytes', 'maxConcurrentCalls']) {
    for (const value of [-1, 0, 1.5, NaN, Infinity, 2 ** 32]) {
      await assert.rejects(LocalEmbedder.load({ ...options, limits: { [name]: value } }), /finite integer/)
    }
  }
  for (const name of ['maxLength', 'batchSize', 'maxBatchTokens', 'threads']) {
    for (const value of [-1, 1.5, NaN, Infinity, 2 ** 32]) {
      await assert.rejects(LocalEmbedder.load({ ...options, [name]: value }), /finite integer/)
    }
  }
  for (const value of [2 ** 32 - 2, 2 ** 32 - 1]) {
    const limits = Object.fromEntries(['maxTexts', 'maxInputBytes', 'maxTokens', 'maxOutputBytes', 'maxConcurrentCalls'].map(name => [name, value]))
    const large = await LocalEmbedder.load({ ...options, limits })
    assert.equal((await large.embedQuery(['cat'])).length, 1)
  }
})
