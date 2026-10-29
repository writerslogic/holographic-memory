'use strict'

// Opt-in: needs a local ONNX embedding model directory and @huggingface/transformers (an optional
// peer dependency of holographic-memory that this checkout does not install).
const test = require('node:test')
const assert = require('node:assert/strict')
const fs = require('node:fs')
const os = require('node:os')
const path = require('node:path')
const { HolographicMemorySystem } = require('holographic-memory')
const { createLocalEmbedder } = require('holographic-memory/semantic')
const { OrganizationalKnowledge } = require('../index.cjs')

const MODEL_DIR = process.env.HMS_TEST_ONNX_MODEL_DIR
const MODEL_ID = process.env.HMS_TEST_ONNX_MODEL_ID ?? 'BAAI/bge-small-en-v1.5'
const QUERY_PREFIX = process.env.HMS_TEST_ONNX_QUERY_PREFIX ?? 'Represent this sentence for searching relevant passages: '

const doc = (id, content) => ({
  document: { id, source_system: 'Wiki', identifier: `urn:said:E${id}`, organizational_unit: ['EGWG'], content },
  source: `${id}.json`,
})

test('a local ONNX embedder ranks by meaning where lexical retrieval finds nothing, and forget removes the document', { skip: !MODEL_DIR && 'set HMS_TEST_ONNX_MODEL_DIR to a local ONNX model directory' }, async () => {
  const embedder = await createLocalEmbedder({ modelPath: MODEL_DIR, modelId: MODEL_ID, revision: 'local', queryPrefix: QUERY_PREFIX })
  const workdir = fs.mkdtempSync(path.join(os.tmpdir(), 'ok-onnx-'))
  try {
    const { model, revision, dimensions } = embedder.space
    const memory = new HolographicMemorySystem(4096, workdir, { embeddingModel: model, embeddingRevision: revision, embeddingDimensions: dimensions })
    const knowledge = new OrganizationalKnowledge(memory, { namespace: 'org', embedder })
    await knowledge.importDocuments([
      doc('charter', 'The working group approved the trust registry charter at the quarterly governance review.'),
      doc('recipe', 'Simmer tomatoes with basil and garlic for twenty minutes, then serve over pasta.'),
      doc('offsite', 'The engineering offsite moves to Thursday because the venue flooded.'),
    ])

    const query = 'preparing an Italian dinner sauce'
    assert.deepEqual(await knowledge.search(query, { lexicalWeight: 1, semanticWeight: 0 }), [])
    const [best] = await knowledge.search(query, { lexicalWeight: 0, semanticWeight: 1 })
    assert.equal(best.identity.id, 'recipe')
    assert.ok(best.semanticScore > 0)

    await knowledge.forget([{ sourceSystem: 'Wiki', id: 'recipe' }], { apply: true })
    const after = await knowledge.search(query, { lexicalWeight: 0, semanticWeight: 1 })
    assert.ok(after.every(hit => hit.identity.id !== 'recipe'))
  } finally {
    await embedder.dispose()
    fs.rmSync(workdir, { recursive: true, force: true })
  }
})
