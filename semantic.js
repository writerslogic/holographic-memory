'use strict'

const { stat } = require('node:fs/promises')
const { resolve } = require('node:path')
const { createHash } = require('node:crypto')

async function localModel(modelPath) {
  if (typeof modelPath !== 'string' || !modelPath) throw new TypeError('modelPath is required')
  const path = resolve(modelPath)
  if (!(await stat(path)).isDirectory()) throw new TypeError('modelPath must be a local directory')
  return path
}

async function transformers() {
  try {
    return await import('@huggingface/transformers')
  } catch (cause) {
    throw new Error('Install @huggingface/transformers to load local ONNX models', { cause })
  }
}

async function createLocalEmbedder({ modelPath, modelId, revision, queryPrefix = '', documentPrefix = '', dtype = 'fp32' }) {
  if (!modelId || !revision) throw new TypeError('modelId and revision must identify the embedding model and preprocessing')
  const path = await localModel(modelPath)
  const preprocessing = createHash('sha256').update(JSON.stringify({ pooling: 'mean', queryPrefix, documentPrefix, dtype })).digest('hex').slice(0, 16)
  const { pipeline } = await transformers()
  const extractor = await pipeline('feature-extraction', path, { local_files_only: true, device: 'cpu', dtype })
  async function encode(texts, prefix) {
    const rows = []
    for (let offset = 0; offset < texts.length; offset += 16) {
      const tensor = await extractor(texts.slice(offset, offset + 16).map(text => prefix + text), { pooling: 'mean', normalize: true })
      rows.push(...tensor.tolist())
    }
    return rows
  }
  try {
    const [probe] = await encode(['dimension probe'], documentPrefix)
    return {
      space: { model: modelId, revision: `${revision}+hms-${preprocessing}`, dimensions: probe.length, normalization: 'l2', metric: 'cosine' },
      encodeDocuments: texts => encode(texts, documentPrefix),
      encodeQuery: async text => (await encode([text], queryPrefix))[0],
      dispose: () => extractor.dispose(),
    }
  } catch (error) {
    await extractor.dispose()
    throw error
  }
}

async function createLocalReranker({ modelPath, dtype = 'fp32' }) {
  const path = await localModel(modelPath)
  const { AutoTokenizer, AutoModelForSequenceClassification } = await transformers()
  const tokenizer = await AutoTokenizer.from_pretrained(path, { local_files_only: true })
  const model = await AutoModelForSequenceClassification.from_pretrained(path, { local_files_only: true, device: 'cpu', dtype })
  const rerank = async (query, passages) => {
    const scores = []
    for (let offset = 0; offset < passages.length; offset += 16) {
      const batch = passages.slice(offset, offset + 16)
      const inputs = await tokenizer(batch.map(() => query), { text_pair: batch, padding: true, truncation: true })
      const { logits } = await model(inputs)
      for (const row of logits.tolist()) {
        if (row.length === 1) scores.push(row[0])
        else if (row.length === 2) scores.push(row[1] - row[0])
        else throw new Error('Reranker must have one relevance logit or two classification logits')
      }
    }
    return scores
  }
  rerank.dispose = () => model.dispose()
  return rerank
}

class DocumentMemory {
  constructor(memory, { embedder, rerank, maxPending = 32 } = {}) {
    if (!Number.isInteger(maxPending) || maxPending < 1 || maxPending > 4096) throw new RangeError('maxPending must be 1..4096')
    if (embedder) {
      const actual = memory.securityStatus().embeddingSpace
      const expected = embedder.space
      if (!actual || !expected || ['model', 'revision', 'dimensions', 'normalization', 'metric'].some(key => actual[key] !== expected[key])) {
        throw new Error('Embedder does not match the memory store embedding space')
      }
    }
    this.memory = memory
    this.embedder = embedder
    this.rerank = rerank
    this.maxPending = maxPending
    this.pending = 0
    this.tail = Promise.resolve()
  }

  enqueue(operation) {
    if (this.pending >= this.maxPending) return Promise.reject(new Error('Document memory queue is full; await pending operations before submitting more'))
    this.pending += 1
    const result = this.tail.then(operation)
    this.tail = result.then(() => undefined, () => undefined)
    return result.finally(() => { this.pending -= 1 })
  }

  memorize(input) {
    return this.enqueue(async () => {
      const chunks = await this.memory.chunkDocument(input)
      const embeddings = this.embedder ? await this.embedder.encodeDocuments(chunks.map(chunk => chunk.text)) : input.embeddings
      return this.memory.memorizeDocument({ ...input, embeddings: embeddings?.map(row => Array.from(row)) })
    })
  }

  async ingest(documents) {
    let count = 0
    for await (const document of documents) {
      await this.memorize(document)
      count += 1
    }
    return count
  }

  search(text, options = {}) {
    return this.enqueue(async () => {
      const k = options.k ?? 10
      const candidateLimit = options.candidateLimit ?? Math.max(k, 100)
      if (!Number.isInteger(k) || k < 1 || k > 1000 || !Number.isInteger(candidateLimit) || candidateLimit < k || candidateLimit > 1000) {
        throw new RangeError('require 1 <= k <= candidateLimit <= 1000')
      }
      const embedding = this.embedder ? Array.from(await this.embedder.encodeQuery(text)) : options.embedding
      const candidates = await this.memory.searchDocuments(text, { ...options, embedding, k: this.rerank ? candidateLimit : k, candidateLimit })
      if (!this.rerank || candidates.length === 0) return candidates
      if (candidates.some(candidate => candidate.text == null)) throw new Error('Reranking requires documents ingested with storeText: true')
      const scores = await this.rerank(text, candidates.map(candidate => candidate.text))
      if (scores.length !== candidates.length || !scores.every(Number.isFinite)) throw new Error('Reranker must return one finite score per candidate')
      return candidates.map((candidate, index) => ({ ...candidate, score: scores[index] }))
        .sort((a, b) => b.score - a.score || a.id.localeCompare(b.id)).slice(0, k)
    })
  }

  delete(id) { return this.enqueue(() => this.memory.deleteDocument(id)) }
  flush() { return this.enqueue(() => this.memory.flush()) }
  maintainIndices() { return this.enqueue(() => this.memory.maintainIndices()) }
}

module.exports = { DocumentMemory, createLocalEmbedder, createLocalReranker }
