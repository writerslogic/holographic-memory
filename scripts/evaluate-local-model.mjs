import { readFile, writeFile, mkdtemp, rm } from 'node:fs/promises'
import { tmpdir, arch, platform } from 'node:os'
import { join } from 'node:path'
import { createHash } from 'node:crypto'
import { spawnSync } from 'node:child_process'
import { fileURLToPath } from 'node:url'
import { HolographicMemorySystem } from '../index.mjs'
import { createLocalEmbedder, createLocalReranker, DocumentMemory } from '../semantic.js'

const args = process.argv.slice(2)
function option(name, fallback) {
  const index = args.indexOf(name)
  return index < 0 ? fallback : args[index + 1]
}
// Dispose native mappings in a child process before removing the temporary store,
// including on Windows where mapped files cannot be removed while open.
if (!args.includes('--worker')) {
  const temporary = await mkdtemp(join(tmpdir(), 'hms-local-eval-'))
  let status = 1
  try {
    const result = spawnSync(process.execPath, [fileURLToPath(import.meta.url), ...args, '--worker', '--temporary-dir', temporary], { stdio: 'inherit' })
    if (result.error) throw result.error
    status = result.status ?? 1
  } finally { await rm(temporary, { recursive: true, force: true }) }
  process.exit(status)
}
const modelPath = option('--model-dir')
const revision = option('--revision')
if (!modelPath || !revision) throw new Error('Usage: node scripts/evaluate-local-model.mjs --model-dir PATH --revision REVISION [--model-id NAME] [--output PATH] [--reranker-dir PATH]')
const input = await readFile(option('--dataset', 'tests/fixtures/relevance.json'), 'utf8')
const dataset = JSON.parse(input)
const embedder = await createLocalEmbedder({ modelPath, modelId: option('--model-id', 'Xenova/all-MiniLM-L6-v2'), revision, dtype: option('--dtype', 'q8') })
const rerank = option('--reranker-dir') ? await createLocalReranker({ modelPath: option('--reranker-dir'), dtype: option('--dtype', 'q8') }) : undefined
if (rerank && !option('--reranker-revision')) throw new Error('--reranker-revision is required for a reproducible report')
const directory = option('--temporary-dir')
try {
  const hms = new HolographicMemorySystem(16384, directory, {
    embeddingModel: embedder.space.model, embeddingRevision: embedder.space.revision, embeddingDimensions: embedder.space.dimensions,
  })
  const memory = new DocumentMemory(hms, { embedder })
  const rerankedMemory = rerank ? new DocumentMemory(hms, { embedder, rerank }) : undefined
  await memory.ingest(dataset.documents)
  const metrics = {}
  for (const [name, search] of [
    ['lexical', text => hms.searchDocuments(text, { k: 5 })],
    ['hybrid', text => memory.search(text, { k: 5 })],
    ...(rerankedMemory ? [['hybrid_reranked', text => rerankedMemory.search(text, { k: 5 })]] : []),
  ]) {
    const latency = []
    let hits1 = 0
    let hits5 = 0
    let reciprocalRank = 0
    for (const query of dataset.queries) {
      const started = performance.now()
      const results = await search(query.text)
      latency.push(performance.now() - started)
      const ids = [...new Set(results.map(hit => hit.documentId))]
      const rank = ids.findIndex(id => query.relevantIds.includes(id))
      hits1 += Number(rank === 0)
      hits5 += Number(rank >= 0 && rank < 5)
      reciprocalRank += rank >= 0 ? 1 / (rank + 1) : 0
    }
    latency.sort((a, b) => a - b)
    metrics[name] = { hit_rate_at_1: hits1 / dataset.queries.length, hit_rate_at_5: hits5 / dataset.queries.length,
      mrr_at_5: reciprocalRank / dataset.queries.length, p95_latency_ms: latency[Math.round((latency.length - 1) * 0.95)] }
  }
  const denseDocuments = await embedder.encodeDocuments(dataset.documents.map(document => document.text))
  const projectionStore = new HolographicMemorySystem(16384, join(directory, 'projection'), {
    embeddingModel: embedder.space.model, embeddingRevision: embedder.space.revision, embeddingDimensions: embedder.space.dimensions,
  })
  for (let i = 0; i < denseDocuments.length; i++) await projectionStore.memorizeVector(dataset.documents[i].id, Float32Array.from(denseDocuments[i]))
  const projectionK = Math.min(5, denseDocuments.length)
  let neighborRecall = 0
  for (const query of dataset.queries) {
    const denseQuery = await embedder.encodeQuery(query.text)
    const exact = denseDocuments.map((vector, i) => ({ id: dataset.documents[i].id, score: vector.reduce((sum, value, j) => sum + value * denseQuery[j], 0) }))
      .sort((a, b) => b.score - a.score || a.id.localeCompare(b.id)).slice(0, projectionK)
    const sparse = await projectionStore.queryVector(Float32Array.from(denseQuery), projectionK)
    const neighbors = new Set(exact.map(hit => hit.id))
    neighborRecall += sparse.filter(hit => neighbors.has(hit.id)).length / projectionK
  }
  const report = { dataset_sha256: createHash('sha256').update(input).digest('hex'),
    model: embedder.space, documents: dataset.documents.length, queries: dataset.queries.length,
    model_artifact_revision: revision, dtype: option('--dtype', 'q8'),
    architecture: arch(), platform: platform(), node: process.version,
    reranker: rerank ? { model: option('--reranker-id', 'Xenova/ms-marco-MiniLM-L-6-v2'), revision: option('--reranker-revision') } : null,
    metrics, projection: { dimensions: 16384, encoder: 'signed-projection-v2', k: projectionK, dense_neighbor_recall_at_k: neighborRecall / dataset.queries.length },
    scope: 'Small curated regression set; not evidence of general retrieval quality or capacity.' }
  const output = `${JSON.stringify(report, null, 2)}\n`
  if (option('--output')) await writeFile(option('--output'), output)
  console.log(output)
  if (metrics.hybrid.hit_rate_at_5 < 0.9) throw new Error('Local-model relevance regression: hit rate at 5 is below 0.9')
} finally {
  await embedder.dispose()
  await rerank?.dispose?.()
}
