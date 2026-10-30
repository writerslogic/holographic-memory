import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { HashingEmbedder, HmsStore } from '../../../integrations/js/packages/core/dist/index.js';

const path = mkdtempSync(join(tmpdir(), 'hms-audit-identity-'));
const embedder = new HashingEmbedder(384);
const store = new HmsStore({
  path, dimensions: 1024, embedder,
  hmsConfig: { embeddingModel: 'different-model', embeddingRevision: 'different-revision' },
});
try {
  console.log(JSON.stringify({ actual: embedder.space, recorded: store.memory.securityStatus().embeddingSpace }));
  await store.remember([{ id: 'one', text: 'alpha beta gamma' }]);
  const hits = await store.recall('alpha beta gamma', { k: 1 });
  console.log(JSON.stringify({ acceptedWrite: true, recalled: hits.map(hit => hit.id) }));
} finally {
  await store.close();
  rmSync(path, { recursive: true });
}

for (const [name, text, options] of [
  ['boundary_whitespace', '  alpha  ', {}],
  ['nonoverlap_chunks', 'alpha beta', { chunkWords: 1, overlapWords: 0 }],
]) {
  const dir = mkdtempSync(join(tmpdir(), 'hms-audit-text-'));
  const candidate = new HmsStore({ path: dir, dimensions: 1024, ...options });
  try {
    await candidate.remember([{ id: 'one', text }]);
    const [record] = await candidate.get(['one']);
    console.log(JSON.stringify({ name, input: text, output: record.text, equal: text === record.text }));
  } finally {
    await candidate.close();
    rmSync(dir, { recursive: true });
  }
}
