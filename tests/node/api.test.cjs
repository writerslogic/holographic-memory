const assert = require('node:assert/strict')
const { test } = require('node:test')
const { mkdtempSync, rmSync } = require('node:fs')
const { tmpdir } = require('node:os')
const { join } = require('node:path')
const { spawnSync } = require('node:child_process')

function run(source, directory) {
  const result = spawnSync(process.execPath, ['-e', source, directory], { cwd: process.cwd(), encoding: 'utf8', timeout: 60000 })
  assert.equal(result.status, 0, result.stderr || result.stdout)
}

test('governance preserves independently owned triples through async maintenance and restart', () => {
  const directory = mkdtempSync(join(tmpdir(), 'warm-node-governance-'))
  try {
    run(`
      const assert = require('node:assert/strict');
      const { WarmMemorySystem } = require('./index.js');
      (async () => {
        const warm = new WarmMemorySystem(4096, process.argv[1], { meaningEnabled: true });
        await warm.memorizeTriplet('owner-a', 'mira', 'works_at', 'acme');
        await warm.memorizeTriplet('owner-b', 'mira', 'works_at', 'acme');
        const report = await warm.governMemoryAsync();
        assert.equal(report.compositesMerged, 0);
        assert.equal(report.idfRefreshed, true);
        assert.equal(warm.vectorCount, 2);
        assert.ok((await warm.multiHopQuery('mira', ['works_at'])).some(hit => hit.entityId === 'acme'));
        await warm.compact();
        await warm.flush();
      })().catch(error => { console.error(error); process.exit(1); });
    `, directory)
    run(`
      const assert = require('node:assert/strict');
      const { WarmMemorySystem } = require('./index.js');
      (async () => {
        const warm = new WarmMemorySystem(4096, process.argv[1], { meaningEnabled: true });
        assert.equal(warm.vectorCount, 2);
        assert.equal(warm.governMemory().compositesMerged, 0);
        assert.ok((await warm.multiHopQuery('mira', ['works_at'])).some(hit => hit.entityId === 'acme'));
        await warm.flush();
      })().catch(error => { console.error(error); process.exit(1); });
    `, directory)
  } finally {
    rmSync(directory, { recursive: true, force: true })
  }
})

test('published APIs preserve structured knowledge and document state across processes', () => {
  const directory = mkdtempSync(join(tmpdir(), 'warm-node-'))
  try {
    run(`
      const assert = require('node:assert/strict');
      const { WarmMemorySystem } = require('./index.js');
      const { DocumentMemory } = require('./semantic.js');
      (async () => {
        const warm = new WarmMemorySystem(4096, process.argv[1], { meaningEnabled: true });
        assert.equal(warm.securityStatus().encryptionActive, false);
        await warm.memorizeTriplet('t1', 'paris', 'capital_of', 'france');
        await warm.memorizeTriplet('t2', 'john', 'father', 'mark');
        await warm.memorizeTriplet('t3', 'mark', 'father', 'bob');
        assert.equal((await warm.structuralQuery(['paris'], ['capital_of'], 'object'))[0].entityId, 'france');
        assert.equal((await warm.multiHopQuery('john', ['father', 'father']))[0].entityId, 'bob');
        const docs = new DocumentMemory(warm);
        await docs.ingest([
          { id: 'guide', text: 'Restore backups from verified snapshots.', sourceUri: 'guide.md', metadata: { project: 'alpha' } },
          { id: 'temporary', text: 'Temporary indexing advice.' },
        ]);
        const [hit] = await docs.search('restoring backups', { filter: { project: 'alpha' } });
        assert.equal(hit.documentId, 'guide');
        assert.equal(hit.sourceUri, 'guide.md');
        assert.ok(hit.text.includes('backups'));
        await docs.delete('temporary');
        const file = require('node:path').join(process.argv[1], 'input.txt');
        require('node:fs').writeFileSync(file, 'File ingestion keeps source offsets. '.repeat(100));
        await warm.memorizeFile('file', file);
        const passages = await warm.searchDocuments('ingestion offsets', { documentIds: ['file'] });
        assert.ok(passages.length > 1);
        assert.equal(passages[0].sourceUri, file);
        await warm.deleteDocument('file');
        await warm.compact();
        await docs.flush();
        await assert.rejects(warm.memorizeVector('bad', new Float32Array([NaN])));
      })().catch(e => { console.error(e); process.exitCode = 1; });
    `, directory)
    run(`
      const assert = require('node:assert/strict');
      (async () => {
        const { WarmMemorySystem } = await import('./index.mjs');
        const { DocumentMemory } = await import('./semantic.js');
        const warm = new WarmMemorySystem(4096, process.argv[1], { meaningEnabled: true });
        const docs = new DocumentMemory(warm);
        assert.equal((await docs.search('backups'))[0].documentId, 'guide');
        assert.equal((await docs.search('temporary')).length, 0);
        assert.equal((await warm.structuralQuery(['paris'], ['capital_of'], 'object'))[0].entityId, 'france');
      })().catch(e => { console.error(e); process.exitCode = 1; });
    `, directory)
  } finally { rmSync(directory, { recursive: true, force: true }) }
})

test('document queue bounds work and recovers after a rejected operation', async () => {
  const { DocumentMemory } = require('../../semantic.js')
  let release
  const blocked = new Promise(resolve => { release = resolve })
  const memory = new DocumentMemory({ flush: () => blocked }, { maxPending: 1 })
  const pending = memory.flush()
  await assert.rejects(memory.flush(), /queue is full/)
  release()
  await pending
  await memory.flush()
  const failing = new DocumentMemory({ deleteDocument: async () => { throw new Error('write failed') }, flush: async () => {} })
  await assert.rejects(failing.delete('id'), /write failed/)
  await failing.flush()
})

test('native document batches and exact index survive invalid updates and restart', () => {
  const directory = mkdtempSync(join(tmpdir(), 'warm-document-index-'))
  const setup = `
    const assert = require('node:assert/strict');
    const { WarmMemorySystem } = require('./index.js');
    const config = { embeddingModel: 'fixture', embeddingRevision: '1', embeddingDimensions: 2 };
    const options = { embedding: [1, 0], k: 5, candidateLimit: 16 };
  `
  try {
    run(`${setup}
      (async () => {
        const warm = new WarmMemorySystem(1024, process.argv[1], config);
        const inputs = Array.from({ length: 128 }, (_, i) => ({ id: 'doc-' + i, text: 'needle ' + i, embeddings: [[Math.cos(i / 100), Math.sin(i / 100)]] }));
        assert.deepEqual(await warm.memorizeDocuments(inputs), Array(128).fill(1));
        const expected = await warm.searchDocuments('', options);
        await warm.trainDocumentIndex();
        assert.equal(warm.documentIndexTrained, true);
        assert.deepEqual(await warm.searchDocuments('', options), expected);
        await assert.rejects(warm.memorizeDocuments([{ ...inputs[0], text: 'uncommitted' }, { id: 'bad', text: 'invalid', embeddings: [[0, 0]] }]));
        assert.deepEqual(await warm.searchDocuments('', options), expected);
        await warm.deleteDocument('doc-0');
        assert.equal(warm.documentIndexTrained, false);
        await warm.compact();
        await warm.flush();
      })().catch(e => { console.error(e); process.exitCode = 1; });
    `, directory)
    run(`${setup}
      (async () => {
        const warm = new WarmMemorySystem(1024, process.argv[1], config);
        assert.equal(warm.documentIndexTrained, false);
        const expected = await warm.searchDocuments('', options);
        assert.equal(expected[0].documentId, 'doc-1');
        assert.ok(expected.every(hit => hit.documentId !== 'doc-0'));
        await warm.trainDocumentIndex();
        assert.equal(warm.documentIndexTrained, false);
        assert.deepEqual(await warm.searchDocuments('', options), expected);
      })().catch(e => { console.error(e); process.exitCode = 1; });
    `, directory)
  } finally { rmSync(directory, { recursive: true, force: true }) }
})

test('a requested security capability is active or construction fails', () => {
  const directory = mkdtempSync(join(tmpdir(), 'warm-security-'))
  try {
    run(`
      const assert = require('node:assert/strict');
      const { WarmMemorySystem } = require('./index.js');
      (async () => {
        const warm = new WarmMemorySystem(4096, process.argv[1], {
          encryptionEnabled: true, encryptionPassphrase: 'test-only-passphrase',
        });
        assert.equal(warm.securityStatus().encryptionActive, true);
        await warm.memorizeDocument({ id: 'private', text: 'Private source document.' });
        await warm.flush();
      })().catch(e => { console.error(e); process.exitCode = 1; });
    `, directory)
    run(`
      const assert = require('node:assert/strict');
      const { WarmMemorySystem } = require('./index.js');
      assert.throws(() => new WarmMemorySystem(4096, process.argv[1]));
      assert.throws(() => new WarmMemorySystem(4096, process.argv[1], {
        encryptionEnabled: true, encryptionPassphrase: 'wrong-passphrase',
      }));
    `, directory)
  } finally { rmSync(directory, { recursive: true, force: true }) }
})
