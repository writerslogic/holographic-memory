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

test('published APIs preserve structured knowledge and document state across processes', () => {
  const directory = mkdtempSync(join(tmpdir(), 'hms-node-'))
  try {
    run(`
      const assert = require('node:assert/strict');
      const { HolographicMemorySystem } = require('./index.js');
      const { DocumentMemory } = require('./semantic.js');
      (async () => {
        const hms = new HolographicMemorySystem(4096, process.argv[1], { meaningEnabled: true });
        assert.equal(hms.securityStatus().encryptionActive, false);
        await hms.memorizeTriplet('t1', 'paris', 'capital_of', 'france');
        await hms.memorizeTriplet('t2', 'john', 'father', 'mark');
        await hms.memorizeTriplet('t3', 'mark', 'father', 'bob');
        assert.equal((await hms.structuralQuery(['paris'], ['capital_of'], 'object'))[0].entityId, 'france');
        assert.equal((await hms.multiHopQuery('john', ['father', 'father']))[0].entityId, 'bob');
        const docs = new DocumentMemory(hms);
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
        await hms.memorizeFile('file', file);
        const passages = await hms.searchDocuments('ingestion offsets', { documentIds: ['file'] });
        assert.ok(passages.length > 1);
        assert.equal(passages[0].sourceUri, file);
        await hms.deleteDocument('file');
        await hms.compact();
        await docs.flush();
        await assert.rejects(hms.memorizeVector('bad', new Float32Array([NaN])));
      })().catch(e => { console.error(e); process.exitCode = 1; });
    `, directory)
    run(`
      const assert = require('node:assert/strict');
      (async () => {
        const { HolographicMemorySystem } = await import('./index.mjs');
        const { DocumentMemory } = await import('./semantic.js');
        const hms = new HolographicMemorySystem(4096, process.argv[1], { meaningEnabled: true });
        const docs = new DocumentMemory(hms);
        assert.equal((await docs.search('backups'))[0].documentId, 'guide');
        assert.equal((await docs.search('temporary')).length, 0);
        assert.equal((await hms.structuralQuery(['paris'], ['capital_of'], 'object'))[0].entityId, 'france');
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

test('a requested security capability is active or construction fails', () => {
  const directory = mkdtempSync(join(tmpdir(), 'hms-security-'))
  try {
    run(`
      const assert = require('node:assert/strict');
      const { HolographicMemorySystem } = require('./index.js');
      (async () => {
        const hms = new HolographicMemorySystem(4096, process.argv[1], {
          encryptionEnabled: true, encryptionPassphrase: 'test-only-passphrase',
        });
        assert.equal(hms.securityStatus().encryptionActive, true);
        await hms.memorizeDocument({ id: 'private', text: 'Private source document.' });
        await hms.flush();
      })().catch(e => { console.error(e); process.exitCode = 1; });
    `, directory)
    run(`
      const assert = require('node:assert/strict');
      const { HolographicMemorySystem } = require('./index.js');
      assert.throws(() => new HolographicMemorySystem(4096, process.argv[1]));
      assert.throws(() => new HolographicMemorySystem(4096, process.argv[1], {
        encryptionEnabled: true, encryptionPassphrase: 'wrong-passphrase',
      }));
    `, directory)
  } finally { rmSync(directory, { recursive: true, force: true }) }
})
