// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0
const assert = require('node:assert/strict')
const { test } = require('node:test')
const { mkdtempSync, rmSync } = require('node:fs')
const { tmpdir } = require('node:os')
const { join } = require('node:path')
const { spawnSync } = require('node:child_process')

function run(source, directory) {
  const result = spawnSync(process.execPath, ['-e', source, directory], { cwd: process.cwd(), encoding: 'utf8', timeout: 120000 })
  assert.equal(result.status, 0, result.stderr || result.stdout)
}

test('frozen native lifecycle matches the Rust and HTTP contract after each restart', () => {
  const directory = mkdtempSync(join(tmpdir(), 'hms-node-lifecycle-'))
  try {
    for (let phase = 0; phase < 3; phase++) {
      run(`
        const assert = require('node:assert/strict');
        const { HolographicMemorySystem } = require('./index.js');
        const fixture = require('./tests/fixtures/native_memory_lifecycle_v1.json');
        (async () => {
          const hms = new HolographicMemorySystem(4096, process.argv[1]);
          for (const [index, step] of fixture.phases[${phase}].entries()) {
            if (step.error) {
              await assert.rejects(hms.memoryCall(step.op, step.request), error => error.message.includes('[' + step.error + ']'));
              continue;
            }
            const result = await hms.memoryCall(step.op, step.request);
            for (const [pointer, expected] of Object.entries(step.expect)) {
              const actual = pointer.slice(1).split('/').reduce((value, key) => value?.[key.replace(/~1/g, '/').replace(/~0/g, '~')], result);
              assert.deepEqual(actual, expected, 'phase ${phase} step ' + index + ': ' + pointer);
            }
          }
          await hms.flush();
        })().catch(error => { console.error(error); process.exit(1); });
      `, directory)
    }
  } finally {
    rmSync(directory, { recursive: true, force: true })
  }
})

test('memory API keeps history, answers as-of queries, explains, forgets and survives a restart', () => {
  const directory = mkdtempSync(join(tmpdir(), 'hms-node-memory-'))
  try {
    run(`
      const assert = require('node:assert/strict');
      const { HolographicMemorySystem } = require('./index.js');
      const { Memory, MemoryError } = require('./memory.js');
      (async () => {
        const hms = new HolographicMemorySystem(4096, process.argv[1]);
        const memory = new Memory(hms, { supersession: 'latest-by-subject' });
        const day = n => new Date(Date.UTC(2025, 0, n));

        const paris = await memory.remember('Alice lives in Paris', { id: 'c1', subject: 'alice/city', validFrom: day(1), entities: ['Alice'] });
        assert.equal(paris.created, true);
        const berlin = await memory.remember({ id: 'c2', text: 'Alice lives in Berlin', subject: 'alice/city', validFrom: day(10) });
        assert.deepEqual(berlin.supersedes, ['c1']);
        assert.equal((await memory.remember('Alice lives in Paris', { id: 'c1', subject: 'alice/city', validFrom: day(1), entities: ['Alice'] })).created, false);

        assert.deepEqual((await memory.recall('where does Alice live')).map(r => r.id), ['c2']);
        assert.deepEqual((await memory.recall('where does Alice live', { asOf: day(5) })).map(r => r.id), ['c1']);
        const all = await memory.recall('Alice', { history: 'all' });
        assert.equal(all.length, 2);
        assert.equal(all.find(r => r.id === 'c1').status, 'superseded');
        assert.equal(all.find(r => r.id === 'c1').validFrom, day(1).getTime());

        const [top] = await memory.recall('Alice Berlin', { explain: true });
        assert.deepEqual(top.explanation.matchedTerms, ['alice', 'berlin']);
        assert.equal(top.explanation.chain[0].id, 'c1');
        assert.equal((await memory.explain('Alice Paris', 'c1', { asOf: day(5) })).status, 'current');

        assert.deepEqual((await memory.history('c2')).map(f => f.id), ['c1', 'c2']);
        assert.equal((await memory.get('c1')).supersededBy[0], 'c2');
        assert.equal(await memory.get('absent'), null);

        await assert.rejects(memory.remember('different', { id: 'c1' }), error => error instanceof MemoryError && error.code === 'conflict');
        await assert.rejects(memory.history('absent'), error => error.code === 'not_found');
        await assert.rejects(memory.recall('x', { k: 0 }), error => error.code === 'invalid_request');
        await assert.rejects(memory.recall('x', { nonsense: 1 }), error => error.code === 'invalid_request');
        await assert.rejects(memory.forget({}), TypeError);
        await assert.rejects(memory.remember('t', { validFrom: 'yesterday' }), TypeError);

        // Facts are documents: the low-level surface still sees them.
        assert.equal((await hms.searchDocuments('Berlin'))[0].documentId, 'c2');

        assert.deepEqual((await memory.forget({ entity: 'Alice' }, { dryRun: true })).forgotten, ['c1']);
        await hms.flush();
      })().catch(error => { console.error(error); process.exit(1); });
    `, directory)
    run(`
      const assert = require('node:assert/strict');
      const { HolographicMemorySystem } = require('./index.js');
      const { Memory } = require('./memory.js');
      (async () => {
        const hms = new HolographicMemorySystem(4096, process.argv[1]);
        const memory = new Memory(hms, { supersession: 'latest-by-subject' });
        assert.deepEqual((await memory.recall('Alice lives')).map(r => r.id), ['c2']);
        assert.equal((await memory.history('c1')).length, 2);
        assert.deepEqual((await memory.forget({ ids: ['c2'] }, { includeChain: true })).forgotten, ['c1', 'c2']);
        assert.deepEqual(await memory.recall('Alice', { history: 'all' }), []);
        assert.deepEqual((await memory.applyForgetting()).forgotten, []);
        assert.deepEqual((await memory.consolidate()).created, []);
        await hms.flush();
      })().catch(error => { console.error(error); process.exit(1); });
    `, directory)
  } finally {
    rmSync(directory, { recursive: true, force: true })
  }
})
