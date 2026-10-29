const assert = require('node:assert/strict')
const { test } = require('node:test')
const { mkdtempSync, rmSync } = require('node:fs')
const { tmpdir } = require('node:os')
const { join } = require('node:path')
const { HolographicMemorySystem } = require('../../index.js')

test('listDocuments pages through stored documents in id order without their text', async () => {
  const directory = mkdtempSync(join(tmpdir(), 'hms-list-'))
  try {
    const hms = new HolographicMemorySystem(4096, directory)
    const ids = ['a', 'a-b', 'b', 'c', 'é']
    for (const id of ids) {
      await hms.memorizeDocument({ id, text: `listing text for ${id}`, version: `v-${id}`, sourceUri: `https://example.org/${id}`, metadata: { kind: 'note' } })
    }

    const seen = []
    let after
    for (;;) {
      const page = await hms.listDocuments(after, 2)
      if (page.length === 0) break
      assert.ok(page.length <= 2)
      seen.push(...page)
      after = page.at(-1).id
    }
    assert.deepEqual(seen.map(document => document.id), ids)
    assert.deepEqual(seen[1], { id: 'a-b', version: 'v-a-b', sourceUri: 'https://example.org/a-b', metadata: { kind: 'note' }, chunkCount: 1 })
    assert.equal('text' in seen[0], false)

    // The cursor is exclusive, and an id that merely extends it still follows it.
    assert.deepEqual((await hms.listDocuments('a', 1)).map(document => document.id), ['a-b'])
    assert.deepEqual(await hms.listDocuments('é'), [])
    assert.equal((await hms.listDocuments()).length, ids.length)

    await hms.deleteDocument('a-b')
    assert.deepEqual((await hms.listDocuments(undefined, 10)).map(document => document.id), ['a', 'b', 'c', 'é'])

    // Out-of-range values must be rejected, not wrapped into range (2 ** 32 + 2 would become 2).
    for (const limit of [0, 1001, -1, 1.5, 2 ** 32 + 2, NaN, Infinity]) {
      await assert.rejects(hms.listDocuments(undefined, limit), /limit must be an integer in 1\.\.=1000/, String(limit))
    }
    assert.equal((await hms.listDocuments(undefined, 1000)).length, 4)
  } finally {
    rmSync(directory, { recursive: true, force: true })
  }
})
