// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0
'use strict'

const assert = require('node:assert/strict')
const { spawnSync } = require('node:child_process')
const fs = require('node:fs')
const os = require('node:os')
const path = require('node:path')
const { before, test } = require('node:test')

const root = path.resolve(__dirname, '..')
const javascript = path.join(root, 'integrations/organizational-knowledge/example/cli.cjs')
let rust

function run(command, args, success = true) {
  const result = spawnSync(command, args, { cwd: root, encoding: 'utf8', maxBuffer: 16 * 1024 * 1024 })
  assert.ifError(result.error)
  if (success) assert.equal(result.status, 0, result.stderr.slice(-4000))
  else assert.notEqual(result.status, 0, 'invalid input succeeded')
  return result.stdout
}

before(() => {
  run('cargo', ['build', '--locked', '--offline', '--example', 'organizational_knowledge'])
  const metadata = JSON.parse(run('cargo', ['metadata', '--locked', '--offline', '--no-deps', '--format-version', '1']))
  rust = path.join(metadata.target_directory, 'debug/examples/organizational_knowledge' + (process.platform === 'win32' ? '.exe' : ''))
})

function fixture(content = 'zebraquartz decision', revision = 'revision-1') {
  return {
    id: 'meeting/"Café"', source_system: 'Zoom', identifier: revision,
    organizational_unit: ['EGWG'], source_url: 'https://example.org/meeting',
    artifact_id: 'sha256:example', raw_files: ['meeting.vtt'], content,
  }
}

test('Rust CLI rejects malformed imports and previews deletion without creating a store', t => {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), 'hms-organizational-cli-'))
  t.after(() => fs.rmSync(directory, { recursive: true, force: true }))
  const store = path.join(directory, 'store')
  const file = path.join(directory, 'document.json')
  for (const value of [
    { ...fixture(), content: null },
    fixture(' \n\t'),
    fixture('x'.repeat(64 * 1024 + 1)),
    { ...fixture(), hms_namespace: 'upstream-value' },
  ]) {
    fs.writeFileSync(file, JSON.stringify(value))
    run(rust, ['import', store, 'toip', file], false)
    assert.equal(fs.existsSync(store), false)
  }
  const preview = run(rust, ['forget', store, 'toip', 'Zoom', fixture().id])
  assert.match(preview, /Dry run:/)
  assert.equal(fs.existsSync(store), false)
  run(rust, ['forget', store, 'toip', 'Zoom', 'invalid\n'], false)
  assert.equal(fs.existsSync(store), false)
})

test('Rust and JavaScript share identities, exact source spans, replacement and deletion', t => {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), 'hms-organizational-interop-'))
  t.after(() => fs.rmSync(directory, { recursive: true, force: true }))
  const store = path.join(directory, 'store')
  const file = path.join(directory, 'document.json')
  const original = ` \nCafé: ${'naïve '.repeat(500)}\nzebraquartz decision.  `
  const document = fixture(original)
  const write = value => fs.writeFileSync(file, JSON.stringify(value))
  const js = args => run(process.execPath, [javascript, ...args])
  const searchRust = (namespace, query) => JSON.parse(run(rust, ['search', store, namespace, query]))
  const searchJs = (namespace, query) => JSON.parse(js(['search', store, namespace, query]))

  write(document)
  run(rust, ['import', store, 'toip', file])
  const [hit] = searchJs('toip', 'zebraquartz')
  assert.deepEqual(hit.identity, { namespace: 'toip', sourceSystem: 'Zoom', id: document.id })
  assert.equal(hit.version, document.identifier)
  assert.equal(hit.sourceUri, document.source_url)
  assert.ok(hit.startByte > 0)
  assert.equal(hit.text, Buffer.from(original).subarray(hit.startByte, hit.endByte).toString('utf8'))
  const metadata = { ...document }
  delete metadata.content
  assert.deepEqual(hit.metadata, { ...metadata, hms_namespace: 'toip' })

  write({ ...fixture(), source_system: 'Confluence' })
  js(['import', store, 'toip', file])
  write(fixture('zebraquartz another organization'))
  js(['import', store, 'other', file])
  assert.equal(searchRust('toip', 'zebraquartz').length, 2)

  write(fixture('amberquartz updated decision', 'revision-2'))
  js(['import', store, 'toip', file])
  assert.equal(searchRust('toip', 'zebraquartz').length, 1)
  assert.equal(searchRust('toip', 'amberquartz')[0].version, 'revision-2')
  run(rust, ['forget', store, 'toip', 'Zoom', document.id])
  assert.equal(searchJs('toip', 'amberquartz').length, 1)
  assert.match(run(rust, ['forget', store, 'toip', 'Zoom', document.id, '--apply']), /Removed: true/)
  assert.equal(searchJs('toip', 'amberquartz').length, 0)
  assert.equal(searchJs('toip', 'zebraquartz').length, 1)
  assert.equal(searchJs('other', 'zebraquartz').length, 1)
  assert.match(run(rust, ['forget', store, 'toip', 'Zoom', document.id, '--apply']), /Removed: false/)
  js(['import', store, 'toip', file])
  assert.equal(searchRust('toip', 'amberquartz').length, 1)
})
