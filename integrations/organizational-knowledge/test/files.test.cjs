'use strict'

const assert = require('node:assert/strict')
const fs = require('node:fs')
const os = require('node:os')
const path = require('node:path')
const { test } = require('node:test')
const { KnowledgeError, readCuratedFiles } = require('../index.cjs')

const MAX_FILE_BYTES = 16 * 1024 * 1024

function withDirectory(run) {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), 'ok-files-'))
  return Promise.resolve(run(directory)).finally(() => fs.rmSync(directory, { recursive: true, force: true }))
}

const failsFor = (promise, code, file) =>
  assert.rejects(promise, error => error instanceof KnowledgeError && error.code === code && (!file || (error.source === file && error.message.includes(file))))

test('files that cannot be read exactly are rejected by name: bad UTF-8, bad JSON, and the 16 MiB limit', () =>
  withDirectory(async directory => {
    const write = (name, data) => {
      const file = path.join(directory, name)
      fs.writeFileSync(file, data)
      return file
    }
    // 0xFF is not UTF-8; lossy decoding would turn it into U+FFFD inside the original text.
    await failsFor(readCuratedFiles([write('utf8.json', Buffer.concat([Buffer.from('{"content":"'), Buffer.from([0xff]), Buffer.from('"}')]))]), 'invalid_document', path.join(directory, 'utf8.json'))
    await failsFor(readCuratedFiles([write('broken.json', '{ not json')]), 'invalid_document', path.join(directory, 'broken.json'))

    const sized = size => {
      const wrapper = Buffer.byteLength('{"a":""}')
      return `{"a":"${'x'.repeat(size - wrapper)}"}`
    }
    for (const size of [MAX_FILE_BYTES - 1, MAX_FILE_BYTES]) {
      const [read] = await readCuratedFiles([write(`size-${size}.json`, sized(size))])
      assert.equal(read.document.a.length, size - Buffer.byteLength('{"a":""}'))
    }
    await failsFor(readCuratedFiles([write('too-big.json', sized(MAX_FILE_BYTES + 1))]), 'invalid_document', path.join(directory, 'too-big.json'))
  }))

test('a directory yields its .json files in name order and rejects empty directories and non-regular .json entries', () =>
  withDirectory(async directory => {
    fs.writeFileSync(path.join(directory, 'b.json'), '{"n":2}')
    fs.writeFileSync(path.join(directory, 'a.json'), '{"n":1}')
    fs.writeFileSync(path.join(directory, 'notes.txt'), 'ignored')
    fs.mkdirSync(path.join(directory, 'nested'))
    fs.writeFileSync(path.join(directory, 'nested', 'c.json'), '{"n":3}')
    const read = await readCuratedFiles([directory])
    assert.deepEqual(read.map(item => [path.basename(item.source), item.document.n]), [['a.json', 1], ['b.json', 2]])

    const empty = path.join(directory, 'empty')
    fs.mkdirSync(empty)
    await failsFor(readCuratedFiles([empty]), 'invalid_request')
    const linked = path.join(directory, 'linked')
    fs.mkdirSync(linked)
    fs.symlinkSync(path.join(directory, 'a.json'), path.join(linked, 'link.json'))
    await failsFor(readCuratedFiles([linked]), 'invalid_request', path.join(linked, 'link.json'))
  }))
