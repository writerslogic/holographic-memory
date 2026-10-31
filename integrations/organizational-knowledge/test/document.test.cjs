'use strict'

const assert = require('node:assert/strict')
const { test } = require('node:test')
const { KnowledgeError, documentId, prepareImport } = require('../index.cjs')
const { adaptDocument } = require('../lib/document.cjs')

const curated = (overrides = {}) => ({
  id: 'zoom-toip-transcript-GMT1',
  source_system: 'Zoom',
  identifier: 'urn:said:Eaaa',
  content: 'Plain text.',
  organizational_unit: ['EGWG'],
  source_identifier: 'GMT1',
  source_url: 'https://example.org/z',
  title: 'Meeting',
  ...overrides,
})
const adapt = overrides => adaptDocument('toip', curated(overrides), 'doc.json')
const rejects = (overrides, field) =>
  assert.throws(() => adapt(overrides), error => error instanceof KnowledgeError && error.code === 'invalid_document' && error.field === field && error.source === 'doc.json')

test('identity is a collision-safe tuple that ignores content revision and source_identifier', () => {
  assert.equal(documentId('toip', 'Zoom', 'meeting-1'), '["toip","Zoom","meeting-1"]')
  assert.notEqual(documentId('a/b', 'c', 'd'), documentId('a', 'b/c', 'd'))
  assert.notEqual(documentId('a', 'b', 'c","d'), documentId('a', 'b","c', 'd'))
  const first = adapt({ identifier: 'urn:said:Eaaa', content: 'one', source_identifier: 'S1' })
  const second = adapt({ identifier: 'urn:said:Ebbb', content: 'two', source_identifier: 'S2' })
  assert.equal(first.documentId, second.documentId)
  assert.notEqual(first.documentId, adapt({ id: 'zoom-toip-chat-GMT1' }).documentId)
})

test('identity components reject blank, control, lone-surrogate and oversize values at the key limit', () => {
  for (const bad of ['', '   ', 'a\u0000b', 'a\nb', 'a\u007fb', 'a\ud800b']) {
    assert.throws(() => documentId('toip', 'Zoom', bad), error => error.code === 'invalid_request')
  }
  const overhead = Buffer.byteLength(JSON.stringify(['toip', 'Zoom', '']))
  for (const size of [1023, 1024, 1025]) {
    const attempt = () => documentId('toip', 'Zoom', 'x'.repeat(size - overhead))
    if (size <= 1024) assert.equal(Buffer.byteLength(attempt()), size)
    else assert.throws(attempt, error => error.code === 'invalid_request')
  }
})

test('a curated document maps to WARM input without altering text or provenance', () => {
  const content = '  \r\n Café 😀 line one\r\nline two  \n'
  const { input, identity } = adapt({ content, extra: { nested: [1, null, 'x'] }, date: '2026-10-01', collected_at: '2026-10-02T00:00:00Z' })
  assert.equal(input.text, content)
  assert.equal(input.version, 'urn:said:Eaaa')
  assert.equal(input.sourceUri, 'https://example.org/z')
  assert.equal(input.storeText, true)
  assert.deepEqual(identity, { namespace: 'toip', sourceSystem: 'Zoom', id: 'zoom-toip-transcript-GMT1' })
  assert.deepEqual(input.metadata, {
    id: 'zoom-toip-transcript-GMT1',
    source_system: 'Zoom',
    identifier: 'urn:said:Eaaa',
    organizational_unit: ['EGWG'],
    source_identifier: 'GMT1',
    source_url: 'https://example.org/z',
    title: 'Meeting',
    extra: { nested: [1, null, 'x'] },
    date: '2026-10-01',
    collected_at: '2026-10-02T00:00:00Z',
    hms_namespace: 'toip',
  })
  assert.equal('content' in input.metadata, false)
  assert.equal('sourceUri' in adapt({ source_url: '' }).input, false)
  assert.equal('sourceUri' in adapt({ source_url: undefined }).input, false)
})

test('malformed documents are rejected with the offending field named', () => {
  for (const field of ['id', 'source_system', 'identifier', 'content', 'organizational_unit']) rejects({ [field]: undefined }, field)
  rejects({ content: '  \n\t ' }, 'content')
  rejects({ content: 'ab\ud800cd' }, 'content')
  rejects({ content: { text: 'nested' } }, 'content')
  rejects({ organizational_unit: ['EGWG', 7] }, 'organizational_unit')
  rejects({ organizational_unit: 'EGWG' }, 'organizational_unit')
  rejects({ title: 12 }, 'title')
  rejects({ source_url: 'javascript:alert(1)' }, 'source_url')
  rejects({ source_url: 'https://user:secret@example.org/' }, 'source_url')
  rejects({ source_url: 'not a url' }, 'source_url')
  // new URL() would normalize these; the stored value is the raw string, so they are rejected.
  for (const url of ['https://exa\nmple.org/x', ' https://example.org/x', 'https://example.org/a b', 'http:example.org']) {
    rejects({ source_url: url }, 'source_url')
  }
  rejects({ hms_namespace: 'other' }, 'hms_namespace')
  rejects({ extra: 'ok\udc00' }, 'metadata.extra')
  rejects({ extra: () => 1 }, 'metadata.extra')
  rejects({ document: { project: 'Trust over IP' }, provenance: { artifact_id: 'sha256:abc' } }, 'document')
  assert.throws(() => adaptDocument('toip', [], 'doc.json'), error => error.code === 'invalid_document')
  assert.throws(() => adaptDocument(' ', curated(), 'doc.json'), KnowledgeError)
})

test('adapter-owned size limits accept limit and reject limit+1', () => {
  const accepts = (overrides, field, limit, build) => {
    for (const size of [limit - 1, limit]) assert.doesNotThrow(() => adapt(build(size)), `${field} ${size}`)
    rejects(build(limit + 1), field)
  }
  // identifier: bytes, not characters
  accepts({}, 'identifier', 1024, size => ({ identifier: 'é'.repeat(size / 2) + (size % 2 ? 'x' : '') }))
  const prefix = 'https://example.org/'
  accepts({}, 'source_url', 8192, size => ({ source_url: prefix + 'a'.repeat(size - prefix.length) }))
  const base = Buffer.byteLength(JSON.stringify(adapt({ pad: '' }).input.metadata))
  accepts({}, 'metadata', 65536, size => ({ pad: 'x'.repeat(size - base) }))
})

test('a __proto__ key is rejected at any depth and never reaches a prototype', () => {
  const parse = text => JSON.parse(JSON.stringify(curated()).replace(/^\{/, text))
  assert.throws(() => adaptDocument('toip', parse('{"__proto__":{"polluted":true},'), 'doc.json'), error => error.code === 'invalid_document' && error.field === 'metadata')
  assert.throws(() => adaptDocument('toip', parse('{"extra":{"__proto__":1},'), 'doc.json'), error => error.code === 'invalid_document' && error.field === 'metadata.extra')
  assert.equal({}.polluted, undefined)
})

test('prepareImport rejects duplicate identities naming both sources, and empty or malformed entry lists', () => {
  const entries = [
    { source: 'a.json', document: curated() },
    { source: 'b.json', document: curated({ identifier: 'urn:said:Ebbb', content: 'newer' }) },
  ]
  assert.throws(
    () => prepareImport('toip', entries),
    error => error.code === 'duplicate_identity' && error.message.includes('a.json') && error.message.includes('b.json'),
  )
  assert.equal(prepareImport('other', [entries[0]]).length, 1)
  assert.throws(() => prepareImport('toip', []), error => error.code === 'invalid_request')
  // The cap applies while draining, so an unbounded source cannot be materialized.
  let drawn = 0
  const endless = (function* () { for (;;) { drawn++; yield { document: curated() } } })()
  assert.throws(() => prepareImport('toip', endless), error => error.code === 'invalid_request')
  assert.equal(drawn, 10_001)
  assert.throws(() => prepareImport('toip', [curated()]), error => error.code === 'invalid_request')
  assert.throws(() => prepareImport('toip', [{ document: curated({ id: '' }) }]), error => error.code === 'invalid_document' && error.source === 'entries[0]')
})
