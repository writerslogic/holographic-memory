'use strict'

const test = require('node:test')
const assert = require('node:assert/strict')
const fs = require('node:fs')
const os = require('node:os')
const path = require('node:path')
const { spawnSync } = require('node:child_process')
const { HolographicMemorySystem } = require('holographic-memory')
const { KnowledgeError, OrganizationalKnowledge, documentId, validateImport } = require('../index.cjs')

const ADAPTER = path.resolve(__dirname, '..')
const ROOT = path.resolve(ADAPTER, '..', '..')
const CLI = path.join(ADAPTER, 'example', 'cli.cjs')
const FIXTURES = path.join(ADAPTER, 'example', 'fixtures')
const DIMENSIONS = 4096
const CHILD_TIMEOUT_MS = 60_000

async function withStore(run) {
  const workdir = fs.mkdtempSync(path.join(os.tmpdir(), 'ok-test-'))
  try {
    return await run(workdir)
  } finally {
    fs.rmSync(workdir, { recursive: true, force: true })
  }
}

const openMemory = directory => new HolographicMemorySystem(DIMENSIONS, directory)

function curated(overrides = {}) {
  return {
    project: 'Test Project',
    organizational_unit: ['Unit A'],
    source_system: 'Zoom',
    source_identifier: 'GMT-0001',
    source_url: '',
    date: '2026-01-01',
    asset_type: 'Meeting',
    identifier_type: 'SAID',
    collected_at: '2026-01-02T00:00:00+00:00',
    id: 'doc-1',
    content_type: 'Transcript',
    title: 'Title',
    identifier: 'urn:said:Etest0001',
    content: 'zebraquartz baseline text',
    ...overrides,
  }
}

const entry = (document, source) => ({ document, source })
const idsOf = hits => [...new Set(hits.map(hit => hit.identity.id))].sort()
const words = count => 'a '.repeat(count - 1) + 'a'
// Chunk n of a document starts 450 words (500 - 50 overlap) after chunk n-1.
const wordsForChunks = chunks => 500 + 450 * (chunks - 1)

function rejectsWith(promise, code, expected = {}, messagePatterns = []) {
  return assert.rejects(promise, error => {
    assert.ok(error instanceof KnowledgeError, `expected KnowledgeError, got ${error}`)
    assert.equal(error.code, code, error.message)
    for (const [key, value] of Object.entries(expected)) assert.equal(error[key], value)
    for (const pattern of messagePatterns) assert.match(error.message, pattern)
    return true
  })
}

test('documents that share a source_identifier remain separate documents', async () => {
  await withStore(async workdir => {
    const knowledge = new OrganizationalKnowledge(openMemory(workdir), { namespace: 'org' })
    const kinds = [
      ['Transcript', 'nimbusone'],
      ['Chat', 'nimbustwo'],
      ['CC', 'nimbusthree'],
      ['AI Summary', 'nimbusfour'],
    ]
    const documents = kinds.map(([contentType, token], index) =>
      curated({
        id: `zoom-${contentType.toLowerCase().replace(' ', '-')}-GMT-0001`,
        source_identifier: 'GMT-0001',
        content_type: contentType,
        identifier: `urn:said:Eshared${index}`,
        content: `zebraquartz ${token}`,
      }),
    )
    const result = await knowledge.importDocuments(documents.map(document => entry(document, document.id)))
    assert.equal(new Set(result.documents.map(document => document.documentId)).size, 4)

    assert.deepEqual(idsOf(await knowledge.search('zebraquartz')), documents.map(document => document.id).sort())
    for (const [index, [, token]] of kinds.entries()) {
      const hits = await knowledge.search(token)
      assert.deepEqual(idsOf(hits), [documents[index].id])
      assert.equal(hits[0].metadata.source_identifier, 'GMT-0001')
    }
  })
})

test('the same source_system and id in two namespaces stay isolated through replace and forget', async () => {
  await withStore(async workdir => {
    const memory = openMemory(workdir)
    const alpha = new OrganizationalKnowledge(memory, { namespace: 'org-a' })
    const beta = new OrganizationalKnowledge(memory, { namespace: 'org-b' })
    const target = { sourceSystem: 'Zoom', id: 'doc-1' }
    await alpha.importDocuments([entry(curated({ identifier: 'urn:said:Ea1', content: 'zebraquartz alphaonly' }))])
    await beta.importDocuments([entry(curated({ identifier: 'urn:said:Eb1', content: 'zebraquartz betaonly' }))])

    const alphaHits = await alpha.search('zebraquartz')
    assert.equal(alphaHits.length, 1)
    assert.equal(alphaHits[0].version, 'urn:said:Ea1')
    assert.equal(alphaHits[0].identity.namespace, 'org-a')
    assert.equal((await beta.search('zebraquartz'))[0].version, 'urn:said:Eb1')
    assert.equal((await alpha.search('betaonly')).length, 0)

    await beta.importDocuments([entry(curated({ identifier: 'urn:said:Eb2', content: 'zebraquartz gammaonly' }))])
    assert.equal((await alpha.search('alphaonly')).length, 1)
    assert.equal((await beta.search('betaonly')).length, 0)
    assert.equal((await beta.search('gammaonly')).length, 1)

    assert.equal((await beta.forget([target], { apply: true })).removed.length, 1)
    assert.equal((await beta.search('zebraquartz')).length, 0)
    assert.equal((await alpha.search('alphaonly')).length, 1)
  })
})

test('delimiter-ambiguous identities do not collide', async () => {
  await withStore(async workdir => {
    const memory = openMemory(workdir)
    const slashed = new OrganizationalKnowledge(memory, { namespace: 'a/b' })
    const plain = new OrganizationalKnowledge(memory, { namespace: 'a' })
    const documents = [
      { owner: slashed, token: 'tokenfirst', document: curated({ source_system: 'c', id: 'd', content: 'tokenfirst' }) },
      { owner: plain, token: 'tokensecond', document: curated({ source_system: 'b/c', id: 'd', content: 'tokensecond' }) },
      { owner: plain, token: 'tokenthird', document: curated({ source_system: 'S', id: 'x","y', content: 'tokenthird' }) },
      { owner: plain, token: 'tokenfourth', document: curated({ source_system: 'S","x', id: 'y', content: 'tokenfourth' }) },
    ]
    const keys = [
      documentId('a/b', 'c', 'd'),
      documentId('a', 'b/c', 'd'),
      documentId('a', 'S', 'x","y'),
      documentId('a', 'S","x', 'y'),
    ]
    assert.equal(new Set(keys).size, 4)

    for (const { owner, document } of documents) await owner.importDocuments([entry(document)])
    for (const [index, { owner, token, document }] of documents.entries()) {
      const hits = await owner.search(token)
      assert.deepEqual([...new Set(hits.map(hit => hit.documentId))], [keys[index]])
      assert.deepEqual(hits[0].identity, {
        namespace: owner.namespace,
        sourceSystem: document.source_system,
        id: document.id,
      })
    }

    await plain.forget([{ sourceSystem: 'S', id: 'x","y' }], { apply: true })
    assert.equal((await plain.search('tokenthird')).length, 0)
    assert.equal((await plain.search('tokenfourth')).length, 1)
  })
})

test('content comes back byte-exact: accents, emoji, CRLF, and surrounding whitespace', async () => {
  await withStore(async workdir => {
    const knowledge = new OrganizationalKnowledge(openMemory(workdir), { namespace: 'org' })
    const unit = 'Café café naïve \u{1F600} zebraquartz\r\nsecond  line   keeps  spacing\r\n'
    const content = `  \r\n\t${unit.repeat(220)}   \r\n`
    await knowledge.importDocuments([entry(curated({ content }))])

    const hits = await knowledge.search('zebraquartz', { k: 50 })
    assert.ok(hits.length >= 3, `expected a multi-chunk document, got ${hits.length} chunk(s)`)
    const original = Buffer.from(content, 'utf8')
    for (const hit of hits) {
      assert.equal(original.subarray(hit.startByte, hit.endByte).toString('utf8'), hit.text)
    }
    const joined = hits.map(hit => hit.text).join('\n')
    assert.ok(joined.includes('\r\n'))
    assert.ok(joined.includes('\u{1F600}'))
    assert.ok(joined.includes('café'))
    assert.ok(joined.includes('second  line   keeps  spacing'))
  })
})

test('metadata, identifier, and source_url map to hit metadata, version, and sourceUri', async () => {
  await withStore(async workdir => {
    const knowledge = new OrganizationalKnowledge(openMemory(workdir), { namespace: 'org' })
    const withUrl = curated({
      id: 'with-url',
      organizational_unit: ['Alpha Group', 'Beta Force'],
      source_url: 'https://wiki.example.org/pages/viewpage.action?pageId=1',
      identifier: 'urn:said:Eexact-version',
      content: 'zebraquartz withurl',
      extra_unknown: { list: [1, 'two', null, true, 0.5], nested: { accent: 'é' } },
    })
    const withoutUrl = curated({ id: 'without-url', source_url: '', content: 'zebraquartz withouturl' })
    await knowledge.importDocuments([entry(withUrl), entry(withoutUrl)])

    const [hit] = await knowledge.search('withurl')
    const { content, ...expectedMetadata } = withUrl
    assert.ok(content)
    assert.deepEqual(hit.metadata, { ...expectedMetadata, hms_namespace: 'org' })
    assert.equal(hit.sourceUri, withUrl.source_url)
    assert.equal(hit.version, 'urn:said:Eexact-version')

    const [bare] = await knowledge.search('withouturl')
    assert.equal(bare.sourceUri, undefined)
    assert.equal(bare.metadata.source_url, '')
    assert.equal(bare.version, withoutUrl.identifier)
  })
})

test('re-importing an identity replaces its content, chunks, and version', async () => {
  await withStore(async workdir => {
    const knowledge = new OrganizationalKnowledge(openMemory(workdir), { namespace: 'org' })
    const long = 'oldtokenalpha filler '.repeat(600)
    const first = await knowledge.importDocuments([entry(curated({ identifier: 'urn:said:Eold', content: long }))])
    assert.ok(first.chunks >= 3)
    assert.equal((await knowledge.search('oldtokenalpha', { k: 50 })).length, first.chunks)

    const second = await knowledge.importDocuments([entry(curated({ identifier: 'urn:said:Enew', content: 'newtokenbeta short' }))])
    assert.equal(second.chunks, 1)
    assert.equal((await knowledge.search('oldtokenalpha', { k: 50 })).length, 0)
    assert.equal((await knowledge.search('filler', { k: 50 })).length, 0)
    const hits = await knowledge.search('newtokenbeta')
    assert.equal(hits.length, 1)
    assert.equal(hits[0].version, 'urn:said:Enew')
  })
})

test('an import containing an invalid document writes nothing', async () => {
  await withStore(async workdir => {
    const knowledge = new OrganizationalKnowledge(openMemory(workdir), { namespace: 'org' })
    await knowledge.importDocuments([entry(curated({ identifier: 'urn:said:Eold', content: 'zebraquartz oldcontent' }))])

    const { identifier, ...missingIdentifier } = curated({ id: 'bad', content: 'zebraquartz badcontent' })
    assert.ok(identifier)
    const tooManyChunks = curated({ id: 'bad', content: words(wordsForChunks(4096) + 1) })
    const invalid = [
      ['adapter rule', missingIdentifier, { field: 'identifier' }],
      ['engine chunk limit', tooManyChunks, {}],
    ]
    for (const [, document, expected] of invalid) {
      await rejectsWith(
        knowledge.importDocuments([
          entry(curated({ identifier: 'urn:said:Enew', content: 'zebraquartz replacement' }), 'replacement.json'),
          entry(curated({ id: 'fresh', content: 'zebraquartz freshdoc' }), 'fresh.json'),
          entry(document, 'bad.json'),
        ]),
        'invalid_document',
        { source: 'bad.json', ...expected },
      )

      const hits = await knowledge.search('zebraquartz', { k: 50 })
      assert.deepEqual(idsOf(hits), ['doc-1'])
      assert.equal(hits[0].version, 'urn:said:Eold')
      assert.equal(hits[0].text, 'zebraquartz oldcontent')
      for (const absent of ['replacement', 'freshdoc', 'badcontent']) assert.equal((await knowledge.search(absent)).length, 0)
    }
  })
})

test('validateImport enforces the engine chunk limit: 4095 and 4096 pass, 4097 is rejected', async () => {
  await withStore(async workdir => {
    const memory = openMemory(workdir)
    const chunkCount = async text => (await memory.chunkDocument({ id: 'probe', text, chunkWords: 500, overlapWords: 50 })).length

    assert.equal(await chunkCount(words(wordsForChunks(3))), 3)
    assert.equal(await chunkCount(words(wordsForChunks(3) + 1)), 4)
    assert.equal(await chunkCount(words(wordsForChunks(4095))), 4095)
    assert.equal(await chunkCount(words(wordsForChunks(4096))), 4096)
    await assert.rejects(chunkCount(words(wordsForChunks(4096) + 1)), /exceeds 4096 chunks/)

    for (const chunks of [4095, 4096]) {
      await validateImport('org', [entry(curated({ content: words(wordsForChunks(chunks)) }), `chunks-${chunks}`)], memory)
    }
    await rejectsWith(
      validateImport('org', [entry(curated({ content: words(wordsForChunks(4096) + 1) }), 'chunks-4097')], memory),
      'invalid_document',
      { source: 'chunks-4097' },
      [/exceeds 4096 chunks/],
    )
  })
})

test('forget: a dry run changes nothing, apply removes only the target, repeating reports it absent', async () => {
  await withStore(async workdir => {
    const knowledge = new OrganizationalKnowledge(openMemory(workdir), { namespace: 'org' })
    await knowledge.importDocuments([
      entry(curated({ id: 'keep', content: 'zebraquartz keepthis' })),
      entry(curated({ id: 'drop', content: 'zebraquartz dropthis' })),
    ])
    const target = { sourceSystem: 'Zoom', id: 'drop' }
    const key = documentId('org', 'Zoom', 'drop')

    const dry = await knowledge.forget([target])
    assert.equal(dry.applied, false)
    assert.deepEqual(dry.targets.map(planned => planned.documentId), [key])
    assert.equal(dry.removed, undefined)
    assert.deepEqual(idsOf(await knowledge.search('zebraquartz')), ['drop', 'keep'])

    const applied = await knowledge.forget([target], { apply: true })
    assert.deepEqual(applied.removed.map(removed => removed.documentId), [key])
    assert.deepEqual(applied.absent, [])
    assert.deepEqual(idsOf(await knowledge.search('zebraquartz')), ['keep'])

    const repeated = await knowledge.forget([target], { apply: true })
    assert.deepEqual(repeated.removed, [])
    assert.deepEqual(repeated.absent.map(absent => absent.documentId), [key])
    assert.deepEqual(idsOf(await knowledge.search('zebraquartz')), ['keep'])
  })
})

test('invalid forget targets are rejected before anything is deleted', async t => {
  await withStore(async workdir => {
    const knowledge = new OrganizationalKnowledge(openMemory(workdir), { namespace: 'org' })
    await knowledge.importDocuments([entry(curated({ content: 'zebraquartz survivor' }))])
    const valid = { sourceSystem: 'Zoom', id: 'doc-1' }

    const cases = {
      'unknown key': [valid, { sourceSystem: 'Zoom', id: 'other', wildcard: '*' }],
      'duplicate target': [valid, { ...valid }],
      'empty list': [],
      'blank id': [valid, { sourceSystem: 'Zoom', id: '  ' }],
    }
    for (const [name, targets] of Object.entries(cases)) {
      await t.test(name, () => rejectsWith(knowledge.forget(targets, { apply: true }), 'invalid_request'))
    }
    assert.equal((await knowledge.search('survivor')).length, 1)
  })
})

function runChild(workdir, store, payload) {
  const script = `
    const { HolographicMemorySystem } = require(${JSON.stringify(path.join(ROOT, 'index.js'))});
    const { OrganizationalKnowledge } = require(${JSON.stringify(path.join(ADAPTER, 'index.cjs'))});
    (async () => {
      const [store, payloadJson] = process.argv.slice(1);
      const { docs = [], forget = [] } = JSON.parse(payloadJson);
      const knowledge = new OrganizationalKnowledge(new HolographicMemorySystem(${DIMENSIONS}, store), { namespace: 'org' });
      if (docs.length) await knowledge.importDocuments(docs.map(document => ({ document, source: document.id })));
      if (forget.length) await knowledge.forget(forget, { apply: true });
      const state = {};
      for (const token of ['persistalpha', 'persistbeta']) {
        const versions = {};
        for (const hit of await knowledge.search(token, { k: 50 })) versions[hit.identity.id] = hit.version;
        state[token] = versions;
      }
      console.log(JSON.stringify(state));
    })().catch(error => { console.error(error); process.exitCode = 1; });
  `
  const result = spawnSync(process.execPath, ['-e', script, store, JSON.stringify(payload)], {
    cwd: workdir,
    encoding: 'utf8',
    timeout: CHILD_TIMEOUT_MS,
  })
  assert.equal(result.status, 0, result.stderr || result.stdout)
  return JSON.parse(result.stdout)
}

test('imports and forgets survive process restarts, and re-importing restores a forgotten document', async () => {
  await withStore(async workdir => {
    const store = path.join(workdir, 'store')
    const alpha = curated({ id: 'alpha', identifier: 'urn:said:Epa', content: 'zebraquartz persistalpha' })
    const beta = curated({ id: 'beta', identifier: 'urn:said:Epb', content: 'zebraquartz persistbeta' })

    const written = runChild(workdir, store, { docs: [alpha, beta], forget: [{ sourceSystem: 'Zoom', id: 'alpha' }] })
    assert.deepEqual(written, { persistalpha: {}, persistbeta: { beta: 'urn:said:Epb' } })

    assert.deepEqual(runChild(workdir, store, {}), written)

    const restored = { persistalpha: { alpha: 'urn:said:Epa' }, persistbeta: { beta: 'urn:said:Epb' } }
    assert.deepEqual(runChild(workdir, store, { docs: [alpha] }), restored)
    assert.deepEqual(runChild(workdir, store, {}), restored)
  })
})

test('search is restricted to the namespace and filters by content_type', async () => {
  await withStore(async workdir => {
    const memory = openMemory(workdir)
    const orgA = new OrganizationalKnowledge(memory, { namespace: 'org-a' })
    const orgB = new OrganizationalKnowledge(memory, { namespace: 'org-b' })
    await orgA.importDocuments([
      entry(curated({ id: 'a-transcript', content_type: 'Transcript', content: 'zebraquartz transcript words' })),
      entry(curated({ id: 'a-chat', content_type: 'Chat', content: 'zebraquartz chat words' })),
    ])
    await orgB.importDocuments([entry(curated({ id: 'b-chat', content_type: 'Chat', content: 'zebraquartz other org words' }))])

    const all = await orgA.search('zebraquartz', { k: 50 })
    assert.deepEqual(idsOf(all), ['a-chat', 'a-transcript'])
    assert.ok(all.every(hit => hit.metadata.hms_namespace === 'org-a' && hit.identity.namespace === 'org-a'))

    const chats = await orgA.search('zebraquartz', { k: 50, filter: { content_type: 'Chat' } })
    assert.deepEqual(idsOf(chats), ['a-chat'])
    assert.equal((await orgA.search('zebraquartz', { filter: { content_type: 'Unknown' } })).length, 0)
  })
})

test('the reserved namespace filter is rejected and organizational_unit matches only the whole array', async () => {
  await withStore(async workdir => {
    const knowledge = new OrganizationalKnowledge(openMemory(workdir), { namespace: 'org-a' })
    await knowledge.importDocuments([entry(curated({ organizational_unit: ['Alpha Group', 'Beta Force'], content: 'zebraquartz units' }))])

    await rejectsWith(knowledge.search('zebraquartz', { filter: { hms_namespace: 'org-b' } }), 'invalid_request')
    await rejectsWith(knowledge.search('zebraquartz', { filter: { hms_namespace: 'org-a' } }), 'invalid_request')

    assert.equal((await knowledge.search('zebraquartz', { filter: { organizational_unit: ['Alpha Group'] } })).length, 0)
    assert.equal((await knowledge.search('zebraquartz', { filter: { organizational_unit: ['Alpha Group', 'Beta Force'] } })).length, 1)
  })
})

test('a duplicate identity within one import is rejected naming both sources and writes nothing', async () => {
  await withStore(async workdir => {
    const knowledge = new OrganizationalKnowledge(openMemory(workdir), { namespace: 'org' })
    await rejectsWith(
      knowledge.importDocuments([
        entry(curated({ id: 'unrelated', content: 'zebraquartz unrelateddoc' }), 'unrelated.json'),
        entry(curated({ id: 'same', content: 'zebraquartz firstcopy' }), 'first.json'),
        entry(curated({ id: 'same', content: 'zebraquartz secondcopy' }), 'second.json'),
      ]),
      'duplicate_identity',
      { source: 'second.json' },
      [/first\.json/, /second\.json/],
    )
    assert.equal((await knowledge.search('zebraquartz', { k: 50 })).length, 0)
  })
})

const cli = (workdir, ...args) =>
  spawnSync(process.execPath, [CLI, ...args], { cwd: workdir, encoding: 'utf8', timeout: CHILD_TIMEOUT_MS })

test('the CLI imports into a new store, searches, previews forget, and applies it', async () => {
  await withStore(async workdir => {
    const store = path.join(workdir, 'store')
    const transcript = 'zoom-example-project-transcript-GMT20260101-100000'
    const wiki = 'confluence-example-project-wiki-page-24609309'
    const search = () => {
      const result = cli(workdir, 'search', store, 'toip', 'quartzmarker')
      assert.equal(result.status, 0, result.stderr)
      return idsOf(JSON.parse(result.stdout))
    }

    assert.equal(fs.existsSync(store), false)
    const imported = cli(workdir, 'import', store, 'toip', FIXTURES)
    assert.equal(imported.status, 0, imported.stderr)
    assert.match(imported.stdout, /Imported 3 document\(s\)/)
    assert.deepEqual(search(), [wiki, transcript].sort())

    const dry = cli(workdir, 'forget', store, 'toip', 'Zoom', transcript)
    assert.equal(dry.status, 0, dry.stderr)
    assert.deepEqual(search(), [wiki, transcript].sort())

    const applied = cli(workdir, 'forget', store, 'toip', 'Zoom', transcript, '--apply')
    assert.equal(applied.status, 0, applied.stderr)
    assert.match(applied.stdout, /Removed: true/)
    assert.deepEqual(search(), [wiki])
  })
})

test('the CLI rejects malformed input before creating the store', async () => {
  await withStore(async workdir => {
    const inputs = path.join(workdir, 'inputs')
    fs.mkdirSync(inputs)
    const { identifier, ...missingIdentifier } = curated({ id: 'no-identifier' })
    assert.ok(identifier)
    const files = {
      'invalid.json': '{ not json',
      'no-identifier.json': JSON.stringify(missingIdentifier),
      'valid.json': JSON.stringify(curated({ id: 'valid' })),
    }
    for (const [name, text] of Object.entries(files)) fs.writeFileSync(path.join(inputs, name), text)
    const input = name => path.join(inputs, name)

    const cases = [
      ['invalid JSON', ['invalid.json'], 'invalid.json'],
      ['missing identifier', ['no-identifier.json'], 'no-identifier.json'],
      ['invalid JSON and missing identifier', ['invalid.json', 'no-identifier.json'], 'invalid.json'],
      ['valid file next to a missing identifier', ['valid.json', 'no-identifier.json'], 'no-identifier.json'],
    ]
    for (const [name, names, named] of cases) {
      const store = path.join(workdir, 'store')
      const result = cli(workdir, 'import', store, 'toip', ...names.map(input))
      assert.notEqual(result.status, 0, `${name}: expected a nonzero exit`)
      assert.ok(result.stderr.includes(named), `${name}: stderr should name ${named}: ${result.stderr}`)
      assert.equal(fs.existsSync(store), false, `${name}: the store directory must not be created`)
    }

    const store = path.join(workdir, 'store')
    for (const args of [['search', store, ' ', 'query'], ['forget', store, 'toip', 'Zoom', 'id\nx', '--apply']]) {
      assert.notEqual(cli(workdir, ...args).status, 0, args.join(' '))
      assert.equal(fs.existsSync(store), false, `${args[0]}: the store directory must not be created`)
    }
  })
})

test('overlapping writes on one instance run in call order instead of failing as partial writes', async () => {
  await withStore(async workdir => {
    const knowledge = new OrganizationalKnowledge(openMemory(workdir), { namespace: 'org' })
    // More concurrent calls than DocumentMemory's pending limit of 32.
    const calls = Array.from({ length: 40 }, (_, index) => knowledge.importDocuments([entry(curated({ id: `doc-${index}`, content: `zebraquartz overlap${index}` }))]))
    const first = knowledge.importDocuments([entry(curated({ id: 'same', identifier: 'urn:said:E1', content: 'zebraquartz firstwins' }))])
    const second = knowledge.importDocuments([entry(curated({ id: 'same', identifier: 'urn:said:E2', content: 'zebraquartz secondwins' }))])
    const results = await Promise.allSettled([...calls, first, second])
    assert.deepEqual(results.filter(result => result.status === 'rejected').map(result => String(result.reason)), [])

    assert.equal(idsOf(await knowledge.search('zebraquartz', { k: 100 })).length, 41)
    assert.equal((await knowledge.search('firstwins')).length, 0)
    assert.equal((await knowledge.search('secondwins'))[0].version, 'urn:said:E2')
  })
})

test('an embedder supplies chunk vectors, and the per-document embedding bound is checked before any write', async () => {
  await withStore(async workdir => {
    const memory = new HolographicMemorySystem(DIMENSIONS, workdir, { embeddingModel: 'test-model', embeddingRevision: 'r1', embeddingDimensions: 8 })
    const vector = text => Array.from({ length: 8 }, (_, index) => (index === (text.includes('alpha') ? 0 : text.includes('beta') ? 1 : 2) ? 1 : 0.01))
    const embedder = { space: memory.securityStatus().embeddingSpace, encodeDocuments: async texts => texts.map(vector), encodeQuery: async text => vector(text) }
    const knowledge = new OrganizationalKnowledge(memory, { namespace: 'org', embedder })
    await knowledge.importDocuments([entry(curated({ id: 'a', content: 'alpha words' })), entry(curated({ id: 'b', content: 'beta words' }))])

    const [best] = await knowledge.search('unmatchedterm beta', { lexicalWeight: 0, semanticWeight: 1 })
    assert.equal(best.identity.id, 'b')
    assert.ok(best.semanticScore > 0.99)

    // 512-dimension vectors: 2048 chunks is exactly 1,048,576 values, the engine's per-document limit.
    const check = chunks => validateImport('org', [entry(curated({ content: words(wordsForChunks(chunks)) }), `chunks-${chunks}`)], memory, { embeddingDimensions: 512 })
    await check(2048)
    await rejectsWith(check(2049), 'invalid_document', { source: 'chunks-2049' }, [/exceed 1048576 values/])
  })
})
