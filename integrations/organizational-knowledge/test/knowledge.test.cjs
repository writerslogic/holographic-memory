'use strict'

const test = require('node:test')
const assert = require('node:assert/strict')
const fs = require('node:fs')
const os = require('node:os')
const path = require('node:path')
const { spawnSync } = require('node:child_process')
const { WarmMemorySystem } = require('writerslogic-warm')
const { KnowledgeError, OrganizationalKnowledge, documentId, prepareImport, validateImport } = require('../index.cjs')

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

const openMemory = directory => new WarmMemorySystem(DIMENSIONS, directory)

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
    const { WarmMemorySystem } = require(${JSON.stringify(path.join(ROOT, 'index.js'))});
    const { OrganizationalKnowledge } = require(${JSON.stringify(path.join(ADAPTER, 'index.cjs'))});
    (async () => {
      const [store, payloadJson] = process.argv.slice(1);
      const { docs = [], forget = [], scope } = JSON.parse(payloadJson);
      const knowledge = new OrganizationalKnowledge(new WarmMemorySystem(${DIMENSIONS}, store), { namespace: 'org' });
      if (docs.length) await knowledge.importDocuments(docs.map(document => ({ document, source: document.id })));
      if (forget.length) await knowledge.forget(forget, { apply: true });
      if (scope) await knowledge.forgetScope(scope, { apply: true });
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

    // The engine's 4096-chunk limit is checked on a throwaway store, which is removed again.
    const scratchRoot = path.join(workdir, 'tmp')
    fs.mkdirSync(scratchRoot)
    fs.writeFileSync(path.join(inputs, 'too-many-chunks.json'), JSON.stringify(curated({ id: 'too-many-chunks', content: words(wordsForChunks(4097)) })))
    const overLimit = spawnSync(process.execPath, [CLI, 'import', path.join(workdir, 'store'), 'toip', input('too-many-chunks.json')], {
      cwd: workdir,
      encoding: 'utf8',
      timeout: CHILD_TIMEOUT_MS,
      env: { ...process.env, TMPDIR: scratchRoot },
    })
    assert.notEqual(overLimit.status, 0)
    assert.match(overLimit.stderr, /too-many-chunks\.json/)
    assert.equal(overLimit.stderr.includes('panic'), false, overLimit.stderr)
    assert.equal(fs.existsSync(path.join(workdir, 'store')), false, 'an over-limit document must not create the store')
    assert.deepEqual(fs.readdirSync(scratchRoot), [], 'the throwaway store must be removed')

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
    const memory = new WarmMemorySystem(DIMENSIONS, workdir, { embeddingModel: 'test-model', embeddingRevision: 'r1', embeddingDimensions: 8 })
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

// Fails one engine write on demand; every call still reaches the real store.
function failingWrites(memory, method) {
  const state = { failAt: Infinity, calls: 0 }
  const proxy = new Proxy(memory, {
    get(target, property) {
      const value = Reflect.get(target, property, target)
      if (property !== method) return typeof value === 'function' ? value.bind(target) : value
      return (...args) => {
        state.calls += 1
        if (state.calls === state.failAt) throw new Error('injected write failure')
        return value.apply(target, args)
      }
    },
  })
  return { proxy, state }
}

test('a failed write reports what was committed, and re-running the import completes it', async () => {
  await withStore(async workdir => {
    const { proxy, state } = failingWrites(openMemory(workdir), 'memorizeDocument')
    const knowledge = new OrganizationalKnowledge(proxy, { namespace: 'org' })
    const entries = ['a', 'b', 'c'].map(id => entry(curated({ id, content: `zebraquartz ${id}word` }), `${id}.json`))
    state.failAt = 2
    const failure = await knowledge.importDocuments(entries).then(() => null, error => error)
    assert.ok(failure instanceof KnowledgeError)
    assert.equal(failure.code, 'partial_failure')
    assert.equal(failure.source, 'b.json')
    assert.deepEqual(failure.committed, [documentId('org', 'Zoom', 'a')])
    assert.match(failure.message, /1 of 3 document\(s\) committed.*injected write failure/)
    assert.deepEqual(idsOf(await knowledge.search('zebraquartz', { k: 10 })), ['a'])

    state.failAt = Infinity
    const result = await knowledge.importDocuments(entries)
    assert.equal(result.documents.length, 3)
    assert.deepEqual(idsOf(await knowledge.search('zebraquartz', { k: 10 })), ['a', 'b', 'c'])
  })
})

test('a failed delete reports what was removed, and repeating the forget finishes it', async () => {
  await withStore(async workdir => {
    const { proxy, state } = failingWrites(openMemory(workdir), 'deleteDocument')
    const knowledge = new OrganizationalKnowledge(proxy, { namespace: 'org' })
    await knowledge.importDocuments(['a', 'b'].map(id => entry(curated({ id, content: `zebraquartz ${id}word` }))))
    const targets = ['a', 'b'].map(id => ({ sourceSystem: 'Zoom', id }))
    state.failAt = 2
    await rejectsWith(knowledge.forget(targets, { apply: true }), 'partial_failure', {}, [/1 of 2 target\(s\)/, /injected write failure/])
    assert.deepEqual(idsOf(await knowledge.search('zebraquartz', { k: 10 })), ['b'])

    state.failAt = Infinity
    const result = await knowledge.forget(targets, { apply: true })
    assert.deepEqual(result.removed.map(target => target.id), ['b'])
    assert.deepEqual(result.absent.map(target => target.id), ['a'])
    assert.equal((await knowledge.search('zebraquartz')).length, 0)
  })
})

const hasListing = typeof WarmMemorySystem.prototype.listDocuments === 'function'
const needsListing = { skip: !hasListing && 'the loaded writerslogic-warm binding has no listDocuments; rebuild it' }
const seedInputs = (namespace, sourceSystem, count, prefix = 'bulk') =>
  prepareImport(namespace, Array.from({ length: count }, (_, index) => entry(curated({ source_system: sourceSystem, id: `${prefix}-${String(index).padStart(5, '0')}`, content: 'zebraquartz bulk' }), `${prefix}-${index}`))).map(item => item.input)
const idsIn = result => result.targets.map(target => target.id).sort()
const liveIds = hits => [...new Set(hits.map(hit => hit.identity?.id ?? 'unparsed'))].sort()

test('forgetScope removes one source system in one namespace and leaves lookalike neighbours alone', needsListing, async () => {
  await withStore(async workdir => {
    const memory = openMemory(workdir)
    const org = new OrganizationalKnowledge(memory, { namespace: 'org' })
    const other = new OrganizationalKnowledge(memory, { namespace: 'org2' })
    const docs = (system, ...ids) => ids.map(id => entry(curated({ source_system: system, id, content: `zebraquartz ${system} ${id}` })))
    await org.importDocuments([...docs('Zoom', 'a', 'b'), ...docs('Zoom2', 'c'), ...docs('Wiki', 'd')])
    await other.importDocuments(docs('Zoom', 'a'))
    // Same id prefix as a namespace document, but not written by the adapter.
    await memory.memorizeDocument({ id: JSON.stringify(['org', 'Zoom', 'foreign']), text: 'zebraquartz foreign', version: 'v', metadata: {} })
    await memory.memorizeDocument({ id: JSON.stringify(['org', 'Zoom', 'a', 'b']), text: 'zebraquartz four parts', version: 'v', metadata: { hms_namespace: 'org' } })

    const preview = await org.forgetScope({ sourceSystem: 'Zoom' })
    assert.equal(preview.applied, false)
    assert.deepEqual(idsIn(preview), ['a', 'b'])
    assert.equal(preview.truncated, false)
    assert.equal((await org.search('zebraquartz', { k: 50 })).length, 5)

    const done = await org.forgetScope({ sourceSystem: 'Zoom' }, { apply: true })
    assert.deepEqual(done.removed.map(target => target.id).sort(), ['a', 'b'])
    assert.deepEqual(done.absent, [])
    // The four-part id carries the namespace metadata but is not an identity this adapter wrote: untouched.
    assert.deepEqual(liveIds(await org.search('zebraquartz', { k: 50 })), ['c', 'd', 'unparsed'])
    assert.deepEqual(idsOf(await other.search('zebraquartz', { k: 50 })), ['a'])
    const left = new Set((await memory.listDocuments(undefined, 100)).map(listed => listed.id))
    assert.ok(left.has(JSON.stringify(['org', 'Zoom', 'foreign'])) && left.has(JSON.stringify(['org', 'Zoom', 'a', 'b'])))

    assert.deepEqual((await org.forgetScope({ sourceSystem: 'Zoom' }, { apply: true })).removed, [])
    const everything = await org.forgetScope({ all: true }, { apply: true })
    assert.deepEqual(everything.removed.map(target => `${target.sourceSystem}/${target.id}`).sort(), ['Wiki/d', 'Zoom2/c'])
    assert.deepEqual(idsOf(await other.search('zebraquartz', { k: 50 })), ['a'])
  })
})

test('forgetScope with keep prunes what an upstream dropped, and a keep list cannot be empty or out of scope', needsListing, async () => {
  await withStore(async workdir => {
    const knowledge = new OrganizationalKnowledge(openMemory(workdir), { namespace: 'org' })
    await knowledge.importDocuments(['a', 'b', 'c', 'd'].map(id => entry(curated({ id, content: `zebraquartz ${id}` }))))
    const keep = [{ sourceSystem: 'Zoom', id: 'b' }, { sourceSystem: 'Zoom', id: 'd' }]

    assert.deepEqual(idsIn(await knowledge.forgetScope({ sourceSystem: 'Zoom', keep })), ['a', 'c'])
    await rejectsWith(knowledge.forgetScope({ sourceSystem: 'Zoom', keep: [] }, { apply: true }), 'invalid_request', { field: 'scope.keep' })
    await rejectsWith(knowledge.forgetScope({ sourceSystem: 'Zoom', keep: [{ sourceSystem: 'Wiki', id: 'x' }] }, { apply: true }), 'invalid_request', { field: 'scope.keep' })
    assert.equal((await knowledge.search('zebraquartz', { k: 10 })).length, 4)

    await knowledge.forgetScope({ sourceSystem: 'Zoom', keep }, { apply: true })
    assert.deepEqual(idsOf(await knowledge.search('zebraquartz', { k: 10 })), ['b', 'd'])
    assert.deepEqual((await knowledge.forgetScope({ sourceSystem: 'Zoom', keep })).targets, [])
  })
})

test('forgetScope pages through more than one listing page and honors limit-1, limit, limit+1', needsListing, async () => {
  await withStore(async workdir => {
    const memory = openMemory(workdir)
    const knowledge = new OrganizationalKnowledge(memory, { namespace: 'org' })
    // Neighbours on both sides of the namespace in id order.
    await new OrganizationalKnowledge(memory, { namespace: 'ora' }).importDocuments([entry(curated({ id: 'before' }))])
    await new OrganizationalKnowledge(memory, { namespace: 'orh' }).importDocuments([entry(curated({ id: 'after' }))])
    await memory.memorizeDocuments(seedInputs('org', 'Zoom', 1005))

    const all = await knowledge.forgetScope({ all: true }, { limit: 100_000 })
    assert.equal(all.targets.length, 1005)
    assert.equal(all.truncated, false)
    assert.equal(new Set(all.targets.map(target => target.documentId)).size, 1005)
    for (const [limit, count, truncated] of [[1004, 1004, true], [1005, 1005, false], [1006, 1005, false]]) {
      const result = await knowledge.forgetScope({ all: true }, { limit })
      assert.deepEqual([result.targets.length, result.truncated], [count, truncated], `limit ${limit}`)
    }

    const first = await knowledge.forgetScope({ all: true }, { apply: true, limit: 3 })
    assert.equal(first.removed.length, 3)
    assert.equal(first.truncated, true)
    assert.equal((await knowledge.forgetScope({ all: true }, { limit: 100_000 })).targets.length, 1002)
    assert.equal((await new OrganizationalKnowledge(memory, { namespace: 'ora' }).search('zebraquartz')).length, 1)
    assert.equal((await new OrganizationalKnowledge(memory, { namespace: 'orh' }).search('zebraquartz')).length, 1)
  })
})

test('forgetScope rejects malformed arguments before deleting anything', needsListing, async () => {
  await withStore(async workdir => {
    const knowledge = new OrganizationalKnowledge(openMemory(workdir), { namespace: 'org' })
    await knowledge.importDocuments([entry(curated({ content: 'zebraquartz keepme' }))])
    const bad = [
      [undefined, {}], [{}, {}], [[], {}], [{ all: true, sourceSystem: 'Zoom' }, {}], [{ all: false }, {}], [{ sourceSystem: ' ' }, {}],
      [{ sourceSystem: 7 }, {}], [{ all: true, unknown: 1 }, {}], [{ all: true, keep: 'doc-1' }, {}], [{ all: true, keep: [{ sourceSystem: 'Zoom' }] }, {}],
      [{ all: true }, { apply: 'yes' }], [{ all: true }, { limit: 0 }], [{ all: true }, { limit: 100_001 }], [{ all: true }, { limit: 1.5 }],
    ]
    for (const [scope, options] of bad) {
      await rejectsWith(knowledge.forgetScope(scope, { ...options, apply: options.apply ?? true }), 'invalid_request')
    }
    // `keep` in the options position must not be ignored: that would delete everything.
    const misplaced = { keep: [{ sourceSystem: 'Zoom', id: 'doc-1' }], apply: true }
    await rejectsWith(knowledge.forgetScope({ sourceSystem: 'Zoom' }, misplaced), 'invalid_request', { field: 'options' })
    await rejectsWith(knowledge.forgetScope({ sourceSystem: 'Zoom' }, null), 'invalid_request', { field: 'options' })
    assert.equal((await knowledge.search('keepme')).length, 1)
  })
})

test('forgetScope reports a binding without listDocuments instead of guessing', async () => {
  await withStore(async workdir => {
    const real = openMemory(workdir)
    const old = new Proxy(real, { get: (target, property) => (property === 'listDocuments' ? undefined : typeof target[property] === 'function' ? target[property].bind(target) : target[property]) })
    await rejectsWith(new OrganizationalKnowledge(old, { namespace: 'org' }).forgetScope({ all: true }), 'unsupported_binding')
  })
})

test('a forgetScope removal survives a restart and a later import restores the document', needsListing, async () => {
  await withStore(async workdir => {
    const store = path.join(workdir, 'store')
    const alpha = curated({ id: 'alpha', identifier: 'urn:said:Epa', content: 'zebraquartz persistalpha' })
    const beta = curated({ id: 'beta', identifier: 'urn:said:Epb', content: 'zebraquartz persistbeta' })
    const written = runChild(workdir, store, { docs: [alpha, beta], scope: { sourceSystem: 'Zoom', keep: [{ sourceSystem: 'Zoom', id: 'beta' }] } })
    assert.deepEqual(written, { persistalpha: {}, persistbeta: { beta: 'urn:said:Epb' } })
    assert.deepEqual(runChild(workdir, store, {}), written)
    assert.deepEqual(runChild(workdir, store, { docs: [alpha] }).persistalpha, { alpha: 'urn:said:Epa' })
  })
})

test('the CLI forget-all and prune preview by default, apply on request, and never create a store', needsListing, async () => {
  await withStore(async workdir => {
    const store = path.join(workdir, 'store')
    const missing = cli(workdir, 'forget-all', store, 'toip', 'Zoom', '--apply')
    assert.equal(missing.status, 0, missing.stderr)
    assert.match(missing.stdout, /does not exist/)
    assert.equal(fs.existsSync(store), false)

    assert.equal(cli(workdir, 'import', store, 'toip', FIXTURES).status, 0)
    const preview = cli(workdir, 'forget-all', store, 'toip', 'Zoom')
    assert.equal(preview.status, 0, preview.stderr)
    assert.match(preview.stdout, /Dry run: would remove 2 document/)
    assert.equal(JSON.parse(cli(workdir, 'search', store, 'toip', 'quartzmarker').stdout).length > 0, true)

    const transcript = path.join(FIXTURES, 'zoom-example-project-transcript-GMT20260101-100000.json')
    const noMatch = cli(workdir, 'prune', store, 'toip', 'Confluence', transcript)
    assert.notEqual(noMatch.status, 0)
    assert.match(noMatch.stderr, /forget-all/)
    const pruned = cli(workdir, 'prune', store, 'toip', 'Zoom', transcript, '--apply')
    assert.equal(pruned.status, 0, pruned.stderr)
    assert.match(pruned.stdout, /Removed 1 document/)
    const cleared = cli(workdir, 'forget-all', store, 'toip', '--all', '--apply')
    assert.match(cleared.stdout, /Removed 2 document/)
    assert.equal(cli(workdir, 'forget-all', store, 'toip', '--all', 'Zoom').status, 2)

    // An existing directory that is not a store must stay untouched too.
    const plain = path.join(workdir, 'plain')
    fs.mkdirSync(plain)
    for (const args of [['forget-all', plain, 'toip', '--all'], ['forget-all', plain, 'toip', '--all', '--apply'], ['search', plain, 'toip', 'word'], ['forget', plain, 'toip', 'Zoom', 'id', '--apply'], ['prune', plain, 'toip', 'Zoom', transcript]]) {
      assert.equal(cli(workdir, ...args).status, 0, args.join(' '))
      assert.deepEqual(fs.readdirSync(plain), [], `${args[0]} must not write into a directory that is not a store`)
    }
  })
})
