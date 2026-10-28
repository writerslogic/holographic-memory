#!/usr/bin/env node
'use strict'

const { HolographicMemorySystem } = require('holographic-memory')
const { OrganizationalKnowledge, documentId, prepareImport, readCuratedFiles } = require('../index.cjs')

// Same dimensions and default configuration as examples/organizational_knowledge.rs,
// so both tools can open the same store directory (one writer at a time).
const DIMENSIONS = 4096

const usage = `usage:
  cli.cjs import STORE NAMESPACE PATH...        PATH: curated .json file or directory of them
  cli.cjs search STORE NAMESPACE QUERY [--k N]
  cli.cjs forget STORE NAMESPACE SOURCE_SYSTEM SOURCE_ID [--apply]`

const fail = message => {
  console.error(message)
  process.exit(2)
}

async function main(argv) {
  const [command, store, namespace, ...rest] = argv
  if (!command || !store || !namespace) fail(usage)

  if (command === 'import') {
    if (rest.length === 0) fail(usage)
    const entries = await readCuratedFiles(rest)
    // Adapter rules are checked before the store is opened, so such input never creates it. The
    // engine's chunking limits can only be checked on an open store; importDocuments does that
    // before its first write, so a document over those limits leaves an empty store behind.
    prepareImport(namespace, entries)
    const knowledge = new OrganizationalKnowledge(new HolographicMemorySystem(DIMENSIONS, store), { namespace })
    const result = await knowledge.importDocuments(entries)
    console.log(`Imported ${result.documents.length} document(s), ${result.chunks} chunk(s)`)
  } else if (command === 'search') {
    documentId(namespace, 'source', 'id') // rejects an unusable namespace before the store opens
    const flag = rest.indexOf('--k')
    const k = flag === -1 ? 10 : Number(rest[flag + 1])
    const words = flag === -1 ? rest : rest.slice(0, flag)
    if (words.length !== 1 || !Number.isInteger(k) || k < 1) fail(usage)
    const knowledge = new OrganizationalKnowledge(new HolographicMemorySystem(DIMENSIONS, store), { namespace })
    const hits = await knowledge.search(words[0], { k })
    console.log(JSON.stringify(hits, null, 2))
  } else if (command === 'forget') {
    const apply = rest.at(-1) === '--apply'
    const [sourceSystem, id, ...extra] = apply ? rest.slice(0, -1) : rest
    if (!sourceSystem || !id || extra.length > 0) fail(usage)
    const key = documentId(namespace, sourceSystem, id)
    if (!apply) {
      // A preview never opens the store.
      console.log(`Dry run: would remove ${key}; pass --apply to execute`)
      return
    }
    const knowledge = new OrganizationalKnowledge(new HolographicMemorySystem(DIMENSIONS, store), { namespace })
    const result = await knowledge.forget([{ sourceSystem, id }], { apply: true })
    console.log(`Removed: ${result.removed.length === 1}`)
  } else {
    fail(usage)
  }
}

main(process.argv.slice(2)).catch(error => {
  console.error(error.message)
  process.exit(1)
})
