#!/usr/bin/env node
'use strict'

const fs = require('node:fs')
const os = require('node:os')
const path = require('node:path')
const { WarmMemorySystem } = require('writerslogic-warm')
const { OrganizationalKnowledge, documentId, readCuratedFiles, validateImport } = require('../index.cjs')

// Same dimensions and default configuration as examples/organizational_knowledge.rs,
// so both tools can open the same store directory (one writer at a time).
const DIMENSIONS = 4096

const usage = `usage:
  cli.cjs import STORE NAMESPACE PATH...        PATH: curated .json file or directory of them
  cli.cjs search STORE NAMESPACE QUERY [--k N]
  cli.cjs forget STORE NAMESPACE SOURCE_SYSTEM SOURCE_ID [--apply]
  cli.cjs forget-all STORE NAMESPACE (SOURCE_SYSTEM | --all) [--limit N] [--apply]
  cli.cjs prune STORE NAMESPACE SOURCE_SYSTEM PATH... [--limit N] [--apply]
                                                keeps the SOURCE_SYSTEM documents in PATH, removes the rest of its documents`

// The engine's chunking limits can only be checked on an open store, and opening one creates it.
// A throwaway store keeps an over-limit document from leaving STORE behind.
async function validateBeforeOpening(namespace, entries) {
  const scratch = fs.mkdtempSync(path.join(os.tmpdir(), 'ok-validate-'))
  try {
    await validateImport(namespace, entries, new WarmMemorySystem(DIMENSIONS, scratch))
  } finally {
    fs.rmSync(scratch, { recursive: true, force: true })
  }
}

const fail = message => {
  console.error(message)
  process.exit(2)
}

function parseFlags(args) {
  const flags = { apply: false, all: false, limit: undefined, positional: [] }
  for (let index = 0; index < args.length; index++) {
    const arg = args[index]
    if (arg === '--apply') flags.apply = true
    else if (arg === '--all') flags.all = true
    else if (arg === '--limit') {
      flags.limit = Number(args[++index])
      if (!Number.isInteger(flags.limit)) fail(usage)
    } else if (arg.startsWith('--')) fail(usage)
    else flags.positional.push(arg)
  }
  return flags
}

// Opening a path creates a store there, even in an existing directory, so read-only and
// removal commands check for one first.
const isStore = store => fs.existsSync(path.join(store, 'store.json'))

async function forgetScope(store, namespace, scope, { apply, limit }) {
  documentId(namespace, 'source', 'id') // rejects an unusable namespace before the store opens
  if (!isStore(store)) return console.log('The store does not exist; nothing to remove')
  const knowledge = new OrganizationalKnowledge(new WarmMemorySystem(DIMENSIONS, store), { namespace })
  const result = await knowledge.forgetScope(scope, limit === undefined ? { apply } : { apply, limit })
  if (!apply) {
    for (const target of result.targets) console.log(target.documentId)
    const more = result.truncated ? '; more match than --limit' : ''
    console.log(`Dry run: would remove ${result.targets.length} document(s)${more}; pass --apply to execute`)
  } else {
    const more = result.truncated ? '; more match, run again to continue' : ''
    console.log(`Removed ${result.removed.length} document(s), ${result.absent.length} already absent${more}`)
  }
}

async function main(argv) {
  const [command, store, namespace, ...rest] = argv
  if (!command || !store || !namespace) fail(usage)

  if (command === 'import') {
    if (rest.length === 0) fail(usage)
    const entries = await readCuratedFiles(rest)
    await validateBeforeOpening(namespace, entries)
    const knowledge = new OrganizationalKnowledge(new WarmMemorySystem(DIMENSIONS, store), { namespace })
    const result = await knowledge.importDocuments(entries)
    console.log(`Imported ${result.documents.length} document(s), ${result.chunks} chunk(s)`)
  } else if (command === 'search') {
    documentId(namespace, 'source', 'id') // rejects an unusable namespace before the store opens
    const flag = rest.indexOf('--k')
    const k = flag === -1 ? 10 : Number(rest[flag + 1])
    const words = flag === -1 ? rest : rest.slice(0, flag)
    if (words.length !== 1 || !Number.isInteger(k) || k < 1) fail(usage)
    if (!isStore(store)) return console.log('[]')
    const knowledge = new OrganizationalKnowledge(new WarmMemorySystem(DIMENSIONS, store), { namespace })
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
    if (!isStore(store)) return console.log('Removed: false')
    const knowledge = new OrganizationalKnowledge(new WarmMemorySystem(DIMENSIONS, store), { namespace })
    const result = await knowledge.forget([{ sourceSystem, id }], { apply: true })
    console.log(`Removed: ${result.removed.length === 1}`)
  } else if (command === 'forget-all') {
    const { positional, apply, all, limit } = parseFlags(rest)
    if (all ? positional.length !== 0 : positional.length !== 1) fail(usage)
    await forgetScope(store, namespace, all ? { all: true } : { sourceSystem: positional[0] }, { apply, limit })
  } else if (command === 'prune') {
    const { positional: [sourceSystem, ...paths], apply, all, limit } = parseFlags(rest)
    if (!sourceSystem || paths.length === 0 || all) fail(usage)
    const entries = await readCuratedFiles(paths)
    await validateBeforeOpening(namespace, entries)
    const keep = entries.filter(({ document }) => document.source_system === sourceSystem).map(({ document }) => ({ sourceSystem, id: document.id }))
    // An empty keep list would remove the whole source system; forget-all says that explicitly.
    if (keep.length === 0) fail(`no ${sourceSystem} documents in the given paths; use forget-all to remove a source system`)
    await forgetScope(store, namespace, { sourceSystem, keep }, { apply, limit })
  } else {
    fail(usage)
  }
}

main(process.argv.slice(2)).catch(error => {
  console.error(error.message)
  process.exit(1)
})
