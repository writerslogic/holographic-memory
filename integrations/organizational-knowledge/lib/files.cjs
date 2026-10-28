'use strict'

const fs = require('node:fs/promises')
const path = require('node:path')
const { KnowledgeError } = require('./errors.cjs')

const MAX_FILE_BYTES = 16 * 1024 * 1024
const MAX_FILES = 10_000
// Parsed documents are held in memory until the import finishes.
const MAX_TOTAL_BYTES = 512 * 1024 * 1024

async function readJsonFile(file, buffer) {
  const handle = await fs.open(file, 'r')
  try {
    let length = 0
    while (length < buffer.length) {
      const { bytesRead } = await handle.read(buffer, length, buffer.length - length, length)
      if (bytesRead === 0) break
      length += bytesRead
    }
    if (length > MAX_FILE_BYTES) throw new KnowledgeError('invalid_document', `${file}: exceeds ${MAX_FILE_BYTES} bytes`, { source: file })
    // Fatal decoding: a lossy replacement character would silently alter the original text.
    const text = new TextDecoder('utf-8', { fatal: true }).decode(buffer.subarray(0, length))
    return { document: JSON.parse(text), length }
  } catch (cause) {
    if (cause instanceof KnowledgeError) throw cause
    throw new KnowledgeError('invalid_document', `${file}: ${cause.message}`, { source: file }, { cause })
  } finally {
    await handle.close()
  }
}

async function expand(target) {
  const info = await fs.stat(target)
  if (!info.isDirectory()) return [target]
  const entries = (await fs.readdir(target, { withFileTypes: true })).filter(entry => entry.name.endsWith('.json'))
  const files = []
  for (const entry of entries.sort((a, b) => (a.name < b.name ? -1 : 1))) {
    if (!entry.isFile()) {
      throw new KnowledgeError('invalid_request', `${path.join(target, entry.name)}: expected a regular file`, { source: path.join(target, entry.name) })
    }
    files.push(path.join(target, entry.name))
  }
  if (files.length === 0) throw new KnowledgeError('invalid_request', `${target}: no .json documents found`, { source: target })
  return files
}

/**
 * Reads curated documents from files or from the `.json` files directly inside directories
 * (not recursive). Each file holds one document. Returns `{ source, document }` entries for
 * `importDocuments`. Files over 16 MiB, more than 512 MiB in total, invalid UTF-8, and invalid
 * JSON are rejected.
 */
async function readCuratedFiles(targets) {
  if (!Array.isArray(targets) || targets.length === 0 || !targets.every(target => typeof target === 'string' && target)) {
    throw new KnowledgeError('invalid_request', 'targets must be a nonempty array of paths')
  }
  const files = []
  for (const target of targets) {
    for (const file of await expand(target)) files.push(file)
    if (files.length > MAX_FILES) throw new KnowledgeError('invalid_request', `more than ${MAX_FILES} files; import in smaller sets`)
  }
  const buffer = Buffer.allocUnsafe(MAX_FILE_BYTES + 1)
  const entries = []
  let total = 0
  for (const file of files) {
    const { document, length } = await readJsonFile(file, buffer)
    total += length
    if (total > MAX_TOTAL_BYTES) throw new KnowledgeError('invalid_request', `${file}: input exceeds ${MAX_TOTAL_BYTES} bytes in total; import in smaller sets`, { source: file })
    entries.push({ source: file, document })
  }
  return entries
}

module.exports = { readCuratedFiles }
