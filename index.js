'use strict'

// Offline model packs for the local embedder, when the optional package is installed.
if (!process.env.HMS_EMBED_MODEL_DIR) {
  try {
    process.env.HMS_EMBED_MODEL_DIR = require('holographic-memory-models').packDir
  } catch {}
}

// Loads the native binding. Order: a locally built `holographic-memory.node` (development
// and the packed tarball used by `npm run test:package`), then the per-platform
// `holographic-memory.<suffix>.node` next to this file, then the optional dependency
// `holographic-memory-<suffix>` that npm installed for this os/cpu/libc.

const { existsSync, readFileSync } = require('node:fs')

function isMusl() {
  // glibc exposes its version through the diagnostic report; musl does not.
  try {
    const header = process.report && process.report.getReport().header
    if (header) return !header.glibcVersionRuntime
  } catch {
    // fall through to the file based check
  }
  try {
    return readFileSync('/usr/bin/ldd', 'utf8').includes('musl')
  } catch {
    return existsSync('/lib/ld-musl-x86_64.so.1') || existsSync('/lib/ld-musl-aarch64.so.1')
  }
}

const abi = {
  darwin: () => '',
  win32: () => '-msvc',
  linux: () => (isMusl() ? '-musl' : '-gnu'),
}[process.platform]

const supportedTargets = [
  'darwin-arm64',
  'darwin-x64',
  'linux-arm64-gnu',
  'linux-arm64-musl',
  'linux-x64-gnu',
  'linux-x64-musl',
  'win32-arm64-msvc',
  'win32-x64-msvc',
]

const suffix = abi ? `${process.platform}-${process.arch}${abi()}` : undefined
const candidates = ['./holographic-memory.node']
if (suffix) candidates.push(`./holographic-memory.${suffix}.node`, `holographic-memory-${suffix}`)

let binding
let lastError
for (const candidate of candidates) {
  try {
    binding = require(candidate)
    break
  } catch (error) {
    lastError = error
  }
}

if (!binding) {
  throw new Error(
    `Unable to load holographic-memory native binding for ${suffix ?? `${process.platform}-${process.arch}`}. ` +
    `Supported targets: ${supportedTargets.join(', ')}. Last error: ${lastError?.message ?? 'none'}`,
  )
}

module.exports = binding
