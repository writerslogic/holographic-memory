'use strict'

const platformPackage = {
  'darwin-arm64': 'holographic-memory-darwin-arm64',
  'darwin-x64': 'holographic-memory-darwin-x64',
  'linux-x64': 'holographic-memory-linux-x64-gnu',
  'win32-x64': 'holographic-memory-win32-x64-msvc',
}[`${process.platform}-${process.arch}`]

const candidates = ['./holographic-memory.node', platformPackage].filter(Boolean)
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
  const supported = Object.keys({
    'darwin-arm64': true,
    'darwin-x64': true,
    'linux-x64': true,
    'win32-x64': true,
  }).join(', ')
  throw new Error(
    `Unable to load holographic-memory native binding for ${process.platform}-${process.arch}. ` +
    `Supported targets: ${supported}. Last error: ${lastError?.message ?? 'none'}`,
  )
}

module.exports = binding
