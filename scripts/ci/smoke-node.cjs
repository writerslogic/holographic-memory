// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0
'use strict'

// Installs the release tarballs into an empty project the way a user's npm would and runs
// a real document search. The root tarball's optionalDependencies are rewritten to the
// local platform tarballs, so npm itself applies the os/cpu/libc filters and must pick
// exactly the package for this machine.
//
// usage: node scripts/ci/smoke-node.cjs <tarballs-dir> --expect <suffix>

const assert = require('node:assert/strict')
const { execFileSync } = require('node:child_process')
const fs = require('node:fs')
const os = require('node:os')
const path = require('node:path')

const argv = process.argv.slice(2)
const expectAt = argv.indexOf('--expect')
const expected = expectAt >= 0 ? argv[expectAt + 1] : undefined
const tarballs = path.resolve(argv.find(arg => !arg.startsWith('--') && arg !== expected))
assert.ok(tarballs && expected, 'usage: smoke-node.cjs <tarballs-dir> --expect <suffix>')

function npm(args, cwd) {
  const windows = process.platform === 'win32'
  // Windows runs npm through its .cmd shim, which needs a shell; the arguments are fixed paths.
  return execFileSync(windows ? 'npm.cmd' : 'npm', args, { cwd, encoding: 'utf8', shell: windows, stdio: ['ignore', 'pipe', 'inherit'] })
}
const forwardSlashes = p => p.split(path.sep).join('/')

const files = fs.readdirSync(tarballs).filter(f => f.endsWith('.tgz'))
const rootTarball = files.find(f => /^writerslogic-warm-\d/.test(f))
assert.ok(rootTarball, `no root tarball in ${tarballs}: ${files.join(', ')}`)
const platformTarballs = files.filter(f => f !== rootTarball)

const work = fs.mkdtempSync(path.join(os.tmpdir(), 'warm-smoke-'))
try {
  const unpacked = path.join(work, 'root')
  fs.mkdirSync(unpacked)
  // Relative names with a fixed cwd: GNU tar on Windows reads `C:` as a remote host.
  fs.copyFileSync(path.join(tarballs, rootTarball), path.join(work, 'root.tgz'))
  execFileSync('tar', ['-xzf', 'root.tgz', '-C', 'root'], { cwd: work })
  const manifestPath = path.join(unpacked, 'package', 'package.json')
  const manifest = JSON.parse(fs.readFileSync(manifestPath, 'utf8'))
  const local = {}
  for (const name of Object.keys(manifest.optionalDependencies)) {
    const file = platformTarballs.find(f => f.startsWith(`${name}-${manifest.version}`))
    assert.ok(file, `no tarball for optional dependency ${name}`)
    local[name] = `file:${forwardSlashes(path.join(tarballs, file))}`
  }
  assert.equal(platformTarballs.length, Object.keys(local).length, 'unexpected extra tarballs')
  manifest.optionalDependencies = local
  fs.writeFileSync(manifestPath, JSON.stringify(manifest, null, 2))
  const repacked = path.join(work, 'writerslogic-warm-smoke.tgz')
  execFileSync('tar', ['-czf', 'writerslogic-warm-smoke.tgz', '-C', 'root', 'package'], { cwd: work })

  const project = path.join(work, 'project')
  fs.mkdirSync(project)
  fs.writeFileSync(path.join(project, 'package.json'), JSON.stringify({ name: 'smoke', private: true }))
  npm(['install', '--no-audit', '--no-fund', '--ignore-scripts', forwardSlashes(repacked)], project)

  const installed = fs.readdirSync(path.join(project, 'node_modules')).filter(n => n.startsWith('writerslogic-warm-'))
  assert.deepEqual(installed, [`writerslogic-warm-${expected}`], 'npm must select exactly this platform package')

  fs.writeFileSync(path.join(project, 'search.cjs'), `
    const assert = require('node:assert/strict');
    (async () => {
      const cjs = require('writerslogic-warm');
      const esm = await import('writerslogic-warm');
      assert.equal(cjs.WarmMemorySystem, esm.WarmMemorySystem);
      const { DocumentMemory } = await import('writerslogic-warm/semantic');
      const warm = new cjs.WarmMemorySystem(4096, './store');
      const memory = new DocumentMemory(warm);
      await memory.memorize({ id: 'manual', text: 'Backups protect stored documents.', sourceUri: 'manual.md' });
      await memory.memorize({ id: 'other', text: 'Rivers carry water to the sea.', sourceUri: 'other.md' });
      const hits = await memory.search('backups');
      assert.equal(hits[0].sourceUri, 'manual.md');
      await memory.flush();
      console.log('search ok on ' + process.platform + '-' + process.arch);
    })().catch(e => { console.error(e); process.exitCode = 1; });
  `)
  execFileSync(process.execPath, ['search.cjs'], { cwd: project, stdio: 'inherit', timeout: 120000 })
  console.log(`smoke passed: ${expected}`)
} finally {
  fs.rmSync(work, { recursive: true, force: true })
}
