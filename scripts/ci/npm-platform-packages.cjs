// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0
'use strict'

// Lays out one npm package directory per native binding built with
// `napi build --platform`: <out>/<suffix>/{package.json,LICENSE,README.md,<binding>.node}.
// The root package.json owns the list of platforms (napi.targets) and the version; the
// optionalDependencies of the root must name exactly those packages at exactly that
// version, so a release can never ship a root package that selects a missing binary.
//
// usage: node scripts/ci/npm-platform-packages.cjs <artifacts-dir> <out-dir> [--allow-partial]

const assert = require('node:assert/strict')
const fs = require('node:fs')
const path = require('node:path')

const root = path.resolve(__dirname, '..', '..')
const rootPackage = JSON.parse(fs.readFileSync(path.join(root, 'package.json'), 'utf8'))

const targets = {
  'aarch64-apple-darwin': { suffix: 'darwin-arm64', os: 'darwin', cpu: 'arm64' },
  'x86_64-apple-darwin': { suffix: 'darwin-x64', os: 'darwin', cpu: 'x64' },
  'aarch64-unknown-linux-gnu': { suffix: 'linux-arm64-gnu', os: 'linux', cpu: 'arm64', libc: 'glibc' },
  'aarch64-unknown-linux-musl': { suffix: 'linux-arm64-musl', os: 'linux', cpu: 'arm64', libc: 'musl' },
  'x86_64-unknown-linux-gnu': { suffix: 'linux-x64-gnu', os: 'linux', cpu: 'x64', libc: 'glibc' },
  'x86_64-unknown-linux-musl': { suffix: 'linux-x64-musl', os: 'linux', cpu: 'x64', libc: 'musl' },
  'aarch64-pc-windows-msvc': { suffix: 'win32-arm64-msvc', os: 'win32', cpu: 'arm64' },
  'x86_64-pc-windows-msvc': { suffix: 'win32-x64-msvc', os: 'win32', cpu: 'x64' },
}

function expectedPackages() {
  return rootPackage.napi.targets.map(triple => {
    const target = targets[triple]
    assert.ok(target, `napi.targets lists ${triple}, which this script does not know`)
    return { triple, ...target, name: `${rootPackage.name}-${target.suffix}` }
  })
}

function findBindings(dir) {
  const found = new Map()
  const walk = current => {
    for (const entry of fs.readdirSync(current, { withFileTypes: true })) {
      const full = path.join(current, entry.name)
      if (entry.isDirectory()) walk(full)
      else if (entry.name.startsWith(`${rootPackage.napi.binaryName}.`) && entry.name.endsWith('.node')) {
        const suffix = entry.name.slice(rootPackage.napi.binaryName.length + 1, -'.node'.length)
        assert.ok(!found.has(suffix), `two bindings for ${suffix}: ${found.get(suffix)} and ${full}`)
        found.set(suffix, full)
      }
    }
  }
  walk(dir)
  return found
}

function verifyRoot(packages) {
  const optional = rootPackage.optionalDependencies ?? {}
  assert.deepEqual(
    Object.keys(optional).sort(),
    packages.map(p => p.name).sort(),
    'root optionalDependencies must list exactly the napi.targets platform packages',
  )
  for (const [name, version] of Object.entries(optional)) {
    assert.equal(version, rootPackage.version, `${name} is pinned to ${version}, root version is ${rootPackage.version}`)
  }
}

function writePackage(pkg, binding, outDir) {
  const dir = path.join(outDir, pkg.suffix)
  fs.rmSync(dir, { recursive: true, force: true })
  fs.mkdirSync(dir, { recursive: true })
  const file = `${rootPackage.napi.binaryName}.${pkg.suffix}.node`
  fs.copyFileSync(binding, path.join(dir, file))
  fs.copyFileSync(path.join(root, 'LICENSE'), path.join(dir, 'LICENSE'))
  fs.writeFileSync(
    path.join(dir, 'README.md'),
    `# ${pkg.name}\n\nNative binding of [${rootPackage.name}](${rootPackage.homepage}) for ${pkg.triple}. ` +
      `Install \`${rootPackage.name}\`, which selects this package automatically.\n`,
  )
  const manifest = {
    name: pkg.name,
    version: rootPackage.version,
    description: `${rootPackage.name} native binding for ${pkg.triple}`,
    license: rootPackage.license,
    author: rootPackage.author,
    homepage: rootPackage.homepage,
    repository: rootPackage.repository,
    os: [pkg.os],
    cpu: [pkg.cpu],
    ...(pkg.libc ? { libc: [pkg.libc] } : {}),
    main: file,
    files: [file, 'README.md', 'LICENSE'],
    engines: rootPackage.engines,
    publishConfig: { access: 'public' },
  }
  fs.writeFileSync(path.join(dir, 'package.json'), `${JSON.stringify(manifest, null, 2)}\n`)
  return dir
}

function main(argv) {
  const [artifacts, outDir] = argv.filter(arg => !arg.startsWith('--'))
  const partial = argv.includes('--allow-partial')
  assert.ok(artifacts && outDir, 'usage: npm-platform-packages.cjs <artifacts-dir> <out-dir> [--allow-partial]')
  const packages = expectedPackages()
  verifyRoot(packages)
  const bindings = findBindings(artifacts)
  const known = new Set(packages.map(p => p.suffix))
  for (const suffix of bindings.keys()) assert.ok(known.has(suffix), `unexpected binding ${suffix}`)
  const missing = packages.filter(p => !bindings.has(p.suffix)).map(p => p.suffix)
  assert.ok(partial || missing.length === 0, `missing bindings: ${missing.join(', ')}`)
  const written = packages.filter(p => bindings.has(p.suffix)).map(p => writePackage(p, bindings.get(p.suffix), outDir))
  console.log(JSON.stringify({ version: rootPackage.version, packages: written.map(d => path.basename(d)), missing }))
}

main(process.argv.slice(2))
