const assert = require('node:assert/strict')
const { execFileSync } = require('node:child_process')
const { mkdtempSync, rmSync, writeFileSync } = require('node:fs')
const { tmpdir } = require('node:os')
const { join } = require('node:path')

const temporary = mkdtempSync(join(tmpdir(), 'hms-package-'))
// Invoke npm's JavaScript CLI directly; Windows cannot execFile a .cmd shim.
const npmCli = process.env.npm_execpath
if (!npmCli) throw new Error('Run this check with npm run test:package')
try {
  const [packed] = JSON.parse(execFileSync(process.execPath, [npmCli, 'pack', '--json', '--ignore-scripts', '--pack-destination', temporary], { encoding: 'utf8' }))
  assert.ok(packed.files.some(file => file.path.endsWith('.node')), 'build the native binding before testing the tarball')
  writeFileSync(join(temporary, 'package.json'), JSON.stringify({ private: true }))
  execFileSync(process.execPath, [npmCli, 'install', '--ignore-scripts', '--omit=optional', '--no-audit', join(temporary, packed.filename)], { cwd: temporary, stdio: 'pipe' })
  execFileSync(process.execPath, ['-e', `
    const assert = require('node:assert/strict');
    (async () => {
      const cjs = require('holographic-memory');
      const esm = await import('holographic-memory');
      assert.equal(cjs.HolographicMemorySystem, esm.HolographicMemorySystem);
      const { DocumentMemory } = await import('holographic-memory/semantic');
      const hms = new cjs.HolographicMemorySystem(4096, './store');
      const memory = new DocumentMemory(hms);
      await memory.memorize({ id: 'manual', text: 'Backups protect stored documents.', sourceUri: 'manual.md' });
      assert.equal((await memory.search('backups'))[0].sourceUri, 'manual.md');
      await memory.flush();
    })().catch(e => { console.error(e); process.exitCode = 1; });
  `], { cwd: temporary, stdio: 'inherit', timeout: 60000 })
  console.log('Packed CommonJS, ESM, native binding, and document APIs passed')
} finally { rmSync(temporary, { recursive: true, force: true }) }
