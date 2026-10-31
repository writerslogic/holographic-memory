// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

const assert = require('node:assert/strict')
const { test } = require('node:test')
const { mkdtempSync, rmSync, cpSync } = require('node:fs')
const { tmpdir } = require('node:os')
const { join } = require('node:path')
const { spawnSync } = require('node:child_process')

function run(directory, legacy, reopen, signing) {
  const source = `
    const assert = require('node:assert/strict');
    const { generateKeyPairSync, createPublicKey, verify } = require('node:crypto');
    const { readFileSync, writeFileSync } = require('node:fs');
    const { join } = require('node:path');
    const { WarmMemorySystem } = require('./index.js');
    (async () => {
      const directory = process.argv[1];
      const legacy = process.argv[2] === 'true';
      const reopen = process.argv[3] === 'true';
      const signing = process.argv[4] === 'true';
      const keyPath = join(directory, 'fixture-signing.key');
      const publicPath = join(directory, 'fixture-public.pem');
      if (!reopen) {
        const { privateKey, publicKey } = generateKeyPairSync('ed25519');
        const bytes = Buffer.from(privateKey.export({ format: 'jwk' }).d, 'base64url');
        writeFileSync(keyPath, bytes, { mode: 0o600 });
        bytes.fill(0);
        writeFileSync(publicPath, publicKey.export({ format: 'pem', type: 'spki' }));
      }
      const publicKey = createPublicKey(readFileSync(publicPath));
      const warm = new WarmMemorySystem(legacy ? 1024 : 256, directory, {
        auditEnabled: true, signingEnabled: signing, signingKeyPath: keyPath,
      });
      if (!reopen) {
        await warm.memorizeText('audit-contract', 'independently verifiable record');
        await warm.delete('audit-contract');
        await warm.compact();
      }
      const entries = (await warm.auditSince(0)).slice(-3);
      assert.deepEqual(entries.map(entry => entry.op), ['memorize', 'delete', 'compact']);
      for (const entry of entries) {
        const message = Buffer.from(entry.signableBytes, 'hex');
        const signature = Buffer.from(entry.signature, 'hex');
        assert.equal(entry.signed, signing);
        assert.equal(signature.length, 64);
        assert.equal(message.length, legacy ? 41 : 70);
        let offset = 0;
        if (legacy) {
          assert.equal(entry.effectId, undefined);
        } else {
          assert.match(entry.effectId, /^[0-9a-f]{32}$/);
          assert.deepEqual(message.subarray(0, 13), Buffer.from('WARM-AUDIT-2\\0'));
          assert.equal(message.subarray(13, 29).toString('hex'), entry.effectId);
          offset = 29;
        }
        assert.equal(message.readBigUInt64LE(offset), BigInt(entry.timestampMs));
        assert.equal(message[offset + 8], { memorize: 1, delete: 2, compact: 3 }[entry.op]);
        assert.equal(message.subarray(offset + 9).toString('hex'), entry.idHash);
        assert.equal(verify(null, message, publicKey, signature), signing);
        if (!signing) assert.deepEqual(signature, Buffer.alloc(64));
        for (const index of [0, legacy ? 8 : 13]) {
          message[index] ^= 1;
          assert.equal(verify(null, message, publicKey, signature), false);
          message[index] ^= 1;
        }
      }
      if (!legacy) assert.equal(new Set(entries.map(entry => entry.effectId)).size, 3);
      for (const timestamp of [0, 1, Number.MAX_SAFE_INTEGER - 1, Number.MAX_SAFE_INTEGER]) {
        assert.ok(Array.isArray(await warm.auditSince(timestamp)));
      }
      for (const timestamp of [-1, 0.5, NaN, Infinity, -Infinity, Number.MAX_SAFE_INTEGER + 1]) {
        await assert.rejects(warm.auditSince(timestamp), /nonnegative safe integer/);
      }
      await warm.flush();
    })().catch(error => { console.error(error); process.exitCode = 1; });
  `
  const result = spawnSync(process.execPath, ['-e', source, directory, String(legacy), String(reopen), String(signing)], {
    cwd: process.cwd(), encoding: 'utf8', timeout: 60000,
  })
  assert.equal(result.status, 0, result.stderr || result.stdout)
}

for (const [legacy, signing] of [[false, true], [true, true], [false, false]]) {
  test(`audit ${signing ? 'signed' : 'unsigned'} entries preserve ${legacy ? 'legacy' : 'delivery-aware'} bytes across restart`, () => {
    const directory = mkdtempSync(join(tmpdir(), 'warm-audit-node-'))
    try {
      if (legacy) cpSync(join(__dirname, '../golden/v4-audit'), directory, { recursive: true })
      run(directory, legacy, false, signing)
      run(directory, legacy, true, signing)
    } finally {
      rmSync(directory, { recursive: true, force: true })
    }
  })
}
