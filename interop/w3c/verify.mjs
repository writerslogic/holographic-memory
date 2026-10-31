// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {inspect} from 'node:util';
import {gunzipSync} from 'node:zlib';
import {DataIntegrityProof} from '@digitalbazaar/data-integrity';
import {driver} from '@digitalbazaar/did-method-key';
import {createVerifyCryptosuite} from '@digitalbazaar/eddsa-jcs-2022-cryptosuite';
import * as Ed25519Multikey from '@digitalbazaar/ed25519-multikey';
import {checkStatus as checkBitstringStatus, decodeList} from '@digitalbazaar/vc-bitstring-status-list';
import * as vc from '@digitalbazaar/vc';
import jsigs from 'jsonld-signatures';

const input = JSON.parse(await readFile(process.argv[2], 'utf8'));
const warmContext = JSON.parse(await readFile(new URL('../../docs/reference/fixtures/warm-v1-context.jsonld', import.meta.url), 'utf8'));
const didKey = driver();
didKey.use({
  multibaseMultikeyHeader: 'z6Mk',
  fromMultibase: Ed25519Multikey.from
});
const documentLoader = async url => {
  if(!url.startsWith('did:key:')) {
    if(url === input.statusList.id) {
      return {documentUrl: url, document: input.statusList};
    }
    if(url === 'https://docs.holographic-memory.com/ns/hms/v1.jsonld') {
      return {documentUrl: url, document: warmContext};
    }
    return vc.defaultDocumentLoader(url);
  }
  return {documentUrl: url, document: await didKey.get({url})};
};
const suite = new DataIntegrityProof({cryptosuite: createVerifyCryptosuite()});
for(const [name, credential] of Object.entries(input)) {
  const result = await jsigs.verify(credential, {
    suite,
    purpose: new jsigs.purposes.AssertionProofPurpose(),
    documentLoader
  });
  assert.equal(result.verified, true, `${name}: ${inspect(result.error, {depth: 6})}`);
  const vcResult = await vc.verifyCredential({
    credential,
    suite,
    documentLoader,
    checkStatus: async options => {
      const result = await checkBitstringStatus(options);
      return {
        ...result,
        verified: result.verified && result.results.every(({status}) => status === false)
      };
    }
  });
  assert.equal(vcResult.verified, name !== 'revokedCredential', `${name} VC model: ${inspect(vcResult, {depth: 6})}`);
  if(name === 'revokedCredential') {
    assert.equal(vcResult.statusResult.results[0].status, true);
  } else if(name === 'credential') {
    assert.equal(vcResult.statusResult.results[0].status, false);
  }
  const tampered = structuredClone(credential);
  if(name !== 'statusList') {
    tampered.credentialSubject.contentHash = '00'.repeat(32);
  } else {
    tampered.credentialSubject.encodedList += 'x';
  }
  const rejected = await jsigs.verify(tampered, {
    suite,
    purpose: new jsigs.purposes.AssertionProofPurpose(),
    documentLoader
  });
  assert.equal(rejected.verified, false, `${name} tampering was accepted`);
}
const encoded = input.statusList.credentialSubject.encodedList;
assert.ok(encoded.startsWith('u'));
const statusList = await decodeList({encodedList: encoded});
assert.ok(statusList.length >= 131072);
assert.equal(statusList.getStatus(0), false);
assert.equal(statusList.getStatus(7), false);
assert.equal(statusList.getStatus(8), true);
const wire = gunzipSync(Buffer.from(encoded.slice(1), 'base64url'));
assert.ok(wire.length >= 16 * 1024);
assert.equal(wire[0], 0);
assert.equal(wire[1] & 0x80, 0x80);
console.log('Digital Bazaar verified VC proofs and status list, accepted index 7, rejected revoked index 8, and rejected tampering');
