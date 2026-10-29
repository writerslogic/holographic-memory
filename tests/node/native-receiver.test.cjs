// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

const assert = require('node:assert/strict')
const { test } = require('node:test')
const { HolographicMemorySystem } = require('../../index.js')

test('native methods reject unwrapped and prototype-spoofed receivers', async () => {
  const receivers = [{}, Object.create(HolographicMemorySystem.prototype), new Date(), null]
  for (const receiver of receivers) {
    assert.throws(() => HolographicMemorySystem.prototype.configureMemory.call(receiver), TypeError)
    await assert.rejects(async () => HolographicMemorySystem.prototype.listDocuments.call(receiver), TypeError)
    const getter = Object.getOwnPropertyDescriptor(HolographicMemorySystem.prototype, 'dimensions').get
    assert.throws(() => getter.call(receiver), /Illegal invocation|Failed to unwrap exclusive reference/)
  }
})
