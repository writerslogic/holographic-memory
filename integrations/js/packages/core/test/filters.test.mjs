// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0
import assert from 'node:assert/strict'
import { test } from 'node:test'
import { matches, parseFilter } from '../dist/index.js'

test('filter semantics match the Python core', () => {
  const f = parseFilter({ a: 1, b: { $in: [1, 2] }, $not: { c: true } })
  assert.ok(matches(f, { a: 1, b: 2, c: false }))
  assert.ok(!matches(f, { a: 1, b: 2, c: true }))
  assert.ok(!matches(parseFilter({ a: 1 }), { a: true }))
  assert.ok(matches(parseFilter({ a: { $ne: 1 } }), {}))
  assert.ok(matches(parseFilter({ 'k.x': 3 }), { k: { x: 3 } }))
  assert.ok(matches(parseFilter({ tags: { $contains: 'x' } }), { tags: ['x', 'y'] }))
  assert.ok(matches(parseFilter({ s: { $contains: 'ell' } }), { s: 'hello' }))
  assert.ok(!matches(parseFilter({ n: { $gt: 1 } }), { n: true }))
  assert.ok(matches(parseFilter({ n: { $exists: false } }), {}))
})

test('array, regex and nested-condition operators', () => {
  assert.ok(matches(parseFilter({ t: { $all: ['a', 'b'] } }), { t: ['b', 'c', 'a'] }))
  assert.ok(!matches(parseFilter({ t: { $all: ['a', 'z'] } }), { t: ['a'] }))
  assert.ok(matches(parseFilter({ t: { $size: 2 } }), { t: [1, 2] }))
  assert.ok(matches(parseFilter({ s: { $regex: 'ab+c', $options: 'i' } }), { s: 'xABBC' }))
  assert.ok(matches(parseFilter({ s: { $not: /^x/ } }), { s: 'yes' }))
  assert.ok(matches(parseFilter({ items: { $elemMatch: { qty: { $gt: 2 }, kind: 'a' } } }), { items: [{ qty: 1, kind: 'a' }, { qty: 3, kind: 'a' }] }))
  assert.ok(!matches(parseFilter({ items: { $elemMatch: { qty: { $gt: 2 }, kind: 'b' } } }), { items: [{ qty: 3, kind: 'a' }] }))
  assert.ok(matches(parseFilter({ $nor: [{ a: 1 }, { b: 2 }] }), { a: 2 }))
})

test('a metadata field named type is a plain equality, not an AST node', () => {
  const f = parseFilter({ type: 'eq', key: 'k' })
  assert.equal(f.type, 'and')
  assert.ok(matches(f, { type: 'eq', key: 'k' }))
})
