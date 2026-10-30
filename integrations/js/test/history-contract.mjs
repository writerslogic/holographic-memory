// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0
import assert from 'node:assert/strict'
import { test } from 'node:test'

export function historyRecovery({ name, store, create, append, read, clear }) {
  for (const boundary of [1, 2, 3]) {
    test(`${name} clear retries after committed deletion ${boundary}`, async () => {
      const s = store()
      const history = create(s)
      await append(history, ['old first', 'old second', 'old third'])
      const forget = s.forget.bind(s)
      let deleted = 0
      s.forget = async (ids) => {
        for (const id of ids) {
          await forget([id])
          if (++deleted === boundary) throw new Error('interrupted clear')
        }
        return ids.length
      }
      await assert.rejects(clear(history), /interrupted clear/)
      s.forget = forget
      const retry = create(s)
      assert.deepEqual(await read(retry), ['old first', 'old second', 'old third'].slice(0, 3 - boundary))
      await clear(retry)
      assert.deepEqual((await s.list()).records, [])
      await append(retry, ['new first'])
      assert.deepEqual(await read(retry), ['new first'])
      await s.close()
    })
  }
  test(`${name} serializes competing session wrappers`, async () => {
    const s = store()
    const a = create(s)
    const b = create(s)
    await Promise.all([append(a, ['first']), append(b, ['second'])])
    assert.deepEqual(await read(a), ['first', 'second'])
    await Promise.all([clear(a), append(b, ['after clear'])])
    assert.deepEqual(await read(a), ['after clear'])
    await s.close()
  })
}
