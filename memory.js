// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0
'use strict'

/** Agent memory over a WarmMemorySystem: remember, recall, forget, explain. See docs/memory-api. */

class MemoryError extends Error {
  constructor(message, code, cause) {
    super(message, cause === undefined ? undefined : { cause })
    this.name = 'MemoryError'
    this.code = code
  }
}

const TIME_FIELDS = ['eventTime', 'validFrom', 'validTo', 'asOf', 'knownAt']

function millis(value, name) {
  if (value === undefined || value === null) return undefined
  if (value instanceof Date) {
    const time = value.getTime()
    if (Number.isNaN(time)) throw new TypeError(`${name} is an invalid Date`)
    return time
  }
  if (typeof value === 'number' && Number.isSafeInteger(value)) return value
  throw new TypeError(`${name} must be a Date or an integer of Unix milliseconds`)
}

function wire(options, extra = {}) {
  const request = { ...options, ...extra }
  for (const field of TIME_FIELDS) {
    if (field in request) {
      const value = millis(request[field], field)
      if (value === undefined) delete request[field]
      else request[field] = value
    }
  }
  for (const key of Object.keys(request)) if (request[key] === undefined) delete request[key]
  return request
}

const SCOPES = {
  ids: ids => ({ kind: 'ids', ids: [...ids] }),
  subject: subject => ({ kind: 'subject', subject }),
  entity: entity => ({ kind: 'entity', entity }),
  ingestedBefore: time => ({ kind: 'ingestedBefore', time: millis(time, 'ingestedBefore') }),
  eventBefore: time => ({ kind: 'eventBefore', time: millis(time, 'eventBefore') }),
  matching: filter => ({ kind: 'matching', filter }),
}

function scope(input) {
  const keys = Object.keys(input ?? {}).filter(key => key in SCOPES && input[key] !== undefined)
  if (keys.length !== 1) {
    throw new TypeError(`forget takes exactly one scope of ${Object.keys(SCOPES).join(', ')}`)
  }
  return SCOPES[keys[0]](input[keys[0]])
}

class Memory {
  /**
   * @param {import('./index').WarmMemorySystem} system
   * @param {object} [config] policies and defaults; omitted fields keep their do-nothing defaults.
   */
  constructor(system, config) {
    if (!system || typeof system.memoryCall !== 'function') throw new TypeError('a WarmMemorySystem is required')
    this.system = system
    system.configureMemory(config ?? null)
  }

  async #call(op, request) {
    try {
      return await this.system.memoryCall(op, request)
    } catch (error) {
      const match = /^\[(\w+)\] ([\s\S]*)$/.exec(String(error?.message ?? error))
      throw match ? new MemoryError(match[2], match[1], error) : error
    }
  }

  async remember(input, options = {}) {
    const fact = typeof input === 'string' ? { ...options, text: input } : input
    const { remembered } = await this.#call('remember', wire(fact))
    return remembered[0]
  }

  async rememberMany(inputs) {
    const { remembered } = await this.#call('remember', { facts: [...inputs].map(input => wire(input)) })
    return remembered
  }

  async recall(query, options = {}) {
    const { results } = await this.#call('recall', wire(options, { query }))
    return results
  }

  async capabilities(contractVersion = 1) {
    return (await this.#call('capabilities', { contractVersion })).capabilities
  }

  async explain(query, id, options = {}) {
    const { explanation } = await this.#call('explain', wire(options, { query, id }))
    return explanation
  }

  async forget(target, options = {}) {
    return this.#call('forget', { ...options, scope: scope(target) })
  }

  async get(id) {
    return (await this.#call('get', { id })).fact
  }

  async history(id) {
    return (await this.#call('history', { id })).history
  }

  applyForgetting(options = {}) {
    return this.#call('applyForgetting', options)
  }

  consolidate(options = {}) {
    return this.#call('consolidate', options)
  }
}

module.exports = { Memory, MemoryError }
