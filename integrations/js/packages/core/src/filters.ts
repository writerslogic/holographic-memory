// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

/**
 * Metadata filter AST, equal in semantics to the Python core.
 *
 * Top-level equality terms ANDed together are pushed to the native engine, which evaluates them
 * before ranking. Every other term (comparison, in, or, not, nested keys, existence) is evaluated
 * here on an over-fetched candidate list, so such filters can return fewer than `k` results.
 */

export type Json = null | boolean | number | string | Json[] | { [key: string]: Json }
export type Metadata = Record<string, unknown>

export type Filter =
  | { type: 'eq'; key: string; value: unknown }
  | { type: 'cmp'; key: string; op: CmpOp; value: unknown }
  | { type: 'and'; terms: Filter[] }
  | { type: 'or'; terms: Filter[] }
  | { type: 'not'; term: Filter }

export type CmpOp = 'ne' | 'gt' | 'gte' | 'lt' | 'lte' | 'in' | 'nin' | 'exists' | 'contains' | 'all' | 'size' | 'regex' | 'elemMatch'
export type FilterLike = Filter | Record<string, unknown> | null | undefined

const MISSING = Symbol('missing')

function lookup(metadata: Metadata, key: string): unknown {
  if (key in metadata) return metadata[key]
  let cur: unknown = metadata
  for (const part of key.split('.')) {
    if (cur !== null && typeof cur === 'object' && !Array.isArray(cur) && part in (cur as object)) {
      cur = (cur as Record<string, unknown>)[part]
    } else return MISSING
  }
  return cur
}

export function deepEqual(a: unknown, b: unknown): boolean {
  if (typeof a !== typeof b) return false
  if (a === null || b === null || typeof a !== 'object') return a === b
  if (Array.isArray(a) !== Array.isArray(b)) return false
  if (Array.isArray(a)) {
    const bb = b as unknown[]
    return a.length === bb.length && a.every((v, i) => deepEqual(v, bb[i]))
  }
  const ao = a as Record<string, unknown>
  const bo = b as Record<string, unknown>
  const ak = Object.keys(ao)
  return ak.length === Object.keys(bo).length && ak.every((k) => k in bo && deepEqual(ao[k], bo[k]))
}

function ordered(a: unknown, b: unknown, op: 'gt' | 'gte' | 'lt' | 'lte'): boolean {
  const ok = (typeof a === 'number' && typeof b === 'number') || (typeof a === 'string' && typeof b === 'string')
  if (!ok) return false
  const x = a as number | string
  const y = b as number | string
  return op === 'gt' ? x > y : op === 'gte' ? x >= y : op === 'lt' ? x < y : x <= y
}

export function matches(filter: Filter, metadata: Metadata): boolean {
  switch (filter.type) {
    case 'eq': {
      const found = lookup(metadata, filter.key)
      return found !== MISSING && deepEqual(found, filter.value)
    }
    case 'and':
      return filter.terms.every((t) => matches(t, metadata))
    case 'or':
      return filter.terms.some((t) => matches(t, metadata))
    case 'not':
      return !matches(filter.term, metadata)
    case 'cmp': {
      const found = lookup(metadata, filter.key)
      const present = found !== MISSING
      const list = (): unknown[] => {
        if (!Array.isArray(filter.value)) throw new TypeError(`${filter.op} takes an array`)
        return filter.value
      }
      switch (filter.op) {
        case 'exists':
          return present === Boolean(filter.value)
        case 'ne':
          return !present || !deepEqual(found, filter.value)
        case 'nin':
          return !present || !list().some((v) => deepEqual(found, v))
        case 'in':
          return present && list().some((v) => deepEqual(found, v))
        case 'gt':
        case 'gte':
        case 'lt':
        case 'lte':
          return present && ordered(found, filter.value, filter.op)
        case 'all':
          return present && Array.isArray(found) && list().every((v) => (found as unknown[]).some((f) => deepEqual(f, v)))
        case 'size':
          return present && Array.isArray(found) && found.length === filter.value
        case 'regex': {
          if (!present || typeof found !== 'string') return false
          const spec = filter.value as { source: string; flags: string }
          return new RegExp(spec.source, spec.flags).test(found)
        }
        case 'elemMatch': {
          if (!present || !Array.isArray(found)) return false
          const sub = filter.value as Filter
          return found.some((el) =>
            el !== null && typeof el === 'object' && !Array.isArray(el) ? matches(sub, el as Metadata) : matches(sub, { '': el }),
          )
        }
        case 'contains':
          if (!present) return false
          if (typeof found === 'string' && typeof filter.value === 'string') return found.includes(filter.value)
          if (Array.isArray(found)) return found.some((v) => deepEqual(v, filter.value))
          return false
      }
    }
  }
}

export function allOf(terms: Filter[]): Filter | undefined {
  const flat: Filter[] = []
  for (const t of terms) {
    if (t.type === 'and') flat.push(...t.terms)
    else flat.push(t)
  }
  if (flat.length === 0) return undefined
  return flat.length === 1 ? flat[0] : { type: 'and', terms: flat }
}

const OPS: Record<string, 'eq' | CmpOp> = {
  $eq: 'eq', $ne: 'ne', $gt: 'gt', $gte: 'gte', $lt: 'lt', $lte: 'lte',
  $in: 'in', $nin: 'nin', $exists: 'exists', $contains: 'contains', $all: 'all', $size: 'size',
}

const SHAPES: Record<string, string> = {
  eq: 'key,type,value',
  cmp: 'key,op,type,value',
  and: 'terms,type',
  or: 'terms,type',
  not: 'term,type',
}

/** True only for the exact AST shapes, so a metadata field named `type` is not mistaken for one. */
function isFilter(v: unknown): v is Filter {
  if (typeof v !== 'object' || v === null) return false
  const t = (v as { type?: unknown }).type
  return typeof t === 'string' && SHAPES[t] === Object.keys(v).sort().join(',')
}

/**
 * Build a filter from a MongoDB-style object: `{a: 1}` is equality, `{a: {$gt: 1}}` a comparison,
 * `$and`/`$or` take arrays, `$not` an object. Several keys are ANDed. A Filter is returned as is.
 */
export function parseFilter(spec: FilterLike): Filter | undefined {
  if (spec === null || spec === undefined) return undefined
  if (isFilter(spec)) return spec
  if (typeof spec !== 'object' || Array.isArray(spec)) throw new TypeError('filter must be an object')
  const terms: Filter[] = []
  for (const [key, value] of Object.entries(spec)) {
    if (key === '$and' || key === '$or') {
      if (!Array.isArray(value)) throw new TypeError(`${key} takes an array of filters`)
      const subs = value.map((v) => parseFilter(v as FilterLike)).filter((f): f is Filter => f !== undefined)
      if (key === '$and') {
        if (subs.length) terms.push({ type: 'and', terms: subs })
      } else terms.push({ type: 'or', terms: subs })
    } else if (key === '$nor') {
      if (!Array.isArray(value)) throw new TypeError('$nor takes an array of filters')
      const subs = value.map((v) => parseFilter(v as FilterLike)).filter((f): f is Filter => f !== undefined)
      terms.push({ type: 'not', term: { type: 'or', terms: subs } })
    } else if (key === '$not') {
      const sub = parseFilter(value as FilterLike)
      if (sub) terms.push({ type: 'not', term: sub })
    } else if (key.startsWith('$')) {
      throw new TypeError(`unsupported top-level operator ${key}`)
    } else if (value !== null && typeof value === 'object' && !Array.isArray(value) &&
      Object.keys(value).length > 0 && Object.keys(value).every((k) => k.startsWith('$'))) {
      const ops = value as Record<string, unknown>
      for (const [opKey, operand] of Object.entries(ops)) {
        if (opKey === '$options') continue
        if (opKey === '$regex') {
          const source = operand instanceof RegExp ? operand.source : String(operand)
          const flags = operand instanceof RegExp ? operand.flags : String(ops['$options'] ?? '')
          terms.push({ type: 'cmp', key, op: 'regex', value: { source, flags } })
          continue
        }
        if (opKey === '$not') {
          const inner = operand instanceof RegExp ? { $regex: operand } : (operand as Record<string, unknown>)
          const sub = parseFilter({ [key]: inner })
          if (sub) terms.push({ type: 'not', term: sub })
          continue
        }
        if (opKey === '$elemMatch') {
          const sub = parseFilter(operand as FilterLike)
          if (sub) terms.push({ type: 'cmp', key, op: 'elemMatch', value: sub })
          continue
        }
        const op = OPS[opKey]
        if (!op) throw new TypeError(`unsupported operator ${opKey}`)
        terms.push(op === 'eq' ? { type: 'eq', key, value: operand } : { type: 'cmp', key, op, value: operand })
      }
    } else terms.push({ type: 'eq', key, value })
  }
  return allOf(terms)
}

/** Separate the part the engine evaluates (top-level equalities) from the residual evaluated here. */
export function splitFilter(filter: Filter | undefined): { pushed: Record<string, unknown>; residual: Filter | undefined } {
  if (!filter) return { pushed: {}, residual: undefined }
  const terms = filter.type === 'and' ? filter.terms : [filter]
  const pushed: Record<string, unknown> = {}
  const rest: Filter[] = []
  for (const t of terms) {
    if (t.type === 'eq' && !t.key.includes('.') && !(t.key in pushed) && isPushable(t.value)) pushed[t.key] = t.value
    else rest.push(t)
  }
  return { pushed, residual: allOf(rest) }
}

/** The engine's equality is exact for scalars; arrays and objects are matched here instead. */
function isPushable(v: unknown): boolean {
  return v === null || ['string', 'number', 'boolean'].includes(typeof v)
}
