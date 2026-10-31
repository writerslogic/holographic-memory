// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0
import { WarmError, type Filter } from '@writerslogic/warm-core'
import type { MetadataFilter, MetadataFilters } from '@llamaindex/core/vector-store'

function list(f: MetadataFilter): unknown[] {
  if (!Array.isArray(f.value)) throw new WarmError(`operator ${f.operator} on "${f.key}" needs an array value`, 'invalid_argument')
  return f.value
}

function one(f: MetadataFilter): Filter {
  const { key, value } = f
  switch (f.operator) {
    case '==':
      return { type: 'eq', key, value }
    case '!=':
      return { type: 'cmp', key, op: 'ne', value }
    case '>':
      return { type: 'cmp', key, op: 'gt', value }
    case '<':
      return { type: 'cmp', key, op: 'lt', value }
    case '>=':
      return { type: 'cmp', key, op: 'gte', value }
    case '<=':
      return { type: 'cmp', key, op: 'lte', value }
    case 'in':
      return { type: 'cmp', key, op: 'in', value: list(f) }
    case 'nin':
      return { type: 'cmp', key, op: 'nin', value: list(f) }
    case 'contains':
      return { type: 'cmp', key, op: 'contains', value }
    case 'text_match':
      return { type: 'cmp', key, op: 'contains', value }
    case 'any':
      return { type: 'or', terms: list(f).map((v): Filter => ({ type: 'cmp', key, op: 'contains', value: v })) }
    case 'all':
      return { type: 'and', terms: list(f).map((v): Filter => ({ type: 'cmp', key, op: 'contains', value: v })) }
    case 'is_empty':
      return {
        type: 'or',
        terms: [
          { type: 'cmp', key, op: 'exists', value: false },
          { type: 'eq', key, value: null },
          { type: 'eq', key, value: '' },
          { type: 'eq', key, value: [] },
        ],
      }
    default:
      throw new WarmError(`unsupported metadata filter operator ${String(f.operator)}`, 'invalid_argument')
  }
}

/**
 * Map LlamaIndex `MetadataFilters` (`and`, default, or `or`) to a core `Filter`. All thirteen
 * operators map; `==` terms ANDed together are evaluated by the engine, the rest after an
 * over-fetch. `text_match` is a substring test on the metadata value, not a full-text query.
 */
export function toFilter(filters: MetadataFilters | undefined): Filter | undefined {
  if (!filters || !filters.filters.length) return undefined
  const terms = filters.filters.map(one)
  if (terms.length === 1) return terms[0]
  return { type: filters.condition === 'or' ? 'or' : 'and', terms }
}
