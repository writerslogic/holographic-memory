// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

export type WarmErrorCode = 'invalid_argument' | 'limit_exceeded' | 'filter_limit' | 'concurrent_mutation' | 'backend' | 'closed'

export class WarmError extends Error {
  constructor(message: string, readonly code: WarmErrorCode, options?: { cause?: unknown }) {
    super(message, options)
    this.name = 'WarmError'
  }
}
