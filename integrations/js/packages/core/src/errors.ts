// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

export type HmsErrorCode = 'invalid_argument' | 'limit_exceeded' | 'backend' | 'closed'

export class HmsError extends Error {
  constructor(message: string, readonly code: HmsErrorCode, options?: { cause?: unknown }) {
    super(message, options)
    this.name = 'HmsError'
  }
}
