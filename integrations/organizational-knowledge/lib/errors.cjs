'use strict'

/**
 * `code` is one of:
 * - `invalid_document`: a curated document failed validation (`source`, `field`).
 * - `invalid_request`: bad arguments, filter, or forget target.
 * - `duplicate_identity`: two documents in one import share a namespaced identity.
 * - `partial_failure`: a write failed after earlier writes committed (`committed` or
 *   `removed`/`absent` list what is already done; re-running is idempotent).
 */
class KnowledgeError extends Error {
  constructor(code, message, details = {}, options = undefined) {
    super(message, options)
    this.name = 'KnowledgeError'
    this.code = code
    this.source = details.source
    this.field = details.field
    this.committed = details.committed
    this.removed = details.removed
    this.absent = details.absent
  }
}

module.exports = { KnowledgeError }
