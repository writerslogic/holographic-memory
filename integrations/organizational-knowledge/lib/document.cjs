'use strict'

const { KnowledgeError } = require('./errors.cjs')

const CHUNK_WORDS = 500
const OVERLAP_WORDS = 50
const NAMESPACE_KEY = 'hms_namespace'
// Match examples/organizational_knowledge.rs so both tools address the same documents.
const MAX_IDENTITY_BYTES = 1024
const MAX_VERSION_BYTES = 1024
const MAX_SOURCE_URI_BYTES = 8192
const MAX_METADATA_BYTES = 64 * 1024
const MAX_CONTENT_BYTES = 8 * 1024 * 1024
const MAX_JSON_DEPTH = 32

const BLANK = /^\p{White_Space}*$/u
const WHITESPACE = /\p{White_Space}/u
const CONTROL = /\p{Cc}/u
const OPTIONAL_STRING_FIELDS = [
  'project',
  'source_identifier',
  'date',
  'asset_type',
  'content_type',
  'title',
  'identifier_type',
  'collected_at',
]

const isPlainObject = value => {
  if (typeof value !== 'object' || value === null || Array.isArray(value)) return false
  const prototype = Object.getPrototypeOf(value)
  return prototype === Object.prototype || prototype === null
}
const bytes = value => Buffer.byteLength(value, 'utf8')
const clip = text => (text.length > 80 ? `${text.slice(0, 77)}...` : text)

function requestFailure(field, message) {
  throw new KnowledgeError('invalid_request', `${field}: ${message}`, { field })
}

function checkIdentityPart(value, field, fail) {
  if (typeof value !== 'string' || BLANK.test(value)) fail(field, 'must be a nonempty string')
  if (!value.isWellFormed()) fail(field, 'must be well-formed Unicode')
  // Control characters would make the key's JSON escaping differ between serializers.
  if (CONTROL.test(value)) fail(field, 'must not contain control characters')
}

/**
 * The HMS document ID for a stable source identity: a compact JSON tuple, so no delimiter
 * can make two different identities collide. Independent of content revision.
 */
function documentId(namespace, sourceSystem, id, fail = requestFailure, names = {}) {
  checkIdentityPart(namespace, names.namespace ?? 'namespace', fail)
  checkIdentityPart(sourceSystem, names.sourceSystem ?? 'sourceSystem', fail)
  checkIdentityPart(id, names.id ?? 'id', fail)
  const key = JSON.stringify([namespace, sourceSystem, id])
  if (bytes(key) > MAX_IDENTITY_BYTES) fail(names.id ?? 'id', `namespaced identity exceeds ${MAX_IDENTITY_BYTES} bytes`)
  return key
}

function parseDocumentId(key) {
  try {
    const parts = JSON.parse(key)
    if (Array.isArray(parts) && parts.length === 3 && parts.every(part => typeof part === 'string')) {
      return { namespace: parts[0], sourceSystem: parts[1], id: parts[2] }
    }
  } catch {
    // Not a key this adapter wrote.
  }
  return null
}

function checkJson(value, path, fail, depth = 0) {
  if (depth > MAX_JSON_DEPTH) fail(path, `nests deeper than ${MAX_JSON_DEPTH} levels`)
  if (typeof value === 'string') {
    if (!value.isWellFormed()) fail(path, 'must be well-formed Unicode')
  } else if (typeof value === 'number') {
    if (!Number.isFinite(value)) fail(path, 'must be a finite number')
  } else if (typeof value === 'boolean' || value === null) {
    // valid as is
  } else if (Array.isArray(value)) {
    for (let i = 0; i < value.length; i++) checkJson(value[i], `${path}[${i}]`, fail, depth + 1)
  } else if (isPlainObject(value)) {
    for (const [key, entry] of Object.entries(value)) {
      if (!key.isWellFormed()) fail(path, 'has a key that is not well-formed Unicode')
      // The binding returns a stored "__proto__" key as the result object's prototype, not data.
      if (key === '__proto__') fail(path, 'has a "__proto__" key, which the engine cannot return as data')
      checkJson(entry, `${path}.${clip(key)}`, fail, depth + 1)
    }
  } else {
    fail(path, 'must be JSON-serializable')
  }
}

function checkSourceUrl(value, fail) {
  if (value === undefined || value === null || value === '') return undefined
  if (typeof value !== 'string') fail('source_url', 'must be a string')
  if (bytes(value) > MAX_SOURCE_URI_BYTES) fail('source_url', `exceeds ${MAX_SOURCE_URI_BYTES} bytes`)
  // new URL() silently strips whitespace and repairs "http:host"; the stored string is the raw value.
  if (CONTROL.test(value) || WHITESPACE.test(value)) fail('source_url', 'must not contain whitespace or control characters')
  if (!/^https?:\/\//i.test(value)) fail('source_url', 'must be an absolute http(s) URL or empty')
  let url
  try {
    url = new URL(value)
  } catch {
    fail('source_url', 'must be an absolute http(s) URL or empty')
  }
  if (url.protocol !== 'http:' && url.protocol !== 'https:') fail('source_url', 'must use http or https')
  if (url.username || url.password) fail('source_url', 'must not embed credentials')
  return value
}

/**
 * Validates one flat curated document (the `data/curated/*.json` shape of the ToIP EGWG
 * prototype) and maps it to an HMS `DocumentInput` without touching storage.
 * The document's own `identifier` (SAID) is kept verbatim as the version; it is not verified.
 */
function adaptDocument(namespace, document, source) {
  const fail = (field, message) => {
    throw new KnowledgeError('invalid_document', `${source}: ${field}: ${message}`, { source, field })
  }
  if (!isPlainObject(document)) fail('document', 'must be a JSON object')
  if (isPlainObject(document.document) || isPlainObject(document.provenance)) {
    fail('document', 'flat curated document expected; the nested model proposed in issue #8 is not supported')
  }
  if (Object.hasOwn(document, NAMESPACE_KEY)) fail(NAMESPACE_KEY, 'is reserved by the adapter')

  const key = documentId(namespace, document.source_system, document.id, fail, {
    namespace: 'namespace',
    sourceSystem: 'source_system',
    id: 'id',
  })
  const identifier = document.identifier
  if (typeof identifier !== 'string' || BLANK.test(identifier)) fail('identifier', 'must be a nonempty string')
  if (!identifier.isWellFormed()) fail('identifier', 'must be well-formed Unicode')
  if (bytes(identifier) > MAX_VERSION_BYTES) fail('identifier', `exceeds ${MAX_VERSION_BYTES} bytes`)

  const content = document.content
  if (typeof content !== 'string') {
    fail('content', 'must be a string; flat curated documents only, the nested model proposed in issue #8 is not supported')
  }
  if (!content.isWellFormed()) fail('content', 'must be well-formed Unicode')
  if (BLANK.test(content)) fail('content', 'must contain text')
  if (bytes(content) > MAX_CONTENT_BYTES) fail('content', `exceeds ${MAX_CONTENT_BYTES} bytes`)

  const units = document.organizational_unit
  if (!Array.isArray(units) || !units.every(unit => typeof unit === 'string')) {
    fail('organizational_unit', 'must be an array of strings')
  }
  for (const field of OPTIONAL_STRING_FIELDS) {
    if (document[field] !== undefined && typeof document[field] !== 'string') fail(field, 'must be a string')
  }
  const sourceUri = checkSourceUrl(document.source_url, fail)

  // A top-level undefined (only possible outside JSON) is absent, as JSON.stringify would treat it.
  const preserved = Object.fromEntries(
    Object.entries(document).filter(([field, value]) => field !== 'content' && value !== undefined),
  )
  checkJson(preserved, 'metadata', fail)
  const metadata = { ...preserved, [NAMESPACE_KEY]: namespace }
  // The engine re-checks the cap; this fails the whole import before any write.
  if (bytes(JSON.stringify(metadata)) > MAX_METADATA_BYTES) fail('metadata', `exceeds ${MAX_METADATA_BYTES} bytes`)

  const input = {
    id: key,
    text: content,
    version: identifier,
    metadata,
    chunkWords: CHUNK_WORDS,
    overlapWords: OVERLAP_WORDS,
    storeText: true,
  }
  if (sourceUri !== undefined) input.sourceUri = sourceUri
  return {
    source,
    documentId: key,
    identity: { namespace, sourceSystem: document.source_system, id: document.id },
    input,
  }
}

module.exports = {
  NAMESPACE_KEY,
  adaptDocument,
  documentId,
  isPlainObject,
  parseDocumentId,
  requestFailure,
}
