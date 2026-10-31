// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0
import { createHash } from 'node:crypto'
import { WarmError } from './errors.js'

interface Passage {
  text?: string | null
  startByte: number
  endByte: number
}

interface Source {
  version: 1
  bytes: number
  sha256: string
  gaps: { startByte: number; endByte: number; text: string }[]
}

const MAX_SOURCE_BYTES = 8 * 1024 * 1024
const digest = (text: Buffer): string => createHash('sha256').update(text).digest('hex')
const corrupt = (): never => { throw new WarmError('stored source text is incomplete or corrupt; reingest the authoritative source', 'backend') }

function part(p: Passage, total: number): Buffer {
  if (!p || typeof p.text !== 'string' || !Number.isSafeInteger(p.startByte) || !Number.isSafeInteger(p.endByte) ||
      p.startByte < 0 || p.endByte <= p.startByte || p.endByte > total) return corrupt()
  const text = Buffer.from(p.text, 'utf8')
  if (text.length !== p.endByte - p.startByte) return corrupt()
  return text
}

// Passages omit boundary and inter-passage whitespace. Retain those exact bytes, the full
// length and digest in the same document mutation, without duplicating covered source text.
export function captureSource(text: string, passages: Passage[]): Source {
  const source = Buffer.from(text, 'utf8')
  if (source.length > MAX_SOURCE_BYTES) throw new WarmError('record exceeds 8 MiB', 'limit_exceeded')
  const gaps: Source['gaps'] = []
  let end = 0
  const gap = (startByte: number, endByte: number): void => {
    gaps.push({ startByte, endByte, text: source.subarray(startByte, endByte).toString('utf8') })
  }
  for (const passage of [...passages].sort((a, b) => a.startByte - b.startByte)) {
    if (!part(passage, source.length).equals(source.subarray(passage.startByte, passage.endByte))) return corrupt()
    if (passage.startByte > end) gap(end, passage.startByte)
    end = Math.max(end, passage.endByte)
  }
  if (end < source.length) gap(end, source.length)
  return { version: 1, bytes: source.length, sha256: digest(source), gaps }
}

export function restoreSource(manifest: unknown, passages: Passage[]): string {
  const source = manifest as Source | undefined
  if (!source || source.version !== 1 || !Number.isSafeInteger(source.bytes) || source.bytes < 0 || source.bytes > MAX_SOURCE_BYTES ||
      typeof source.sha256 !== 'string' || !/^[a-f0-9]{64}$/.test(source.sha256) || !Array.isArray(source.gaps) || source.gaps.length > 4097) {
    return corrupt()
  }
  const parts = [...passages, ...source.gaps].map((p) => ({ ...p, bytes: part(p, source.bytes) }))
    .sort((a, b) => a.startByte - b.startByte)
  const text = Buffer.alloc(source.bytes)
  let end = 0
  for (const p of parts) {
    if (p.startByte > end) return corrupt()
    const overlap = Math.max(0, Math.min(end, p.endByte) - p.startByte)
    if (!p.bytes.subarray(0, overlap).equals(text.subarray(p.startByte, p.startByte + overlap))) return corrupt()
    p.bytes.copy(text, p.startByte + overlap, overlap)
    end = Math.max(end, p.endByte)
  }
  if (end !== source.bytes || digest(text) !== source.sha256) return corrupt()
  return text.toString('utf8')
}
