// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

import type { EncryptionOptions, Kdf, WasmApi } from "./types.js";

/**
 * Encrypted snapshot envelope (all integers big-endian):
 *
 *   "HMSE" | version u8 | kdf u8 | p1 u32 | p2 u32 | p3 u32 | salt[16] | nonce[12] | ciphertext+tag
 *
 * kdf 1 = PBKDF2-HMAC-SHA256 (p1 = iterations); kdf 2 = Argon2id (p1 = memory KiB,
 * p2 = passes, p3 = lanes). The whole header is the AES-256-GCM additional data, so the KDF
 * parameters cannot be altered without failing authentication.
 */
const MAGIC = [0x48, 0x4d, 0x53, 0x45];
const VERSION = 1;
const HEADER_BYTES = 4 + 1 + 1 + 12 + 16 + 12;
const TAG_BYTES = 16;
const KDF_IDS: Record<Kdf, number> = { pbkdf2: 1, argon2id: 2 };

export const DEFAULT_PBKDF2_ITERATIONS = 600_000;
export const DEFAULT_ARGON2_MEMORY_KIB = 19_456;
export const DEFAULT_ARGON2_PASSES = 2;
export const DEFAULT_ARGON2_LANES = 1;

// Bounds applied to parameters read from untrusted bytes, so a forged header cannot ask for
// minutes of key stretching or gigabytes of memory.
const MAX_ITERATIONS = 10_000_000;
const MAX_MEMORY_KIB = 1 << 20;
const MAX_PASSES = 64;
const MAX_LANES = 16;

export interface KdfParams {
  kdf: Kdf;
  p1: number;
  p2: number;
  p3: number;
}

/** A derived key plus everything needed to write more envelopes under it. */
export interface CipherState {
  params: KdfParams;
  salt: Uint8Array;
  key: CryptoKey;
}

/** WebCrypto wants ArrayBuffer-backed views; ours never are shared memory. */
const buf = (bytes: Uint8Array): Uint8Array<ArrayBuffer> => bytes as Uint8Array<ArrayBuffer>;

function subtle(): SubtleCrypto {
  const s = globalThis.crypto?.subtle;
  if (!s) throw new Error("WebCrypto (crypto.subtle) is unavailable in this runtime");
  return s;
}

function randomBytes(length: number): Uint8Array {
  return globalThis.crypto.getRandomValues(new Uint8Array(length));
}

function passphraseBytes(passphrase: string | Uint8Array): Uint8Array {
  const bytes = typeof passphrase === "string" ? new TextEncoder().encode(passphrase) : passphrase;
  if (bytes.length === 0) throw new Error("the passphrase must not be empty");
  return bytes;
}

function checkParams(params: KdfParams): void {
  const { kdf, p1, p2, p3 } = params;
  const ok =
    kdf === "pbkdf2"
      ? Number.isInteger(p1) && p1 >= 1 && p1 <= MAX_ITERATIONS
      : Number.isInteger(p1) &&
        p1 >= 8 &&
        p1 <= MAX_MEMORY_KIB &&
        Number.isInteger(p2) &&
        p2 >= 1 &&
        p2 <= MAX_PASSES &&
        Number.isInteger(p3) &&
        p3 >= 1 &&
        p3 <= MAX_LANES;
  if (!ok) throw new Error(`invalid ${kdf} parameters`);
}

export function paramsFromOptions(options: EncryptionOptions): KdfParams {
  const kdf = options.kdf ?? "pbkdf2";
  if (kdf !== "pbkdf2" && kdf !== "argon2id") throw new Error(`unknown kdf ${String(kdf)}`);
  const params: KdfParams =
    kdf === "pbkdf2"
      ? { kdf, p1: options.iterations ?? DEFAULT_PBKDF2_ITERATIONS, p2: 0, p3: 0 }
      : {
          kdf,
          p1: options.memoryKiB ?? DEFAULT_ARGON2_MEMORY_KIB,
          p2: options.passes ?? DEFAULT_ARGON2_PASSES,
          p3: options.lanes ?? DEFAULT_ARGON2_LANES,
        };
  checkParams(params);
  return params;
}

export async function deriveKey(
  passphrase: string | Uint8Array,
  salt: Uint8Array,
  params: KdfParams,
  wasm: () => Promise<WasmApi>,
): Promise<CryptoKey> {
  checkParams(params);
  const secret = passphraseBytes(passphrase);
  const usages: KeyUsage[] = ["encrypt", "decrypt"];
  if (params.kdf === "argon2id") {
    const api = await wasm();
    const raw = api.argon2idKey(secret, salt, params.p1, params.p2, params.p3);
    try {
      return await subtle().importKey("raw", buf(raw), "AES-GCM", false, usages);
    } finally {
      raw.fill(0);
    }
  }
  const material = await subtle().importKey("raw", buf(secret), "PBKDF2", false, ["deriveKey"]);
  return subtle().deriveKey(
    { name: "PBKDF2", hash: "SHA-256", salt: buf(salt), iterations: params.p1 },
    material,
    { name: "AES-GCM", length: 256 },
    false,
    usages,
  );
}

/** Derives a fresh key (random salt) for a new encrypted store. */
export async function newCipher(
  options: EncryptionOptions,
  wasm: () => Promise<WasmApi>,
): Promise<CipherState> {
  const params = paramsFromOptions(options);
  const salt = randomBytes(16);
  return { params, salt, key: await deriveKey(options.passphrase, salt, params, wasm) };
}

export function isEncrypted(bytes: Uint8Array): boolean {
  return bytes.length >= 4 && MAGIC.every((b, i) => bytes[i] === b);
}

function writeHeader(state: CipherState, nonce: Uint8Array): Uint8Array {
  const header = new Uint8Array(HEADER_BYTES);
  const view = new DataView(header.buffer);
  header.set(MAGIC, 0);
  header[4] = VERSION;
  header[5] = KDF_IDS[state.params.kdf];
  view.setUint32(6, state.params.p1);
  view.setUint32(10, state.params.p2);
  view.setUint32(14, state.params.p3);
  header.set(state.salt, 18);
  header.set(nonce, 34);
  return header;
}

/** Seals `plain` under the state's key with a fresh random 96-bit nonce. */
export async function seal(state: CipherState, plain: Uint8Array): Promise<Uint8Array> {
  const nonce = randomBytes(12);
  const header = writeHeader(state, nonce);
  const body = new Uint8Array(
    await subtle().encrypt({ name: "AES-GCM", iv: buf(nonce), additionalData: buf(header) }, state.key, buf(plain)),
  );
  const out = new Uint8Array(header.length + body.length);
  out.set(header, 0);
  out.set(body, header.length);
  return out;
}

/** Parses an envelope's header without decrypting it. */
export function readHeader(bytes: Uint8Array): { params: KdfParams; salt: Uint8Array } {
  if (!isEncrypted(bytes)) throw new Error("not an encrypted writerslogic-warm snapshot");
  if (bytes.length < HEADER_BYTES + TAG_BYTES) throw new Error("encrypted snapshot is truncated");
  if (bytes[4] !== VERSION) throw new Error(`unsupported encrypted snapshot version ${bytes[4]}`);
  const kdf = (Object.keys(KDF_IDS) as Kdf[]).find((name) => KDF_IDS[name] === bytes[5]);
  if (!kdf) throw new Error(`unknown kdf id ${bytes[5]}`);
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const params: KdfParams = { kdf, p1: view.getUint32(6), p2: view.getUint32(10), p3: view.getUint32(14) };
  checkParams(params);
  return { params, salt: bytes.slice(18, 34) };
}

export async function open(state: CipherState, bytes: Uint8Array): Promise<Uint8Array> {
  const header = bytes.subarray(0, HEADER_BYTES);
  const nonce = bytes.subarray(34, 46);
  try {
    return new Uint8Array(
      await subtle().decrypt(
        { name: "AES-GCM", iv: buf(nonce), additionalData: buf(header) },
        state.key,
        buf(bytes.subarray(HEADER_BYTES)),
      ),
    );
  } catch {
    throw new Error("decryption failed: wrong passphrase or corrupted snapshot");
  }
}

/** Derives the key for an existing envelope from its own header. */
export async function cipherFor(
  bytes: Uint8Array,
  passphrase: string | Uint8Array,
  wasm: () => Promise<WasmApi>,
): Promise<CipherState> {
  const { params, salt } = readHeader(bytes);
  return { params, salt, key: await deriveKey(passphrase, salt, params, wasm) };
}
