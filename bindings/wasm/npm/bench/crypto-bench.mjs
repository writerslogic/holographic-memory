// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

// WebCrypto AES-256-GCM against AES-256-GCM compiled into the module (feature `inwasm-aes`),
// and the two key derivations. Builds the feature variant into a scratch directory first.
//   node bench/crypto-bench.mjs <scratch-dir>   (prints JSON)

import { spawnSync } from "node:child_process";
import { createRequire } from "node:module";
import { mkdirSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "../..");
const scratch = resolve(process.argv[2] ?? "scratch-crypto");
mkdirSync(scratch, { recursive: true });
const targetDir = `${process.env.WARM_WASM_TARGET_DIR ?? "./.warm-target-wasm"}-aes`;
const env = { ...process.env, CARGO_TARGET_DIR: targetDir, RUSTFLAGS: '--cfg getrandom_backend="wasm_js" -C target-feature=+simd128' };
const sh = (cmd, args) => {
  const r = spawnSync(cmd, args, { env, stdio: ["ignore", "ignore", "inherit"] });
  if (r.status !== 0) throw new Error(`${cmd} failed`);
};
sh("cargo", ["build", "--release", "--target", "wasm32-unknown-unknown", "--features", "inwasm-aes", "--manifest-path", join(root, "Cargo.toml")]);
const version = /name = "wasm-bindgen"\nversion = "([^"]+)"/.exec((await import("node:fs")).readFileSync(join(root, "Cargo.lock"), "utf8"))[1];
const bindgen = join(process.env.HOME, "Library/Caches/.wasm-pack", `wasm-bindgen-cargo-install-${version}`, "wasm-bindgen");
sh(bindgen, ["--target", "nodejs", "--out-dir", scratch, "--out-name", "warm_aes", join(targetDir, "wasm32-unknown-unknown/release/warm_wasm.wasm")]);
const wasm = createRequire(import.meta.url)(join(scratch, "warm_aes.js"));

const subtle = globalThis.crypto.subtle;
const key = crypto.getRandomValues(new Uint8Array(32));
const nonce = crypto.getRandomValues(new Uint8Array(12));
const aad = new Uint8Array(46);
const cryptoKey = await subtle.importKey("raw", key, "AES-GCM", false, ["encrypt", "decrypt"]);

async function time(fn, repeats) {
  await fn();
  const samples = [];
  for (let i = 0; i < repeats; i++) {
    const t = performance.now();
    await fn();
    samples.push(performance.now() - t);
  }
  samples.sort((a, b) => a - b);
  return samples[Math.floor(samples.length / 2)];
}

const out = { runtime: `node ${process.version}`, wasmBytes: null, aes: [], kdf: {} };
for (const mib of [1, 10, 50]) {
  const data = new Uint8Array(mib * 1024 * 1024);
  for (let i = 0; i < data.length; i += 65536) crypto.getRandomValues(data.subarray(i, Math.min(i + 65536, data.length)));
  const repeats = mib >= 50 ? 3 : 7;
  const sealed = new Uint8Array(await subtle.encrypt({ name: "AES-GCM", iv: nonce, additionalData: aad }, cryptoKey, data));
  const viaWasm = wasm.aesGcmSeal(key, nonce, aad, data);
  if (Buffer.compare(sealed, viaWasm) !== 0) throw new Error("the two AES-GCM paths disagree");
  out.aes.push({
    mib,
    webcryptoEncryptMs: await time(() => subtle.encrypt({ name: "AES-GCM", iv: nonce, additionalData: aad }, cryptoKey, data), repeats),
    webcryptoDecryptMs: await time(() => subtle.decrypt({ name: "AES-GCM", iv: nonce, additionalData: aad }, cryptoKey, sealed), repeats),
    wasmEncryptMs: await time(() => wasm.aesGcmSeal(key, nonce, aad, data), repeats),
    wasmDecryptMs: await time(() => wasm.aesGcmOpen(key, nonce, aad, sealed), repeats),
  });
}
const salt = crypto.getRandomValues(new Uint8Array(16));
const pass = new TextEncoder().encode("correct horse battery staple");
const material = await subtle.importKey("raw", pass, "PBKDF2", false, ["deriveBits"]);
out.kdf.pbkdf2_600000_webcryptoMs = await time(() => subtle.deriveBits({ name: "PBKDF2", hash: "SHA-256", salt, iterations: 600000 }, material, 256), 5);
out.kdf.argon2id_19456KiB_t2_p1_wasmMs = await time(() => wasm.argon2idKey(pass, salt, 19456, 2, 1), 5);
out.wasmMemoryBytes = wasm.memoryBytes();
console.log(JSON.stringify(out));
