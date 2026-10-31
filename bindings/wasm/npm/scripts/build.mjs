// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

// Builds the package: the wasm module once per variant (scalar, SIMD), wasm-bindgen glue for
// every target (web, bundler, nodejs, deno), wasm-opt, the TypeScript layer, and a size report.
//
//   node scripts/build.mjs [--skip-wasm] [--variants scalar,simd] [--targets web,bundler,nodejs,deno]

import { spawnSync } from "node:child_process";
import { copyFileSync, existsSync, mkdirSync, readFileSync, renameSync, rmSync, statSync, writeFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { brotliCompressSync, gzipSync } from "node:zlib";

const here = dirname(fileURLToPath(import.meta.url));
const pkgRoot = resolve(here, "..");
const crateRoot = resolve(pkgRoot, "..");
const dist = join(pkgRoot, "dist");
const targetBase = process.env.WARM_WASM_TARGET_DIR ?? "./.warm-target-wasm";

const args = process.argv.slice(2);
const option = (name, fallback) => {
  const at = args.indexOf(name);
  return at >= 0 ? args[at + 1] : fallback;
};
const variants = option("--variants", "scalar,simd").split(",");
const targets = option("--targets", "web,bundler,nodejs,deno").split(",");
const skipWasm = args.includes("--skip-wasm");

// rustc's default wasm32-unknown-unknown feature set; SIMD is added only to the SIMD variant,
// so wasm-opt fails on a scalar build that contains a SIMD instruction.
const BASE_FEATURES = [
  "--enable-bulk-memory",
  "--enable-nontrapping-float-to-int",
  "--enable-sign-ext",
  "--enable-mutable-globals",
  "--enable-multivalue",
  "--enable-reference-types",
];
const GETRANDOM = '--cfg getrandom_backend="wasm_js"';

function run(command, commandArgs, options = {}) {
  const result = spawnSync(command, commandArgs, { stdio: "inherit", ...options });
  if (result.status !== 0) {
    throw new Error(`${command} ${commandArgs.join(" ")} exited with ${result.status ?? result.signal}`);
  }
}

function wasmOptBinary() {
  const local = join(pkgRoot, "node_modules", ".bin", "wasm-opt");
  if (existsSync(local)) return local;
  throw new Error("wasm-opt not found: run `npm install` in bindings/wasm/npm (binaryen)");
}

// The wasm-bindgen CLI must match the crate's wasm-bindgen version exactly; wasm-pack keeps the
// matching binary in its cache (run `wasm-pack build` once, or `cargo install wasm-bindgen-cli`).
function bindgenBinary() {
  const lock = readFileSync(join(crateRoot, "Cargo.lock"), "utf8");
  const version = /name = "wasm-bindgen"\nversion = "([^"]+)"/.exec(lock)?.[1];
  if (!version) throw new Error("wasm-bindgen is not in bindings/wasm/Cargo.lock");
  const cache = process.env.WARM_WASM_PACK_CACHE ?? join(process.env.HOME, "Library/Caches/.wasm-pack");
  const found = join(cache, `wasm-bindgen-cargo-install-${version}`, "wasm-bindgen");
  if (!existsSync(found)) {
    throw new Error(`wasm-bindgen ${version} CLI not found; run: cargo install wasm-bindgen-cli --version ${version}`);
  }
  return found;
}

const glueDirName = (target) => (target === "nodejs" ? "node" : target);
const sizes = {};

if (!skipWasm) {
  rmSync(join(dist, "wasm"), { recursive: true, force: true });
  for (const variant of variants) {
    const simd = variant === "simd";
    const env = {
      ...process.env,
      CARGO_TARGET_DIR: `${targetBase}-${variant}`,
      RUSTFLAGS: `${GETRANDOM}${simd ? " -C target-feature=+simd128" : ""}`,
    };
    // One cargo build per variant; wasm-bindgen then emits each target from the same module.
    run("cargo", ["build", "--release", "--target", "wasm32-unknown-unknown", "--manifest-path", join(crateRoot, "Cargo.toml")], { env });
    const built = join(env.CARGO_TARGET_DIR, "wasm32-unknown-unknown", "release", "warm_wasm.wasm");
    for (const target of targets) {
      const out = join(dist, "wasm", glueDirName(target), variant);
      mkdirSync(out, { recursive: true });
      run(bindgenBinary(), ["--target", target, "--out-dir", out, "--out-name", "warm_wasm", built]);
      const wasm = join(out, "warm_wasm_bg.wasm");
      const before = statSync(wasm).size;
      run(wasmOptBinary(), ["-Oz", ...BASE_FEATURES, ...(simd ? ["--enable-simd"] : []), "--strip-producers", wasm, "-o", wasm]);
      const bytes = readFileSync(wasm);
      sizes[`${glueDirName(target)}/${variant}`] = {
        unoptimizedBytes: before,
        bytes: bytes.length,
        gzipBytes: gzipSync(bytes, { level: 9 }).length,
        brotliBytes: brotliCompressSync(bytes).length,
      };
      if (target === "nodejs") renameSync(join(out, "warm_wasm.js"), join(out, "warm_wasm.cjs"));
    }
  }
  mkdirSync(dist, { recursive: true });
  writeFileSync(join(dist, "wasm-sizes.json"), JSON.stringify(sizes, null, 2) + "\n");
}

run(join(pkgRoot, "node_modules", ".bin", "tsc"), ["-p", join(pkgRoot, "tsconfig.json")], { cwd: pkgRoot });
copyFileSync(resolve(pkgRoot, "../../../LICENSE"), join(pkgRoot, "LICENSE"));
if (existsSync(join(dist, "wasm-sizes.json"))) console.log(readFileSync(join(dist, "wasm-sizes.json"), "utf8"));
