# Integration Guide

HMS is designed to be highly portable. While this crate provides high-performance Node.js bindings, the core logic is accessible to other systems through several paths.

## 1. Rust Integration (Library)

Since the project uses a decoupled architecture, you can use HMS as a standard Rust library in other Rust projects.

Add it to your `Cargo.toml`:
```toml
[dependencies]
holographic-memory = { git = "https://github.com/writerslogic/holographic-memory", default-features = false }
```

Usage in Rust:
```rust
use hms_core::{HmsCore, EntangledHVec};

fn main() {
    let hms = HmsCore::new(10000, None, None).unwrap();
    let vec = hms.encode_text("hello world");
    hms.memorize("id1".to_string(), vec).unwrap();
}
```

## 2. Cross-Language Principles

If you are implementing HMS in a language not supported by Rust's FFI, you can implement the **Binary Spatter Code (BSC)** principles:

### Encoding
1. **N-Grams**: Break text into chunks of 3 characters.
2. **Deterministic Hashing**: Use a seeded PRNG (like `StdRng` with character code as seed) to generate a high-dimensional bit vector for each character.
3. **Permutation**: Apply a cyclic shift (rotate bits) based on the character's position in the N-gram.
4. **Binding**: XOR the character vectors within the N-gram.
5. **Bundling**: Apply a majority rule (bit-count) across all N-gram vectors to produce the final document vector.

## 3. WebAssembly (Wasm)

You can compile the `core` module to Wasm using `wasm-pack` for use in browser-based environments or Edge workers.

```bash
wasm-pack build -- --no-default-features
```

## 4. REST API / Microservice

The recommended way to integrate with non-Rust/Node environments (Python, Go, Ruby) is to wrap this crate in a small Express or Fastify service and communicate over JSON-RPC or REST.
