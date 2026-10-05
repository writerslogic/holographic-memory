# Agent rules for holographic-memory

@[CLAUDE.md](CLAUDE.md)

`CLAUDE.md` is the project rule file for every agent, not only Claude. If the include above did not expand, read it before doing anything else.

## Gate before any commit

```sh
cargo fmt -- --check
cargo clippy --all-targets -- -D warnings
cargo clippy --all-targets --all-features -- -D warnings
cargo test --locked
cargo test --locked --all-features
```

## Security claims

`docs/SECURITY.md` and `docs/PRIVACY.md` are the authority on what HMS does and does not protect. README and package metadata must not claim more than they do. PoSME, RATS attestation verification, and search over encrypted data are not implemented; do not describe them as features.
