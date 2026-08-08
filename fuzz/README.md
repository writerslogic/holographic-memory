# Fuzzing

Install `cargo-fuzz`, then run:

```sh
cargo fuzz run wire_inputs
```

Targets must remain deterministic and convert arbitrary input into bounded allocations.
