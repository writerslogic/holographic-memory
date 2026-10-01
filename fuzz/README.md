<img src="../assets/logo.png" width="88" alt="Holographic Memory System" align="left">

<h1>HMS fuzzing</h1>
<p><strong>Deterministic, allocation-bounded fuzz targets for the Holographic Memory System wire formats.</strong></p>

<br clear="left">

Install `cargo-fuzz`, then run:

```sh
cargo fuzz run wire_inputs
```

Targets must remain deterministic and convert arbitrary input into bounded allocations.
