The final ordinal full-corpus COUNT check uses the same immutable 5M index,
962 queries, and saved Tantivy counts as the Linux harness, copied and verified
by the coordinator. It runs on Apple M4 / macOS, 64-bit slots, Rust 1.96 native
release, four Rayon threads, nice19, and is a correctness check only. The passed
x86 protocol and 2,886-result exhaustive checks remain required independent
receipts. This relocation avoids the Linux shared-memory queue; it establishes
no x86 full-corpus performance or RSS claim.

Driver source: query_kernel.rs, linked to ordinal source commit
7e216c6a5afd5a0f518c619fb95b1bb990e08aab. A private target contains the native
executable. The run records all answers, query/reference/binary hashes, peak RSS,
and open/total wall time solely to account for resources. Start only after the
coordinator confirms the copied fixture's checksums:

    uv run --no-project python native_count.py /path/to/verified-count-data /path/to/native-driver

Build query_kernel.rs with the pinned crate path, serde, serde_json and anyhow,
using release LTO, codegen-units=1 and RUSTFLAGS="-C target-cpu=native". Pass the
verified fixture directory and resulting executable as the script arguments.
