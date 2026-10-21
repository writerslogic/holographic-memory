# holographic-vsa-models

Model packs for the local embedder in `holographic-vsa`. Installing `holographic-vsa[embed]` pulls this
package so `holographic_memory.Embedder()` works with no network. Each pack is verified against a
sha256 pinned in the library before use.

Build (maintainers): `./build_packs.sh /path/to/hms-embed` (a binary built with
`--features local-embed-download`), then `python -m build`.
