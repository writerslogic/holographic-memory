# Quickstart examples

The code behind the [five-minute quickstart](../../docs/quickstart/index.md). The same corpus and
queries drive all three languages, and CI runs each one on every pull request.

| File | Run from the repository root |
|------|------------------------------|
| `rust/` | `cargo run --release --manifest-path examples/quickstart/rust/Cargo.toml` |
| `node/quickstart.mjs` | `npm ci --omit=optional && npm run build && node examples/quickstart/node/quickstart.mjs` (Node 24+) |
| `python/quickstart.py` | start `hms-server` first; see the header of the file |

## Data

- `corpus.jsonl`: the 30 articles of the Universal Declaration of Human Rights (1948), taken
  from the English text on Wikisource, which reproduces the United Nations document. It is a
  United Nations document issued with a UN symbol and is in the public domain worldwide under
  UN Administrative Instruction ST/AI/189/Add.9/Rev.2.
- `queries.json`: eight questions, each with the article that answers it. They were written once
  and are not tuned; the quickstart is a demonstration, not a benchmark.
- `expected-search.txt`, `expected-search-python.txt`: the result tables the programs print.
  CI diffs the programs' output against them.

## The encryption check

The Rust and Node programs scan every file of an encrypted store and assert that none holds the
passage text, `documents.text` included (see
[privacy limits](../../docs/concepts/privacy-limits.md)).
