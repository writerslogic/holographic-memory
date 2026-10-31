# Agent Memory Leaderboard Cycle 2 adapter

`adapter.py` implements the participant-hosted Add and Search contract. The
original submitted version was `0.6.1-s-aml-553ddfd`. The query repair in this
directory preserves the original native engine and handles long Search queries
by lossless UTF-8 partitioning and deterministic result fusion.

The deployed image uses native source from commit
`12c80decb094d57ee13ba187a01bf1de8788ca2f`. `modal_deploy.py` packages
the checkout from which it is invoked, so running it directly from a newer
`main` builds a different native engine. Verify the native `src/`, `Cargo.toml`,
and `Cargo.lock` against that commit before reproducing the deployment. The
query repair's reviewed source hashes, test log, and live deployment record are
in `/Volumes/A/warm-coordination/reports/aml/` on the development host.

Run the focused local contract checks with:

```sh
cd benchmarks/public/aml_cycle2
WARM_BIN=/path/to/exact-s/warm-server WARM_QUERY_TEST_BINARY=/path/to/exact-s/warm-server \
  uv run --no-project --python 3.12 \
  --with fastapi==0.162.0 --with httpx==0.28.1 --with uvicorn==0.54.0 \
  python -m unittest -v test_adapter test_query test_local_host experimental.test_evidence_selection
```

Keep the evaluation key and endpoint authentication token out of Git. The
adapter takes the token from `AML_TOKEN`; the Modal deployment obtains it from
the existing `warm-aml-cycle2-token` Secret. Health is unauthenticated. Add
returns only after the native write and Volume commit complete.

For a host with a persistent local filesystem, `local_host.py` starts the
same adapter and a pinned native binary on loopback addresses. It reads the
authentication token from a private `AML_TOKEN=...` file, verifies the binary
SHA-256 before launch, and drains the native server on shutdown. Native journal
transactions sync before Add returns; the separate Modal Volume commit is not
needed on a local disk. Put a process supervisor around the launcher and expose
only its adapter port through a TLS proxy. Never expose the unauthenticated
native server port.

```sh
uv run --script benchmarks/public/aml_cycle2/local_host.py \
  --binary /path/to/pinned/warm-server --binary-sha256 SHA256 \
  --data-dir /private/persistent/stores --token-file /private/aml-token.env
```

The local lifecycle test uses synthetic data and checks Add, shutdown, reopen,
Search, authentication, and user isolation. Pass `WARM_QUERY_TEST_BINARY` to run
it against the pinned binary.

`gce_install.sh` installs this same local host on a dedicated Ubuntu 24.04
Compute Engine VM. It checks out the pinned native commit, overlays the
reviewed adapter and launcher from the pinned integration commit, builds the
server with Rust 1.96, and installs a loopback-only systemd service. It does
not install a token or expose a public port. Provision the token privately and
configure a TLS reverse proxy only after checking the VM's identity, source,
disk lifetime, and firewall rules.

Copy `runtime-requirements.txt` alongside the installer. Fresh installations use
the hash-locked runtime dependencies in that file. Regenerate it from the repository root:

```sh
uv pip compile --python-version 3.12 --generate-hashes --no-annotate --no-header \
  benchmarks/public/aml_cycle2/runtime-requirements.in \
  -o benchmarks/public/aml_cycle2/runtime-requirements.txt
```

This refresh changes the runtime of new installations, not the environment
recorded by historical runs.


## Optional source provenance

For a fresh store, `local_host.py --source-provenance` enables native metadata
with `aml_request_id`, `aml_message_index` and `aml_part_index` on each fact.
Programmatic callers can use `create_app(..., source_provenance=True)` or
`add_facts(payload, source_provenance=True)`. These fields identify the input
request, message and chunk; they do not infer speakers, event identities or dates.
Original fact IDs, literal text, chunk boundaries and Search behavior are preserved.

Provenance is disabled by default. Native idempotency fingerprints include
metadata, so enabling it against previously stored facts can cause a conflict on
retry. Use a fresh data directory and keep the setting consistent across restarts;
this option does not backfill or overwrite existing records. The deployed Modal
configuration retains the default.

## Experimental evidence selection

`experimental/evidence_selection.py` is an offline, standard-library selector.
Each claim has OR alternatives of whole AND witnesses, represented by original
source IDs. The selector preserves designated claims already supported by the
baseline, maximizes completed claims, then baseline retention, then candidate
order. Its exact search covers unions of at most one witness per claim followed
by baseline fill, with at most 2,048 candidates, eight claims, 64 witnesses, ten
output IDs, 32,768 states and 1,000,000 transitions. Resource exhaustion raises an
error instead of returning partial evidence. The active adapter does not import it.

From this directory, run its structural boundary and exhaustive-oracle tests with:

```sh
uv run --no-project --python 3.12 python -m unittest -v experimental.test_evidence_selection
```

The caller must establish source ownership, semantic sufficiency, witness
minimality, contextual validity and coverage. The selector does not infer them.
`experimental/semantic_cases.json` records ten synthetic regression specifications
for actor binding, relative dates, redundant evidence, event status, missing
identity bridges and later corrections. These cases are expectations for future
semantic checks; passing the structural selector tests does not establish that
WARM or a model solves them.

`experimental/findings.json` preserves the controlled experiment's aggregate
results, source-archive hashes and interpretation limits; `citation_matches.csv`
records the 24 questions' required and matched citation counts for each arm.
On 24 exposed development questions, citation hits rose from 16 to 17 with no
citation losses. The additional citation supplied no new temporal support, nine
matched search pairs showed no directed-search advantage, and the sole changed
reader case introduced an unsupported date. The six preselected reader pairs
were unchanged at 4/6 strict in a blinded qualitative review. No configuration
qualified for promotion or held-out confirmation. Full source records, prompts,
model caches and inference ledgers remain in the original research archive.
