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
in `/Volumes/A/hms-coordination/reports/aml/` on the development host.

Run the focused local contract checks with:

```sh
cd benchmarks/public/aml_cycle2
HMS_QUERY_TEST_BINARY=/path/to/exact-s/hms-server uv run --no-project --python 3.12 \
  --with fastapi==0.119.0 --with httpx==0.28.1 --with uvicorn==0.38.0 \
  python -m unittest -v test_adapter.py test_query.py
```

Keep the evaluation key and endpoint authentication token out of Git. The
adapter takes the token from `AML_TOKEN`; the Modal deployment obtains it from
the existing `hms-aml-cycle2-token` Secret. Health is unauthenticated. Add
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
  --binary /path/to/pinned/hms-server --binary-sha256 SHA256 \
  --data-dir /private/persistent/stores --token-file /private/aml-token.env
```

The local lifecycle test uses synthetic data and checks Add, shutdown, reopen,
Search, authentication, and user isolation. Pass `HMS_QUERY_TEST_BINARY` to run
it against the pinned binary.
