# Contributing to HMS

Thank you for helping. HMS is a Rust vector-symbolic memory engine with Node.js bindings, a
Python client and an HTTP server, licensed under Apache-2.0. There is no CLA: you keep your
copyright and certify your right to contribute with a sign-off on each commit (see
[Sign off your commits](#sign-off-your-commits-dco)).

Participation is covered by the [Code of Conduct](CODE_OF_CONDUCT.md). Report vulnerabilities
privately as described in [SECURITY.md](SECURITY.md), never in a public issue.

## Ways to help

- Pick an issue labelled [good first issue](https://github.com/writerslogic/holographic-memory/labels/good%20first%20issue).
  Each one states the files involved and how to check the result.
- Ask or answer questions in [Discussions](https://github.com/writerslogic/holographic-memory/discussions).
- Report a bug with the smallest input that reproduces it. Include the OS, Rust and Node
  versions, and the enabled Cargo features.
- Open an issue before a large change so the approach can be agreed first.

## Set up

You need Rust 1.89 or newer (the MSRV), and for the bindings Node.js 22 or newer (24 for the quickstart, which uses `node:sqlite`) and Python 3.10 or newer.

```sh
git clone https://github.com/writerslogic/holographic-memory
cd holographic-memory
cargo test                          # default features, no native dependencies
cargo test --all-features           # everything CI builds
npm ci --omit=optional && npm run build && npm test   # Node bindings
```

If your cargo `target-dir` is shared with other builds and you see
`Blocking waiting for file lock`, set a private one: `export CARGO_TARGET_DIR=$HOME/.cache/hms-target`.

`default = []`: nothing is on unless asked for. A change that compiles under only one feature
combination is not done. Check the combinations the [CI matrix](.github/workflows/ci.yml) builds.

## Before you open a pull request

Run what CI runs:

```sh
cargo fmt -- --check
cargo clippy --all-targets -- -D warnings
cargo clippy --all-targets --all-features -- -D warnings
cargo test --locked
cargo test --locked --all-features
cargo +1.89 check --no-default-features        # MSRV
```

CI uses the newest stable clippy, which is stricter than older releases. Update with
`rustup update stable` if a lint appears only in CI.

Also:

- **Tests.** New logic, bug fixes and security-relevant changes come with an automated test that
  fails without the change. Test behaviour, not getters or your own implementation. If a test is
  not feasible, say why in the PR.
- **Defects.** Reproduce first, fix the cause, and verify through the same surface the user hit:
  a UI bug needs a UI check, an export bug needs a real round trip.
- **Numbers.** Every number in the README or the docs traces to a committed file under
  [`benchmarks/results/`](benchmarks/results). Never write a figure from memory. If your change
  moves a number, regenerate the results file and say so. Report where HMS loses as plainly as
  where it wins. Capacity and scaling claims state the tested range; they do not extrapolate.
  The block between `<!-- BENCH:START -->` and `<!-- BENCH:END -->` in the README is generated;
  do not edit it by hand.
- **Style.** Follow `cargo fmt` and fix every clippy warning. Add comments only for a constraint
  or a reason that is not obvious from the code. JavaScript uses two-space indentation.
- **Security features.** With the `security` feature in scope: zeroize key material after cipher
  initialisation, compare secrets in constant time, never branch on secret values, and validate
  public keys and algorithm parameters before use.
- **Docs.** Pages live under `docs/`; put new pages in the folder for their area (`docs/api/`,
  `docs/security/`, ...) and they join the navigation without editing `mkdocs.yml`. Preview with
  `pip install -r docs/requirements.txt && mkdocs serve`. The build fails on any broken link.
- **New files** start with the licence header: `// Copyright 2024-2026 WritersLogic Contributors`
  and `// SPDX-License-Identifier: Apache-2.0` (`#` comments for Python, YAML, TOML and shell).

## Sign off your commits (DCO)

HMS uses the [Developer Certificate of Origin](https://developercertificate.org/) instead of a
CLA. By signing off you certify that you wrote the change or have the right to submit it under
the project licence. Add a trailer to every commit:

```
Signed-off-by: Your Name <you@example.com>
```

`git commit -s` adds it from your `user.name` and `user.email`. To add it to commits you have
already made: `git rebase --signoff origin/main`, then `git push --force-with-lease`.

A workflow checks every non-merge commit in a pull request from a contributor without write
access. Maintainers and bots are exempt from the check, not from the certificate. Maintainers
squash-merge, so your commits do not need a GPG or SSH signature.

## Commits and pull requests

- Commit subjects read `<type>: <imperative description>`, one line. Types: `feat`, `fix`,
  `refactor`, `test`, `docs`, `perf`, `security`, `chore`.
- Keep a pull request to one logical change, and fill in the template. Link the issue it closes.
- Rebase on `main` before requesting review. Resolve every review thread, and every bot finding
  (CodeQL, code quality), by fixing it or by replying with why it is a false positive.
- Do not commit generated binaries, `*.lock` scratch files, or evidence directories over a few
  megabytes; large benchmark evidence stays out of git and the results file references it.

## Release process

Maintainers cut releases; see the Releasing section of the documentation. Contributors do not
tag, publish packages, or edit version numbers.
