## What and why
<!-- The change and the problem it solves. Link the issue: "Closes #123". -->

## Type of change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation
- [ ] Performance
- [ ] Refactor or chore

## Checklist
- [ ] Every commit is signed off (`git commit -s`); see [CONTRIBUTING.md](../CONTRIBUTING.md#sign-off-your-commits-dco)
- [ ] `cargo fmt -- --check`, `cargo clippy --all-targets -- -D warnings` and `cargo test` pass, and so do the same with `--all-features`
- [ ] A bug fix or new logic has a test that fails without the change, or the description says why not
- [ ] A change that depends on a Cargo feature builds under the feature combinations CI runs
- [ ] Any number in the docs or README traces to a committed file under `benchmarks/results/`, with the tested range stated
- [ ] Documentation updated where behavior changed
- [ ] New files carry the Apache-2.0 header

## How I verified it
<!-- The command you ran and what it showed. For a UI or export change, the real surface. -->
