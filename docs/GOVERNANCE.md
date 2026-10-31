<!-- Copyright 2024-2026 WritersLogic Contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Governance

WARM is maintained by WritersLogic, Inc. under a single-maintainer model. This page states how
decisions are made and who may do what, as it works today and as the repository enforces it.

## Roles

| Role | Who | Can |
| --- | --- | --- |
| Maintainer | `@dcondrey` (listed in `.github/CODEOWNERS`) | merge to `main`, cut releases, approve environments (`release`, `crates-io`, `npm`, `pypi`), triage security reports, change repository settings |
| Committer (write access) | granted by the maintainer, one account today | open branches in the repository and review; cannot bypass branch protection |
| Contributor | anyone | open issues, discussions in issues, and pull requests from a fork under the [DCO](../CONTRIBUTING.md#developer-certificate-of-origin) |
| Security reporter | anyone | report privately through GitHub Security Advisories or the address in [SECURITY.md](../SECURITY.md) |

Becoming a committer: a contributor with several merged, non-trivial pull requests that show
they follow [CONTRIBUTING.md](../CONTRIBUTING.md) can ask the maintainer in an issue; access is
granted by the maintainer and recorded by the repository's collaborator list.

## How decisions are made

- Changes land only through pull requests into `main`. Branch protection requires signed
  commits, resolved review conversations, a review from a code owner, and up-to-date branches;
  the maintainer may merge their own pull requests once CI is green.
- Design and scope questions are discussed in the issue before code is written; the maintainer
  decides, and the reason is recorded in the issue or the pull request.
- Anything that changes a public API, a stored format or a security property follows the
  compatibility rules in [production-readiness.md](production-readiness.md) and needs a test.
- Performance and capacity claims follow the measurement rules in the README: every number
  traces to a committed results file under `benchmarks/results/`.
- Releases follow [RELEASING.md](RELEASING.md); only the maintainer pushes release tags and
  approves the publishing environments.

## Conduct and security

- Behaviour in the project is governed by the [Code of Conduct](../CODE_OF_CONDUCT.md); reports
  go to the maintainer.
- Vulnerabilities are handled under [SECURITY.md](../SECURITY.md) and
  [docs/SECURITY.md](SECURITY.md).

## Continuity

There is one maintainer. The project cannot currently continue unaided if that person becomes
unavailable beyond the repository's organisation-level access: WritersLogic, Inc. owns the
repository, and organisation owners can grant access, but no second maintainer is designated and
no second person holds release credentials. Naming a second maintainer, and a second person who can
approve the publishing environments, is the open item for closing this gap.
