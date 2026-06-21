# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| v0.5.x  | :white_check_mark: |
| v0.4.x  | :white_check_mark: |
| < v0.4  | :x:                |

## Reporting a Vulnerability

We take the security of HMS seriously. If you believe you have found a security vulnerability, please do not open a public issue.

Please report vulnerabilities by:
- Opening a [draft security advisory](https://github.com/writerslogic/holographic-memory/security/advisories/new) on GitHub
- Contacting the WritersLogic security team at security@writerslogic.com

We will acknowledge your report within 48 hours and provide a timeline for a fix if applicable.

## Security Practices

- All dependencies are audited with `cargo-deny`
- CI runs `cargo clippy` with `-D warnings`
- Apache-2.0 licensed with full provenance tracking
