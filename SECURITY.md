# Security Policy

## Supported Versions

| Version | Supported          |
|---------|--------------------|
| 0.2.x   | :white_check_mark: |
| 0.1.x   | security fixes only (EOL 2026-12-31) |

## Reporting a Vulnerability

Please report security issues **privately** by emailing
[sachncs@gmail.com](mailto:sachncs@gmail.com). Do not file a public
GitHub issue for suspected vulnerabilities.

Include:

- A description of the issue and its impact.
- Steps to reproduce (a minimal script is best).
- The kvfold version, Python version, and OS.

You can expect an initial acknowledgement within 72 hours. We'll work
with you on a fix timeline and coordinate disclosure.

## Data and telemetry boundary

kvfold is a local library. It does not transmit model weights, prompts, KV
cache tensors, generated text, or usage telemetry. Optional integration tests
may download public models from Hugging Face, and users' own benchmark scripts
may write tensors or outputs to paths they select. Treat those artifacts as
potentially sensitive and do not attach them to public issues.

Dependency vulnerabilities are scanned in CI with `pip-audit` and npm's
production audit. Reports for upstream projects should be sent to their
security channels as well as optionally notifying this project when the
affected dependency is in kvfold's supported runtime range.

## Out-of-scope

- Issues in upstream dependencies (transformers, torch, vLLM). File
  those with the relevant upstream project.
- Denial-of-service from running on adversarial inputs at scale —
  compression ratios are tunable; pick what fits your threat model.
