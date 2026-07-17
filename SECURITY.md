# Security Policy

## Supported Versions

| Version | Supported          |
|---------|--------------------|
| 0.1.x   | :white_check_mark: |

## Reporting a Vulnerability

Please report security issues **privately** by emailing
[sachncs@gmail.com](mailto:sachncs@gmail.com). Do not file a public
GitHub issue for suspected vulnerabilities.

Include:

- A description of the issue and its impact.
- Steps to reproduce (a minimal script is best).
- The kvcompress version, Python version, and OS.

You can expect an initial acknowledgement within 72 hours. We'll work
with you on a fix timeline and coordinate disclosure.

## Out-of-scope

- Issues in upstream dependencies (transformers, torch, vLLM). File
  those with the relevant upstream project.
- Denial-of-service from running on adversarial inputs at scale —
  compression ratios are tunable; pick what fits your threat model.