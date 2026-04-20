# Security Policy

## Supported Versions

Security fixes are applied to the latest minor release. Older versions receive
fixes only when the upgrade path is non-trivial.

| Version | Supported |
| ------- | --------- |
| 0.9.x   | Yes       |
| < 0.9   | No        |

## Reporting a Vulnerability

If you believe you have found a security vulnerability in snapvec, please
report it privately rather than opening a public issue.

Preferred channel: email **stffens@gmail.com** with subject line
`[snapvec security] <short description>`.

Please include:
- A description of the issue and its potential impact.
- Steps to reproduce (minimal code sample if possible).
- The snapvec version, Python version, and operating system.

You should receive an initial acknowledgement within 7 days. After triage,
we will coordinate a fix and a disclosure timeline with you.

## Scope

In scope:
- Memory safety issues in Cython kernels (`snapvec/_fast.pyx`).
- Deserialization vulnerabilities in the on-disk index format
  (`snapvec/_file_format.py`, `save` / `load`).
- Numerical correctness issues that produce silent data loss.

Out of scope:
- Resource exhaustion on adversarial inputs (snapvec is a library, not a
  service; the caller owns input validation).
- Issues in third-party dependencies (NumPy, Cython) unless snapvec's usage
  is the trigger.
