# Security Policy

## Reporting a vulnerability

Please report security vulnerabilities privately rather than opening a public
issue:

- Use GitHub's private vulnerability reporting — "Report a vulnerability" under
  the repository's **Security** tab
  (<https://github.com/ornlneutronimaging/NeuNorm/security/advisories/new>), or
- Email the maintainers: zhangc@ornl.gov, bilheuxjm@ornl.gov.

We will acknowledge your report and work with you on a fix and coordinated
disclosure.

## Supported versions

Security fixes target the latest release (the 2.x line). The 1.x series is
legacy and not actively maintained.

## Dependency scanning

Dependencies are scanned in CI with pip-audit and Grype, and kept current via
Dependabot and monthly pixi lockfile refreshes.
