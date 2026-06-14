# API stability promise

NeuroDB is meant to be built on. This document states what you can rely on.

## Versioning

- **`/v1` is stable.** Within `/v1` we only make *additive, backward-compatible*
  changes: new endpoints, new optional request fields, new response fields. Code
  written against `/v1` keeps working.
- **Breaking changes go to `/v2`.** A change that removes or renames a field,
  changes a type, or alters existing semantics ships under a new `/v2` prefix.
  `/v1` continues to be served during a documented overlap period.
- The Python client (`neurodb_client`, re-exported from `neurodb`) follows the
  same promise: public names are stable within a major version.

## Deprecated: unversioned routes

The original unversioned data routes (`/memories`, `/stats`, `/flush`, `/embed`,
…) are **deprecated aliases** of their `/v1` equivalents. They:

- still work today,
- emit a `Deprecation: true` header and a `Link: </v1>; rel="successor-version"`
  header on every response, and
- are **scheduled for removal in the next major release**.

Migrate by prefixing data paths with `/v1`. System probes (`/health`, `/ready`,
`/metrics`, `/version`) are unversioned by design and are not deprecated.

## Data format

The on-disk `.npz` manifest carries a `version`. NeuroDB migrates older files
forward automatically on load (see `neurodb/migrations.py`) and refuses files
written by a newer build with a clear upgrade message. Every version transition
has an explicit, tested migration — your data survives upgrades, or fails loudly
with instructions, never silently.

## Readiness contract

`/ready` returns `503` until the most recent persist succeeded, so a load
balancer will pull a node that cannot durably save (e.g. a full disk) out of
rotation.
