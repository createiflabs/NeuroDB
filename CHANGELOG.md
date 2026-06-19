# Changelog

All notable changes to NeuroDB are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/) and the project aims for
[Semantic Versioning](https://semver.org/). The `/v1` HTTP surface is the
stability boundary (see [docs/API_STABILITY.md](docs/API_STABILITY.md)).

## [Unreleased]

### Added
- **Approximate search** (`"approx": true`, cosine) — an optional HNSW candidate
  pre-filter (`pip install 'neurodb[ann]'`) cuts cosine search from `O(N·d)` to
  `O(M·d + log N)` while recovering ≥90% of the exact top-k; exact remains the
  default and the source of truth.
- **Write-ahead log durability** — every write/delete is `fsync`'d to a WAL
  before the call returns and replayed on boot, so a `kill -9` loses no
  acknowledged write. Toggle with `NEURODB_WAL` (default on).
- **Logical JSONL export/import** (`neurodb export` / `neurodb import`,
  `neurodb.portability`) — portable, NumPy-version-independent data interchange,
  so data is never trapped in a version-specific binary format.
- **Batched similarity search** (`Memory.search_batch`) — many queries share one
  matmul.
- **Completion benchmark + honest accuracy doc** ([`benchmarks/`](benchmarks/),
  [docs/ACCURACY.md](docs/ACCURACY.md)) — a reproducible head-to-head against
  scikit-learn imputers and a candid where-it-works / where-it-doesn't writeup.

### Changed
- **Ingest is now amortized `O(1)`** — a geometric-growth backing buffer replaces
  the per-write `np.vstack`, which made loading a dataset one record at a time
  `O(N²)`.
- **Batch writes are atomic** — a single invalid item rejects the whole batch
  with nothing partially applied.

### Fixed
- **Windows backup/restore** — `os.fsync` on a read-only handle raised `EBADF`;
  the staged file is opened read-write before `fsync`.

### Documented
- The concurrency model and honest GIL ceiling, the WAL durability contract, and
  the single-node / vertical-scale-first scope.
