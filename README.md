<div align="center">

# 🧠 NeuroDB

**A content-addressable store powered by Modern Hopfield networks.**

*Writing a pattern is appending a vector. Retrieval is a single attention step.*

[![CI](https://github.com/createiflabs/NeuroDB/actions/workflows/ci.yml/badge.svg)](https://github.com/createiflabs/NeuroDB/actions/workflows/ci.yml)
[![Docker Pulls](https://img.shields.io/docker/pulls/createiflabs/neurodb)](https://hub.docker.com/r/createiflabs/neurodb)
[![Image Size](https://img.shields.io/docker/image-size/createiflabs/neurodb/latest)](https://hub.docker.com/r/createiflabs/neurodb)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](pyproject.toml)

</div>

NeuroDB stores patterns (vectors) as the rows of a matrix and recalls them by
content, not by key. It is built on **Modern Hopfield networks**
([Ramsauer et al., 2020](https://arxiv.org/abs/2008.02217)), whose update rule is
a single attention step:

```
p  = softmax(β · X · q)     # how strongly a query q attends to each stored pattern
x* = Xᵀ · p                 # the recalled (reconstructed) pattern
```

With a high inverse-temperature `β`, recall snaps to the single closest stored
pattern (exact content-addressable memory and pattern completion); with a low
`β`, it returns a soft blend. From this one primitive NeuroDB offers:

- 🧲 **Associative recall & pattern completion** — query with a full, partial or
  noisy vector and get the completed pattern back (`/complete`, with an optional
  field `mask`).
- 🚨 **Per-field anomaly detection** — recall the pattern an input *should* match,
  then report exactly which fields deviate and by how much (`/anomaly`).
- 🔎 **Similarity search** — classic nearest-neighbour retrieval by cosine
  similarity (`/search`).
- ✍️ **Append-only writes** — storing a pattern is just appending a row; no
  training, no index rebuild.
- 💾 **Single-file persistence** — the entire store is one `.npz` file that loads
  on startup and autosaves.
- 🐳 **Container-native** — a slim, non-root, multi-arch image.

---

## Quick start

### 🐳 Docker

```bash
docker run -d --name neurodb -p 8000:8000 -v neurodb_data:/data createiflabs/neurodb:latest
```

Open **http://localhost:8000** for the dashboard or **/docs** for the OpenAPI UI.

### 🧱 Docker Compose

```bash
docker compose up -d
```

### 🐍 From source

```bash
pip install -r requirements-dev.txt -e .
python -m neurodb              # serves on http://0.0.0.0:8000
```

---

## Walkthrough

### Python client (recommended)

A thin, dependency-free client ships in this repo (`neurodb_client`, stdlib only).
It's the recommended way to use NeuroDB from Python; the raw HTTP below works from
any language.

```python
import neurodb_client as neurodb   # or: from neurodb import connect, ValidationReport

db = neurodb.connect("http://localhost:8000", api_key=None)
mem = db.create("sensors", dimension=3, beta=12,
                fields=["temperature", "humidity", "pressure"], normalize="zscore")
mem.write([{"vector": [20, 50, 1013]}, {"vector": [21, 52, 1012]}])
print(mem.anomaly([20, 95, 1013])["fields"][0])      # → which field is weird, by how many σ
print(mem.anomaly_batch([[20, 95, 1013], [21, 52, 1012]]))  # batch, one matmul
print(mem.capacity()["status"])                       # healthy | crowded | saturated
```

Typed exceptions (`NotFound`, `BadRequest`, `Unauthorized`) map the server's
error envelope.

### Create a memory

A *memory* is one Hopfield associative memory: a fixed dimension, an inverse
temperature `beta`, optional per-dimension `fields` (handy for anomaly reports on
structured records), and a `normalize` mode (see
[Normalization](#normalization)).

```bash
curl -X POST localhost:8000/memories -H 'content-type: application/json' -d '{
  "name": "sensors", "dimension": 3, "beta": 12,
  "fields": ["temperature", "humidity", "pressure"]
}'
```

> Because `fields` are given, this memory defaults to `normalize: "zscore"` — each
> dimension is standardized before recall so a large-magnitude field (here
> `pressure ~1013`) can't drown out small ones (`temperature ~20`). Pass
> `"normalize": "none"` to opt out, or `"l2"` for unit-direction embeddings.

### Write patterns (append vectors)

```bash
curl -X POST localhost:8000/memories/sensors/patterns -H 'content-type: application/json' -d '{
  "items": [
    {"vector": [20, 50, 1013]}, {"vector": [21, 52, 1012]},
    {"vector": [19, 48, 1014]}, {"vector": [20, 51, 1013]}
  ]
}'
```

### Pattern completion

Know only the temperature? Mark dimension `0` as known and let NeuroDB complete
the rest:

```bash
curl -X POST localhost:8000/memories/sensors/complete -H 'content-type: application/json' \
  -d '{"query": [20, 0, 0], "mask": [0]}'
# → reconstruction ≈ [20.0, 50.25, 1013.0]   (humidity/pressure recalled in raw units)
```

### Per-field anomaly detection

```bash
curl -X POST localhost:8000/memories/sensors/anomaly -H 'content-type: application/json' \
  -d '{"query": [20, 95, 1013]}'
# → score 43.0, z_score 29.1; top field
#   {"name": "humidity", "value": 95, "expected": 52, "deviation": 43, "z_deviation": 29.1}
```

With `zscore`, each field's `z_deviation` is "how many standard deviations off",
so anomalies are comparable *across* fields (humidity here is ~29σ out); fields
are ranked by `z_deviation`. The raw `deviation` and `score` are still reported in
original units for backward compatibility.

### Batch recall (anomaly / completion at throughput)

Anomaly detection is a stream/bulk job, so score a whole batch in **one matmul**
(not one HTTP call per record). Each item may carry an `id` that is echoed back
so callers can correlate results:

```bash
curl -X POST localhost:8000/memories/sensors/anomaly/batch -H 'content-type: application/json' \
  -d '{"items": [{"id": "r1", "vector": [20, 95, 1013]}, {"id": "r2", "vector": [21, 52, 1012]}]}'
# → {"results": [ {<same shape as /anomaly>, "id": "r1"}, ... ], "count": 2}
```

`/complete/batch` is the same envelope. A batch result for each item is identical
to the single-query call; the cap per request is `NEURODB_MAX_BATCH`.

### Dataset validation

Per-field anomaly scoring composes into a one-call **dataset check**:
`Memory.validate(records, threshold=...)` streams records through the batch
anomaly endpoint, flags any field whose `z_deviation` exceeds `threshold`, and
returns a `ValidationReport` — plain data that drops straight into a data
pipeline.

```python
from neurodb import connect

db = connect("http://localhost:8000")
mem = db.memory("sensors")

report = mem.validate(
    [{"id": "r1", "vector": [20, 95, 1013]}, {"id": "r2", "vector": [21, 52, 1012]}],
    threshold=3.0,            # σ; defaults to 3.0
    fields=None,              # optionally restrict to a subset of field names
)
print(report.summary())       # NeuroDB validation [sensors]: 1/2 passed (50.0%), 1 failed, threshold=3.0
print(report.ok)              # False  (also: bool(report))
report.to_dict()              # JSON-friendly: records, per-field stats, pass_rate
```

It is pure client-side composition over `/anomaly/batch` — no extra server
endpoint. Each run emits one local telemetry event; see **Telemetry** below.

### Integrations

Optional adapters wrap `validate(...)` for common data-stack tools. They live in
`neurodb_client.integrations` and are guarded behind extras — install only what
you use:

```bash
pip install 'neurodb[great-expectations]'   # or [airflow], [dagster], [integrations]
```

```python
# Great Expectations — a callable returning a GE-style {success, result}
from neurodb_client.integrations.great_expectations import neurodb_expectation
check = neurodb_expectation(mem, threshold=3.0)
check(records)

# Airflow — fails the task when records fail validation
from neurodb_client.integrations.airflow import NeuroDBValidateOperator
NeuroDBValidateOperator(task_id="validate", memory=mem, records=records)

# Dagster — an asset check on the upstream asset
from neurodb_client.integrations.dagster import neurodb_asset_check
neurodb_asset_check(asset=my_asset, memory=mem, records=records)
```

Each adapter raises a clear `ImportError` (naming the extra) if its library
isn't installed, so importing the subpackage never drags the heavy deps in.

### Telemetry

Validation runs emit a small `TelemetryEvent` (counts + pass rate, no payloads).
The default sink is a **no-op** — nothing is collected and nothing leaves the
machine. NeuroDB ships no remote endpoint; forwarding is a seam you opt into:

```python
from neurodb_client import telemetry

agg = telemetry.LocalAggregator()       # in-process counters
telemetry.set_sink(agg)
...                                      # run validations
print(agg.snapshot())                   # {"runs": 3, "records": 900, "failures": 12, ...}

# To ship elsewhere, implement emit() and register it:
class MySink:
    def emit(self, event): requests.post(MY_URL, json=event.to_dict())
telemetry.set_sink(MySink())
```

### Filtered recall

`complete` and `anomaly` (and their batch forms) accept the same metadata
`filter` as `search`, to score a record only against patterns of the same type —
recall runs over the filtered subset, normalized by the full-memory statistics:

```bash
curl -X POST localhost:8000/memories/sensors/anomaly -H 'content-type: application/json' \
  -d '{"query": [20, 95, 1013], "filter": {"site": "warehouse-A"}}'
```

### Updating records

Records change. `POST .../patterns` with an existing `id` already **upserts** it.
For partial edits use `PATCH` — `vector` replaces the row, `metadata` shallow-merges
(or replaces with `merge_metadata: false`):

```bash
curl -X PATCH localhost:8000/memories/sensors/patterns/r1 -H 'content-type: application/json' \
  -d '{"metadata": {"label": "checked"}}'
```

### Capacity / saturation diagnostics

Modern Hopfield memories have finite capacity: past it, distinct patterns'
attractors merge and recall *silently* returns blends. NeuroDB knows it's a
Hopfield network and can warn you:

```bash
curl localhost:8000/memories/sensors/capacity
# → {"status": "healthy|crowded|saturated", "self_recall_fail_fraction": ...,
#    "max_pairwise_similarity": ..., "suggested_beta": ..., ...}
```

`self_recall_fail_fraction` is the honest signal — the fraction of sampled
patterns that can't even retrieve *themselves* at the memory's β. A compact
version appears per memory in `/stats`, and `/health` reports a
`saturated_memories` count.

> **Worked benchmark:** [`examples/anomaly_benchmark/`](examples/anomaly_benchmark/)
> runs per-field anomaly attribution on a realistic multi-scale dataset and
> compares detection quality with scikit-learn's IsolationForest — matching its
> ROC-AUC (~0.93) with **zero training** while naming the offending field and its
> σ deviation.

### Similarity search

```bash
curl -X POST localhost:8000/memories/sensors/search -H 'content-type: application/json' \
  -d '{"query": [20, 50, 1013], "k": 3}'
```

### Normalization

The Hopfield similarity step is a dot product `X·q`, so without normalization the
field with the largest numeric magnitude dominates the softmax and smaller fields
become invisible — an anomaly detector's worst case. Each memory picks one mode at
creation via `normalize`:

| Mode       | What it does                              | Use when                                                                 |
| ---------- | ----------------------------------------- | ------------------------------------------------------------------------ |
| `"zscore"` | per-dimension standardize `(x − μ) / σ`   | **structured records** (sensors, profiles) — fields on disparate scales  |
| `"l2"`     | per-row L2 normalize to unit length       | text/embedding vectors where direction matters, magnitude doesn't        |
| `"none"`   | raw dot product (pre-normalization)       | vectors that are already comparably scaled / unit-norm                    |

- **Default:** `"zscore"` when `fields` is given, else `"none"`. Loaded legacy
  memories (files without the key) default to `"none"`, so existing data behaves
  exactly as before.
- `zscore` runs recall in standardized space and **de-normalizes the
  reconstruction back to raw units**; `anomaly` additionally reports
  `z_deviation`/`z_score` (deviations measured in σ). `mean`/`std` are recomputed
  from the matrix on every write — never persisted — and exposed per memory in
  `/stats` for debugging.
- `l2` reconstructions live on the unit sphere and are **directional, not in
  original magnitude units**.
- `search` always ranks by cosine on the **raw** vectors regardless of mode
  (cosine already normalizes magnitude away), so search results are unaffected.
- **Tuning `beta`:** after `zscore`, similarities are on a standardized scale (per
  field roughly ±a few σ), so a *much* smaller `beta` than for raw records avoids
  saturating the softmax onto a single row. Start around `beta` 1–4 and raise it
  for sharper recall.

### Bring your own embeddings (raw HTTP)

The `neurodb_client` above removes this boilerplate, but here's the dependency-free
HTTP path (works from any language) with your own embedding model:

```python
import urllib.request, json
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
dim = model.get_sentence_embedding_dimension()

def call(path, body):
    req = urllib.request.Request(f"http://localhost:8000{path}",
        data=json.dumps(body).encode(), headers={"content-type": "application/json"})
    return json.load(urllib.request.urlopen(req))

call("/memories", {"name": "kb", "dimension": dim, "beta": 16})
docs = ["NeuroDB recalls patterns by content.", "Paris is the capital of France."]
call("/memories/kb/patterns", {"items": [
    {"id": str(i), "vector": model.encode(d).tolist(), "metadata": {"text": d}}
    for i, d in enumerate(docs)]})
hit = call("/memories/kb/complete", {"query": model.encode("memory by content").tolist()})
print(hit["top"]["metadata"]["text"])
```

### Demo text endpoints (built-in embedder)

For zero-setup demos, NeuroDB ships a lightweight lexical embedder. Create a
memory whose dimension matches it (`256` by default) and push text directly:

```bash
curl -X POST localhost:8000/memories/demo/texts -H 'content-type: application/json' \
  -d '{"items": [{"text": "golden retrievers are friendly dogs"}]}'

curl -X POST localhost:8000/memories/demo/recall/text -H 'content-type: application/json' \
  -d '{"text": "friendly puppy", "k": 3}'   # returns the attention distribution
```

> The built-in embedder is *lexical*, not neural — great for demos, but for real
> semantics generate embeddings with your own model and push raw vectors.

---

## API reference

| Method   | Path                                  | Description                                |
| -------- | ------------------------------------- | ------------------------------------------ |
| `GET`    | `/`                                   | Web dashboard                              |
| `GET`    | `/health` · `/version`                | Liveness & engine info (public)            |
| `GET`    | `/docs`                               | Interactive OpenAPI docs                   |
| `GET`    | `/stats`                              | Per-memory statistics                      |
| `POST`   | `/memories`                           | Create a memory                            |
| `GET`    | `/memories` · `/memories/{n}`         | List / inspect memories                    |
| `DELETE` | `/memories/{n}`                       | Delete a memory                            |
| `POST`   | `/memories/{n}/patterns`              | Append patterns (existing id upserts)      |
| `GET`    | `/memories/{n}/patterns/{id}`         | Fetch a pattern                            |
| `PATCH`  | `/memories/{n}/patterns/{id}`         | Update a pattern (vector / metadata)       |
| `DELETE` | `/memories/{n}/patterns/{id}`         | Delete a pattern                           |
| `POST`   | `/memories/{n}/complete`              | Recall / pattern completion                |
| `POST`   | `/memories/{n}/complete/batch`        | Batch completion (one matmul)              |
| `POST`   | `/memories/{n}/search`                | Nearest patterns (cosine)                  |
| `POST`   | `/memories/{n}/anomaly`               | Per-field anomaly detection                |
| `POST`   | `/memories/{n}/anomaly/batch`         | Batch anomaly detection (one matmul)       |
| `GET`    | `/memories/{n}/capacity`              | Hopfield capacity / saturation report      |
| `POST`   | `/memories/{n}/texts`                 | Append text (built-in embedder)            |
| `POST`   | `/memories/{n}/search/text`           | Text similarity search                     |
| `POST`   | `/memories/{n}/recall/text`           | Text recall (attention distribution)       |
| `POST`   | `/embed`                              | Embed text → vector                        |

---

## Configuration

All settings are environment variables:

| Variable                     | Default               | Description                                                        |
| ---------------------------- | --------------------- | ------------------------------------------------------------------ |
| `NEURODB_DATA_FILE`          | `./data/neurodb.npz`  | Single file the whole store persists to.                           |
| `NEURODB_HOST`               | `0.0.0.0`             | Bind address.                                                      |
| `NEURODB_PORT`               | `8000`                | Bind port.                                                         |
| `NEURODB_AUTOSAVE_INTERVAL`  | `5`                   | Seconds between autosaves when dirty.                              |
| `NEURODB_MAX_BATCH`          | `1024`                | Max items in one `/anomaly/batch` or `/complete/batch` request.    |
| `NEURODB_API_KEY`            | _(unset)_             | Required on data routes when set (see auth below).                 |
| `NEURODB_ALLOW_ANONYMOUS`    | `false`               | Allow running without a key. **Required** if `NEURODB_API_KEY` is unset, otherwise the server refuses to start. |
| `NEURODB_CORS_ORIGINS`       | _(empty)_             | Comma-separated allowed CORS origins. Empty = no cross-origin.     |
| `NEURODB_RATE_LIMIT_PER_MINUTE` | `600`              | Per-client (API key, else IP) request budget; `0` disables.        |
| `NEURODB_MAX_REQUEST_BYTES`  | `8388608`             | Reject larger request bodies with `413`.                           |
| `NEURODB_FAIL_ON_CORRUPT_LOAD` | `false`             | Fail startup on an unreadable data file instead of quarantining.   |
| `NEURODB_EMBEDDING_DIM`      | `256`                 | Dimension of the built-in text embedder.                           |
| `NEURODB_LOG_LEVEL`          | `info`                | Log level.                                                         |
| `NEURODB_LOG_FORMAT`         | `json`                | `json` (structured, with request ids) or `text`.                  |

### Security & operations

NeuroDB is **secure by default**: with no `NEURODB_API_KEY` it refuses to start
unless `NEURODB_ALLOW_ANONYMOUS=1` is set explicitly. When a key is set, send
`X-API-Key: <key>` or `Authorization: Bearer <key>`. CORS is closed unless you
list origins, request bodies are size-capped, and a per-client rate limit is
applied. `/health`, `/version`, `/ready`, `/metrics` and `/` stay public.

- **API**: served under `/v1` (the unversioned paths still work but send a
  `Deprecation` header).
- **Liveness/readiness**: `/health` (cheap liveness) and `/ready` (503 until the
  last persist succeeded).
- **Metrics**: Prometheus exposition at `/metrics`.
- **Durability**: see [docs/OPERATIONS.md](docs/OPERATIONS.md); deployment guide
  in [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md).

---

## How it works

Patterns are the rows of a `float32` matrix `X`. Every operation is the same
single Hopfield attention step `p = softmax(β·X·q)`, computed as one vectorised
NumPy matmul (`O(N·d)`), followed by `x* = Xᵀ·p`:

- **complete** runs the step; with a `mask`, the known dimensions are clamped and
  only the unknown ones are filled (iterate with `steps > 1` for sharper recall).
- **anomaly** recalls `x*` for the input and reports the per-dimension residual
  `|q − x*|`, naming the worst fields.
- **search** ranks patterns by cosine similarity.

Per-memory [normalization](#normalization) (`zscore`/`l2`) runs the attention step
in normalized space and de-normalizes the result, so disparately-scaled fields
contribute comparably; `search` always ranks on the raw vectors.

The batch endpoints run the same step over an `(M, D)` query block as a single
matmul (`softmax(β·Q·Xᵀ)` → `weights·X`), so scoring a stream of records is
vectorized, not a Python loop. `capacity` reports the network's saturation
(self-recall failure fraction) — the Hopfield-specific signal that recall is
degrading into blends, which a plain vector store can't tell you.

The whole store — every memory's matrix, ids and metadata — serialises to a
single `.npz` file. Each save snapshots every memory under its lock, writes a
temp file, `fsync`s it, and atomically renames it into place, so a crash never
leaves a torn file and the persisted matrix always matches its ids. Writes are
buffered in memory and persisted by (a) the periodic autosave when dirty,
(b) an explicit `POST /v1/flush` (synchronous + durable), and (c) graceful
shutdown. A `kill -9` can therefore lose up to `NEURODB_AUTOSAVE_INTERVAL`
seconds of un-flushed writes — call `/v1/flush` when you need a hard durability
point. A corrupt data file on startup is quarantined (never deleted) and the
store starts empty, unless `NEURODB_FAIL_ON_CORRUPT_LOAD=1`.

> **Scope:** NeuroDB favours exactness and simplicity over billion-scale ANN.
> It's a clean, correct associative-memory layer that's easy to reason about.

---

## Performance & limits

Recall is an exact full scan — every query is an `O(N·D)` matmul — so latency
grows **linearly with the number of patterns**. Representative single-node numbers
(commodity laptop CPU; run `python examples/benchmark_perf/run.py` for your own):

| N patterns | D dims | single `anomaly` | `anomaly_batch` (M=256) | memory |
|-----------:|-------:|-----------------:|------------------------:|-------:|
| 1,000      | 32     | ~0.1 ms          | ~5 ms                   | ~0.25 MiB |
| 10,000     | 64     | ~0.3 ms          | ~28 ms                  | ~5 MiB |
| 50,000     | 128    | ~3 ms            | ~0.5 s                  | ~49 MiB |
| 100,000    | 256    | ~8 ms            | ~1.0 s                  | ~195 MiB |

Single-record anomaly stays sub-10 ms into the 100k range; batch latency and
memory scale with N. A `zscore`/`l2` memory keeps a cached normalized matrix, so
it costs roughly **2× raw** (`N·D·4` bytes). Past a few hundred thousand patterns
per memory on one node, shard the population across memories or reach for a
purpose-built ANN index — NeuroDB is built for exact recall and per-field
attribution at single-node scale.

Set resource ceilings so a runaway writer is rejected (HTTP `413`) instead of
OOM-crashing the process — reads keep serving:

| Env var | Default | Effect |
|---|---|---|
| `NEURODB_MAX_PATTERNS_PER_MEMORY` | `1000000` | reject writes past this count (`0`=unlimited) |
| `NEURODB_MAX_TOTAL_BYTES` | unset | reject writes past this estimated footprint |
| `NEURODB_MEMORY_PRESSURE_PCT` | `90` | `/health` flips `memory_pressure` at this % of budget |

`/v1/stats` reports `approx_bytes` and `pct_of_budget`. For backup/restore,
upgrades, the slowlog, and the API-stability promise, see
[`docs/OPERATIONS.md`](docs/OPERATIONS.md) and
[`docs/API_STABILITY.md`](docs/API_STABILITY.md).

---

## Reference collections

A **collection** is a portable, signed bundle — a curated reference population
plus its baseline, schema, and provenance — that loads as a ready-to-score
memory. Map your table to its schema and start validating on day one; no
clean-baseline-of-your-own required.

```bash
python examples/collections/build_sample.py          # build the toy sample
neurodb collection info examples/collections/sample_service_health.ndcoll
neurodb collection load examples/collections/sample_service_health.ndcoll
```

The **format and tooling are open** (build/sign/verify/load + a synthetic-
generation framework with realism diagnostics); *specific* domain-validated,
attested collections are licensed content. See
[`docs/COLLECTIONS.md`](docs/COLLECTIONS.md).

---

## Development

```bash
pip install -r requirements-dev.txt -e .
ruff check .        # lint
pytest -q           # tests
python -m neurodb   # run locally   (or: make install && make test && make run)
python examples/quickstart.py   # end-to-end demo against a running server
```

CI lints and tests on Python 3.10–3.12, then builds a multi-arch
(`linux/amd64`, `linux/arm64`) image and publishes it to Docker Hub
(`createiflabs/neurodb`) on pushes to `main` and version tags.

---

## License

MIT © 2026 createif labs
