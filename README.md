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

### Create a memory

A *memory* is one Hopfield associative memory: a fixed dimension, an inverse
temperature `beta`, and optional per-dimension `fields` (handy for anomaly
reports on structured records).

```bash
curl -X POST localhost:8000/v1/memories -H 'content-type: application/json' -d '{
  "name": "sensors", "dimension": 3, "beta": 12,
  "fields": ["temperature", "humidity", "pressure"]
}'
```

### Write patterns (append vectors)

```bash
curl -X POST localhost:8000/v1/memories/sensors/patterns -H 'content-type: application/json' -d '{
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
curl -X POST localhost:8000/v1/memories/sensors/complete -H 'content-type: application/json' \
  -d '{"query": [20, 0, 0], "mask": [0]}'
# → reconstruction ≈ [20.0, 51.9, 1012.9]
```

### Per-field anomaly detection

```bash
curl -X POST localhost:8000/v1/memories/sensors/anomaly -H 'content-type: application/json' \
  -d '{"query": [20, 95, 1013]}'
# → score 47.0; top field {"name": "humidity", "value": 95, "expected": 48, "deviation": 47}
```

### Similarity search

```bash
curl -X POST localhost:8000/v1/memories/sensors/search -H 'content-type: application/json' \
  -d '{"query": [20, 50, 1013], "k": 3}'
```

### Bring your own embeddings (Python)

```python
import urllib.request, json
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
dim = model.get_sentence_embedding_dimension()

def call(path, body):
    req = urllib.request.Request(f"http://localhost:8000{path}",
        data=json.dumps(body).encode(), headers={"content-type": "application/json"})
    return json.load(urllib.request.urlopen(req))

call("/v1/memories", {"name": "kb", "dimension": dim, "beta": 16})
docs = ["NeuroDB recalls patterns by content.", "Paris is the capital of France."]
call("/v1/memories/kb/patterns", {"items": [
    {"id": str(i), "vector": model.encode(d).tolist(), "metadata": {"text": d}}
    for i, d in enumerate(docs)]})
hit = call("/v1/memories/kb/complete", {"query": model.encode("memory by content").tolist()})
print(hit["top"]["metadata"]["text"])
```

### Demo text endpoints (built-in embedder)

For zero-setup demos, NeuroDB ships a lightweight lexical embedder. Create a
memory whose dimension matches it (`256` by default) and push text directly:

```bash
curl -X POST localhost:8000/v1/memories/demo/texts -H 'content-type: application/json' \
  -d '{"items": [{"text": "golden retrievers are friendly dogs"}]}'

curl -X POST localhost:8000/v1/memories/demo/recall/text -H 'content-type: application/json' \
  -d '{"text": "friendly puppy", "k": 3}'   # returns the attention distribution
```

> The built-in embedder is *lexical*, not neural — great for demos, but for real
> semantics generate embeddings with your own model and push raw vectors.

---

## Use cases

Four primitives — recall, completion, anomaly, search — cover a wide range of
jobs. The snippets below reuse the `call()` helper and an `embed(text) →
list[float]` from [Bring your own embeddings](#bring-your-own-embeddings-python).

### 🧠 Long-term memory for AI agents

Give an agent a content-addressable memory: it appends observations as vectors
and later recalls the most relevant one by *meaning*, not by key. `/complete`
(or `/recall/text`) returns the recalled item plus an attention distribution —
how strongly the cue matches each stored memory.

```python
call("/v1/memories", {"name": "agent", "dimension": dim, "beta": 16})
call("/v1/memories/agent/patterns", {"items": [
    {"id": "m1", "vector": embed("user prefers dark mode"),  "metadata": {"note": "user prefers dark mode"}},
    {"id": "m2", "vector": embed("we deploy every Friday"),  "metadata": {"note": "we deploy every Friday"}},
]})
hit = call("/v1/memories/agent/complete", {"query": embed("when do we ship?")})
print(hit["top"]["metadata"]["note"])      # → "we deploy every Friday"
```

### 🔎 RAG / semantic retrieval

Store document-chunk embeddings with metadata, then fetch the most relevant
chunks to ground an LLM answer. Metadata filters scope the search — per tenant,
per source, by recency:

```bash
curl -X POST localhost:8000/v1/memories/docs/search -H 'content-type: application/json' -d '{
  "query": [/* query embedding */], "k": 5,
  "filter": {"tenant": "acme", "year": {"$gte": 2024}}
}'
```

### 🧩 Record completion & imputation

Fill the missing fields of a partial record from the nearest stored prototypes —
a sensor reading with a dead channel, a half-filled form, a sparse profile. Mark
the *known* dimensions; the Hopfield step completes the rest:

```bash
# Known: temperature only (dim 0). NeuroDB infers humidity + pressure.
curl -X POST localhost:8000/v1/memories/sensors/complete -H 'content-type: application/json' \
  -d '{"query": [20, 0, 0], "mask": [0]}'        # → ≈ [20.0, 51.9, 1012.9]
```

### 🚨 Anomaly & fault detection

Recall the "normal" pattern an input *should* resemble, then report exactly
which fields deviate and by how much — telemetry drift, fraud features, server
metrics. Name the dimensions and the report reads in plain English:

```bash
curl -X POST localhost:8000/v1/memories/sensors/anomaly -H 'content-type: application/json' \
  -d '{"query": [20, 95, 1013]}'
# → top field {"name": "humidity", "expected": 48, "value": 95, "deviation": 47}
```

### 🪞 Deduplication & "more like this"

Before inserting, search for the nearest existing vector — a near-`1.0` cosine
score means a likely duplicate. The same query powers item-to-item
recommendations ("more like this"), optionally scoped by a metadata `filter`:

```bash
curl -X POST localhost:8000/v1/memories/catalog/search -H 'content-type: application/json' \
  -d '{"query": [/* item embedding */], "k": 1}'
# score ≈ 1.0 → near-duplicate; lower → genuinely novel
```

### ✨ Canonicalization & fuzzy lookup

Snap a noisy or partial query to the closest canonical entry — normalize a messy
address, product name, or SKU to a known record. Store the canonical forms, then
recall with a high `beta` so the result snaps to the single best match.

> **Choosing `beta`:** high `beta` (≈ 16–50) gives crisp, single-match recall
> (memory, dedup, canonicalization); low `beta` returns a soft blend of related
> patterns (useful for smoothing or "average of similar records").

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
| `POST`   | `/memories/{n}/patterns`              | Append patterns                            |
| `GET`    | `/memories/{n}/patterns/{id}`         | Fetch a pattern                            |
| `DELETE` | `/memories/{n}/patterns/{id}`         | Delete a pattern                           |
| `POST`   | `/memories/{n}/complete`              | Recall / pattern completion                |
| `POST`   | `/memories/{n}/search`                | Nearest patterns (cosine)                  |
| `POST`   | `/memories/{n}/anomaly`               | Per-field anomaly detection                |
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
