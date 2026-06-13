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
# GitHub Container Registry (published by CI on every push to main):
docker run -d --name neurodb -p 8000:8000 -v neurodb_data:/data ghcr.io/createiflabs/neurodb:latest

# Docker Hub (published once the DOCKERHUB_TOKEN secret is configured):
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
curl -X POST localhost:8000/memories -H 'content-type: application/json' -d '{
  "name": "sensors", "dimension": 3, "beta": 12,
  "fields": ["temperature", "humidity", "pressure"]
}'
```

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
# → reconstruction ≈ [20.0, 51.9, 1012.9]
```

### Per-field anomaly detection

```bash
curl -X POST localhost:8000/memories/sensors/anomaly -H 'content-type: application/json' \
  -d '{"query": [20, 95, 1013]}'
# → score 47.0; top field {"name": "humidity", "value": 95, "expected": 48, "deviation": 47}
```

### Similarity search

```bash
curl -X POST localhost:8000/memories/sensors/search -H 'content-type: application/json' \
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

| Variable                    | Default               | Description                                   |
| --------------------------- | --------------------- | --------------------------------------------- |
| `NEURODB_DATA_FILE`         | `./data/neurodb.npz`  | Single file the whole store persists to.      |
| `NEURODB_HOST`              | `0.0.0.0`             | Bind address.                                 |
| `NEURODB_PORT`              | `8000`                | Bind port.                                    |
| `NEURODB_AUTOSAVE_INTERVAL` | `5`                   | Seconds between autosaves when dirty.         |
| `NEURODB_API_KEY`           | _(unset)_             | If set, all data routes require this key.      |
| `NEURODB_CORS_ORIGINS`      | `*`                   | Comma-separated allowed CORS origins.         |
| `NEURODB_EMBEDDING_DIM`     | `256`                 | Dimension of the built-in text embedder.      |
| `NEURODB_LOG_LEVEL`         | `info`                | Log level.                                    |

When `NEURODB_API_KEY` is set, send `X-API-Key: <key>` or `Authorization: Bearer <key>`.

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
single `.npz` file, written atomically and reloaded on startup.

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
(`linux/amd64`, `linux/arm64`) image and publishes it to GHCR (always) and Docker
Hub (when `DOCKERHUB_TOKEN` is set).

---

## License

MIT © 2026 createif labs
