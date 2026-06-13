<div align="center">

# 🧠 NeuroDB

**A lightweight, container-native vector database for AI memory and semantic search.**

[![CI](https://github.com/createiflabs/NeuroDB/actions/workflows/ci.yml/badge.svg)](https://github.com/createiflabs/NeuroDB/actions/workflows/ci.yml)
[![Docker Pulls](https://img.shields.io/docker/pulls/createiflabs/neurodb)](https://hub.docker.com/r/createiflabs/neurodb)
[![Image Size](https://img.shields.io/docker/image-size/createiflabs/neurodb/latest)](https://hub.docker.com/r/createiflabs/neurodb)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](pyproject.toml)

</div>

NeuroDB stores high-dimensional vectors (embeddings) alongside arbitrary JSON
metadata and serves fast nearest-neighbour similarity search over a clean REST
API. It's the long-term **memory layer** for AI apps: semantic search,
retrieval-augmented generation (RAG), recommendations and deduplication.

- ⚡ **Fast** — vectorised cosine / dot / Euclidean search powered by NumPy.
- 🧩 **Simple** — one tiny service, a REST API, an OpenAPI spec and a live dashboard.
- 💾 **Durable** — collections persist to disk and reload on startup.
- 🏷️ **Filterable** — Mongo-style metadata filters (`$in`, `$gte`, `$lt`, …).
- 🔐 **Secure-ready** — optional API-key auth on all data routes.
- 🐳 **Container-native** — a slim, non-root, multi-arch image: `createiflabs/neurodb`.
- 🪶 **Zero-config demo** — a built-in lexical text embedder so you can try it instantly.

---

## Quick start

### 🐳 Docker (recommended)

```bash
# GitHub Container Registry (published by CI on every push to main):
docker run -d --name neurodb -p 8000:8000 -v neurodb_data:/data ghcr.io/createiflabs/neurodb:latest

# Docker Hub (published once the DOCKERHUB_TOKEN secret is configured):
docker run -d --name neurodb -p 8000:8000 -v neurodb_data:/data createiflabs/neurodb:latest
```

Then open **http://localhost:8000** for the dashboard, or
**http://localhost:8000/docs** for interactive API docs.

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

## The dashboard

Browse to the root URL for a built-in playground: paste documents, embed and
store them, then run semantic queries and watch them rank by similarity. The
demo uses NeuroDB's lightweight **lexical** embedder so it works with no model
downloads — for production semantics, generate embeddings with your model of
choice and push the raw vectors via the API.

---

## Using the API

### 1. Create a collection

```bash
curl -X POST localhost:8000/collections \
  -H 'content-type: application/json' \
  -d '{"name": "documents", "dimension": 256, "metric": "cosine"}'
```

### 2. Upsert vectors with metadata

```bash
curl -X POST localhost:8000/collections/documents/vectors \
  -H 'content-type: application/json' \
  -d '{"items": [
        {"id": "doc-1", "vector": [0.12, 0.04, ...], "metadata": {"lang": "en"}},
        {"id": "doc-2", "vector": [0.31, 0.92, ...], "metadata": {"lang": "de"}}
      ]}'
```

### 3. Search (with an optional metadata filter)

```bash
curl -X POST localhost:8000/collections/documents/search \
  -H 'content-type: application/json' \
  -d '{"vector": [0.10, 0.05, ...], "k": 5, "filter": {"lang": "en"}}'
```

### Bring your own embeddings (Python)

```python
import urllib.request, json
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")   # 384-dim
dim = model.get_sentence_embedding_dimension()

def call(path, body):
    req = urllib.request.Request(
        f"http://localhost:8000{path}",
        data=json.dumps(body).encode(),
        headers={"content-type": "application/json"},
    )
    return json.load(urllib.request.urlopen(req))

call("/collections", {"name": "kb", "dimension": dim, "metric": "cosine"})
docs = ["NeuroDB is a vector database.", "Paris is the capital of France."]
call("/collections/kb/vectors", {"items": [
    {"id": str(i), "vector": model.encode(d).tolist(), "metadata": {"text": d}}
    for i, d in enumerate(docs)
]})
hits = call("/collections/kb/search",
            {"vector": model.encode("which db stores embeddings?").tolist(), "k": 2})
print(hits["results"][0]["metadata"]["text"])
```

### Demo text endpoints (built-in embedder)

If a collection's dimension matches the built-in embedder (`256` by default),
you can skip embedding yourself:

```bash
curl -X POST localhost:8000/collections/demo/texts \
  -H 'content-type: application/json' \
  -d '{"items": [{"text": "golden retrievers are friendly dogs"}]}'

curl -X POST localhost:8000/collections/demo/search/text \
  -H 'content-type: application/json' \
  -d '{"text": "friendly dog breeds", "k": 3}'
```

---

## API reference

| Method   | Path                                   | Description                          |
| -------- | -------------------------------------- | ------------------------------------ |
| `GET`    | `/`                                    | Web dashboard                        |
| `GET`    | `/health`                              | Liveness + counts (public)           |
| `GET`    | `/version`                             | Version & embedder info (public)     |
| `GET`    | `/docs`                                | Interactive OpenAPI docs             |
| `GET`    | `/stats`                               | Per-collection statistics            |
| `POST`   | `/collections`                         | Create a collection                  |
| `GET`    | `/collections`                         | List collections                     |
| `GET`    | `/collections/{name}`                  | Collection info                      |
| `DELETE` | `/collections/{name}`                  | Delete a collection                  |
| `POST`   | `/collections/{name}/persist`          | Force-flush to disk                  |
| `POST`   | `/collections/{name}/vectors`          | Upsert vectors                       |
| `GET`    | `/collections/{name}/vectors/{id}`     | Fetch a vector                       |
| `DELETE` | `/collections/{name}/vectors/{id}`     | Delete a vector                      |
| `POST`   | `/collections/{name}/search`           | Nearest-neighbour search             |
| `POST`   | `/collections/{name}/texts`            | Upsert text (built-in embedder)      |
| `POST`   | `/collections/{name}/search/text`      | Text search (built-in embedder)      |
| `POST`   | `/embed`                               | Embed text → vector                  |

---

## Configuration

All settings are environment variables:

| Variable                    | Default  | Description                                      |
| --------------------------- | -------- | ------------------------------------------------ |
| `NEURODB_DATA_DIR`          | `./data` | Where collections are persisted.                 |
| `NEURODB_HOST`              | `0.0.0.0`| Bind address.                                    |
| `NEURODB_PORT`              | `8000`   | Bind port.                                        |
| `NEURODB_AUTOSAVE_INTERVAL` | `5`      | Seconds between autosaves of dirty collections.  |
| `NEURODB_API_KEY`           | _(unset)_| If set, all data routes require this key.        |
| `NEURODB_CORS_ORIGINS`      | `*`      | Comma-separated allowed CORS origins.            |
| `NEURODB_EMBEDDING_DIM`     | `256`    | Dimension of the built-in text embedder.         |
| `NEURODB_LOG_LEVEL`         | `info`   | Log level.                                        |

When `NEURODB_API_KEY` is set, send it as `X-API-Key: <key>` or
`Authorization: Bearer <key>`.

---

## How it works

Each collection keeps its vectors in a contiguous `float32` matrix, so a search
is a single vectorised NumPy operation (`O(N·D)` brute force) followed by a
partial top-`k` sort. This is exact, dependency-light and plenty fast into the
hundreds of thousands of vectors. Collections persist as a `vectors.npy` matrix
plus a `meta.json` sidecar and reload automatically on startup.

> **Scope:** NeuroDB favours correctness and simplicity over billion-scale ANN
> indexes. If you outgrow brute-force search, it's a clean base to add an HNSW
> index behind the same API.

---

## Development

```bash
pip install -r requirements-dev.txt -e .
ruff check .        # lint
pytest -q           # tests
python -m neurodb   # run locally

# or use the Makefile:
make install && make test && make run
```

Build the image locally:

```bash
docker build -t createiflabs/neurodb:dev .
docker run --rm -p 8000:8000 createiflabs/neurodb:dev
```

CI (GitHub Actions) lints and tests on Python 3.10–3.12, then builds and pushes
a multi-arch (`linux/amd64`, `linux/arm64`) image to Docker Hub on pushes to
`main` and version tags.

---

## License

MIT © 2026 createif labs
