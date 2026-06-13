"""End-to-end NeuroDB example using only the Python standard library.

Start the server first (``python -m neurodb`` or ``docker run ... neurodb``),
then run:  ``python examples/quickstart.py``
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

BASE = os.environ.get("NEURODB_URL", "http://localhost:8000")


def call(path: str, body: dict | None = None, method: str = "GET") -> dict:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        f"{BASE}{path}",
        data=data,
        method="POST" if body is not None and method == "GET" else method,
        headers={"content-type": "application/json"},
    )
    with urllib.request.urlopen(req) as resp:
        return json.load(resp)


def main() -> None:
    info = call("/version")
    dim = info["embedding_dim"]
    print(f"Connected to NeuroDB {info['version']} (embedder dim={dim})")

    # Fresh collection sized for the built-in text embedder.
    try:
        call("/collections/quickstart", method="DELETE")
    except urllib.error.HTTPError:
        pass
    call("/collections", {"name": "quickstart", "dimension": dim, "metric": "cosine"})

    documents = [
        "Golden retrievers are friendly, loyal dog breeds.",
        "Python is a popular language for data science and machine learning.",
        "The Eiffel Tower is a famous landmark in Paris, France.",
        "Vector databases store embeddings for fast semantic search.",
    ]
    call(
        "/collections/quickstart/texts",
        {"items": [{"text": d} for d in documents]},
    )
    print(f"Stored {len(documents)} documents.\n")

    for query in ["friendly dog breeds", "learning to program in python", "travel to france"]:
        hits = call("/collections/quickstart/search/text", {"text": query, "k": 2})
        print(f"Query: {query!r}")
        for hit in hits["results"]:
            print(f"   {hit['score']:.4f}  {hit['metadata']['text']}")
        print()


if __name__ == "__main__":
    main()
