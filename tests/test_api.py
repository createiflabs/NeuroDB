import pytest
from fastapi.testclient import TestClient

from neurodb.config import Settings
from neurodb.server import create_app


@pytest.fixture()
def client(tmp_path):
    settings = Settings(data_dir=str(tmp_path), autosave_interval=0.0)
    with TestClient(create_app(settings)) as test_client:
        yield test_client


def test_health_is_public(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert "version" in body


def test_collection_crud_and_vector_search(client):
    created = client.post("/collections", json={"name": "docs", "dimension": 3, "metric": "cosine"})
    assert created.status_code == 201, created.text
    assert created.json()["count"] == 0

    upserted = client.post(
        "/collections/docs/vectors",
        json={
            "items": [
                {"id": "a", "vector": [1, 0, 0], "metadata": {"t": "a"}},
                {"id": "b", "vector": [0, 1, 0], "metadata": {"t": "b"}},
            ]
        },
    )
    assert upserted.status_code == 200
    assert upserted.json()["upserted"] == 2

    found = client.post("/collections/docs/search", json={"vector": [1, 0, 0], "k": 1})
    assert found.status_code == 200
    results = found.json()["results"]
    assert results[0]["id"] == "a"

    listing = client.get("/collections")
    assert listing.json()["collections"][0]["name"] == "docs"


def test_text_endpoints_rank_semantically(client):
    dim = client.get("/version").json()["embedding_dim"]
    client.post("/collections", json={"name": "mem", "dimension": dim, "metric": "cosine"})

    docs = [
        "the cat sat on the mat",
        "dogs are loyal animals",
        "a feline rested on the rug",
    ]
    client.post(
        "/collections/mem/texts",
        json={"items": [{"text": d} for d in docs]},
    )

    response = client.post(
        "/collections/mem/search/text", json={"text": "a cat on the rug", "k": 3}
    )
    assert response.status_code == 200
    results = response.json()["results"]
    assert len(results) == 3
    ranked = [r["metadata"]["text"] for r in results]
    # The unrelated "dogs" document should rank last.
    assert ranked[-1] == "dogs are loyal animals"


def test_missing_collection_returns_404(client):
    assert client.get("/collections/nope").status_code == 404


def test_dimension_mismatch_returns_400(client):
    client.post("/collections", json={"name": "c", "dimension": 3})
    response = client.post("/collections/c/vectors", json={"items": [{"vector": [1, 0]}]})
    assert response.status_code == 400


def test_delete_vector(client):
    client.post("/collections", json={"name": "c", "dimension": 2})
    client.post("/collections/c/vectors", json={"items": [{"id": "x", "vector": [1, 0]}]})
    assert client.delete("/collections/c/vectors/x").status_code == 200
    assert client.get("/collections/c/vectors/x").status_code == 404


def test_api_key_protects_data_routes(tmp_path):
    settings = Settings(data_dir=str(tmp_path), autosave_interval=0.0, api_key="secret")
    with TestClient(create_app(settings)) as client:
        assert client.get("/health").status_code == 200  # public
        assert client.get("/collections").status_code == 401  # protected, no key
        ok = client.get("/collections", headers={"X-API-Key": "secret"})
        assert ok.status_code == 200
        bearer = client.get("/collections", headers={"Authorization": "Bearer secret"})
        assert bearer.status_code == 200
