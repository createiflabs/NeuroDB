def test_health_and_version(client):
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"
    version = client.get("/version").json()
    assert version["engine"] == "modern-hopfield"


def test_write_search_and_complete(client):
    created = client.post("/memories", json={"name": "m", "dimension": 3, "beta": 30})
    assert created.status_code == 201, created.text
    assert created.json()["beta"] == 30

    written = client.post(
        "/memories/m/patterns",
        json={"items": [{"id": "a", "vector": [1, 0, 0]}, {"id": "b", "vector": [0, 1, 0]}]},
    )
    assert written.json()["written"] == 2

    search = client.post("/memories/m/search", json={"query": [1, 0, 0], "k": 1})
    assert search.json()["results"][0]["id"] == "a"

    complete = client.post("/memories/m/complete", json={"query": [0.8, 0.2, 0]})
    body = complete.json()
    assert body["top"]["id"] == "a"
    assert len(body["reconstruction"]) == 3


def test_anomaly_endpoint_names_field(client):
    client.post(
        "/memories",
        json={"name": "r", "dimension": 3, "beta": 30, "fields": ["age", "income", "score"]},
    )
    client.post(
        "/memories/r/patterns",
        json={"items": [{"vector": [1, 1, 1]}, {"vector": [1, 1, 1]}]},
    )
    anomaly = client.post("/memories/r/anomaly", json={"query": [1, 5, 1]}).json()
    assert anomaly["fields"][0]["name"] == "income"
    assert anomaly["score"] > 0


def test_text_endpoints_recall(client):
    dim = client.get("/version").json()["embedding_dim"]
    client.post("/memories", json={"name": "mem", "dimension": dim, "beta": 8})
    docs = [
        "golden retrievers are friendly loyal dogs",
        "python is a popular programming language",
        "the eiffel tower stands in paris france",
    ]
    client.post("/memories/mem/texts", json={"items": [{"text": d} for d in docs]})

    search = client.post("/memories/mem/search/text", json={"text": "friendly dog", "k": 3}).json()
    assert search["results"][0]["metadata"]["text"] == "golden retrievers are friendly loyal dogs"

    recall = client.post("/memories/mem/recall/text", json={"text": "friendly dog", "k": 3}).json()
    assert recall["top"]["metadata"]["text"] == "golden retrievers are friendly loyal dogs"
    assert 0.0 <= recall["top"]["weight"] <= 1.0


def test_empty_memory_recall_returns_200(client):
    # A fresh, empty memory must not error on recall/complete/anomaly — the
    # dashboard's first "Recall" should show an empty result, not a 400.
    client.post("/memories", json={"name": "m", "dimension": 3})
    complete = client.post("/memories/m/complete", json={"query": [1, 0, 0]})
    assert complete.status_code == 200
    assert complete.json()["weights"] == []
    anomaly = client.post("/memories/m/anomaly", json={"query": [1, 0, 0]})
    assert anomaly.status_code == 200
    assert anomaly.json()["fields"] == []


def test_bad_filter_operator_returns_400(client):
    client.post("/memories", json={"name": "m", "dimension": 2})
    client.post("/memories/m/patterns", json={"items": [{"vector": [1, 0], "metadata": {"p": 1}}]})
    resp = client.post("/memories/m/search", json={"query": [1, 0], "filter": {"p": {"$in": "x"}}})
    assert resp.status_code == 400


def test_missing_memory_returns_404(client):
    assert client.get("/memories/nope").status_code == 404


def test_dimension_mismatch_returns_400(client):
    client.post("/memories", json={"name": "m", "dimension": 3})
    response = client.post("/memories/m/patterns", json={"items": [{"vector": [1, 0]}]})
    assert response.status_code == 400


def test_api_key_protects_data_routes(auth_client):
    assert auth_client.get("/health").status_code == 200
    assert auth_client.get("/memories").status_code == 401
    assert auth_client.get("/memories", headers={"X-API-Key": "secret"}).status_code == 200
    bearer = auth_client.get("/memories", headers={"Authorization": "Bearer secret"})
    assert bearer.status_code == 200
