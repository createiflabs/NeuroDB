"""End-to-end test of the Python client against the in-process app.

The client's transport is pointed at FastAPI's TestClient so the full
create → write → recall → batch → capacity → update → delete flow runs without a
real socket.
"""

from __future__ import annotations

import pytest

from neurodb_client import BadRequest, Client, NotFound, connect


@pytest.fixture()
def db(client) -> Client:
    """A NeuroDB client whose transport drives the in-process TestClient."""

    def transport(method, url, headers, data, timeout):
        resp = client.request(method, url, content=data, headers=headers)
        return resp.status_code, resp.content

    # base_url empty → url passed to transport is exactly the request path.
    return connect("", transport=transport)


def test_connect_signature_is_friendly():
    c = connect("http://localhost:8000", api_key="k")
    assert c.base_url == "http://localhost:8000"
    assert c.api_key == "k"


def test_full_flow(db):
    mem = db.create(
        "sensors", dimension=3, beta=2.0,
        fields=["temperature", "humidity", "pressure"], normalize="zscore",
    )
    assert mem.info["normalize"] == "zscore"

    mem.write(
        [
            {"id": "0", "vector": [20, 50, 1013]},
            {"id": "1", "vector": [21, 52, 1012]},
            {"id": "2", "vector": [19, 48, 1014]},
            {"id": "3", "vector": [20, 51, 1013]},
        ]
    )

    # single anomaly
    a = mem.anomaly([20, 95, 1013])
    assert a["fields"][0]["name"] == "humidity"

    # batch anomaly with id echo
    batch = mem.anomaly_batch([{"id": "q", "vector": [20, 95, 1013]}, [21, 52, 1012]])
    assert batch["results"][0]["id"] == "q"
    assert batch["count"] == 2

    # completion
    c = mem.complete([20, 0, 0], mask=[0])
    assert c["reconstruction"][0] == pytest.approx(20.0, abs=1e-3)

    # capacity diagnostic
    cap = mem.capacity()
    assert cap["status"] in {"healthy", "crowded", "saturated"}

    # update + read back
    mem.update("0", metadata={"flag": True})
    assert mem.get("0")["metadata"]["flag"] is True

    # rebind to existing memory
    again = db.memory("sensors")
    assert again.info["count"] == 4

    db.delete("sensors")


def test_typed_errors(db):
    with pytest.raises(NotFound):
        db.memory("does-not-exist")

    db.create("m", dimension=2)
    with pytest.raises(NotFound):
        db.memory("m").get("missing-id")
    with pytest.raises(BadRequest):
        db.memory("m").write([{"vector": [1, 2, 3]}])  # wrong dimension → 400
