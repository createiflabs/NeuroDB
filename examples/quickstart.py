"""End-to-end NeuroDB demo using only the Python standard library.

Start the server first (``python -m neurodb`` or ``docker run ... neurodb``),
then run:  ``python examples/quickstart.py``

It demonstrates the Modern Hopfield primitives: append patterns, complete a
partial query, and detect a per-field anomaly.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

BASE = os.environ.get("NEURODB_URL", "http://localhost:8000")


def call(path: str, body: dict | None = None, method: str | None = None) -> dict:
    data = json.dumps(body).encode() if body is not None else None
    verb = method or ("POST" if body is not None else "GET")
    req = urllib.request.Request(
        f"{BASE}{path}", data=data, method=verb, headers={"content-type": "application/json"}
    )
    with urllib.request.urlopen(req) as resp:
        return json.load(resp)


def main() -> None:
    info = call("/version")
    print(f"Connected to NeuroDB {info['version']} | engine={info['engine']}\n")

    # A memory of structured records: [temperature, humidity, pressure].
    try:
        call("/memories/sensors", method="DELETE")
    except urllib.error.HTTPError:
        pass
    call(
        "/memories",
        {"name": "sensors", "dimension": 3, "beta": 12,
         "fields": ["temperature", "humidity", "pressure"]},
    )

    # Write some "normal" readings — writing is appending a vector.
    normals = [[20, 50, 1013], [21, 52, 1012], [19, 48, 1014], [20, 51, 1013]]
    call("/memories/sensors/patterns", {"items": [{"vector": v} for v in normals]})
    print(f"Appended {len(normals)} normal readings.\n")

    # Pattern completion: we only know the temperature (dimension 0).
    completed = call("/memories/sensors/complete", {"query": [20, 0, 0], "mask": [0]})
    recon = completed["reconstruction"]
    print("Pattern completion from temperature=20:")
    print(f"   -> humidity ~= {recon[1]:.1f}, pressure ~= {recon[2]:.1f}\n")

    # Per-field anomaly detection: humidity is wildly off.
    anomaly = call("/memories/sensors/anomaly", {"query": [20, 95, 1013]})
    print(f"Anomaly scan of [20, 95, 1013]  (overall score {anomaly['score']:.1f}):")
    for field in anomaly["fields"][:3]:
        print(
            f"   {field['name']:<12} value={field['value']:.0f} "
            f"expected={field['expected']:.0f} deviation={field['deviation']:.1f}"
        )


if __name__ == "__main__":
    main()
