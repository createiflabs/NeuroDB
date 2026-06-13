# Deploying NeuroDB

NeuroDB is a **single-node, single-writer** service: all state lives in one
process's memory and is persisted to one data file. Scale reads horizontally;
never run more than one writer against the same data file.

## Quick start (Docker Compose)

```bash
cp .env.example .env        # set NEURODB_API_KEY, CORS origins, etc.
docker compose up -d
curl -s localhost:8000/health
```

The shipped `docker-compose.yml` runs read-only-rootfs, drops all capabilities,
sets `no-new-privileges`, and applies CPU/memory limits. It enables
`NEURODB_ALLOW_ANONYMOUS=1` for local convenience — **remove that and set
`NEURODB_API_KEY` for any networked deployment** (the server otherwise refuses
to start without a key).

## Single-worker constraint

The store is in-process NumPy state, so it **must run with exactly one worker**.
`python -m neurodb` pins `workers=1` and refuses to start if `WEB_CONCURRENCY>1`.
Do not put multiple workers/replicas behind one data file — they would diverge
and corrupt it.

### Scaling reads

Run N read-only replicas, each mounting a read-only copy/snapshot of the data
file and reloading on change; send all writes to the single primary.

## Health, readiness, metrics

- `GET /health` — liveness (container `HEALTHCHECK` / k8s livenessProbe).
- `GET /ready` — readiness; returns 503 until the last persist succeeded. Use it
  as the load-balancer / k8s readinessProbe.
- `GET /metrics` — Prometheus exposition (request counts/latency, memory/pattern
  gauges, save results).

## Graceful shutdown

On `SIGTERM` the lifespan handler cancels autosave and performs a final durable
save. Give the container enough `stop_grace_period` for that save to finish on
large stores (it scales with data size).

## Capacity planning

Resident memory ≈ `total_patterns × dimension × 4 bytes` (float32), plus a
transient bump during a save (a snapshot copy of the largest memory). Size
container memory limits with headroom above that.

## TLS

Terminate TLS at a reverse proxy / ingress in front of NeuroDB and forward
`X-Forwarded-For` only from trusted proxies.
