# Operations runbook

## Durability model

Writes are acknowledged from memory and persisted by:

1. **Autosave** — every `NEURODB_AUTOSAVE_INTERVAL` seconds (default 5s) when
   dirty.
2. **Explicit flush** — `POST /v1/flush` snapshots and `fsync`s synchronously;
   it returns only once the data is durably on disk. Call it before a planned
   restart or whenever you need a hard durability point.
3. **Graceful shutdown** — a final durable save on `SIGTERM`.

Each save writes a temp file, `fsync`s it, then atomically renames it over the
data file (and `fsync`s the directory on POSIX). A crash can lose at most the
writes since the last of the above events.

## Backup & restore

The data file is a single atomically-replaced `.npz`, so copying it while the
process is running yields a consistent snapshot.

```bash
# Backup (safe at any time; the rename is atomic)
cp /data/neurodb.npz /backups/neurodb-$(date +%F).npz

# Restore
docker compose stop
cp /backups/neurodb-2026-06-13.npz /data/neurodb.npz
docker compose start
curl -s localhost:8000/ready && curl -s -H "X-API-Key: $KEY" localhost:8000/v1/stats
```

## Corrupt-file recovery

On startup an unreadable or mismatched data file is **quarantined** to
`neurodb.npz.corrupt-<id>` (never deleted) and the store starts empty. Set
`NEURODB_FAIL_ON_CORRUPT_LOAD=1` to fail startup instead (so an orchestrator
surfaces the problem rather than silently serving an empty store). Investigate
the quarantined file before discarding it.

## Monitoring

Scrape `/metrics`. Useful signals:

- `neurodb_save_total{result="error"}` increasing — persistence is failing
  (disk full / permissions); `/ready` will also flip to 503.
- `neurodb_http_request_duration_seconds` — latency.
- `neurodb_patterns_total` / `neurodb_memories` — store growth.

Alert on `/ready` returning 503 and on any `save_total{result="error"}`.
