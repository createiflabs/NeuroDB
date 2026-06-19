# Operations runbook

## Durability model

Every write/delete is appended to a **write-ahead log** (`<data-file>.wal`) and
`fsync`'d **before the request returns**, so an acknowledged write survives a
`kill -9`: on boot the store loads the last `.npz` snapshot and replays the WAL,
recovering operations not yet snapshotted (regression:
`wal_replay_recovers_uncommitted_stores`).

The `.npz` snapshot is a periodic **compaction checkpoint** that folds the WAL
in and truncates it, taken by:

1. **Autosave** — every `NEURODB_AUTOSAVE_INTERVAL` seconds (default 5s) when
   dirty.
2. **Explicit flush** — `POST /v1/flush` snapshots and `fsync`s synchronously;
   it returns only once the data is durably on disk.
3. **Graceful shutdown** — a final durable save on `SIGTERM`.

Each snapshot rotates the WAL aside, writes a temp file, `fsync`s it, atomically
renames it over the data file (and `fsync`s the directory on POSIX), then
discards the rotated WAL segment — a crash at any step replays cleanly on boot
(idempotent replay). Set `NEURODB_WAL=0` to disable the log; durability then
falls back to snapshot-only (a crash loses writes since the last save above).

## Backup & restore

Use the built-in tooling — it takes the snapshot under the store lock (so matrix
rows always agree with ids/metadata) and validates a source before it ever
replaces live data.

```bash
# Backup (consistent point-in-time snapshot; dest may be a file or directory)
neurodb backup /backups/                       # → /backups/neurodb-backup-<ts>.npz
# or over HTTP (admin: same API key as the data API)
curl -s -X POST -H "X-API-Key: $KEY" localhost:8000/v1/backup
curl -s -X POST -H "X-API-Key: $KEY" "localhost:8000/v1/backup?download=true" -o snap.npz

# Restore (validates the source first; preserves the current file as *.pre-restore)
docker compose stop
neurodb restore /backups/neurodb-backup-<ts>.npz
docker compose start
curl -s localhost:8000/ready && curl -s -H "X-API-Key: $KEY" localhost:8000/v1/stats
```

Remote/off-box targets are deliberately your call — ship the snapshot with your
own tool (no cloud SDKs in core):

```bash
# Cron sidecar: snapshot every 15 min and push wherever you like.
*/15 * * * * neurodb backup /backups/ && aws s3 cp --recursive /backups s3://my-bucket/neurodb/
```

Restore is CLI-only (too dangerous to expose over HTTP). A corrupt or
newer-format source is refused **before** the live file is touched.

## Upgrades & data migration

The data file carries a manifest `version`. On startup an older file is migrated
forward in memory automatically and the upgraded version is stamped on the next
save; a file written by a *newer* NeuroDB is refused with an upgrade message
(restore an older backup or upgrade the binary). To migrate eagerly:

```bash
neurodb migrate        # loads, migrates, and rewrites at the current version
```

## Resource limits

To stop a runaway writer OOM-killing the process, set ceilings — writes past
them are rejected with `413` **before** allocating, while reads keep serving:

- `NEURODB_MAX_PATTERNS_PER_MEMORY` (default 1,000,000; `0` = unlimited)
- `NEURODB_MAX_TOTAL_BYTES` (default unset/unlimited) — estimated total footprint
- `NEURODB_MEMORY_PRESSURE_PCT` (default 90) — `/health` flips `memory_pressure`

`/v1/stats` reports `approx_bytes` and `pct_of_budget` per memory and in
aggregate. Note `zscore`/`l2` memories cost roughly **double** (raw matrix plus a
cached normalized matrix), and that cache counts against the byte budget.

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
- `neurodb_http_request_duration_seconds` — per-route latency.
- `neurodb_batch_size` — batch-request size distribution (spot misuse / tune
  `NEURODB_MAX_BATCH`).
- `neurodb_ops_total{op="write|update|delete"}` — mutation rate.
- `neurodb_patterns_total` / `neurodb_memories` — store growth.

For "why is this one request slow", inspect the **slowlog**: any data op slower
than `NEURODB_SLOWLOG_MS` (default 200ms) is recorded (timing + shape only, never
contents) and the recent ring buffer is at `GET /v1/slowlog` (admin-guarded).

Alert on `/ready` returning 503, on any `save_total{result="error"}`, and on
`/health` reporting `memory_pressure: true`.
