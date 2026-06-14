# Collections: format, tooling, and the open-core line

A **collection** is a portable, versioned, integrity-checked bundle that loads
into NeuroDB as a ready-to-score memory: the reference population ("what normal
looks like") plus its baseline, schema, and provenance. A user maps their table
columns to the collection's schema and starts scoring on day one — no
"bring-your-own clean baseline" step.

**The format is open. Specific collections are licensed content.** Anyone can
build and load collections with the open tooling below; a curated, attested
collection (e.g. DRV-Prüfkatalog) is a paid, signed artifact. The engine is free;
the validated dataset is the product.

## Bundle format (`.ndcoll`)

A zip with three members:

| member | contents |
|---|---|
| `collection.json` | manifest: `format_version`, name, dimension, beta, normalize, **schema**, **baseline**, **criteria**, **provenance**, optional **attestation** |
| `patterns.npy` | the reference matrix (engine float32 layout) |
| `signature.json` | optional Ed25519 signature over the other two members |

- **schema** — ordered fields (name/type/unit) + column→dimension map, so a
  collection is self-describing.
- **baseline** — precomputed zscore mean/std, recommended `beta`, per-field
  thresholds, and a build-time capacity report (so a buyer sees it's healthy).
- **provenance** — required (§ below). The loader rejects unprovenanced data.
- **attestation** — optional domain-partner sign-off, independently signed.
- `format_version` is migratable (mirrors the store's manifest migrations), so a
  bundle built today still loads after the format evolves.

## Tooling (all open)

```bash
neurodb collection info   <bundle>     # schema/criteria/provenance/attestation/capacity (no patterns)
neurodb collection verify <bundle>     # signature status
neurodb collection load   <bundle>     # materialize into the data file
```

HTTP: `POST /v1/collections/load` and `GET /v1/collections/{name}` (admin —
guarded by the data API key). Library: `neurodb.collections` (build/read/info/
verify/sign) and `neurodb.synthesis` (generation framework).

Try it with the bundled toy sample:

```bash
python examples/collections/build_sample.py
neurodb collection info examples/collections/sample_service_health.ndcoll
```

## Provenance (compliance + selling point)

Every collection carries a machine- and human-readable provenance block. For the
v1 **synthetic-first** position it states plainly that the data is generated, not
collected, and contains no personal data (GDPR Art. 4(1)) — the clean legal
basis. The loader **requires** a well-formed provenance block; the format also
supports an `anonymized`/`mixed` source with a documented methodology, but v1
ships synthetic only (don't ship anonymized-real without a legal opinion).

## The correctness trap (the core quality bar)

A collection is only valuable if its **distribution** — spread, per-field
variance, cross-field correlation — matches real valid data, not merely if each
record passes the criteria. Independent uniform/normal draws produce a "too
clean" population that flags valid real records as anomalies. The
`neurodb.synthesis` framework generates *correlated* populations and ships
diagnostics (`diagnose`, `is_too_clean`) that catch degeneracy; `calibrate` fits
marginals/correlations from a sample returning **parameters only** (no records
retained). The domain partner attests exactly this: *the population's
distribution is realistic for a valid population* — not just "records are valid."

> Note: the Hopfield capacity metric rewards *separable* data; a genuinely dense,
> realistic population can read "saturated" even though it is correct. Realism is
> judged by the synthesis diagnostics and the attestation scope, not by capacity
> alone.

## Signing & licensing seam

Bundles are signed with Ed25519 (pure-stdlib, shipped in the open core). The
loader verifies integrity (tamper-evidence) and authenticity (who built it); a
community user can self-sign. **Licensing is enforced at distribution, not by
crippling the loader**: the open core loads any validly-signed (or unsigned
community) bundle — you only hand the signed commercial bundle to paying
customers. The `CollectionLicense` seam (`neurodb.collections.license`) lets a
future commercial edition add load-time entitlement checks without forking; the
open default (`SignatureOnlyLicense`) verifies the signature only.

## Repo / licensing discipline (critical)

- **Open (this MIT repo):** the bundle format, loader, `info`, synthesis
  *framework*, provenance/attestation/signature *mechanisms*, verification, and a
  tiny clearly-labeled toy sample.
- **Commercial (NEVER in this repo):** the specific DRV-Prüfkatalog criteria
  spec, the tuned/calibrated generator parameters for it, and the built, signed,
  attested DRV bundle. Committing any of these here would relicense them
  irrevocably under MIT — they live in a private repo / artifact store and are
  distributed only to customers.

## Catalog-readiness (build for one, design for many)

v1 ships one collection (DRV-Prüfkatalog), but the format, loader, provenance,
attestation, and signing are **collection-agnostic** — collection #2 needs no
format change, only new content + a new attestation. The `criteria` namespace and
the stable, versioned bundle format are the only things required now to avoid a
future redesign; no DRV specifics are hard-coded into the format or loader.
