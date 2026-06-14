"""Open synthetic-generation framework for building reference collections.

The *framework* is open (capability — it lets anyone build collections); the
*specific* tuned generator for a licensed criteria set (e.g. DRV-Prüfkatalog) is
commercial content built privately.

The core requirement it serves is the **correctness trap**: a reference
population is only valuable if its *distribution* (spread + cross-field
correlations) matches real valid data — not merely if each record passes the
criteria. Independent uniform/normal draws produce a "too clean" population that
flags valid real records as anomalies. This framework generates *correlated*
populations and ships diagnostics that catch degeneracy.
"""

from __future__ import annotations

from .generator import (
    CriteriaSpec,
    FieldSpec,
    calibrate,
    diagnose,
    generate,
    is_too_clean,
)

__all__ = [
    "FieldSpec",
    "CriteriaSpec",
    "generate",
    "diagnose",
    "is_too_clean",
    "calibrate",
]
