# Copyright 2026
# SPDX-License-Identifier: Apache-2.0
"""Severity & confidence scoring rubric.

Scoring is *defined, not vibes*, so reports are consistent across models/runs:

  severity   = f(impact, reachability, exploitability, privilege_required)
  confidence = f(validator verdict, evidence quality, votes) − hedge penalty

Also assigns stable SARIF rule IDs (so dashboards/baselines stay consistent) and
maps categories to CWE/OWASP where not already set.
"""
from __future__ import annotations

from .findings import Finding

# Category → (CWE, OWASP) defaults when the model didn't supply them.
_CATEGORY_MAP: dict[str, tuple[str, str]] = {
    "injection": ("CWE-89", "A03:2021-Injection"),
    "xss": ("CWE-79", "A03:2021-Injection"),
    "authz": ("CWE-285", "A01:2021-Broken Access Control"),
    "authn": ("CWE-287", "A07:2021-Identification and Authentication Failures"),
    "crypto": ("CWE-327", "A02:2021-Cryptographic Failures"),
    "deserialization": ("CWE-502", "A08:2021-Software and Data Integrity Failures"),
    "ssrf": ("CWE-918", "A10:2021-Server-Side Request Forgery"),
    "path_traversal": ("CWE-22", "A01:2021-Broken Access Control"),
    "secrets": ("CWE-798", "A07:2021-Identification and Authentication Failures"),
    "logic": ("CWE-840", "A04:2021-Insecure Design"),
    "misconfig": ("CWE-16", "A05:2021-Security Misconfiguration"),
    "csrf": ("CWE-352", "A01:2021-Broken Access Control"),
}

_REACH_MULT = {"reachable": 1.0, "unclear": 0.7, "unreachable": 0.4}
# `informational` MUST be a bucket. Without it, _bucket_to_score fell through to its
# 0.4 default — the MEDIUM score, identical to what an unparseable string gets — so a
# model answering `"severity": "info"` on a trivial issue came out of enrich() as
# **medium** and could fail CI. Ordered high→low; the informational floor is below 0.0 so
# nothing can ever be promoted INTO it or out of it by the reachability multiplier.
_SEV_BUCKETS = (("critical", 0.85), ("high", 0.65), ("medium", 0.4), ("low", 0.05),
                ("informational", 0.0))


def enrich(finding: Finding, *, hedge_penalty: bool = True,
           language_aware_fp: bool = True, language: str = "") -> Finding:
    """Apply CWE/OWASP mapping, recompute severity+confidence, set SARIF id.

    **Idempotent.** This used to recompute severity *from the already-recomputed
    severity*, so every extra call multiplied by the reachability factor again:
    a critical/unclear finding decayed critical → high → medium → low. That fired
    for real on ``--resume`` (finalize enriches → persists the decayed value →
    the next run restores it → enriches again), silently dropping genuine
    criticals below the reporting threshold. We now remember the ORIGINAL
    model-assigned severity/confidence on the finding the first time through and
    always derive from that baseline, so N calls == 1 call.
    """
    # CWE/OWASP mapping.
    if finding.category in _CATEGORY_MAP:
        cwe, owasp = _CATEGORY_MAP[finding.category]
        finding.cwe = finding.cwe or cwe
        finding.owasp = finding.owasp or owasp

    # Latch the pre-enrichment inputs once, so re-enrichment is a pure function of
    # (base_severity, base_confidence, reachable, votes, hedged) rather than of how
    # many times enrich() has already run.
    if finding.base_severity is None:
        finding.base_severity = finding.severity
    if finding.base_confidence is None:
        finding.base_confidence = finding.confidence

    # Severity by reachability. An UNREACHABLE finding has no attacker-controlled
    # path to it — it isn't exploitable as-is, so it's code hygiene, not a live
    # vulnerability. We label it with the explicit 'informational' tier rather than
    # stamping a misleading medium/high on something we simultaneously call
    # "not reachable". Reachable/unclear findings keep the normal reachability-modulated
    # severity.
    if finding.reachable == "unreachable":
        finding.severity = "informational"
    elif str(finding.base_severity or "").lower() == "informational":
        # Already at the floor. The reachability multiplier can only ever REDUCE severity
        # (it is <= 1.0), so an explicitly informational finding must stay informational —
        # it must never be promoted into a tier that can gate CI.
        finding.severity = "informational"
    else:
        base = _bucket_to_score(finding.base_severity)
        score = base * _REACH_MULT.get(finding.reachable, 0.7)
        finding.severity = _score_to_bucket(score)

    # Confidence: votes raise it; hedging + (optionally) memory-unsafe langs lower it.
    conf = finding.base_confidence
    conf += min(0.3, 0.05 * max(0, finding.votes - 1))
    if hedge_penalty and finding.hedged:
        conf -= 0.2
    if language_aware_fp and language in ("c", "cpp"):
        conf -= 0.1  # higher FP base rate in memory-unsafe languages
    finding.confidence = max(0.0, min(1.0, conf))

    finding.sarif_rule_id = sarif_rule_id(finding)
    return finding


def sarif_rule_id(finding: Finding) -> str:
    """Stable SARIF rule id: CWE if known, else category."""
    if finding.cwe:
        return f"WildGinger-{finding.cwe.replace('CWE-', 'CWE')}"
    return f"WildGinger-{finding.category.upper().replace('/', '_')}"


def passes_threshold(finding: Finding, threshold: str) -> bool:
    # 'informational' (unreachable / code-hygiene) ranks 0 → below 'low', so it never
    # meets a real severity threshold and never gates CI.
    order = {"informational": 0, "low": 1, "medium": 2, "high": 3, "critical": 4}
    return order.get(finding.severity, 0) >= order.get(threshold, 2)


def _bucket_to_score(sev: str) -> float:
    for name, lo in _SEV_BUCKETS:
        if sev.lower() == name:
            # midpoint of the bucket for a stable starting score
            return min(1.0, lo + 0.1)
    # An UNKNOWN string is not a medium finding. Defaulting to 0.4 silently promoted
    # every severity the normalizer did not recognize (including "informational", which
    # was simply missing from the table) into the medium bucket.
    return 0.05


def _score_to_bucket(score: float) -> str:
    for name, lo in _SEV_BUCKETS:
        if score >= lo:
            return name
    return "low"
