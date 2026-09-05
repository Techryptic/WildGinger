# Copyright 2026
# SPDX-License-Identifier: Apache-2.0
"""Finding model, dedup fingerprints & lifecycle.

A :class:`Finding` is the unit of output. It carries location, category/CWE,
severity, confidence, votes, hedge flag, reachability, lifecycle status, source
(llm/rule/seed), and a **stable fingerprint** used for dedup + baselines.

Identity design (this is the thing that controls duplicate noise):

  * :meth:`Finding.fingerprint` keys on ``file + category + cwe + enclosing symbol``.
    It deliberately does NOT include the cited snippet or raw line numbers: the same
    bug quoted with different whitespace, or shifted by an edit above it, must keep
    the same identity or dedupe/baselines/suppressions all break.
  * :meth:`Finding.location_key` is the coarser "same place, same attack class" key
    used to collapse the classic "20 findings on one endpoint" flood before the
    expensive judge panel runs.

``symbol`` is populated by whoever has the index (the hunt stage, and the rules
pass in the orchestrator). When it is unknown we fall back to a coarse line bucket,
which is less stable but still better than keying on quoted text.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, asdict
from typing import Any, Optional

LIFECYCLE = ("new", "confirmed", "accepted_risk", "false_positive", "fixed", "suppressed")

REACHABILITY = ("reachable", "unreachable", "unclear")

_HEDGE_RE_WORDS = ("possibly", "potentially", "could", "might", "may ", "in theory",
                   "perhaps", "seems", "appears to")

# Fallback line-bucket size used ONLY when a finding has no enclosing symbol.
# Wider than the old 10-line window so two reports of one module-level issue a few
# lines apart still collapse; symbol-based keys are preferred whenever available.
_LINE_BUCKET = 25


@dataclass
class Finding:
    file_rel: str
    line_start: int
    line_end: int
    category: str                  # attack class or 'novel/other'
    title: str
    description: str
    severity: str = "medium"       # low | medium | high | critical
    confidence: float = 0.5        # 0..1
    cwe: Optional[str] = None
    owasp: Optional[str] = None
    votes: int = 1
    hedged: bool = False
    reachable: str = "unclear"
    status: str = "new"
    source: str = "llm"            # llm | rule | seed
    remediation: Optional[str] = None
    cited_snippet: str = ""
    # Name of the enclosing function/method/class, resolved from the symbol index at
    # parse time. This is the STABLE part of a finding's identity — it survives
    # reformatting and line shifts, which raw line numbers and quoted text do not.
    # Empty when the location couldn't be tied to a symbol (module-level code).
    symbol: str = ""
    sarif_rule_id: Optional[str] = None
    evidence: list[str] = field(default_factory=list)
    # Deterministically-built malicious sample request for reachable endpoint
    # findings (method/path/params/curl/payload_note/derivation/citation). None when
    # the finding isn't inside a resolvable HTTP handler. See endpoints.py.
    sample_request: Optional[dict[str, Any]] = None
    # Per-judge verdicts from the validation panel: [{role, verdict, reachable,
    # evidence}, ...]. Populated by stages.validate and drained into the
    # `finding_evidence` table by the orchestrator once the finding has a row id.
    # Audit detail only — never part of the fingerprint.
    judge_verdicts: list[dict[str, str]] = field(default_factory=list)
    id: Optional[int] = None       # assigned when persisted
    # The model's ORIGINAL severity/confidence, latched by scoring.enrich() on its
    # first call. enrich() derives from these instead of from the current values, so
    # calling it twice (finalize, or a --resume round-trip through the DB) can't
    # decay a critical down to low. Not part of the fingerprint and not reported.
    base_severity: Optional[str] = None
    base_confidence: Optional[float] = None

    def __post_init__(self) -> None:
        # Every ``dedup_key`` this finding has already been PERSISTED under. Not a
        # dataclass field on purpose: ``asdict`` (report.json) must not see it, and a
        # set is not JSON-serializable.
        #
        # Why it exists: ``fingerprint()`` includes ``cwe``, and ``scoring.enrich``
        # fills a missing CWE from the category map at finalize. So the row written
        # during the pass (cwe=None) and the row written at finalize (cwe='CWE-89')
        # had DIFFERENT keys, and the UNIQUE(dedup_key) upsert inserted a SECOND row
        # instead of updating the first. On ``--resume`` both rows came back and the
        # same bug was reported twice, with its votes split. The same thing happens to
        # any finding that is later merged away by dedupe. Tracking the old keys lets
        # the persist step delete the rows it has superseded.
        self.superseded_keys: set[str] = set()


    # ── derived ────────────────────────────────────────────────────────────────
    def _location_token(self) -> str:
        """The stable location component of this finding's identity.

        Prefers the enclosing symbol name. Falls back to a coarse line bucket only
        when no symbol is known (module-level code, or an index miss) — a bucket is
        less stable across edits, but far more stable than quoted source text.
        """
        if self.symbol:
            return f"sym:{self.symbol}"
        return f"blk:{self.line_start // _LINE_BUCKET}"

    def fingerprint(self) -> str:
        """Stable identity: file + category + cwe + enclosing symbol.

        Excludes the cited snippet ON PURPOSE. Keying on quoted text meant the SAME
        bug reported with slightly different quoting produced a different fingerprint,
        so duplicates survived dedupe and baselines/suppressions broke on any edit
        that reformatted the line.
        """
        sig_basis = (f"{self.file_rel}|{self.category}|{self.cwe or ''}"
                     f"|{self._location_token()}")
        return hashlib.sha256(sig_basis.encode("utf-8")).hexdigest()[:16]

    def short_id(self) -> str:
        """Stable, human-friendly display id: ``F-`` + the first 8 hex chars of the
        full fingerprint.

        8, not 4. Four hex chars is a 16-bit space: measured against real fingerprints
        the first collision appears at finding **#196**, and 5000 findings produce 191
        colliding ids. That is not cosmetic — ``short_id`` is the PoC filename stem
        (``pocs/<id>__<slug>.md``), so two findings that collide on both id and slug
        (routine for rule findings, which share titles) silently OVERWRITE each other's
        PoC while ``generate_pocs`` reports both as written; it is also the id printed in
        report.md and report.json, and the prefix ``scan mark`` resolves — where a
        collision surfaces as "ambiguous id" for the id the report just told you to use.
        32 bits keeps ids readable while making a collision implausible at any realistic
        finding count.
        """
        return f"F-{self.fingerprint()[:8]}"

    def slug(self, *, max_len: int = 60) -> str:
        """Filesystem-safe slug of the title (for flat PoC filenames)."""
        import re as _re
        base = (self.title or self.category or "finding").lower()
        base = _re.sub(r"[^a-z0-9]+", "-", base).strip("-")
        return (base[:max_len].rstrip("-")) or "finding"

    def location_key(self) -> str:
        """Coarser identity that collapses the "20 findings on the same endpoint,
        same attack class" noise: file + category + enclosing symbol.

        Different attack classes at the same spot stay separate (SQLi vs XSS are
        genuinely different bugs). Keying on the SYMBOL rather than a fixed line
        bucket fixes the old boundary bug: with ``line_start // 10``, findings at
        lines 9 and 11 of the same function landed in different buckets and never
        merged. A function is the natural unit of "the same place".
        """
        basis = f"{self.file_rel}|{self.category}|{self._location_token()}"
        return hashlib.sha256(basis.encode("utf-8")).hexdigest()[:16]

    def dedup_key(self) -> str:
        return self.fingerprint()


    def to_dict(self) -> dict[str, Any]:
        """Serializable view for report.json.

        ``judge_verdicts`` is deliberately dropped: it's audit detail that lives in
        the ``finding_evidence`` table and the JSONL audit log, and inlining a blob of
        judge transcripts per finding would bloat the report. The human-readable
        summaries are already in ``evidence``.
        """
        d = asdict(self)
        d.pop("judge_verdicts", None)
        # Internal scoring baselines (see Finding.base_severity) — implementation
        # detail of enrich() idempotency, not part of the published schema.
        d.pop("base_severity", None)
        d.pop("base_confidence", None)
        d["fingerprint"] = self.fingerprint()
        return d


def resolve_symbol(index, file_rel: str, line: int) -> str:
    """Best-effort enclosing-symbol name for ``file_rel:line`` (``""`` if unknown).

    Kept here (rather than at each call site) so every producer of a Finding stamps
    identity the same way. ``index`` is duck-typed on ``enclosing_symbol`` so tests
    and the rules pass can pass a light stand-in. Never raises: an index miss simply
    yields ``""`` and the fingerprint falls back to a line bucket.
    """
    if index is None or not file_rel:
        return ""
    try:
        sym = index.enclosing_symbol(file_rel, int(line))
    except Exception:  # noqa: BLE001 — identity must never break a scan
        return ""
    return getattr(sym, "name", "") or ""


def detect_hedging(text: str) -> bool:
    """True if the text leans on speculative/hedge language."""
    low = text.lower()
    return any(w in low for w in _HEDGE_RE_WORDS)


# ── dedup + vote counting ───────────────────────────────────────

def _absorb(existing: Finding, other: Finding) -> None:
    """Fold ``other`` into ``existing`` in place (shared by both merge passes).

    Rules: votes accumulate; the strongest claim wins (severity, then confidence);
    reachability is optimistic (any judge saying "reachable" wins, because that's the
    dangerous case); evidence concatenates; and we never LOSE detail that only the
    absorbed copy had (snippet / symbol / cwe / remediation).
    """
    existing.votes += other.votes
    # The absorbed copy may already have a row on disk under its own key (findings are
    # persisted per-cell and after every pass). Inherit its identities so the persist
    # step can delete the rows this merge has superseded — otherwise they linger,
    # invisible to every report, until a --resume reloads them as duplicates.
    existing.superseded_keys |= other.superseded_keys | {other.fingerprint()}
    if _severity_rank(other.severity) > _severity_rank(existing.severity):
        existing.severity = other.severity
        # The stronger claim's wording describes the issue better.
        existing.title = other.title or existing.title
        existing.description = other.description or existing.description
    existing.confidence = max(existing.confidence, other.confidence)
    if other.reachable == "reachable":
        existing.reachable = "reachable"
    existing.evidence.extend(other.evidence)
    # Backfill anything the representative happens to be missing. Without this a
    # snippet-less representative could swallow the only copy that HAD a citation,
    # leaving a finding with no evidence to show in the report.
    if not existing.cited_snippet and other.cited_snippet:
        existing.cited_snippet = other.cited_snippet
    if not existing.symbol and other.symbol:
        existing.symbol = other.symbol
    if not existing.cwe and other.cwe:
        existing.cwe = other.cwe
    if not existing.remediation and other.remediation:
        existing.remediation = other.remediation


def merge_findings(findings: list[Finding]) -> list[Finding]:
    """Collapse findings sharing a fingerprint; accumulate votes; keep the
    highest-severity / highest-confidence representative."""
    by_key: dict[str, Finding] = {}
    for f in findings:
        key = f.fingerprint()
        if key not in by_key:
            by_key[key] = f
            continue
        _absorb(by_key[key], f)
    return list(by_key.values())


def merge_by_location(findings: list[Finding]) -> list[Finding]:
    """Aggressively collapse the "20 findings on the same endpoint, same attack
    class" noise using ``location_key`` (file + category + enclosing symbol).

    This runs BEFORE validation so we don't spend expensive judge calls on 20 near
    -identical variants of the same issue. Votes accumulate (more detections =
    higher confidence), and the strongest representative is kept.
    """
    by_key: dict[str, Finding] = {}
    for f in findings:
        key = f.location_key()
        if key not in by_key:
            by_key[key] = f
            continue
        _absorb(by_key[key], f)
    return list(by_key.values())


_SEV_ORDER = {"low": 1, "medium": 2, "high": 3, "critical": 4}


def _severity_rank(sev: str) -> int:
    return _SEV_ORDER.get(sev.lower(), 0)


def sort_canonical(findings: list[Finding]) -> list[Finding]:
    """Return ``findings`` in a stable, run-independent order.

    Both thread pools collect results via ``as_completed``, so the in-memory order
    depends on thread scheduling — two identical scans produced reports whose
    findings were shuffled, which makes diffing runs painful. Sorting by
    (file, line, category, fingerprint) makes output byte-comparable across runs.
    The trailing fingerprint is the tie-breaker that guarantees a TOTAL order.
    """
    return sorted(
        findings,
        key=lambda f: (f.file_rel, f.line_start, f.line_end, f.category,
                       f.fingerprint()),
    )
