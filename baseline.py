# Copyright 2026
# SPDX-License-Identifier: Apache-2.0
"""Baselines, suppressions & finding lifecycle.

* ``baseline.json`` — a snapshot of known finding fingerprints. In ``new_only``
  mode, findings already in the baseline are kept out of the CI-gating set (they're
  marked ``status='accepted_risk'``-equivalent for reporting "not new").
* ``suppressions.yaml`` — per-finding mutes with reason/owner/expiration; expired
  suppressions automatically re-activate.

Returns the reportable findings plus counts of suppressed / baselined items.

.. warning::
   Fingerprints changed when :meth:`findings.Finding.fingerprint` moved off the
   cited snippet and onto the enclosing symbol (it had to: quoting the same bug
   slightly differently produced a different fingerprint, so baselines broke on any
   reformat). **Any baseline.json or suppressions.yaml written before that change
   must be regenerated**, otherwise every entry silently stops matching and
   previously-baselined findings reappear as new. There is no automatic conversion —
   the old key simply cannot be recomputed.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import yaml

from .findings import Finding
from .scoring import passes_threshold
from .settings import Settings


@dataclass
class Suppression:
    fingerprint: str
    reason: str
    owner: str
    expires: float | None  # epoch seconds; None = never


# Bumped whenever Finding.fingerprint() changes shape. A baseline written under an
# older scheme cannot be matched or converted, so we warn instead of silently
# treating every previously-known finding as brand new.
FINGERPRINT_SCHEME = 2


def load_baseline(path: str | None) -> set[str]:
    if not path:
        return set()
    p = Path(path)
    if not p.exists():
        return set()
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"⚠ baseline {p} could not be read ({exc}); treating every finding as "
              f"new. Fix or regenerate it with `scan baseline`.")
        return set()
    if not isinstance(data, dict):
        # A bare list is the shape people hand-write. Say so instead of silently
        # treating a populated baseline as empty (which turns CI red on known findings).
        print(f"⚠ baseline {p} must be a JSON object with a 'fingerprints' list, got "
              f"{type(data).__name__}; ignoring it. Regenerate with `scan baseline`.")
        return set()
    scheme = data.get("fingerprint_scheme", 1)
    if scheme != FINGERPRINT_SCHEME:
        print(f"⚠ baseline {p} was written with fingerprint scheme v{scheme}, but this "
              f"build uses v{FINGERPRINT_SCHEME}. Its entries CANNOT match, so every "
              f"finding will look new. Regenerate the baseline from a current run.")
        return set()
    fps = data.get("fingerprints", [])
    if not isinstance(fps, list):
        print(f"⚠ baseline {p} has a non-list 'fingerprints'; ignoring it.")
        return set()
    return {str(x) for x in fps if isinstance(x, (str, int))}


def write_baseline(path: str, findings: list[Finding]) -> None:
    """Write a baseline from live Finding objects."""
    write_baseline_fingerprints(path, [f.fingerprint() for f in findings])


def write_baseline_fingerprints(path: str, fingerprints: list[str], *,
                                merge: bool = True) -> tuple[int, int]:
    """Write a baseline from bare fingerprint strings. Returns ``(added, retained)``.

    Split out so ``scan baseline`` can bootstrap from a previous run's report.json
    without needing to rebuild Finding objects. Both entry points share this one
    writer so the on-disk shape (and the scheme stamp that ``load_baseline``
    validates) can never drift between them.

    ``merge=True`` (the default) UNIONs with whatever is already on disk. This used to
    be a blind overwrite, which quietly destroyed the baseline on its second use:
    ``scan baseline`` harvests report.json, and report.json only contains findings that
    were NOT already baselined (that is the point of ``new_only``) — so re-running the
    documented "accept today's findings" workflow replaced {a,b,c} with {d}, and every
    previously-triaged finding came back as net-new and turned CI red.
    """
    existing = load_baseline(path) if merge else set()
    incoming = {str(x) for x in fingerprints}
    fps = sorted(existing | incoming)
    Path(path).write_text(
        json.dumps({"fingerprint_scheme": FINGERPRINT_SCHEME, "fingerprints": fps},
                   indent=2),
        encoding="utf-8")
    return len(incoming - existing), len(existing)


def load_suppressions(path: str | None) -> dict[str, Suppression]:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    try:
        data = yaml.safe_load(p.read_text(encoding="utf-8")) or []
    except (OSError, yaml.YAMLError) as exc:
        print(f"⚠ suppressions {p} could not be parsed ({exc}); no suppressions "
              f"applied.")
        return {}
    if isinstance(data, dict):
        # `suppressions:\n  - fingerprint: ...` is the most natural way to write this
        # file, and iterating a dict yields its KEYS (strings) — so `entry.get` raised
        # AttributeError, which is not a KeyError, escaped write_reports, and killed
        # the whole report with a traceback. Accept the nested shape.
        data = data.get("suppressions", data.get("entries", []))
    if not isinstance(data, list):
        print(f"⚠ suppressions {p} must be a list of entries (or a 'suppressions:' "
              f"key containing one); got {type(data).__name__}. Ignoring it.")
        return {}
    out: dict[str, Suppression] = {}
    for entry in data:
        if not isinstance(entry, dict):
            print(f"⚠ suppressions {p}: skipping non-mapping entry {entry!r}.")
            continue
        try:
            exp = entry.get("expires")
            exp_ts = _parse_date(exp) if exp else None
            out[str(entry["fingerprint"])] = Suppression(
                fingerprint=str(entry["fingerprint"]),
                reason=str(entry.get("reason", "")),
                owner=str(entry.get("owner", "")),
                expires=exp_ts,
            )
        except (KeyError, TypeError, AttributeError, ValueError) as exc:
            # A malformed entry must never take the report down with it.
            print(f"⚠ suppressions {p}: skipping unusable entry ({exc}).")
            continue
    return out


@dataclass
class GateResult:
    """The single gating decision for a run.

    ``reportable`` is the CI-gating set (exit code 1) and must stay exactly what
    ``report.json`` lists. ``informational`` is the reachability-demoted tier: it is
    reported for the record but never gates.
    """
    reportable: list[Finding]
    informational: list[Finding]
    suppressed: int
    baselined: int


def apply_baseline_and_suppressions(
    settings: Settings, findings: list[Finding]
) -> GateResult:
    """Split ``findings`` into (reportable, informational) + suppression counts.

    The ``informational`` tier exists because the reachability gate demotes findings it
    proves unreachable to ``severity='informational'`` and promises, in the evidence it
    attaches, that they "still appear in the report's informational section". They did
    not: ``informational`` ranks 0 and the lowest legal ``output.severity_threshold``
    ranks 1, so the threshold check below deleted every one of them from report.json,
    report.md and SARIF — while the console still printed "N demoted to informational"
    and the summary said "Informational: 0". Routing them here keeps that promise
    without letting code-hygiene items fail anyone's build.
    """
    baseline = load_baseline(settings.baseline.file)
    suppressions = load_suppressions(settings.suppressions.file)
    now = time.time()

    reportable: list[Finding] = []
    informational: list[Finding] = []
    suppressed = 0
    baselined = 0
    for f in findings:
        if f.status == "false_positive":
            continue
        if not passes_threshold(f, settings.output.severity_threshold):
            # 'informational' is not "below the bar" — it is the reachability gate's
            # deliberate verdict. Keep it for the report, out of the gating set. Still
            # honour an active suppression, so muting a fingerprint mutes it everywhere.
            if f.severity == "informational":
                sup = suppressions.get(f.fingerprint())
                if not (sup and (sup.expires is None or sup.expires > now)):
                    informational.append(f)
            continue
        fp = f.fingerprint()
        sup = suppressions.get(fp)
        if sup and (sup.expires is None or sup.expires > now):
            f.status = "suppressed"
            suppressed += 1
            continue
        if settings.baseline.mode == "new_only" and fp in baseline:
            baselined += 1
            continue  # not new → don't gate CI on it
        reportable.append(f)
    return GateResult(reportable=reportable, informational=informational,
                      suppressed=suppressed, baselined=baselined)


def _parse_date(value) -> float | None:
    """Accept epoch seconds or 'YYYY-MM-DD'."""
    if isinstance(value, (int, float)):
        return float(value)
    try:
        import datetime
        return datetime.datetime.strptime(str(value), "%Y-%m-%d").timestamp()
    except (ValueError, TypeError):
        return None
