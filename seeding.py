# Copyright 2026
# SPDX-License-Identifier: Apache-2.0
"""Hybrid seeding — import Fortify / SARIF findings.

Off by default. When enabled, imports existing SAST findings (Fortify SARIF/FPR
export or generic SARIF) as **seed findings** so the LLM can triage them (cut FPs)
and focus on what static analysis missed. Import-only: reads a file the other tool
already produced; no network, no running external tools.

Returns a list of seed :class:`Finding` objects (source='seed'). The orchestrator
adds them to the findings pool and routes them through validate → trace like any
other finding (the LLM acts as a triage/gap-fill layer).
"""
from __future__ import annotations

import json
from pathlib import Path

from .findings import Finding

_SARIF_LEVEL_TO_SEV = {"error": "high", "warning": "medium", "note": "low", "none": "low"}


def _normalize_sarif_uri(uri: str) -> str:
    """SARIF ``artifactLocation.uri`` → the root-relative posix form the ledger uses.

    SARIF URIs are legitimately written as ``file:///abs/path``, percent-encoded,
    backslashed, or ``./``-prefixed. The raw value used to be stored as ``file_rel``, so
    it never matched ``discovery.rel_path`` — meaning an imported seed could NEVER dedupe
    against the LLM finding for the same bug, silently defeating the entire point of
    hybrid seeding.
    """
    import urllib.parse

    raw = str(uri or "").strip()
    if not raw:
        return ""
    if raw.startswith("file:"):
        parsed = urllib.parse.urlparse(raw)
        raw = urllib.parse.unquote(parsed.path or "")
    else:
        raw = urllib.parse.unquote(raw)
    raw = raw.replace("\\", "/")
    while raw.startswith("./"):
        raw = raw[2:]
    return raw.lstrip("/")


def load_seeds(settings) -> list[Finding]:
    """Load all configured seed sources into Finding objects (source='seed')."""
    seeds: list[Finding] = []
    s = settings.seeding
    if s.mode == "llm_only":
        return seeds
    if s.fortify_sarif:
        seeds.extend(_load_sarif(s.fortify_sarif, origin="fortify"))
    for path in s.generic_sarif:
        seeds.extend(_load_sarif(path, origin="sarif"))
    # SCA stays BlackDuck's job; we don't import dependency CVEs as code findings.
    if not seeds:
        print("⚠ seeding: hybrid/triage mode is enabled but 0 seeds were imported. "
              "Check seeding.fortify_sarif / seeding.generic_sarif paths.")
    return seeds


def _load_sarif(path: str, *, origin: str) -> list[Finding]:
    p = Path(path)
    if not p.exists():
        # SILENCE here meant a typo'd path produced zero seeds with no warning and no
        # audit event: a user who enabled hybrid seeding got none of their SAST findings
        # and no signal at all that the import had not happened.
        print(f"⚠ seeding: {origin} SARIF not found at {path} — 0 seeds imported.")
        return []
    try:
        doc = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"⚠ seeding: could not read {origin} SARIF {path} ({exc}) — 0 seeds.")
        return []
    if not isinstance(doc, dict):
        # A top-level array made `doc.get` raise AttributeError, which the
        # JSONDecodeError-only guard did not catch — aborting ALL seed import.
        print(f"⚠ seeding: {origin} SARIF {path} is not a JSON object — 0 seeds.")
        return []

    out: list[Finding] = []
    for run in doc.get("runs", []) or []:
        if not isinstance(run, dict):
            continue
        # Map ruleId → CWE if the driver provided it.
        rule_cwe: dict[str, str] = {}
        # `run.get("tool", {})` returned None for `"tool": null`, and the `or {}` was
        # applied one level too late to help.
        driver = ((run.get("tool") or {}).get("driver") or {})
        for rule in (driver.get("rules") or []):
            if not isinstance(rule, dict):
                continue
            props = rule.get("properties", {}) or {}
            cwe = props.get("cwe")
            if cwe:
                rule_cwe[rule.get("id", "")] = cwe
        for res in (run.get("results") or []):
            if not isinstance(res, dict):
                continue
            level = str(res.get("level", "warning"))
            msg = (res.get("message", {}) or {}).get("text", "")
            rid = res.get("ruleId", "")
            loc = _first_location(res)
            out.append(Finding(
                file_rel=_normalize_sarif_uri(loc[0]),
                line_start=loc[1],
                line_end=loc[2],
                category="seed",
                title=f"[{origin}:{rid}]"[:120],
                description=msg[:2000],
                severity=_SARIF_LEVEL_TO_SEV.get(level, "medium"),
                confidence=0.5,  # unverified until the LLM triages it
                cwe=rule_cwe.get(rid),
                cited_snippet="",
                source="seed",
                status="new",
            ))
    return out


def _first_location(result: dict) -> tuple[str, int, int]:
    locs = result.get("locations", [])
    if locs:
        phys = locs[0].get("physicalLocation", {}) or {}
        art = phys.get("artifactLocation", {}) or {}
        region = phys.get("region", {}) or {}
        uri = art.get("uri", "")
        start = int(region.get("startLine", 1) or 1)
        end = int(region.get("endLine", start) or start)
        return uri, start, end
    return "", 1, 1
