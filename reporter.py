# Copyright 2026
# SPDX-License-Identifier: Apache-2.0
"""Reporting — JSON / SARIF / Markdown / executive summary.

Writes machine-readable + human-readable outputs to ``out_dir``. SARIF is built by
hand against the documented 2.1.0 schema (no hard dependency on sarif-om, which can
be swapped in later). Findings carry stable rule IDs so dashboards/baselines stay
consistent run-to-run.

Baselines/suppressions filtering is applied here before output.
"""
from __future__ import annotations

import json
from pathlib import Path

from .findings import Finding
from .settings import Settings

SARIF_VERSION = "2.1.0"
SARIF_SCHEMA = "https://json.schemastore.org/sarif-2.1.0.json"
_SEV_TO_SARIF = {"critical": "error", "high": "error", "medium": "warning",
                 "low": "note", "informational": "note"}


def write_reports(settings: Settings, findings: list[Finding], *,
                  coverage_pct: float, out_dir: Path,
                  run_stats: dict | None = None,
                  gate=None) -> dict:
    """Write all configured report formats. Returns a small summary dict.

    ``run_stats`` (from :mod:`runstats`) is rendered at the TOP of report.md and
    embedded in report.json so tokens/cost/runtime/cache/model-config travel with
    the results file, not just the console.

    ``gate`` is a :class:`baseline.GateResult`. Pass the one the caller already
    computed: the orchestrator needs the same split to decide the exit code, to pick
    which findings get a (paid) LLM-resolved sample request, and to pick which get a
    PoC file — and evaluating it twice meant report.json and the exit code were derived
    from two separate passes over baseline.json/suppressions.yaml. When it is None we
    compute it here, so standalone/test callers keep working.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Apply baseline/suppressions (lazy import; optional).
    if gate is None:
        from .baseline import apply_baseline_and_suppressions
        gate = apply_baseline_and_suppressions(settings, findings)
    reportable = gate.reportable
    informational = gate.informational
    suppressed, baselined = gate.suppressed, gate.baselined

    # The reachability-demoted tier is REPORTED but never gates CI, so it is appended
    # after the gating set rather than mixed into it.
    documented = reportable + informational

    formats = set(settings.output.formats)
    if "json" in formats:
        _write_json(out_dir / "report.json", reportable, coverage_pct, settings,
                    run_stats=run_stats, informational=informational,
                    suppressed=suppressed, baselined=baselined)
    if "sarif" in formats:
        _write_sarif(out_dir / "report.sarif", documented)
    if "markdown" in formats:
        _write_markdown(out_dir / "report.md", documented, coverage_pct,
                        run_stats=run_stats)
    # The executive summary is Markdown too, so it follows the same format gate —
    # writing it unconditionally meant `formats: [json]` still produced a stray .md.
    # ``exec_summary`` is accepted as an explicit opt-in for JSON/SARIF-only setups.
    if "markdown" in formats or "exec_summary" in formats:
        _write_exec_summary(out_dir / "executive_summary.md", documented, coverage_pct,
                            suppressed=suppressed, baselined=baselined,
                            run_stats=run_stats)

    return {"reportable": len(reportable), "informational": len(informational),
            "suppressed": suppressed, "baselined": baselined}


def _coverage_gap_note(coverage_pct: float) -> str:
    if coverage_pct >= 1.0:
        return "Coverage complete (100%)."
    return (f"⚠ COVERAGE INCOMPLETE ({coverage_pct*100:.1f}%). Some cells were not "
            f"analyzed to the thoroughness bar — see audit log for gaps.")


def _write_json(path: Path, findings: list[Finding], coverage_pct: float,
                settings: Settings, *, run_stats: dict | None = None,
                informational: list[Finding] | None = None,
                suppressed: int = 0, baselined: int = 0) -> None:
    informational = informational or []
    doc = {
        "tool": "WildGinger",
        "config_hash": settings.config_hash(),
        "prompts_version": settings.versioning.prompts_version,
        # End-of-run metrics (tokens/cost/runtime/cache + the model-config line) so
        # they travel with the machine-readable results, not just the console.
        "run_stats": run_stats or {},
        "coverage_pct": round(coverage_pct, 4),
        "coverage_complete": coverage_pct >= 1.0,
        "finding_count": len(findings),
        # Counts of what was deliberately EXCLUDED from `findings`. Without these a
        # consumer of report.json cannot tell "clean" from "everything was suppressed".
        "suppressed_count": suppressed,
        "baselined_count": baselined,
        # Each finding carries a short display 'id' (F-<4hex>, stable run-to-run)
        # that matches its flat pocs/<id>__<slug>.md file, plus the full
        # 'fingerprint' used internally for dedupe/baseline/suppression.
        "findings": [{**f.to_dict(), "id": f.short_id()} for f in findings],
        # Reachability-demoted, code-hygiene items. Deliberately a SEPARATE key: they
        # are part of the record but must never be counted in `finding_count` or gate
        # CI, and mixing them into `findings` would do both.
        "informational_count": len(informational),
        "informational": [{**f.to_dict(), "id": f.short_id()} for f in informational],
    }
    path.write_text(json.dumps(doc, indent=2), encoding="utf-8")


def _write_sarif(path: Path, findings: list[Finding]) -> None:
    rules_seen: dict[str, dict] = {}
    results = []
    for f in findings:
        rid = f.sarif_rule_id or f"WildGinger-{f.category.upper()}"
        if rid not in rules_seen:
            rules_seen[rid] = {
                "id": rid,
                "name": f.category,
                "shortDescription": {"text": f.title or f.category},
                "properties": {"cwe": f.cwe, "owasp": f.owasp},
            }
        results.append({
            "ruleId": rid,
            "level": _SEV_TO_SARIF.get(f.severity, "warning"),
            "message": {"text": f"{f.title}: {f.description}"[:2000]},
            "locations": [{
                "physicalLocation": {
                    "artifactLocation": {"uri": f.file_rel},
                    "region": {"startLine": max(1, f.line_start),
                               "endLine": max(1, f.line_end)},
                }
            }],
            "properties": {
                "severity": f.severity, "confidence": round(f.confidence, 2),
                "votes": f.votes, "reachable": f.reachable, "hedged": f.hedged,
                "source": f.source, "status": f.status,
                **({"sample_request": f.sample_request} if getattr(f, "sample_request", None) else {}),
            },
        })

    sarif = {
        "$schema": SARIF_SCHEMA,
        "version": SARIF_VERSION,
        "runs": [{
            "tool": {"driver": {
                "name": "WildGinger",
                "rules": list(rules_seen.values()),
            }},
            "results": results,
        }],
    }
    path.write_text(json.dumps(sarif, indent=2), encoding="utf-8")


def _write_markdown(path: Path, findings: list[Finding], coverage_pct: float,
                    *, run_stats: dict | None = None) -> None:
    lines = ["# Vulnerability Scan Report", ""]
    if run_stats:
        from .runstats import markdown_block
        lines += markdown_block(run_stats)
    lines += [_coverage_gap_note(coverage_pct), ""]
    # Split externally-reachable (actionable) findings from informational ones a
    # user of the product could never reach (e.g. a server-only .properties secret).
    reachable = [f for f in findings if f.reachable != "unreachable"]
    informational = [f for f in findings if f.reachable == "unreachable"]
    lines += ["## Reachable findings", ""]
    _render_findings(lines, reachable)
    if informational:
        lines += ["", "---", "",
                  "## Informational — not externally reachable (code hygiene, not "
                  "exploitable as-is)",
                  "",
                  "_These were flagged in code but judged **not reachable** from any "
                  "external entry point (no attacker-controlled path reaches them — "
                  "e.g. dead/internal code or server-only config). They are code "
                  "hygiene, **not live vulnerabilities**, and are excluded from the "
                  "severity counts and CI gating. Worth cleaning up; not urgent._", ""]
        _render_findings(lines, informational)
    path.write_text("\n".join(lines), encoding="utf-8")


def _render_findings(lines: list[str], findings: list[Finding]) -> None:
    order = {"critical": 0, "high": 1, "medium": 2, "low": 3, "informational": 4}
    if not findings:
        lines.append("_None._")
        return
    # Severity first (most urgent at the top), then the canonical location order so
    # equal-severity findings don't shuffle between otherwise-identical runs.
    for f in sorted(findings, key=lambda x: (order.get(x.severity, 9), x.file_rel,
                                            x.line_start, x.category,
                                            x.fingerprint())):
        lines += [
            f"### `{f.short_id()}` — [{f.severity.upper()}] {f.title or f.category}",
            f"- **PoC:** `pocs/{f.short_id()}__{f.slug()}.md`",
            f"- **Location:** `{f.file_rel}:{f.line_start}-{f.line_end}`",
            f"- **Category:** {f.category}  |  **CWE:** {f.cwe or 'n/a'}  "
            f"|  **OWASP:** {f.owasp or 'n/a'}",
            f"- **Confidence:** {f.confidence:.2f}  |  **Votes:** {f.votes}  "
            f"|  **Reachable:** {f.reachable}  |  **Source:** {f.source}",
            "",
            f.description or "",
            "",
        ]
        if f.cited_snippet:
            lines += ["```", f.cited_snippet, "```", ""]
        sr = getattr(f, "sample_request", None)
        if sr and sr.get("curl"):
            note = sr.get("payload_note", "")
            deriv = sr.get("derivation", "")
            cite = sr.get("citation", "")
            meta = " · ".join(x for x in [note, f"derivation: {deriv}" if deriv else "",
                                          f"from {cite}" if cite else ""] if x)
            lines += [
                f"**Sample attack request** ({sr.get('method','')} {sr.get('path','')}):",
                "```bash",
                sr["curl"],
                "```",
            ]
            if meta:
                lines += [f"_{meta}_", ""]
            else:
                lines += [""]
        if f.remediation:
            lines += [f"**Remediation:** {f.remediation}", ""]




def _write_exec_summary(path: Path, findings: list[Finding], coverage_pct: float,
                        *, suppressed: int, baselined: int,
                        run_stats: dict | None = None) -> None:
    counts = {"critical": 0, "high": 0, "medium": 0, "low": 0, "informational": 0}
    for f in findings:
        counts[f.severity] = counts.get(f.severity, 0) + 1
    reachable = sum(1 for f in findings if f.reachable == "reachable")
    # Exploitable = everything except the informational (unreachable) code-hygiene items.
    exploitable = len(findings) - counts["informational"]
    lines = [
        "# Executive Summary", "",
        _coverage_gap_note(coverage_pct), "",
        f"- **Total reportable findings:** {len(findings)} "
        f"({exploitable} exploitable + {counts['informational']} informational)",
        f"- Critical: {counts['critical']} | High: {counts['high']} "
        f"| Medium: {counts['medium']} | Low: {counts['low']} "
        f"| Informational: {counts['informational']}",
        f"- **Reachable (attacker-exposed):** {reachable}",
        f"- Suppressed: {suppressed} | In baseline (not new): {baselined}",
        "",
    ]
    # Cost/runtime/model config belong here too: this is the file a reviewer or
    # manager actually opens, and "what did this run cost and with which models" is
    # the first question asked. It previously only existed in report.md/json.
    if run_stats:
        from .runstats import markdown_block
        lines += markdown_block(run_stats)
    lines += [
        "Findings are model judgments verified by adversarial review + reachability "
        "tracing + citation checks. Confirm via the generated human-run PoCs before "
        "remediation (no code was executed by the scanner).",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
