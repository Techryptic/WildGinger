# Copyright 2026
# SPDX-License-Identifier: Apache-2.0
"""Stage prompt templates.

Each stage prompt is narrow and ends with a strict structured-output contract so
the harness can parse it deterministically (and verify it). Findings are emitted
as a JSON array inside <findings>...</findings>; verdicts inside <verdict>...; etc.
"""
from __future__ import annotations

# ── Recon ────────────────────────────────────────────────────────────────────
RECON_PROMPT = """\
You are a reconnaissance agent. Explore the repository using your read-only tools
(list_files, read_file, search_code, find_definition, find_callers) and produce a
partition of the attack surface for parallel vulnerability hunters.

Goal: identify 5-15 distinct, independent **focus areas** (subsystems that process
untrusted input or perform sensitive operations) so parallel hunters don't
converge on the same code.

Known attack surface hints (from a deterministic pre-scan):
{attack_surface_hint}

When done, emit EXACTLY one block:

<focus_areas>
[
  {{"name": "<short name>", "scope": "<file/dir or symbol prefix>", "why": "<entry point / sink / trust boundary>"}}
]
</focus_areas>

Emit the block once. Make each entry self-contained enough to hand to a hunter.
"""

# ── Hunt: directed ─────────────────────────────────────────────────────────────
# One-line "what to look for" per attack class, rendered into HUNT_DIRECTED_PROMPT
# via {class_guidance}. The lens text is weighted by real-world bug-bounty
# frequency/severity (analysis of 11k+ disclosed findings): the tainted-flow family
# carries the severity, authz the impact, information disclosure the volume — so the
# high-value patterns are named explicitly instead of implied by the class name.
CLASS_LENSES: dict[str, str] = {
    "injection": "SQL/OS-command/template/code injection — unsanitized input reaching an interpreter",
    "path_traversal": "file reads/writes outside the intended root (../, absolute paths, archive slip)",
    "ssrf": "server-side requests to attacker-influenced URLs; also open redirects (unvalidated Location)",
    "xss": "unescaped user input rendered into HTML/JS/SVG (stored, reflected, DOM)",
    "deserialization": "pickle/yaml.load/unmarshal of untrusted data",
    "authn": "session/login/token forgery, weak reset flows, missing rate limits on auth endpoints",
    "authz": "missing or mis-scoped authorization on objects/actions (IDOR, privilege escalation)",
    "csrf": "state-changing routes without anti-CSRF token / SameSite protection",
    "crypto": "weak algorithms/modes, hardcoded keys, insecure random for security purposes",
    "logic": "broken business invariants — races, negative amounts, price confusion; unbounded resource use (ReDoS, unlimited pagination)",
    "misconfig": "insecure defaults, debug endpoints, permissive CORS, verbose errors",
    "secrets": "hardcoded credentials; ALSO info disclosure — stack traces, exposed .git/env, PII in responses",
}


def class_guidance(attack_class: str) -> str:
    """Render the per-class lens lines for HUNT_DIRECTED_PROMPT's ``{class_guidance}``.

    A cell's attack_class may be a comma-joined GROUP label (attack_class_groups);
    expand every class in it. Unknown classes get a bare line — never silently dropped.
    """
    lines = []
    for cls in (c.strip() for c in str(attack_class or "").split(",")):
        if not cls:
            continue
        lens = CLASS_LENSES.get(cls)
        lines.append(f"  - {cls}: {lens}" if lens else f"  - {cls}")
    return "\n".join(lines) or "  - (any real vulnerability)"


HUNT_DIRECTED_PROMPT = """\
You are a security vulnerability hunter. Investigate the following narrow task
THOROUGHLY using your read-only tools. Follow calls across files (find_definition /
find_callers) — do not stop at one file.

Attack classes: **{attack_class}**
Focus area: {focus_area}
Primary file: {file_rel} (lines {start_line}-{end_line})

Method:
1. Read the primary code with read_file. Then follow every relevant call/definition.
2. Look specifically for these issues, but report ANY real vulnerability:
{class_guidance}
3. For each finding, CITE the exact file + line range and quote the vulnerable line
   verbatim (we verify citations against the source — fabricated citations are rejected).
4. Quality bar: prefer reachable, exploitable issues. If you find only a weak/low
   signal, KEEP LOOKING before concluding.

When done, emit EXACTLY one block (empty array if genuinely nothing found):

<findings>
[
  {{"file": "{file_rel}", "line_start": 0, "line_end": 0,
    "category": "<the class from the list above that best fits this finding>",
    "title": "...", "description": "...", "severity": "low|medium|high|critical",
    "confidence": 0.0, "cwe": "CWE-XXX", "cited_snippet": "<verbatim line(s)>",
    "remediation": "..."}}
]
</findings>

If you genuinely find nothing after thorough analysis, emit <findings>[]</findings>.
"""

# ── Hunt: open-ended ───────────────────────────────────────────────────────────
HUNT_OPEN_PROMPT = """\
You are a senior security researcher doing OPEN-ENDED review. You are NOT limited
to a checklist — find anything suspicious: business-logic flaws, auth bypasses,
broken invariants, race conditions, novel or chained issues. Reason like an
attacker and follow code across files with your read-only tools.

Focus area: {focus_area}
Primary file: {file_rel} (lines {start_line}-{end_line})

Cite exact file + line range and quote vulnerable lines verbatim (citations are
verified against source). Use category "novel/other" if it fits no standard class.

When done, emit EXACTLY one block (empty array if genuinely nothing found):

<findings>
[
  {{"file": "{file_rel}", "line_start": 0, "line_end": 0, "category": "novel/other",
    "title": "...", "description": "...", "severity": "low|medium|high|critical",
    "confidence": 0.0, "cwe": "CWE-XXX", "cited_snippet": "<verbatim line(s)>",
    "remediation": "..."}}
]
</findings>
"""

# ── Validate (adversarial; DIFFERENT model; blind to hunter reasoning) ──────────
VALIDATE_PROMPT = """\
You are an adversarial validator. A hunter (whose reasoning you CANNOT see) claims
the following vulnerability. Your job is to DISPROVE it. Investigate the cited code
independently with your read-only tools. You may NOT invent new findings.

Claimed finding:
  file: {file_rel}
  lines: {line_start}-{line_end}
  category: {category}
  title: {title}
  description: {description}
  cited snippet: {cited_snippet}

Decide:
- CONFIRMED  — the vulnerability is real and the citation is accurate.
- REJECTED   — it is a false positive (explain why: sanitized input, unreachable,
               misread code, etc.).
- UNCERTAIN  — cannot determine with available context.

Emit EXACTLY one block:

<verdict>
{{"verdict": "CONFIRMED|REJECTED|UNCERTAIN", "reachable": "reachable|unreachable|unclear",
  "evidence": "<your independent evidence, citing lines>"}}
</verdict>
"""

# ── Semantic dedupe (group same-root-cause findings BEFORE the judges) ──────────
DEDUPE_PROMPT = """\
You are a triage assistant removing DUPLICATE security findings before expensive
judging. Below is a numbered list of candidate findings (id, file, lines, category,
title). Group findings that describe the SAME underlying vulnerability / root cause —
even when they are phrased differently, span adjacent lines, or appear in different
files along the same call chain (e.g. a sink reported once at the handler and once at
the data layer).

Rules:
- Only group findings that are genuinely the SAME issue. Different attack classes at
  the same location (e.g. SQL injection vs. XSS) are NOT duplicates.
- A finding that is unique must appear in its own group of size 1.
- Every id must appear in EXACTLY one group.

Candidate findings:
{items}

Emit EXACTLY one block — a JSON array of groups, each group an array of ids that are
duplicates of each other (keep the clearest/most-severe id FIRST in each group):

<groups>
[[0,3],[1],[2,4,5]]
</groups>
"""

# ── PoC route resolution (LLM fallback; VERIFIED against source) ────────────────
ROUTE_RESOLVE_PROMPT = """\
You are helping build a reproduction request for a confirmed security finding so a
red teamer can test it quickly. Find the HTTP entry point (route) that reaches the
vulnerable code, using your read-only tools (read_file, search_code, find_definition,
find_callers). Trace the call chain UP from the finding to the request handler.

Finding:
  file: {file_rel}
  lines: {line_start}-{line_end}
  category: {category}
  title: {title}
  cited snippet: {cited_snippet}

Known entry-point/source hints (from a deterministic pre-scan):
{source_hints}

CRITICAL RULES:
- Report the ACTUAL route as written in the code. Do NOT invent or guess a path.
- You MUST cite the exact `file:line` where the route/decorator is defined
  (e.g. the line with `@app.route("/x")` / `@GetMapping(...)` / `app.post("/x", ...)`).
- If you cannot find a real route in the code, set "path" to null. Guessing is worse
  than admitting you couldn't resolve it.

Emit EXACTLY one block:

<route>
{{"method": "GET|POST|PUT|DELETE|PATCH", "path": "/actual/route/from/code",
  "route_citation": "app.py:41",
  "headers": ["Cookie: session=<...>"],
  "body": "field=value (or empty string)",
  "payload_note": "<how the malicious input is delivered>",
  "confidence": "high|medium|low"}}
</route>

If no route can be found, emit: <route>{{"path": null}}</route>
"""

# ── Trace (reachability; separate question) ─────────────────────────────────────
TRACE_PROMPT = """\
You are a reachability analyst. For the confirmed finding below, determine whether
attacker-controlled input can actually REACH the vulnerable code from an external
entry point. Use find_callers / find_definition to follow the call chain back to a
source (HTTP route, CLI, deserialization, message handler, etc.).

Finding:
  file: {file_rel}
  lines: {line_start}-{line_end}
  title: {title}

Known entry-point/source hints:
{source_hints}

Emit EXACTLY one block:

<reachability>
{{"reachable": "reachable|unreachable|unclear", "path": "<entrypoint -> ... -> sink>",
  "evidence": "<citations>"}}
</reachability>
"""
