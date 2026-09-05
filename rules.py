# Copyright 2026
# SPDX-License-Identifier: Apache-2.0
"""Deterministic local rules.

Pure-Python/regex checks that catch high-confidence issues regardless of model
behavior. Each match becomes a deterministic finding (CWE-tagged) and a ledger
anchor the LLM later enriches/validates. No network, no execution.

Built-in rule packs cover banned/dangerous APIs, weak crypto, unsafe
deserialization, insecure randomness, disabled cert verification, and hardcoded
debug flags. Org-specific anti-patterns can be loaded from a YAML file
(``rules.org_patterns``) so institutional knowledge marries the open-ended hunt.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class Rule:
    id: str
    description: str
    cwe: str
    severity: str          # low | medium | high | critical
    pattern: re.Pattern


@dataclass
class RuleHit:
    rule_id: str
    description: str
    cwe: str
    severity: str
    file_rel: str
    line: int
    snippet: str


# Built-in rules. Patterns are conservative to keep precision high (these become
# findings without LLM review). The LLM adds context; these provide the floor.
_BUILTIN: list[Rule] = [
    Rule("DANGEROUS_EVAL", "Use of eval/exec on dynamic input", "CWE-95", "high",
         re.compile(r"\b(?:eval|exec)\s*\(")),
    Rule("OS_SYSTEM", "Shell command execution", "CWE-78", "high",
         re.compile(r"\b(?:os\.system|subprocess\.(?:call|run|Popen)\s*\([^)]*shell\s*=\s*True"
                    r"|Runtime\.getRuntime\(\)\.exec|child_process\.exec)\b")),
    Rule("PICKLE_LOADS", "Unsafe deserialization via pickle", "CWE-502", "high",
         re.compile(r"\bpickle\.loads?\s*\(")),
    Rule("YAML_LOAD", "Unsafe yaml.load (use safe_load)", "CWE-502", "high",
         re.compile(r"\byaml\.load\s*\((?![^)]*Loader\s*=\s*yaml\.SafeLoader)")),
    Rule("WEAK_HASH", "Weak hash for security (MD5/SHA1)", "CWE-327", "medium",
         re.compile(r"\b(?:hashlib\.)?(?:md5|sha1)\s*\(")),
    Rule("WEAK_CIPHER", "Weak cipher mode (ECB/DES)", "CWE-327", "medium",
         re.compile(r"\b(?:DES|ARC4|ECB)\b")),
    Rule("INSECURE_RANDOM", "Insecure RNG for security context", "CWE-330", "low",
         re.compile(r"\b(?:random\.(?:random|randint|choice)|Math\.random)\s*\(")),
    Rule("TLS_VERIFY_OFF", "Disabled TLS/cert verification", "CWE-295", "high",
         re.compile(r"(?i)verify\s*=\s*False|rejectUnauthorized\s*:\s*false|"
                    r"CURLOPT_SSL_VERIFYPEER\s*,\s*0")),
    Rule("DEBUG_TRUE", "Debug mode enabled", "CWE-489", "low",
         re.compile(r"(?i)\bdebug\s*=\s*True\b|app\.debug\s*=\s*True")),
    Rule("HARDCODED_TMP", "Insecure hardcoded temp path", "CWE-377", "low",
         re.compile(r"['\"](?:/tmp/|/var/tmp/)[^'\"]*['\"]")),
]


def load_org_rules(path: str | Path | None) -> list[Rule]:
    """Load org-specific anti-patterns from YAML.

    Expected schema (list of dicts):
      - id: AUTH_BYPASS_X
        description: "legacy auth header trusted without validation"
        cwe: CWE-287
        severity: high
        pattern: "X-Internal-User\\s*:"
    """
    if not path:
        return []
    p = Path(path)
    if not p.exists():
        return []
    try:
        data = yaml.safe_load(p.read_text(encoding="utf-8")) or []
    except (OSError, yaml.YAMLError) as exc:
        print(f"⚠ org rules {p} could not be parsed ({exc}); no org rules loaded.")
        return []
    if isinstance(data, dict):
        # `rules:\n  - id: ...` is the most natural way to write this file, and
        # iterating a dict yields its KEYS — so `entry["id"]` raised TypeError, which
        # was NOT in the except clause below and NOT caught at the call site either,
        # aborting the whole scan with a traceback (and, per the exit-code contract,
        # exiting 1 as though vulnerabilities had been found).
        data = data.get("rules", data.get("org_rules", []))
    if not isinstance(data, list):
        print(f"⚠ org rules {p} must be a list of rules (or a 'rules:' key containing "
              f"one); got {type(data).__name__}. No org rules loaded.")
        return []
    rules: list[Rule] = []
    skipped = 0
    for entry in data:
        if not isinstance(entry, dict):
            skipped += 1
            continue
        try:
            pattern = entry["pattern"]
            if not isinstance(pattern, str):
                raise TypeError(f"pattern must be a string, got "
                                f"{type(pattern).__name__}")
            _reject_catastrophic(str(entry.get("id", "?")), pattern)
            rules.append(Rule(
                id=str(entry["id"]),
                description=str(entry.get("description", "")),
                cwe=str(entry.get("cwe", "CWE-Other")),
                severity=_normalize_severity(entry.get("severity")),
                pattern=re.compile(pattern),
            ))
        except (KeyError, TypeError, AttributeError, ValueError, re.error) as exc:
            print(f"⚠ org rules {p}: skipping rule {entry.get('id', '?')!r} ({exc}).")
            skipped += 1
            continue
    if skipped:
        print(f"⚠ org rules {p}: {skipped} of {len(data)} entries were unusable.")
    elif not rules and data:
        print(f"⚠ org rules {p} produced no usable rules.")
    return rules


# Nested quantifiers over an overlapping alphabet are the classic catastrophic
# -backtracking shape. run_rules applies every pattern to every line of every file with
# no timeout, so one such pattern is an unbounded hang in the deterministic pre-scan,
# before any LLM call, with no progress output: measured 45 s on a 31-CHARACTER line for
# `(a+)+$`. The ten built-in patterns are all linear; only the user-supplied ones need
# this guard.
_CATASTROPHIC_RE = re.compile(r"\([^()]*[+*]\)\s*[+*]")


def _reject_catastrophic(rule_id: str, pattern: str) -> None:
    if _CATASTROPHIC_RE.search(pattern):
        raise ValueError(
            "pattern has a nested quantifier (e.g. `(a+)+`), which can backtrack "
            "catastrophically and hang the whole pre-scan; rewrite it without the "
            "inner repetition")


_SEVERITY_ALIASES = {"info": "informational", "informational": "informational",
                     "low": "low", "minor": "low", "medium": "medium",
                     "moderate": "medium", "high": "high", "major": "high",
                     "severe": "high", "critical": "critical", "crit": "critical",
                     "blocker": "critical"}


def _normalize_severity(value) -> str:
    """Map an org-supplied severity onto the buckets the rest of the system knows.

    An unnormalized `severity: CRITICAL` was read as `medium` by the pre-enrich severity
    gate (`sev_rank.get("CRITICAL", 2)`) and only self-corrected after enrichment.
    """
    return _SEVERITY_ALIASES.get(str(value or "").strip().lower(), "medium")


def run_rules(files: list[tuple[str, str]], *, org_rules: list[Rule] | None = None) -> list[RuleHit]:
    """Apply all rules to ``(file_rel, text)`` tuples; return deterministic hits."""
    rules = _BUILTIN + (org_rules or [])
    hits: list[RuleHit] = []
    for file_rel, text in files:
        for lineno, line in enumerate(text.splitlines(), start=1):
            for rule in rules:
                if rule.pattern.search(line):
                    hits.append(RuleHit(
                        rule_id=rule.id, description=rule.description, cwe=rule.cwe,
                        severity=rule.severity, file_rel=file_rel, line=lineno,
                        snippet=line.strip()[:200],
                    ))
    return hits
