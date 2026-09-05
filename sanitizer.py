# Copyright 2026
# SPDX-License-Identifier: Apache-2.0
"""Sanitization — protect sensitive data before it leaves the host.

Runs on every tool result before it returns to the model. **Option B redaction:**
the real secret value is NEVER stored — we replace it with a length-stable
placeholder and record only ``placeholder + file + line`` (the value can be
re-read from the local file for reporting if ever needed).

Detectors: regex rule packs (API keys, AWS keys, private keys, tokens, passwords,
connection strings, emails) + Shannon-entropy for high-entropy strings.

Policy modes:
  * ``redact`` (default) — replace matches with placeholders.
  * ``block``  — if a chunk still trips a detector, refuse to return its content.
  * ``off``    — passthrough (approved/air-gapped repos only).

The sanitizer also exposes :func:`scan_secrets` so a deterministic local finding
can be emitted independently of the LLM.
"""
from __future__ import annotations

import hashlib
import math
import re
from re import error
from dataclasses import dataclass, field

# (label, pattern). group(0) is redacted. Patterns favor recall; entropy backs them.
_SECRET_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("aws_access_key", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")),
    ("aws_secret_key", re.compile(r"\b(?<![A-Za-z0-9/+=])[A-Za-z0-9/+=]{40}(?![A-Za-z0-9/+=])\b")),
    ("private_key", re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA |PGP )?PRIVATE KEY-----")),
    ("generic_api_key", re.compile(r"\b(?:sk|pk|rk)_(?:live|test)_[A-Za-z0-9]{16,}\b")),
    ("bearer_token", re.compile(r"\b[Bb]earer\s+[A-Za-z0-9\-._~+/]{20,}=*")),
    ("jwt", re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b")),
    ("password_assign", re.compile(
        r"(?i)\b(?:password|passwd|pwd|secret|token|api[_-]?key)\b\s*[:=]\s*"
        r"['\"]([^'\"]{6,})['\"]")),
    ("connection_string", re.compile(
        r"\b(?:mongodb|postgres(?:ql)?|mysql|redis|amqp)://[^\s'\"]+")),
    ("email", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")),
]

# Tokens long enough to entropy-check.
_TOKEN_RE = re.compile(r"[A-Za-z0-9/+=_\-]{20,}")

# Git object ids and common content hashes are hex-only and fixed-length. The
# `aws_secret_key` rule ("any 40 chars of base64 alphabet") matched every single
# git SHA-1 in a repo, redacting commit ids as if they were AWS credentials — noisy
# and actively misleading, since the placeholder was labelled aws_secret_key.
# Hex-only strings at hash lengths carry no secret by themselves.
_HEX_HASH_RE = re.compile(r"(?i)\A[0-9a-f]{7,}\Z")
# Only lengths that are unambiguously digest-shaped. Deliberately excludes short
# lengths (7-12) that a real short password/key could occupy: `api_key =
# "abcdef123456"` is 12 hex chars and IS a secret, so exempting that length would
# have re-introduced a leak. git short-SHAs stay redacted as a result, which is the
# safe direction to err.
_HASH_LENGTHS = frozenset({32, 40, 56, 64, 96, 128})


def _is_benign_hash(value: str) -> bool:
    """True for a bare hex digest (git SHA, md5/sha256 of content, etc.).

    Deliberately narrow: hex-only AND a recognized digest length. A real secret that
    happens to be hex-only and exactly 40 chars is possible but rare, and the
    entropy rule still covers genuinely random-looking material of other shapes.
    """
    return len(value) in _HASH_LENGTHS and bool(_HEX_HASH_RE.match(value))


@dataclass
class Redaction:
    """A single redaction event. The VALUE is never retained."""
    label: str
    placeholder: str
    line: int


@dataclass
class SanitizeResult:
    text: str
    redactions: list[Redaction] = field(default_factory=list)
    blocked: bool = False

    @property
    def changed(self) -> bool:
        return bool(self.redactions) or self.blocked


def shannon_entropy(s: str) -> float:
    if not s:
        return 0.0
    counts: dict[str, int] = {}
    for ch in s:
        counts[ch] = counts.get(ch, 0) + 1
    n = len(s)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


def _placeholder(label: str, value: str) -> str:
    """Length-stable, deterministic placeholder. Does NOT encode the value
    (uses a short salted hash purely for uniqueness within a run)."""
    tag = hashlib.sha256(value.encode("utf-8")).hexdigest()[:6]
    base = f"<{label.upper()}_{tag}>"
    # Keep length stable so line/column offsets don't shift downstream.
    if len(base) < len(value):
        base = base + "*" * (len(value) - len(base))
    return base[: max(len(value), len(base))]


@dataclass
class Sanitizer:
    mode: str = "redact"               # redact | block | off
    entropy_threshold: float = 4.0
    quarantine_extensions: tuple[str, ...] = (".env", ".pem", ".key", ".p12")
    audit: object = None               # optional Audit for anomaly logging

    def sanitize(self, text: str, file_rel: str = "") -> str:
        """Sanitize text for return to the model. Convenience wrapper that returns
        just the text (used as the Navigator's sanitize callback)."""
        return self.sanitize_full(text, file_rel).text

    def sanitize_full(self, text: str, file_rel: str = "") -> SanitizeResult:
        if self.mode == "off":
            return SanitizeResult(text=text)

        # Quarantine whole files by extension (e.g. .env, .pem).
        low = file_rel.lower()
        if any(low.endswith(ext) for ext in self.quarantine_extensions):
            return SanitizeResult(
                text=f"[QUARANTINED — '{file_rel}' is a sensitive data file; "
                     f"contents withheld. A 'hardcoded secret/data file' finding "
                     f"is recorded locally.]",
                blocked=self.mode == "block",
            )

        redactions: list[Redaction] = []
        out_lines: list[str] = []
        for lineno, line in enumerate(text.splitlines(), start=1):
            new_line = self._redact_line(line, lineno, file_rel, redactions)
            out_lines.append(new_line)
        result_text = "\n".join(out_lines)

        if self.mode == "block" and redactions:
            return SanitizeResult(
                text=f"[BLOCKED — content in '{file_rel}' tripped {len(redactions)} "
                     f"sensitive-data detector(s); not transmitted.]",
                redactions=redactions,
                blocked=True,
            )
        return SanitizeResult(text=result_text, redactions=redactions)

    def _redact_line(self, line: str, lineno: int, file_rel: str,
                     redactions: list[Redaction]) -> str:
        # 1) Pattern-based redaction.
        for label, pat in _SECRET_PATTERNS:
            def _sub(m: re.Match, _label=label) -> str:
                # Redact only the captured SECRET when the pattern isolates one
                # (group 1), not the whole match. `password_assign` matches the entire
                # `password = "hunter2"` statement, so redacting group(0) erased the
                # very line the finding is about — the report then cited code that had
                # been replaced by a placeholder, and coverage's citation check
                # rejected the finding outright.
                value = m.group(1) if m.groups() and m.group(1) else m.group(0)
                if _is_benign_hash(value):
                    return m.group(0)
                ph = _placeholder(_label, value)
                redactions.append(Redaction(label=_label, placeholder=ph, line=lineno))
                # Preserve the surrounding syntax (keyword, operator, quotes) and swap
                # only the value, so the vulnerability stays visible and citable.
                # Replace the captured GROUP's span, not every occurrence of its text.
                # `str.replace` is global within the match, so when the identifier equals
                # the value — `password = "password"`, `secret = "secret"`, pervasive in
                # fixtures and CI configs — the VARIABLE NAME was replaced too. The model
                # then saw `<PASSWORD_ASSIGN_x> = "<PASSWORD_ASSIGN_x>"`, could not
                # describe the finding, and any citation of that line could never match
                # the raw file (the citation gate compares against unsanitized source),
                # so the finding was silently dropped.
                try:
                    lo = m.start(1) - m.start(0)
                    hi = m.end(1) - m.start(0)
                except (IndexError, error):
                    return m.group(0).replace(value, ph)
                whole = m.group(0)
                return whole[:lo] + ph + whole[hi:]
            line = pat.sub(_sub, line)

        # 2) Entropy-based redaction for stray high-entropy tokens.
        def _entropy_sub(m: re.Match) -> str:
            tok = m.group(0)
            if _is_benign_hash(tok):
                return tok
            if shannon_entropy(tok) >= self.entropy_threshold:
                ph = _placeholder("highentropy", tok)
                redactions.append(Redaction(label="high_entropy", placeholder=ph, line=lineno))
                return ph
            return tok
        line = _TOKEN_RE.sub(_entropy_sub, line)
        return line

    # ── deterministic secret-scan (independent local findings) ─────────────────
    def scan_secrets(self, text: str, file_rel: str) -> list[Redaction]:
        """Return secret detections WITHOUT modifying text. Used to emit a
        deterministic local finding even if the LLM never comments."""
        res = self.sanitize_full(text, file_rel)
        return res.redactions

    # ── sanitization-bypass anomaly logging ───────────────────
    def check_bypass_attempt(self, agent_id: str, tool: str, args: dict) -> bool:
        """Detect a model trying to decode/unmask a redaction placeholder.

        E.g. base64-decoding a masked string or searching for the placeholder tag.
        Logs an anomaly and returns True if suspicious.
        """
        blob = " ".join(str(v) for v in args.values()).lower()
        suspicious = (
            "base64" in blob and ("decode" in blob or "b64decode" in blob)
        ) or "_redacted" in blob or re.search(r"<[a-z_]+_[0-9a-f]{6}", blob) is not None
        if suspicious and self.audit is not None:
            try:
                self.audit.anomaly("sanitization_bypass_attempt",
                                   agent_id=agent_id, tool=tool, args=args)
            except Exception:  # noqa: BLE001
                pass
        return bool(suspicious)
