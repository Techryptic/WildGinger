# Copyright 2026
# SPDX-License-Identifier: Apache-2.0
"""Deterministic attack-surface / source-sink inventory.

A lightweight, pure-Python pass that gives the LLM **structured facts** to reason
over (it still reasons freely on top). We detect, per file:
  * entrypoints / sources — where untrusted input enters (routes, handlers, CLI,
    deserialization, file uploads, message consumers), and
  * sinks — dangerous operations (exec, SQL, filesystem, network, template render).

Files containing entrypoints or sinks are **trust boundaries**; the orchestrator
boosts their ledger cell weights so hunters spend disproportionate budget there.

This is intentionally NOT a full taint engine. It is
a regex/keyword inventory — high value, low complexity, no false-precision claims.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

# (label, compiled pattern). Patterns are broad on purpose: recall > precision for
# *hints*. The LLM and validator separate signal from noise downstream.
_SOURCE_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("http_route", re.compile(r"@(?:app|router|blueprint)\.(?:route|get|post|put|delete|patch)\b"
                              r"|@(?:Get|Post|Put|Delete|Patch)Mapping\b"
                              r"|app\.(?:get|post|put|delete|patch)\s*\(")),
    ("flask_request", re.compile(r"\brequest\.(?:args|form|json|data|values|files|cookies)\b")),
    ("django_request", re.compile(r"\brequest\.(?:GET|POST|FILES|body|COOKIES)\b")),
    ("express_req", re.compile(r"\breq\.(?:params|query|body|headers|cookies)\b")),
    ("cli_args", re.compile(r"\b(?:sys\.argv|argparse|os\.environ|process\.argv)\b")),
    ("deserialization", re.compile(r"\b(?:pickle\.loads|yaml\.load|marshal\.loads|"
                                   r"ObjectInputStream|unserialize|JSON\.parse)\b")),
    ("file_upload", re.compile(r"\b(?:MultipartFile|FileUpload|request\.files|multer)\b")),
    ("message_consumer", re.compile(r"\b(?:on_message|@KafkaListener|@RabbitListener|consume)\b")),
    ("rpc_handler", re.compile(r"\b(?:grpc|@GrpcMethod|rpc\s+\w+)\b")),
]

_SINK_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("command_exec", re.compile(r"\b(?:os\.system|subprocess\.(?:run|call|Popen)|exec|eval|"
                                r"Runtime\.exec|child_process\.(?:exec|spawn)|popen)\b")),
    ("sql", re.compile(r"\b(?:execute|executemany|cursor\.execute|createQuery|rawQuery|"
                       r"\.query\s*\()\b|SELECT\s.+\sFROM\s")),
    ("filesystem", re.compile(r"\b(?:open|fopen|readFile|writeFile|os\.remove|Path\()\b")),
    ("network", re.compile(r"\b(?:requests\.(?:get|post)|urllib|http\.client|fetch|axios|"
                           r"socket\.)\b")),
    ("template", re.compile(r"\b(?:render_template_string|Template\(|innerHTML|dangerouslySetInnerHTML)\b")),
    ("crypto_weak", re.compile(r"\b(?:md5|sha1|DES|ECB|Random\(\))\b")),
]

_SANITIZER_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("escape", re.compile(r"\b(?:escape|sanitize|clean|bleach|htmlspecialchars|"
                          r"parameterize|quote|prepared)\b")),
]


@dataclass
class SurfaceHit:
    file_rel: str
    line: int
    kind: str        # source | sink | sanitizer
    label: str


@dataclass
class AttackSurface:
    sources: list[SurfaceHit] = field(default_factory=list)
    sinks: list[SurfaceHit] = field(default_factory=list)
    sanitizers: list[SurfaceHit] = field(default_factory=list)
    # Accumulated during analyze() so trust_boundary_files() is O(1) instead of a
    # re-walk of both hit lists.
    _boundary_files: set[str] = field(default_factory=set, repr=False)

    def trust_boundary_files(self) -> set[str]:
        """Files that contain a source or a sink — weight these higher."""
        if self._boundary_files:
            return set(self._boundary_files)
        files: set[str] = set()
        for h in self.sources:
            files.add(h.file_rel)
        for h in self.sinks:
            files.add(h.file_rel)
        return files


def analyze(files: list[tuple[str, str]]) -> AttackSurface:
    """Build an :class:`AttackSurface` from ``(file_rel, text)`` tuples.

    At most ONE hit per (line, kind). Every matching pattern used to append its own
    ``SurfaceHit``, so a single line that looks like nine different request sources
    produced nine identical-location objects: measured 153k hits on a 2k-file corpus,
    extrapolating to ~750k objects (~150 MB) retained for the whole scan to serve a
    30-item prompt hint and a set of filenames.

    Deliberately NOT capped. ``trace.entrypoint_symbols`` needs every distinct source
    LOCATION to build the entrypoint set, and dropping some would make real handlers
    look unreachable — i.e. it would gate out real bugs. Deduplicating is free;
    truncating is not.
    """
    surface = AttackSurface()
    for file_rel, text in files:
        for lineno, line in enumerate(text.splitlines(), start=1):
            src = _scan_line(surface.sources, _SOURCE_PATTERNS, file_rel, lineno, line,
                             "source")
            snk = _scan_line(surface.sinks, _SINK_PATTERNS, file_rel, lineno, line,
                             "sink")
            _scan_line(surface.sanitizers, _SANITIZER_PATTERNS, file_rel, lineno, line,
                       "sanitizer")
            if src or snk:
                surface._boundary_files.add(file_rel)
    return surface


def _scan_line(bucket: list[SurfaceHit], patterns, file_rel: str, lineno: int,
               line: str, kind: str) -> bool:
    """Append at most one hit for this line/kind. Returns True if it matched."""
    for label, pat in patterns:
        if pat.search(line):
            bucket.append(SurfaceHit(file_rel=file_rel, line=lineno, kind=kind,
                                     label=label))
            return True
    return False
