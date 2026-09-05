# Copyright 2026
# SPDX-License-Identifier: Apache-2.0
"""File discovery.

Walks the pre-staged target folder (pure ``pathlib`` / ``os.walk`` — no ``find``),
classifies each file by language + artifact category (code / iac / config), hashes
its raw bytes, counts lines, records mtime (for incremental mode), and assigns a
priority weight (risky paths first).

Everything here is cross-platform and read-only.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

from . import fileio
from .paths import relative_to_root
from .settings import ScopeConfig

# Extension → (language, category). Categories: code | iac | config.
_EXT_MAP: dict[str, tuple[str, str]] = {
    # code
    ".py": ("python", "code"),
    ".js": ("javascript", "code"),
    ".jsx": ("javascript", "code"),
    ".ts": ("typescript", "code"),
    ".tsx": ("typescript", "code"),
    ".java": ("java", "code"),
    ".go": ("go", "code"),
    ".c": ("c", "code"),
    ".h": ("c", "code"),
    ".cc": ("cpp", "code"),
    ".cpp": ("cpp", "code"),
    ".cxx": ("cpp", "code"),
    ".hpp": ("cpp", "code"),
    ".cs": ("csharp", "code"),
    ".rb": ("ruby", "code"),
    ".php": ("php", "code"),
    ".rs": ("rust", "code"),
    # iac
    ".tf": ("terraform", "iac"),
    ".tfvars": ("terraform", "iac"),
    # config
    ".yaml": ("yaml", "config"),
    ".yml": ("yaml", "config"),
    ".json": ("json", "config"),
    ".toml": ("toml", "config"),
    ".xml": ("xml", "config"),
    ".ini": ("ini", "config"),
    ".env": ("dotenv", "config"),
}

# Filenames (case-insensitive) that map to iac/config regardless of extension.
_FILENAME_MAP: dict[str, tuple[str, str]] = {
    "dockerfile": ("dockerfile", "iac"),
    "docker-compose.yml": ("compose", "iac"),
    "docker-compose.yaml": ("compose", "iac"),
    "jenkinsfile": ("jenkins", "iac"),
    ".gitlab-ci.yml": ("gitlabci", "iac"),
    # A bare `.env` has NO suffix (`Path(".env").suffix == ""`), so the ".env" entry in
    # _EXT_MAP could never match it — the single highest-value secrets file in a repo was
    # classified as "unknown" (or skipped outright with scan_unknown_extensions: false),
    # which also meant sanitization.quarantine_extensions never fired for it.
    ".env": ("dotenv", "config"),
    ".env.local": ("dotenv", "config"),
    ".env.production": ("dotenv", "config"),
    ".env.prod": ("dotenv", "config"),
    ".env.development": ("dotenv", "config"),
    ".env.dev": ("dotenv", "config"),
    ".env.test": ("dotenv", "config"),
    ".npmrc": ("npmrc", "config"),
    ".netrc": ("netrc", "config"),
}

# Path substrings that elevate priority (auth/API/data layers, parsers).
_RISKY_PATH_HINTS = (
    "auth", "login", "session", "token", "password", "crypto", "secret",
    "api", "controller", "route", "handler", "endpoint", "rpc", "graphql",
    "admin", "upload", "deserial", "parse", "exec", "command", "query", "sql",
)

# Content hints (dangerous functions) that elevate priority when present.
_RISKY_CONTENT = re.compile(
    r"\b(eval|exec|system|popen|subprocess|pickle\.loads|os\.system|Runtime\.exec|"
    r"deserialize|unserialize|yaml\.load|child_process)\b"
)


@dataclass
class DiscoveredFile:
    path: Path                 # absolute
    rel_path: str              # forward-slash, relative to root (ledger key)
    language: str
    category: str
    sha256: str
    line_count: int
    size_bytes: int
    mtime: float
    weight: float = 1.0


@dataclass
class DiscoveryResult:
    files: list[DiscoveredFile] = field(default_factory=list)
    skipped: int = 0
    skipped_reasons: dict[str, int] = field(default_factory=dict)

    def _skip(self, reason: str) -> None:
        self.skipped += 1
        self.skipped_reasons[reason] = self.skipped_reasons.get(reason, 0) + 1


def classify(path: Path) -> tuple[str, str] | None:
    """Return (language, category) for a file by extension/name, or None if the
    extension isn't recognized. (Unknown-but-text fallback is decided in
    ``discover`` using config, so this stays a pure lookup.)"""
    name = path.name.lower()
    if name in _FILENAME_MAP:
        return _FILENAME_MAP[name]
    # Dockerfile.* variants.
    if name.startswith("dockerfile"):
        return ("dockerfile", "iac")
    # GitHub Actions live under .github/workflows/*.yml — caller's path check.
    ext = path.suffix.lower()
    return _EXT_MAP.get(ext)



def _excluded(rel_parts: tuple[str, ...], exclude_dirs: set[str]) -> bool:
    return any(part in exclude_dirs for part in rel_parts)


def discover(root: Path, scope: ScopeConfig) -> DiscoveryResult:
    """Walk ``root`` and return all in-scope files with metadata + priority weights."""
    result = DiscoveryResult()
    exclude_dirs = set(scope.exclude_dirs)
    include_languages = set(scope.include_languages)
    max_bytes = scope.max_file_kb * 1024
    wanted_categories = set(scope.artifact_categories)
    binary_exts = {e.lower() for e in getattr(scope, "binary_extensions", [])}
    scan_unknown = getattr(scope, "scan_unknown_extensions", True)


    for dirpath, dirnames, filenames in os.walk(root):
        # Prune excluded dirs in-place so os.walk doesn't descend into them.
        dirnames[:] = [d for d in dirnames if d not in exclude_dirs]
        for fname in filenames:
            fpath = Path(dirpath) / fname
            try:
                rel = relative_to_root(root, fpath)
            except ValueError:
                continue
            rel_parts = tuple(rel.split("/"))
            if _excluded(rel_parts, exclude_dirs):
                continue

            # Cheap denylist first: obvious media/binaries skipped without reading.
            if fpath.suffix.lower() in binary_exts:
                result._skip("binary_ext")
                continue

            classification = classify(fpath)
            if classification is None:
                # Unknown extension. Option A: scan it anyway IF it's text and the
                # config allows it (generic "unknown"/code path). Otherwise skip.
                if not scan_unknown:
                    result._skip("unsupported_type")
                    continue
                language, category = "unknown", "code"
            else:
                language, category = classification

            # Category filter applies to recognized files; "unknown" code is always
            # in scope when scan_unknown is on (so coverage is complete).
            if classification is not None:
                if category not in wanted_categories:
                    result._skip("category_excluded")
                    continue
                # 'code' files honor include_languages; iac/config always considered.
                if category == "code" and language not in include_languages:
                    result._skip("language_excluded")
                    continue

            try:
                stat = fpath.stat()
            except OSError:
                result._skip("stat_error")
                continue
            if stat.st_size > max_bytes:
                result._skip("too_large")
                continue
            if fileio.is_probably_binary(fpath):
                result._skip("binary")
                continue


            try:
                meta = fileio.read_text_meta(fpath, max_bytes=max_bytes)
            except (OSError, ValueError):
                result._skip("read_error")
                continue

            weight = _priority_weight(rel, meta.text)
            result.files.append(DiscoveredFile(
                path=fpath,
                rel_path=rel,
                language=language,
                category=category,
                sha256=meta.sha256,
                line_count=meta.line_count,
                size_bytes=meta.size_bytes,
                mtime=stat.st_mtime,
                weight=weight,
            ))
    # Highest priority first.
    result.files.sort(key=lambda f: f.weight, reverse=True)
    return result


def _priority_weight(rel_path: str, text: str) -> float:
    """Heuristic priority: risky paths + dangerous-function content scan higher."""
    weight = 1.0
    low = rel_path.lower()
    for hint in _RISKY_PATH_HINTS:
        if hint in low:
            weight += 0.5
            break
    if _RISKY_CONTENT.search(text):
        weight += 1.0
    return weight
