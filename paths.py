# Copyright 2026
# SPDX-License-Identifier: Apache-2.0
"""Cross-platform path helpers + path-jailing.

Everything path-related goes through here so portability rules live in ONE place
(Windows / macOS / Linux). No `if os.name == "nt"` branches anywhere else.

The single most important function is :func:`resolve_in_jail`, the security
boundary: every file the model can touch via the navigation tools must resolve to
a path *inside* the target root. ``Path.resolve()`` collapses ``..`` and follows
symlinks on every OS, so an attacker-supplied ``../../etc/passwd`` (or a symlink
escape) is caught here.
"""
from __future__ import annotations

from pathlib import Path, PurePath


class PathJailError(ValueError):
    """Raised when a requested path resolves outside the target root."""


def to_path(value: str | Path) -> Path:
    """Normalize any string/Path into a ``Path``. Never string-concatenate paths
    elsewhere — always start here."""
    return value if isinstance(value, Path) else Path(value)


def resolve_root(root: str | Path) -> Path:
    """Resolve the target root once, up front. Must exist and be a directory."""
    p = to_path(root).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"target root does not exist: {p}")
    if not p.is_dir():
        raise NotADirectoryError(f"target root is not a directory: {p}")
    return p


def resolve_target(target: str | Path) -> tuple[Path, str | None]:
    """Resolve ``scope.path`` (or ``--path``) into ``(root, only_file)``.

    A directory behaves exactly as before: ``(dir, None)``.

    A FILE yields ``(parent_directory, "name.py")`` — the scan root becomes the folder the
    file lives in, and ``only_file`` narrows what gets HUNTED to that one file. Pointing
    at a file used to fail with ``NotADirectoryError``, which is a reasonable thing for
    someone to try ("audit just this file").

    Why the root is the PARENT and not the file: the value of this harness is cross-file
    reasoning — ``find_definition`` following a call into ``store.py``, reachability
    walking back to an entrypoint in ``app.py``. Rooting at the file alone would leave the
    symbol index, call graph and attack surface containing exactly one file, so every
    lookup would miss and reachability would answer "unclear" for everything. The whole
    folder is therefore indexed and readable; only the ledger is narrowed.

    The consequence to be explicit about: the path JAIL is the parent directory, so the
    model can still READ sibling files. A single-file scope limits what is *analyzed*, not
    what is *visible*. If you need true isolation, copy the file into its own directory.
    """
    p = to_path(target).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"target path does not exist: {p}")
    if p.is_dir():
        return p, None
    if not p.is_file():
        # A socket / device / broken symlink: not scannable, and the error should say
        # which of the two supported shapes it failed to be.
        raise NotADirectoryError(
            f"target path is neither a directory nor a regular file: {p}")
    parent = p.parent
    if not parent.is_dir():
        raise NotADirectoryError(f"target file has no readable parent directory: {p}")
    return parent, p.name


def resolve_in_jail(root: Path, requested: str | Path) -> Path:
    """Resolve ``requested`` (which may be relative to ``root`` or absolute) and
    guarantee the result is inside ``root``.

    Raises :class:`PathJailError` if the resolved path escapes the jail. This is
    the security boundary for every read-only navigation tool.
    """
    root = root.resolve()
    req = to_path(requested)
    # Treat absolute requests as rooted at the jail (defense in depth): an agent
    # asking for "/etc/passwd" should resolve under root, not the real FS root.
    if req.is_absolute():
        # Strip the anchor (drive/root) so it becomes relative to the jail.
        req = Path(*req.parts[1:]) if len(req.parts) > 1 else Path()
    candidate = (root / req).resolve()
    if not _is_relative_to(candidate, root):
        raise PathJailError(
            f"path escapes target root: requested={requested!r} resolved={candidate}"
        )
    return candidate


def _is_relative_to(child: Path, parent: Path) -> bool:
    """``Path.is_relative_to`` backport-safe check (3.8 compatible)."""
    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return False


def canonical_rel(root: Path, requested: str) -> str | None:
    """Normalize an UNTRUSTED path into the exact ``file_rel`` form the ledger uses.

    Returns a root-relative, forward-slash path, or None if it escapes the jail or
    cannot be resolved.

    Models write paths every way imaginable — ``app/x.py``, ``./app/x.py``,
    ``/app/x.py``, ``app\\x.py`` — and the raw string was previously stored as-is.
    ``./app/x.py`` and ``/app/x.py`` even PASSED the citation gate, because that gate goes
    through :func:`resolve_in_jail` which normalizes them, while every other consumer
    keyed on the exact string. The consequences compounded: ``enclosing_symbol`` missed,
    so ``symbol`` was empty and the fingerprint fell back to a line bucket (no dedupe
    against the canonically-spelled copy, unstable baselines); ``(SELECT id FROM files
    WHERE path=?)`` returned NULL so ``findings.file_id`` was NULL; the resume join then
    produced ``file_rel=''`` — a third identity, multiplying per resume; no
    ``sample_request`` and no PoC; and SARIF got a ``uri`` that breaks code-scanning
    ingestion.
    """
    try:
        return relative_to_root(root, resolve_in_jail(root, requested))
    except (PathJailError, ValueError, OSError):
        return None


def relative_to_root(root: Path, path: Path) -> str:
    """Return a stable, forward-slash relative path for display/storage so the
    coverage ledger keys are identical across operating systems."""
    rel = path.resolve().relative_to(root.resolve())
    return PurePath(rel).as_posix()
