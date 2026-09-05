# Copyright 2026
# SPDX-License-Identifier: Apache-2.0
"""Symbol + call-graph index.

Powers the ``find_definition`` / ``find_callers`` navigation tools and seeds the
ledger's function-level coverage. The default ``auto`` backend uses tree-sitter
when a grammar is available, with a **pure-Python regex indexer** as the per-file
fallback for supported languages. Both populate the same :class:`Index` interface.

The regex indexer is intentionally conservative: it recognizes common
function/method/class definition forms per language. False negatives degrade to
"line-window only" coverage for that symbol; they never cause incorrect results.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

# Per-language definition patterns. Each yields the symbol name in group 1.
# Kept deliberately simple/robust; precision over recall is fine here.
_DEF_PATTERNS: dict[str, list[re.Pattern]] = {
    "python": [
        re.compile(r"^\s*def\s+([A-Za-z_]\w*)\s*\("),
        re.compile(r"^\s*class\s+([A-Za-z_]\w*)\s*[\(:]"),
    ],
    "javascript": [
        re.compile(r"^\s*function\s+([A-Za-z_$][\w$]*)\s*\("),
        re.compile(r"^\s*(?:export\s+)?(?:async\s+)?function\s+([A-Za-z_$][\w$]*)"),
        re.compile(r"^\s*(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*=\s*(?:async\s*)?\("),
        re.compile(r"^\s*class\s+([A-Za-z_$][\w$]*)"),
    ],
    "java": [
        re.compile(r"^\s*(?:public|private|protected|static|final|abstract|\s)+"
                   r"[\w<>\[\],\s]+\s+([A-Za-z_]\w*)\s*\([^;]*\)\s*\{?"),
        re.compile(r"^\s*(?:public|private|protected|\s)*class\s+([A-Za-z_]\w*)"),
    ],
    "go": [
        re.compile(r"^\s*func\s+(?:\([^)]*\)\s*)?([A-Za-z_]\w*)\s*\("),
        re.compile(r"^\s*type\s+([A-Za-z_]\w*)\s+"),
    ],
    "c": [
        re.compile(r"^[A-Za-z_][\w\s\*]+\s+([A-Za-z_]\w*)\s*\([^;]*\)\s*\{?\s*$"),
    ],
    "cpp": [
        re.compile(r"^[A-Za-z_][\w\s\*:<>,&]+\s+([A-Za-z_]\w*)\s*\([^;]*\)\s*\{?\s*$"),
        re.compile(r"^\s*class\s+([A-Za-z_]\w*)"),
    ],
}
# typescript/jsx/tsx reuse javascript patterns.
_DEF_PATTERNS["typescript"] = _DEF_PATTERNS["javascript"] + [
    re.compile(r"^\s*(?:public|private|protected|\s)*([A-Za-z_$][\w$]*)\s*\([^;]*\)\s*[:{]"),
]

_IDENT = re.compile(r"[A-Za-z_$][\w$]*")


@dataclass
class Symbol:
    name: str
    file_rel: str
    start_line: int   # 1-based
    end_line: int     # 1-based (best-effort)


@dataclass
class Index:
    """In-memory symbol + call-graph index over discovered files."""
    # name → list of definitions (a name may be defined in several files)
    definitions: dict[str, list[Symbol]] = field(default_factory=dict)
    # file_rel → ordered list of symbols defined in it
    by_file: dict[str, list[Symbol]] = field(default_factory=dict)
    # callee_name → set of caller file_rel (cheap file-level call graph)
    callers: dict[str, set[str]] = field(default_factory=dict)
    # callee_name → set of caller SYMBOL names (precise, function-level call graph;
    # excludes a symbol referencing itself). Powers real source→sink reachability.
    symbol_callers: dict[str, set[str]] = field(default_factory=dict)
    # Memoization for function_callers(); see there. repr=False so a debug print of an
    # Index does not dump the whole resolved call graph.
    _callers_cache: dict[str, list[Symbol]] = field(default_factory=dict, repr=False)
    # Filled by build_index: {"tree_sitter": n, "regex": n, "none": n} and the languages
    # for which no symbol extraction was possible at all.
    backend_stats: dict[str, int] = field(default_factory=dict)
    unsupported_languages: list[str] = field(default_factory=list)

    def add_definition(self, sym: Symbol) -> None:
        self.definitions.setdefault(sym.name, []).append(sym)
        self.by_file.setdefault(sym.file_rel, []).append(sym)
        self._callers_cache.clear()

    def find_definition(self, name: str) -> list[Symbol]:
        return self.definitions.get(name, [])

    def find_callers(self, name: str) -> list[str]:
        """File-level callers (backward-compatible: files that reference ``name``)."""
        return sorted(self.callers.get(name, set()))

    def function_callers(self, name: str) -> list[Symbol]:
        """Precise, function-level callers: the Symbol definitions of every function
        whose body references ``name`` (excluding ``name`` referencing itself).

        Memoized. This sits inside two BFS inner loops (reachability and the endpoint
        resolver) and re-sorted a set that can hold tens of thousands of names on every
        call — ``symbol_callers['get']`` was measured at 36 000 entries on a 2k-file
        corpus. The index is immutable after :func:`build_index`, so the result is
        stable; the cache is invalidated by :meth:`add_definition` for safety.
        """
        hit = self._callers_cache.get(name)
        if hit is not None:
            return hit
        out: list[Symbol] = []
        for caller_name in sorted(self.symbol_callers.get(name, set())):
            out.extend(self.definitions.get(caller_name, []))
        self._callers_cache[name] = out
        return out

    def enclosing_symbol(self, file_rel: str, line: int,
                         *, decorator_lookback: int = 3) -> "Symbol | None":
        """Return the symbol whose line range covers ``line`` in ``file_rel``.

        Two-stage match so decorator look-back never pollutes body matching:
          1. **Body match** — a symbol whose actual ``[start_line, end_line]`` range
             contains ``line``. On overlap, the tightest (smallest) range wins.
          2. **Decorator region** — if no body matched, a line sitting just ABOVE a
             ``def`` (e.g. ``@app.route`` one line up) maps to that function; the
             nearest def within ``decorator_lookback`` lines below wins.

        Returns ``None`` when the line can't be tied to any symbol (callers treat
        that as 'unclear' and fail open rather than gating the finding out).
        """
        syms = self.by_file.get(file_rel, [])
        # 1) Body match (tightest range wins).
        best: "Symbol | None" = None
        for sym in syms:
            if sym.start_line <= line <= sym.end_line:
                if best is None or (sym.end_line - sym.start_line) < (best.end_line - best.start_line):
                    best = sym
        if best is not None:
            return best
        # 2) Decorator region: line is just above a def (nearest def below wins).
        best_gap: int | None = None
        for sym in syms:
            gap = sym.start_line - line
            if 0 < gap <= decorator_lookback and (best_gap is None or gap < best_gap):
                best, best_gap = sym, gap
        return best


    def functions_in(self, file_rel: str) -> list[Symbol]:
        return self.by_file.get(file_rel, [])


# ── tree-sitter backend (optional, precise) ────────────────────────────────────
# Maps our language names to tree-sitter grammar names.
_TS_LANG = {
    "python": "python", "javascript": "javascript", "typescript": "typescript",
    "java": "java", "go": "go", "c": "c", "cpp": "cpp", "ruby": "ruby",
    "php": "php", "rust": "rust", "csharp": "c_sharp",
}
# tree-sitter node types that represent a named definition, per grammar family.
_TS_DEF_NODES = {
    "function_definition", "function_declaration", "method_definition",
    "method_declaration", "class_definition", "class_declaration",
    "function_item", "impl_item", "constructor_declaration",
}


def _ts_parser(language: str):
    """Return a tree-sitter parser for a language, or None if unavailable.
    Prefers tree-sitter-language-pack, then tree-sitter-languages."""
    ts_name = _TS_LANG.get(language)
    if not ts_name:
        return None
    try:
        from tree_sitter_language_pack import get_parser  # type: ignore
        return get_parser(ts_name)
    except Exception:  # noqa: BLE001
        try:
            from tree_sitter_languages import get_parser  # type: ignore
            return get_parser(ts_name)
        except Exception:  # noqa: BLE001
            return None


def _ts_definitions(file_rel: str, language: str, text: str) -> list[Symbol] | None:
    """Extract precise symbol definitions via tree-sitter, or None if unavailable."""
    parser = _ts_parser(language)
    if parser is None:
        return None
    try:
        data = text.encode("utf-8", errors="replace")
        tree = parser.parse(data)
    except Exception:  # noqa: BLE001
        return None

    syms: list[Symbol] = []

    def _name_of(node) -> str | None:
        for child in node.children:
            if child.type in ("identifier", "name", "field_identifier",
                              "type_identifier", "constant"):
                return data[child.start_byte:child.end_byte].decode("utf-8", "replace")
        # tree-sitter also exposes a 'name' field on many def nodes:
        try:
            n = node.child_by_field_name("name")
            if n is not None:
                return data[n.start_byte:n.end_byte].decode("utf-8", "replace")
        except Exception:  # noqa: BLE001
            pass
        return None

    def _walk(node) -> None:
        if node.type in _TS_DEF_NODES:
            name = _name_of(node)
            if name:
                syms.append(Symbol(name=name, file_rel=file_rel,
                                   start_line=node.start_point[0] + 1,
                                   end_line=node.end_point[0] + 1))
        for child in node.children:
            _walk(child)

    _walk(tree.root_node)
    return syms


def build_index(files: list[tuple[str, str, str]], *, backend: str = "auto") -> Index:
    """Build an :class:`Index` from ``(file_rel, language, text)`` tuples.

    ``backend``: 'auto' (tree-sitter if importable, else regex), 'tree_sitter'
    (force, but still fall back per-file if a grammar is missing), or 'regex'.

    Pass 1 records every definition (precise end lines via tree-sitter when
    available; best-effort via regex otherwise). Pass 2 records file-level
    references; pass 3 records references within each symbol's body.
    """
    index = Index()
    use_ts = backend in ("auto", "tree_sitter")
    # Per-backend tallies so the (large) consequences of a silent downgrade are visible.
    stats = {"tree_sitter": 0, "regex": 0, "none": 0}
    unsupported: set[str] = set()

    # Pass 1: definitions.
    for file_rel, language, text in files:
        defs_in_file: list[Symbol] = []

        # Try tree-sitter first (precise start/end lines) when enabled.
        if use_ts:
            ts = _ts_definitions(file_rel, language, text)
            if ts is not None:
                for sym in ts:
                    index.add_definition(sym)
                stats["tree_sitter"] += 1
                continue  # tree-sitter succeeded for this file

        # Regex fallback.
        patterns = _DEF_PATTERNS.get(language)
        if not patterns:
            # NO symbol support at all for this language. That is not a small gap: with no
            # enclosing symbol, every fingerprint degrades to a 25-line bucket (unstable
            # baselines, over-aggressive merging of genuinely distinct bugs), chunking
            # falls back to fixed windows, the `functions` table stays empty so
            # proof-of-read coverage is unmeasurable, and reachability answers "unclear"
            # for everything — so the gate pays its full cost and demotes nothing. It used
            # to happen silently for csharp/php/ruby/rust whenever tree-sitter was absent.
            stats["none"] += 1
            if language:
                unsupported.add(language)
            continue
        stats["regex"] += 1
        lines = text.splitlines()
        for i, line in enumerate(lines, start=1):
            for pat in patterns:
                m = pat.match(line)
                if m:
                    defs_in_file.append(Symbol(name=m.group(1), file_rel=file_rel,
                                               start_line=i, end_line=i))
                    break
        # Best-effort end lines: a symbol spans until the next symbol's start - 1.
        for j, sym in enumerate(defs_in_file):
            sym.end_line = (defs_in_file[j + 1].start_line - 1
                            if j + 1 < len(defs_in_file) else len(lines))
            index.add_definition(sym)


    # Pass 2: cheap file-level call graph (after all definitions are known).
    known = set(index.definitions)
    for file_rel, _language, text in files:
        seen: set[str] = set()
        for token in _IDENT.findall(text):
            if token in known and token not in seen:
                seen.add(token)
                index.callers.setdefault(token, set()).add(file_rel)

    # Pass 3: PRECISE, function-level call graph. For each defined symbol, scan the
    # identifier tokens inside its own body range and record an edge for every OTHER
    # known symbol it references (self-references are excluded so a dead function
    # never looks like its own caller). This is what makes real source→sink
    # reachability possible (walk callers UP to an attack-surface entrypoint).
    # STREAMED, one file at a time. Materializing `{file_rel: text.splitlines()}` for
    # the whole corpus held a SECOND full copy of every source line in memory, on top of
    # the caller's own `files` list, the orchestrator's `_file_text`, and estimate's copy
    # — four concurrent full-corpus copies at peak, which is a real OOM risk on a large
    # monorepo (and if it happens inside estimate() the failure is swallowed into a
    # fabricated "~$0.0" cost projection).
    for file_rel, _language, text in files:
        symbols = index.by_file.get(file_rel)
        if not symbols:
            continue
        lines = text.splitlines()
        for sym in symbols:
            # Body = the symbol's own definition line through its end line. Skip the
            # very first line's own name token by only counting references, then
            # discarding self below.
            body = "\n".join(lines[sym.start_line - 1:sym.end_line])
            for token in set(_IDENT.findall(body)):
                if token in known and token != sym.name:
                    index.symbol_callers.setdefault(token, set()).add(sym.name)
    index._callers_cache.clear()
    index.backend_stats = dict(stats)
    index.unsupported_languages = sorted(unsupported)
    if stats["none"]:
        print(f"⚠ Symbol index: {stats['none']} file(s) have NO symbol support "
              f"(language(s): {', '.join(index.unsupported_languages) or 'unknown'}).\n"
              f"  For those files findings fall back to coarse line buckets, "
              f"proof-of-read cannot be measured,\n"
              f"  and reachability is always 'unclear'. Install tree-sitter grammars for "
              f"them to fix this.")
    if backend == "tree_sitter" and stats["tree_sitter"] == 0 and files:
        # An explicit request for precision that silently became the regex fallback (or
        # nothing at all) is worse than an error, because every downstream number quietly
        # gets less trustworthy.
        raise RuntimeError(
            "indexing.backend is 'tree_sitter' but no tree-sitter grammar could parse "
            "ANY file — every symbol would come from the regex fallback (or be missing "
            "entirely). Install tree_sitter + tree_sitter_languages, or set "
            "indexing.backend: auto to accept the fallback explicitly.")
    return index
