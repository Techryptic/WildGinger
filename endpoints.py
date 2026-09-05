# Copyright 2026
# SPDX-License-Identifier: Apache-2.0
"""Deterministic endpoint + malicious-sample-request builder.

For a confirmed, externally-reachable finding whose enclosing function is an HTTP
entry point, we build a **malicious** sample request *deterministically* from the
code we already parse — no LLM calls, no cost, always consistent with the actual
route. The result is attached to the finding as ``sample_request`` and rendered in
the report + used to make the PoC stub concrete.

Everything here is best-effort and fail-open: if we can't confidently resolve the
method/path we return ``None`` rather than emit a bogus request. The host is always
a ``<BASE_URL>`` placeholder (never a real target), matching the inert-PoC stance.
"""
from __future__ import annotations

import re
import shlex
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote

from .audit import BudgetExceeded
from .findings import Finding
from .indexing import Index, Symbol

# ── route-decorator / handler patterns ─────────────────────────────────────────
# Flask/FastAPI-style:  @app.route("/x", methods=["POST"])  ·  @app.get("/x")
_ROUTE_METHOD_PATH = re.compile(
    r"@\w+\.(route|get|post|put|delete|patch)\s*\(\s*['\"]([^'\"]+)['\"]"
    r"(?:[^)]*methods\s*=\s*\[([^\]]*)\])?", re.IGNORECASE)
# Spring-style:  @GetMapping("/x")  @RequestMapping(value="/x", method=...)
_SPRING_MAPPING = re.compile(
    r"@(Get|Post|Put|Delete|Patch|Request)Mapping\s*\(\s*(?:value\s*=\s*)?['\"]([^'\"]+)['\"]",
    re.IGNORECASE)
# The verb of a @RequestMapping lives in a separate `method = ...` attribute:
#   method = RequestMethod.POST      method = {RequestMethod.PUT, RequestMethod.POST}
#   method = POST                    method=RequestMethod.DELETE
# Matched in two steps so a bare verb elsewhere in the window can't be mistaken for
# the attribute: first isolate the `method = <...>` clause, then read verbs from it.
_SPRING_METHOD_CLAUSE = re.compile(
    r"\bmethod\s*=\s*(\{[^}]*\}|[A-Za-z_.]+)", re.IGNORECASE)
_SPRING_METHOD_VERB = re.compile(
    r"\b(GET|POST|PUT|DELETE|PATCH|HEAD|OPTIONS)\b", re.IGNORECASE)


def _spring_methods(window: str) -> list[str]:
    """HTTP verbs declared in a Spring ``method = ...`` attribute ([] if absent)."""
    clause = _SPRING_METHOD_CLAUSE.search(window)
    if not clause:
        return []
    return _SPRING_METHOD_VERB.findall(clause.group(1))
# Express-style:  app.post("/x", ...)  router.get('/x', ...)
# The receiver is constrained to router-ish names and the path must start with '/'.
# A bare `\w+\.(get|post)\(` matched ANY method call taking a string literal, so
# `os.environ.get("/etc/shadow")` and `d.get("x")` were both reported as real HTTP
# routes — the generated PoC then advertised `GET /etc/shadow` as an endpoint of the
# application under test.
_EXPRESS_ROUTE = re.compile(
    r"\b(?:app|application|router|routes?|server|api|srv|bp|blueprint|r)\s*\.\s*"
    r"(get|post|put|delete|patch)\s*\(\s*['\"](/[^'\"]*)['\"]", re.IGNORECASE)

# ── parameter accessors inside a handler body ──────────────────────────────────
# Each yields (param_name, location). Location = query|body|path|header|cookie.
_PARAM_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"request\.args(?:\.get)?\s*[\(\[]\s*['\"]([^'\"]+)['\"]"), "query"),
    (re.compile(r"request\.form(?:\.get)?\s*[\(\[]\s*['\"]([^'\"]+)['\"]"), "body"),
    (re.compile(r"request\.values(?:\.get)?\s*[\(\[]\s*['\"]([^'\"]+)['\"]"), "query"),
    (re.compile(r"request\.cookies(?:\.get)?\s*[\(\[]\s*['\"]([^'\"]+)['\"]"), "cookie"),
    (re.compile(r"request\.GET(?:\.get)?\s*[\(\[]\s*['\"]([^'\"]+)['\"]"), "query"),
    (re.compile(r"request\.POST(?:\.get)?\s*[\(\[]\s*['\"]([^'\"]+)['\"]"), "body"),
    (re.compile(r"\breq\.query\.([A-Za-z_]\w*)"), "query"),
    (re.compile(r"\breq\.body\.([A-Za-z_]\w*)"), "body"),
    (re.compile(r"\breq\.params\.([A-Za-z_]\w*)"), "path"),
    (re.compile(r"@RequestParam(?:\([^)]*)?\s+\w+\s+([A-Za-z_]\w*)"), "query"),
]
# A raw request body sink (e.g. pickle.loads(request.data)) — no named param.
_RAW_BODY = re.compile(r"request\.data|request\.body|req\.rawBody")

# ── malicious payloads per attack class ─────────────────────────────────────────
# Kept short + classic so the intent is obvious in a report. URL-encoded at render.
_PAYLOADS: dict[str, str] = {
    "injection": "' OR '1'='1",
    "sql": "' OR '1'='1",
    "sqli": "' OR '1'='1",
    "command_injection": "; id",
    "command": "; id",
    "path_traversal": "../../../../etc/passwd",
    "xss": "<script>alert(1)</script>",
    "ssrf": "http://169.254.169.254/latest/meta-data/",
    "open_redirect": "//evil.example.com",
    "deserialization": "<serialized-gadget-payload>",
    "authz": "<other-users-id>",
    "authn": "<forged-or-empty-credential>",
    "csrf": "<auto-submit-form-from-attacker-origin>",
    "secrets": "<n/a>",
}
_DEFAULT_PAYLOAD = "<malicious-value>"


@dataclass
class Endpoint:
    method: str
    path: str
    params: list[dict[str, str]]        # [{name, in, type}]
    raw_body: bool                      # true when handler reads raw request body
    citation: str                       # file:line-range of the handler


def _payload_for(category: str) -> str:
    return _PAYLOADS.get((category or "").lower(), _DEFAULT_PAYLOAD)


def _extract_route(lines: list[str], sym: Symbol, *, lookback: int = 4) -> tuple[str, str] | None:
    """Find (method, path) from a route decorator/registration at or just above the
    handler's definition line. Returns None if no route can be resolved."""
    lo = max(0, sym.start_line - 1 - lookback)
    hi = min(len(lines), sym.start_line + 1)  # include the def line itself (express)
    window = "\n".join(lines[lo:hi])

    m = _ROUTE_METHOD_PATH.search(window)
    if m:
        verb, path, methods = m.group(1), m.group(2), m.group(3)
        if methods:
            method = _pick_method(re.findall(r"['\"]([A-Za-z]+)['\"]", methods))
        elif verb.lower() in ("get", "post", "put", "delete", "patch"):
            method = verb.upper()
        else:
            method = "GET"
        return method, path

    m = _SPRING_MAPPING.search(window)
    if m:
        kind, path = m.group(1), m.group(2)
        if kind.lower() == "request":
            # @RequestMapping carries the verb in a `method = RequestMethod.POST`
            # attribute. Defaulting to GET produced a curl that exercises the WRONG
            # verb — a reviewer following the PoC would get a 405 and dismiss a real
            # finding. Fall back to GET only when no method attribute is present
            # (which is genuinely Spring's default).
            declared = _spring_methods(window)
            method = _pick_method(declared) if declared else "GET"
        else:
            method = kind.upper()
        return method, path

    m = _EXPRESS_ROUTE.search(window)
    if m:
        return m.group(1).upper(), m.group(2)
    return None


# Ranked by how interesting the verb is for an attacker: a state-changing verb makes
# the better PoC when a handler is registered for several. (Flask's
# methods=["GET","POST"] and Spring's method={GET,POST} both allow multiples, and
# taking merely the FIRST listed was arbitrary.)
_METHOD_PRIORITY = ("POST", "PUT", "PATCH", "DELETE", "GET", "HEAD", "OPTIONS")


def _pick_method(candidates: list[str]) -> str:
    """Choose the most attack-relevant HTTP verb from those a route accepts."""
    upper = {c.upper() for c in candidates if c}
    for verb in _METHOD_PRIORITY:
        if verb in upper:
            return verb
    return next(iter(sorted(upper)), "GET")


def _extract_params(body: str) -> tuple[list[dict[str, str]], bool]:
    params: list[dict[str, str]] = []
    seen: set[str] = set()
    for pat, loc in _PARAM_PATTERNS:
        for name in pat.findall(body):
            if name and name not in seen:
                seen.add(name)
                params.append({"name": name, "in": loc, "type": "str"})
    raw_body = bool(_RAW_BODY.search(body))
    return params, raw_body


_MAX_WALK_DEPTH = 15


def resolve_endpoint(finding: Finding, index: Index, file_text: str,
                     file_texts: dict[str, str] | None = None) -> Endpoint | None:
    """Deterministically resolve the HTTP endpoint for a finding.

    First tries the finding's own enclosing function (the common case: the bug is
    IN the handler). If that function has no route decorator — e.g. the sink lives in
    a data-access / helper layer (``store.py`` / ``helpers.py``) — we walk the
    function-level call graph UP toward a caller that IS an HTTP handler, so sink
    findings still resolve the real method/path/params. Returns None only when no
    reachable handler can be found (fail-open: no fabricated request)."""
    sym = index.enclosing_symbol(finding.file_rel, finding.line_start)
    if sym is None:
        return None
    lines = file_text.splitlines()

    # 1) Direct: the finding's own function is the route handler.
    route = _extract_route(lines, sym)
    if route is not None:
        method, path = route
        body = "\n".join(lines[sym.start_line - 1:sym.end_line])
        params, raw_body = _extract_params(body)
        citation = f"{finding.file_rel}:{sym.start_line}-{sym.end_line}"
        return Endpoint(method=method, path=path, params=params,
                        raw_body=raw_body, citation=citation)

    # 2) Sink in a lower layer: walk callers UP to an HTTP handler.
    texts = file_texts or {finding.file_rel: file_text}
    return _resolve_via_callers(sym, index, texts)


def _lines_for(index: Index, texts: dict[str, str], sym: Symbol) -> list[str]:
    return (texts.get(sym.file_rel, "")).splitlines()


# A symbol name referenced by this many functions is a generic verb (`get`, `run`,
# `handle`), not a useful call-graph edge: the name-based index links every definition of
# it to every referencing function. Expanding such a name explores most of the repo for
# no information, and on a 2k-file corpus `function_callers('get')` returns ~36 000
# symbols. Skipping them makes the walk bounded without changing any real answer.
_MAX_CALLERS_PER_NAME = 200


def _resolve_via_callers(origin: Symbol, index: Index,
                         texts: dict[str, str]) -> Endpoint | None:
    """BFS up the function-level call graph from ``origin`` to the first caller that
    carries a route decorator. Cycle-guarded and depth-capped."""
    from collections import deque

    visited: set[str] = set()
    # Whole-file line lists are reused across every symbol in the same file. Without
    # this, one hop through a generic name re-`splitlines()` the containing file once per
    # caller symbol — tens of thousands of whole-file splits (hundreds of MB of transient
    # strings) to read a 5-line window.
    lines_cache: dict[str, list[str]] = {}

    def _lines(sym: Symbol) -> list[str]:
        cached = lines_cache.get(sym.file_rel)
        if cached is None:
            cached = _lines_for(index, texts, sym)
            lines_cache[sym.file_rel] = cached
        return cached

    # deque, not list: `list.pop(0)` is O(n), making the frontier drain O(n^2).
    frontier: deque[tuple[str, int]] = deque([(origin.name, 0)])
    while frontier:
        name, depth = frontier.popleft()
        if name in visited or depth > _MAX_WALK_DEPTH:
            continue
        visited.add(name)
        callers = index.function_callers(name)
        if len(callers) > _MAX_CALLERS_PER_NAME:
            continue
        for caller in callers:
            lines = _lines(caller)
            if not lines:
                continue
            route = _extract_route(lines, caller)
            if route is not None:
                method, path = route
                body = "\n".join(lines[caller.start_line - 1:caller.end_line])
                params, raw_body = _extract_params(body)
                citation = f"{caller.file_rel}:{caller.start_line}-{caller.end_line}"
                return Endpoint(method=method, path=path, params=params,
                                raw_body=raw_body, citation=citation)
            frontier.append((caller.name, depth + 1))
    return None


def _auth_headers(ep: Endpoint, category: str) -> list[str]:
    """Headers an analyst needs to exercise the endpoint. If the handler reads a
    ``session``/auth cookie, include a Cookie header (forged for authn/authz; the
    VICTIM's valid session for csrf — the attack rides on it)."""
    headers: list[str] = []
    cookie_params = [p for p in ep.params if p["in"] == "cookie"]
    cat = (category or "").lower()
    if cookie_params or cat in ("authn", "authz", "csrf"):
        name = cookie_params[0]["name"] if cookie_params else "session"
        if cat in ("authn", "authz"):
            headers.append(f"Cookie: {name}=<forged-token e.g. base64({{\"id\":1,"
                           f"\"role\":\"admin\"}})+'.anything'>")
        else:
            headers.append(f"Cookie: {name}=<a valid session from POST /api/login>")
    return headers


def _build_curl(ep: Endpoint, payload: str, category: str) -> tuple[str, str, list[str], str]:
    """Return (curl_string, payload_note, headers, body). Injects the malicious
    payload into the most relevant param; falls back to a body/query placeholder."""
    base = "https://<BASE_URL>"
    headers = _auth_headers(ep, category)
    body_str = ""
    # Prefer a query/path param for GET; a body param for POST/PUT/PATCH.
    query_params = [p for p in ep.params if p["in"] in ("query", "path")]
    body_params = [p for p in ep.params if p["in"] == "body"]

    target = None
    if ep.method == "GET":
        target = query_params[0] if query_params else (ep.params[0] if ep.params else None)
    else:
        target = body_params[0] if body_params else (ep.params[0] if ep.params else None)

    path = ep.path
    # Substitute a path variable (<id> / {id} / :id) if that's the target.
    def _fill_path(p: str, name: str, value: str) -> str:
        p = re.sub(rf"<(?:[^:>]+:)?{re.escape(name)}>", quote(value, safe=""), p)
        p = re.sub(rf"\{{{re.escape(name)}\}}", quote(value, safe=""), p)
        p = re.sub(rf":{re.escape(name)}\b", quote(value, safe=""), p)
        return p

    note_target = target["name"] if target else "the request body"
    # Every value interpolated into the shell command goes through shlex.quote.
    # Hand-wrapping in single quotes was broken for exactly the payloads that matter:
    # the SQLi payload `' OR '1'='1` closed the quote early, so the emitted curl
    # parsed as several arguments and the reviewer's copy-paste silently tested
    # something else entirely. `body_str` stays the RAW logical value (that's what
    # callers/report renderers want); only the shell string is quoted.
    header_flags = "".join(f" -H {shlex.quote(h)}" for h in headers)
    # The verb is a property of the ROUTE, never of which branch we take below. This
    # branch is chosen on "has no body to send", which is also true of a POST/PUT/DELETE
    # route whose parameters are all path/query (Express `router.post('/user/:id')`
    # reading `req.params.id`, Flask `methods=["POST"]` reading `request.args`). Those
    # used to be emitted as `curl -i <url>` with no `-X`, so the reviewer's copy-pasted
    # "proof" exercised GET instead: normally a 404/405 — making a real finding look
    # unreproducible — or, worse, a DIFFERENT handler registered on the same path, which
    # proves nothing about the bug we reported. `-X GET` is omitted deliberately: it is
    # noise, and it changes curl's redirect handling.
    verb_flag = "" if ep.method == "GET" else f" -X {ep.method}"

    if ep.method == "GET" or (not body_params and not ep.raw_body):
        if target and target["in"] == "path":
            path = _fill_path(path, target["name"], payload)
            url = f"{base}{path}"
        elif target:
            url = f"{base}{path}?{target['name']}={quote(payload, safe='')}"
        else:
            url = f"{base}{path}"
        curl = f"curl -i{verb_flag}{header_flags} {shlex.quote(url)}"
    else:
        url = f"{base}{path}"
        if ep.raw_body and not body_params:
            body_str = payload
            curl = (f"curl -i -X {ep.method}{header_flags} {shlex.quote(url)} "
                    f"--data-binary {shlex.quote(payload)}")
            note_target = "raw request body"
        else:
            field = target["name"] if target else "input"
            body_str = f"{field}={payload}"
            curl = (f"curl -i -X {ep.method}{header_flags} {shlex.quote(url)} "
                    f"--data {shlex.quote(body_str)}")
    note = f"{category} via `{note_target}`"
    return curl, note, headers, body_str


def build_sample_request(finding: Finding, index: Index,
                         file_text: str,
                         file_texts: dict[str, str] | None = None,
                         ctx: Any = None,
                         surface: Any = None) -> dict[str, Any] | None:
    """Build a malicious ``sample_request`` dict for an endpoint finding, or None.

    Deterministic FIRST (direct handler + caller-graph walk). If that fails AND
    ``ctx`` is supplied AND ``poc.llm_route_resolution`` is on, an LLM fallback tries
    to find the route — but its answer is VERIFIED against source before use (an
    unverified/guessed route is discarded). Never emits a fabricated request.
    """
    ep = resolve_endpoint(finding, index, file_text, file_texts)
    if ep is not None:
        derivation = "derived" if (ep.params or ep.raw_body) else "inferred"
    else:
        ep = _resolve_endpoint_llm(
            finding, index, file_texts or {finding.file_rel: file_text}, ctx,
            surface)
        if ep is None:
            return None
        derivation = "llm-verified"

    payload = _payload_for(finding.category)
    curl, note, headers, body = _build_curl(ep, payload, finding.category)
    return {
        "method": ep.method,
        "path": ep.path,
        "params": ep.params,
        "headers": headers,
        "body": body,
        "curl": curl,
        "payload_note": note,
        "derivation": derivation,
        "citation": ep.citation,
    }


# ── LLM route-resolution fallback (verified against source) ─────────────────────

def _verify_route_citation(citation: str, path: str,
                           file_texts: dict[str, str]) -> bool:
    """Confirm the LLM's cited ``file:line`` actually contains the claimed route path
    in the source (a few lines of tolerance). Rejects fabricated citations."""
    if not citation or not path:
        return False
    try:
        file_rel, _, line_s = citation.partition(":")
        line = int(re.findall(r"\d+", line_s)[0])
    except (ValueError, IndexError):
        return False
    text = file_texts.get(file_rel)
    if not text:  # tolerate a slightly different relative path (basename match)
        for fr, t in file_texts.items():
            if fr.endswith(file_rel) or file_rel.endswith(fr):
                text = t
                break
    if not text:
        return False
    lines = text.splitlines()
    window = "\n".join(lines[max(0, line - 4):min(len(lines), line + 3)])
    return path in window  # the exact route path must appear at the cited location


def _resolve_endpoint_llm(finding: Finding, index: Index,
                          file_texts: dict[str, str], ctx: Any,
                          surface: Any = None):
    """Ask a model to find the HTTP route reaching ``finding``, then VERIFY its cited
    route line against source. Returns a verified Endpoint, or None (→ generic sketch).
    Fail-open: missing ctx / disabled flag / error / unverifiable answer → None.
    """
    if ctx is None:
        return None
    settings = getattr(ctx, "settings", None)
    poc_cfg = getattr(settings, "poc", None) if settings else None
    if not poc_cfg or not getattr(poc_cfg, "llm_route_resolution", False):
        return None
    role = getattr(poc_cfg, "route_resolver_role", "trace")
    if role not in getattr(settings, "roles", {}):
        return None

    from .agent import find_last_json
    from .prompts.templates import ROUTE_RESOLVE_PROMPT

    # The attack surface must be handed in explicitly. Reading it off ``ctx`` via
    # getattr never worked: ScanContext has no ``_surface`` field (only the
    # Orchestrator does), so the hints were always empty.
    hints = _llm_source_hints(surface)
    prompt = ROUTE_RESOLVE_PROMPT.format(
        file_rel=finding.file_rel, line_start=finding.line_start,
        line_end=finding.line_end, category=finding.category,
        title=finding.title or finding.category,
        cited_snippet=(finding.cited_snippet or "")[:400], source_hints=hints,
    )
    # This is the slow path: a tool-using agent run PER finding. Announce it before
    # it starts — the orchestrator's per-finding line only prints on completion, so
    # without this a multi-minute agent run looks exactly like a hang.
    print(f"  [sample] {finding.file_rel} — deterministic walk failed; "
          f"asking the model…")
    try:
        agent = ctx.make_agent(role, f"route-{finding.fingerprint()}", with_tools=True)
        result = agent.run(prompt)
        if result.error:
            ctx.audit.event("route_llm_error", error=result.error)
            return None
        parsed = find_last_json(result.tagged("route") or result.final_text)
    except BudgetExceeded:
        # The hard cost ceiling must reach the orchestrator, which stops enriching
        # further findings and writes the partial report. Catching it in the blanket
        # handler below made the orchestrator's own BudgetExceeded handler dead code.
        raise
    except Exception as exc:  # noqa: BLE001 — never break finalize
        try:
            ctx.audit.event("route_llm_exception", error=str(exc))
        except Exception:  # noqa: BLE001
            pass
        return None

    if not isinstance(parsed, dict):
        return None
    path = parsed.get("path")
    if not path or not isinstance(path, str):
        return None  # model admitted it couldn't find one
    citation = str(parsed.get("route_citation", ""))
    if not _verify_route_citation(citation, path, file_texts):
        ctx.audit.event("route_llm_unverified", path=path, citation=citation)
        return None  # discard unverified — fall back to generic sketch

    method = str(parsed.get("method", "GET")).upper()
    if method not in ("GET", "POST", "PUT", "DELETE", "PATCH"):
        method = "GET"
    params: list[dict[str, str]] = []
    body = parsed.get("body")
    if body and "=" in str(body):
        params.append({"name": str(body).split("=", 1)[0], "in": "body", "type": "str"})
    ctx.audit.event("route_llm_verified", path=path, citation=citation,
                    confidence=parsed.get("confidence", ""))
    return Endpoint(method=method, path=path, params=params,
                    raw_body=bool(body) and "=" not in str(body), citation=citation)


def _llm_source_hints(surface: Any, *, max_items: int = 25) -> str:
    if surface is None:
        return "  (no deterministic entry-point hints available)"
    try:
        lines = [f"  {h.label} {h.file_rel}:{h.line}" for h in surface.sources[:max_items]]
    except Exception:  # noqa: BLE001
        return "  (no deterministic entry-point hints available)"
    return "\n".join(lines) if lines else "  (no deterministic sources detected)"
