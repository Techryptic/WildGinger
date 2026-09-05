# Copyright 2026
# SPDX-License-Identifier: Apache-2.0
"""Orchestrator: taskflow, budgets, and pass/convergence loop.

Drives the whole scan:
  discovery → index → attack-surface → deterministic rules → seed ledger →
  preflight → cost confirmation → recon →
  pass loop (hunt → validate → trace → dedupe) until coverage-complete + converged
  or a budget ceiling, then hands findings to the reporter.

Budget ceilings (time / cost / passes) are SAFETY LIMITS, not targets. The normal
exit is "coverage complete + converged". On a ceiling we checkpoint (state is
already durable in SQLite) and exit COVERAGE_INCOMPLETE.

Concurrency: hunters run in a ThreadPoolExecutor; all DB writes funnel through the
single-writer queue, and cells are claimed via leases so two hunters never collide.
"""
from __future__ import annotations

import time
from concurrent.futures import CancelledError, ThreadPoolExecutor, as_completed
from pathlib import Path

from . import attack_surface, coverage, discovery, fileio, indexing, rules, scoring
from .audit import Audit, BudgetExceeded
from .context import ScanContext, build_system_prompt_for
from .db import Database
from .findings import Finding, merge_findings, resolve_symbol, sort_canonical
from .paths import resolve_target
from .sanitizer import Sanitizer
from .settings import Settings
from .stages import dedupe as dedupe_stage
from .stages import hunt as hunt_stage
from .stages import recon as recon_stage
from .stages import trace as trace_stage
from .stages import validate as validate_stage

EXIT_OK = 0
EXIT_FINDINGS = 1
EXIT_COVERAGE_INCOMPLETE = 2
EXIT_FATAL = 3

# How long a claimed cell stays leased to one hunter. Named here because two preflight
# checks compare configuration against it (a per-request timeout longer than the lease, or
# far more hunters than an endpoint's concurrency, both let a lease lapse mid-hunt).
_CELL_LEASE_SECONDS = 600


from dataclasses import dataclass as _dataclass


def _row_int(row, key: str, default: int = 0) -> int:
    """Read an int column from a sqlite3.Row / dict / stub without raising."""
    try:
        value = row[key]
    except (IndexError, KeyError, TypeError):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _row_float(row, key: str, default: float = 0.0) -> float:
    try:
        value = row[key]
    except (IndexError, KeyError, TypeError):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _recon_scopes(focus_areas: list) -> list[str]:
    """Root-relative path prefixes named by recon's focus areas.

    ``scope`` is free text ("app/api", "app/api/users.py", "UserService"), so we keep
    only values that look like a path and normalize them; a bare symbol name matches
    nothing and is harmlessly ignored.
    """
    out: list[str] = []
    for area in focus_areas or []:
        if not isinstance(area, dict):
            continue
        scope = str(area.get("scope") or "").strip().replace("\\", "/")
        while scope.startswith("./"):
            scope = scope[2:]
        scope = scope.strip("/")
        if not scope or ".." in scope:
            continue
        if "/" not in scope and "." not in scope:
            continue  # a bare symbol name, not a path
        if scope not in out:
            out.append(scope)
    return out[:50]


def _error_kind(err: str) -> str:
    """Coarse failure CLASS (no raw detail), for grouping failures by root cause.

    Distinct from :func:`_classify_error`, which embeds the original message for
    display — that makes every error string unique and useless for counting. This
    returns a stable bucket so the systemic-abort heuristic can tell "40 cells hit the
    same auth problem" from "40 cells failed for 12 different reasons".
    """
    low = (err or "").lower()
    if "throttl" in low or "toomanyrequests" in low:
        return "rate-limited by the model API"
    if "accessdenied" in low or "not authorized" in low or "forbidden" in low:
        return "model/AWS access denied (permissions or model access)"
    if "resource" in low and "notfound" in low:
        return "model/profile not found (check the ARN)"
    if "unsupported" in low or "temperature" in low:
        return "model rejected a request parameter"
    if "validationexception" in low or "invalid" in low:
        return "invalid request to the model (check model id/region)"
    if "timeout" in low or "timed out" in low:
        return "model call timed out"
    if "unparseable" in low or "parse" in low:
        return "model reply couldn't be parsed into findings"
    if "unauthorized" in low or "401" in low or "apigee" in low:
        return "gateway auth rejected (check APIGEE_KEY)"
    if "connection" in low or "urlerror" in low or "dns" in low:
        return "network/connection failure"
    return "other"


def _classify_error(err: str) -> str:
    """Turn a raw exception string into a short, human-readable CLI reason."""
    low = (err or "").lower()
    if "throttl" in low or "toomanyrequests" in low or "http 429" in low:
        return f"rate-limited by the model API (auto-retried) — {err[:160]}"
    if "temperature" in low:
        return f"model rejected a parameter — {err[:160]}"
    # OpenAI-compatible / OpenRouter failures that a retry can never fix. Named
    # specifically because each has a different, actionable cause and they are otherwise
    # indistinguishable from a generic 4xx.
    if "insufficient credits" in low or "http 402" in low:
        return (f"the account is out of credits (no retry can help) — "
                f"top it up or switch models — {err[:160]}")
    if "api_key_env" in low or "http 401" in low:
        return (f"the endpoint rejected the API key — check the env var named by "
                f"models.<name>.api_key_env — {err[:160]}")
    if "no endpoint for this model id" in low or "no endpoints found" in low:
        return (f"no provider serves that model id with the parameters requested "
                f"(check the model name, or relax routing.require_parameters) — "
                f"{err[:160]}")
    if "is the endpoint running" in low or "connection refused" in low:
        return (f"nothing is listening on the configured base_url — is the local model "
                f"server running? — {err[:160]}")
    if "accessdenied" in low or "not authorized" in low or "forbidden" in low:
        return f"model/AWS access denied (check permissions/model access) — {err[:160]}"
    if "profilenotfound" in low or "could not be found" in low and "profile" in low:
        return (f"the AWS profile does not exist — set aws.profile: null to use the "
                f"default credential chain (env vars / instance role / Bedrock API key) "
                f"— {err[:160]}")
    if "resource" in low and "notfound" in low:
        return f"model/profile not found (check ARN) — {err[:160]}"
    if "validationexception" in low or "invalid" in low:
        return f"invalid request to the model (check model id/region) — {err[:160]}"
    if "timeout" in low or "timed out" in low:
        return f"model call timed out (will retry) — {err[:160]}"
    if "unparseable" in low or "parse" in low:
        return "model reply couldn't be parsed into findings (re-queued)"
    return err[:200]


@_dataclass
class PassResult:

    """Outcome of one pass — confirmed findings + failure stats for the loop."""
    total: int = 0
    analyzed: int = 0
    shallow: int = 0
    # Of `analyzed`, how many were settled on the shallow-plateau rule: the model read the
    # code and found nothing, repeatedly. Counted as covered (so the loop can converge)
    # but reported separately so "100% coverage" is never overstated.
    thin: int = 0
    failed: int = 0
    confirmed: int = 0
    last_error: str | None = None
    systemic_failure: bool = False



class Budget:
    """Tracks the safety ceilings."""

    def __init__(self, settings: Settings, audit: Audit):
        self.settings = settings
        self.audit = audit
        self.start = time.time()
        self.passes = 0

    def exceeded(self) -> str | None:
        b = self.settings.budget
        if (time.time() - self.start) / 60.0 >= b.max_time_minutes:
            return "max_time"
        if self.audit.cost.cost_usd >= b.max_cost_usd > 0:
            return "max_cost"
        if self.passes >= b.max_passes:
            return "max_passes"
        return None


class Orchestrator:
    def __init__(self, settings: Settings, *, ci: bool = False,
                 limit: int | None = None, assume_yes: bool = False,
                 resume: bool = False, resume_force: bool = False):
        self.settings = settings
        self.ci = ci
        self.limit = limit
        self.assume_yes = assume_yes
        self.resume = resume
        self.resume_force = resume_force
        # Wall-clock start of the whole scan (for the end-of-run Runtime metric).
        self._scan_start = time.time()
        # `scope.path` may name a DIRECTORY (the usual case) or a single FILE. A file
        # roots the scan at its parent — so the index, call graph and attack surface still
        # cover the whole folder and cross-file navigation works — while `_only_file`
        # narrows what is actually hunted. See paths.resolve_target.
        self.root, self._only_file = resolve_target(settings.scope.path)
        out_dir = Path(settings.output.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        self.db = Database(out_dir / "scan_state.db")
        self.db.initialize()
        # Install per-model prices from config so cost is REAL (not $0) and the
        # hard kill-switch can actually fire.
        from . import audit as _audit_mod
        _audit_mod.set_prices({
            mp.model_id: (mp.price_per_1m_input, mp.price_per_1m_output,
                          mp.price_per_1m_cache_read, mp.price_per_1m_cache_write)
            for mp in settings.models.values()
        })
        # MLflow experiment tracking + tracing (monitoring.mlflow.enabled). A
        # disabled/no-op logger is returned when off or if mlflow isn't importable
        # — never breaks a scan. Wired into Audit so it captures EVERY LLM call
        # (Bedrock + GPT). The experiment is named per scanned target (or via the
        # CLI --experiment), and the stage prompt templates are registered for the
        # Prompts page.
        from .monitoring import build_logger
        target_name = self.root.name or "target"
        try:
            from .prompts import templates as _tpl
            _prompts = {
                "recon": _tpl.RECON_PROMPT, "hunt_directed": _tpl.HUNT_DIRECTED_PROMPT,
                "hunt_open": _tpl.HUNT_OPEN_PROMPT, "validate": _tpl.VALIDATE_PROMPT,
                "trace": _tpl.TRACE_PROMPT,
            }
        except Exception:  # noqa: BLE001
            _prompts = None
        self.mlflow = build_logger(settings, target_name=target_name, prompts=_prompts)
        self.mlflow.start()

        self.audit = Audit(
            out_dir,
            restrict_permissions=settings.governance.data_protection.restrict_permissions,
            hard_cost_ceiling=settings.budget.max_cost_usd,
            mlflow=self.mlflow,
        )
        self.findings: list[Finding] = []
        # Identity of THIS scan process, for the single-scan advisory lock.
        import os as _os
        import uuid as _uuid
        self._run_id = f"{_os.getpid()}:{_uuid.uuid4().hex[:8]}"
        self._holds_run_lock = False
        # Set when a --resume finds the scan already complete: finalize then skips every
        # step that would cost money to reproduce an identical report.
        self._no_model_calls_left = False
        # Wall-clock deadline for this run, set once the pass loop starts.
        self._deadline: float | None = None
        # Guards double report-writing when the cost ceiling is hit during finalize.
        self._finalized = False
        # Exit code computed by _finalize_partial, so an aborted run still returns the
        # documented code for what it found instead of a hardcoded 2/3.
        self._partial_exit_code: int | None = None
        # Current pass number, mirrored here so _persist_findings can stamp judge
        # evidence rows with the pass that produced them.
        self._pass_no = 0
        # Set by the SIGINT/SIGTERM handler; the pass loop stops at its next
        # checkpoint so Ctrl-C produces a report instead of losing everything.
        self._interrupted = False
        # Populated during _run_inner once discovery+indexing have run. Initialized
        # here because _finalize_partial() can fire BEFORE that (a cost ceiling or
        # interrupt during recon/preflight), and it reads them.
        self._surface = None
        self._index = None
        self._file_text: dict[str, str] = {}
        self._file_lang: dict[str, str] = {}



    # ── interrupt handling ────────────────────────────────────────────────────
    def _install_signal_handlers(self):
        """Trap Ctrl-C / SIGTERM so an interrupt CHECKPOINTS instead of losing work.

        cli.py's handler calls ``sys.exit(130)``, which raises SystemExit — that
        bypassed the BudgetExceeded handler, so Ctrl-C threw away every finding and
        wrote no report. Here we set a flag and let the pass loop stop cleanly at its
        next checkpoint; a SECOND interrupt exits immediately for the impatient.

        Returns the previous handlers so :meth:`run` can restore them (signal
        handlers are process-global, and only the main thread may install them).
        """
        import signal

        def _handler(signum, _frame):
            if self._interrupted:
                print("\n\u23f9  Second interrupt — cancelling all queued work now "
                      "(in-flight hunters, already paid for, are allowed to finish; "
                      "state is durable in SQLite).")
                raise KeyboardInterrupt
            self._interrupted = True
            self.audit.event("interrupt_requested", signal=int(signum))
            print(f"\n\u23f8  Interrupt received (signal {signum}) — finishing the "
                  f"current work, then writing reports. Press Ctrl-C again to abort "
                  f"now.")

        previous = {}
        for sig in (signal.SIGINT, getattr(signal, "SIGTERM", None)):
            if sig is None:
                continue
            try:
                previous[sig] = signal.signal(sig, _handler)
            except (ValueError, OSError, AttributeError):
                # Not the main thread, or unsupported on this platform — the scan
                # still works, it just won't checkpoint on interrupt.
                continue
        return previous

    @staticmethod
    def _restore_signal_handlers(previous) -> None:
        import signal
        for sig, handler in (previous or {}).items():
            try:
                signal.signal(sig, handler)
            except (ValueError, OSError, AttributeError):
                continue

    # ── main ─────────────────────────────────────────────────────────────────
    def run(self) -> int:
        previous_handlers = self._install_signal_handlers()
        try:
            return self._run_inner()
        except SystemExit as exc:
            # A deliberate, already-explained abort (e.g. --resume refused because the
            # config changed). Honour the code it chose; never turn it into a traceback.
            return int(exc.code) if isinstance(exc.code, int) else EXIT_FATAL
        except BudgetExceeded as exc:
            print(f"\n\U0001F6D1 COST KILL-SWITCH: {exc}")
            # Finalize what we already found (enrich → sample requests → PoCs →
            # reports) so an aborted run still shows curls + reproduction sketches.
            self.audit.event("budget_abort")
            code = self._finalize_partial()
            # Findings outrank the abort reason: if the partial report lists something
            # at/above the threshold, CI must see 1. Hardcoding 3 here meant a
            # report.json full of criticals exited "fatal" and a pipeline that treats
            # 3 as infrastructure noise shipped them.
            return EXIT_FATAL if code in (None, EXIT_OK) else code
        except KeyboardInterrupt:
            # Hard abort (second Ctrl-C, or an interrupt outside the pass loop). Still
            # try to salvage a report — the findings are already in memory.
            print("\n\u23f9  Interrupted — writing a partial report from what we found.")
            self.audit.event("interrupted")
            code = self._finalize_partial()
            return EXIT_COVERAGE_INCOMPLETE if code in (None, EXIT_OK) else code
        except Exception as exc:  # noqa: BLE001
            # ANY other failure — a reporter bug, a full disk, a writer-queue timeout,
            # a sqlite error — used to propagate out of run() → cmd_scan → main() and
            # surface as a traceback with **exit 1**, which is the documented code for
            # "findings at/above threshold". A harness crash was therefore
            # indistinguishable from a real vulnerability in CI. Salvage what we can and
            # return the FATAL code.
            import traceback
            self.audit.event("fatal_error", error=str(exc),
                             error_type=type(exc).__name__,
                             traceback=traceback.format_exc()[-4000:])
            print(f"\n\u274c FATAL: {type(exc).__name__}: {exc}")
            print("   Writing whatever was already found, then exiting 3. Full "
                  "traceback is in the audit log (audit/events.jsonl).")
            code = self._finalize_partial()
            return EXIT_FATAL if code in (None, EXIT_OK) else code
        finally:
            self._restore_signal_handlers(previous_handlers)
            if getattr(self, "_holds_run_lock", False):
                self.db.release_run_lock(self._run_id)
            self.mlflow.end()
            self.audit.close()
            self.db.close()


    def _run_inner(self) -> int:
        s = self.settings
        # `config_file` alongside `config_hash`: the hash proves two runs used the SAME
        # config, but once several files exist (config.yaml, config_api.yaml) only the
        # path answers "which one was this?" — the question asked when two reports differ.
        self.audit.event("scan_start", config_hash=s.config_hash(),
                         config_file=s.config_source,
                         prompts_version=s.versioning.prompts_version)

        # ONE scan per out_dir. Concurrent runs re-seed each other's live cells and pay
        # for the same hunts twice; no per-row guard can undo that.
        ok, holder = self.db.acquire_run_lock(self._run_id)
        if not ok:
            print(f"\n⚠ Another scan appears to be using this output directory "
                  f"({s.output.out_dir}); its lock is held by '{holder}'.\n"
                  f"  Two scans in one out_dir re-seed each other's cells and pay for "
                  f"the same work twice.\n"
                  f"  Wait for it to finish, use a different --output, or (if that run "
                  f"is dead) delete\n"
                  f"  {Path(s.output.out_dir) / 'scan_state.db'} or wait for the lock to "
                  f"expire.")
            self.audit.event("run_lock_denied", holder=holder)
            return EXIT_FATAL
        self._holds_run_lock = True

        # E10: the --resume config guard, moved to the FRONT. It used to live inside
        # _resume_from_checkpoint, which runs after discovery, the full file read,
        # build_index, attack_surface, rules, seeding, seed_ledger (which has already
        # MUTATED the ledger), recon (an LLM call) and preflight (one per model) — so a
        # mistyped config cost real tokens and left the ledger re-seeded before aborting.
        if self.resume and not self.resume_force:
            cp = self.db.get_checkpoint()
            cp_hash = cp["config_hash"] if cp is not None else None
            if cp_hash and cp_hash != s.config_hash():
                print(f"⚠ Resume refused: config changed since checkpoint "
                      f"(was {cp_hash}, now {s.config_hash()}). "
                      f"Re-run with --resume-force to override, or drop --resume for a "
                      f"fresh scan.")
                self.audit.event("resume_refused", was=cp_hash, now=s.config_hash())
                return EXIT_FATAL

        # 1) Discovery
        disc = discovery.discover(self.root, s.scope)
        # Loudly report what was skipped and why (nothing is silently invisible).
        if disc.skipped:
            summary = ", ".join(f"{k}({v})" for k, v in sorted(disc.skipped_reasons.items()))
            self.audit.event("discovery_skips", total=disc.skipped, reasons=disc.skipped_reasons)
            print(f"Discovered {len(disc.files)} file(s); skipped {disc.skipped}: {summary}")
        if not disc.files:
            self.audit.event("nothing_to_scan")
            print("Nothing to scan (no supported files).")
            # Write an empty-but-valid report set. Returning here without writing left a
            # PREVIOUS run's report.json/report.sarif/pocs in out_dir, so a CI job that
            # publishes them after a green exit republished last week's findings as
            # current — and this path is easy to hit with a bad --path or an over-broad
            # exclude_dirs.
            self._write_empty_reports("nothing to scan (no supported files)")
            return EXIT_OK

        # Single-file scope: the target has to be something discovery actually accepted.
        # Without this check a targeted file that was skipped (unsupported language, over
        # scope.max_file_kb, binary) seeded zero cells and the run reported a clean,
        # 100%-covered, exit-0 scan of nothing — a false pass, from a plausible typo.
        if self._only_file is not None:
            discovered_names = {df.rel_path for df in disc.files}
            if self._only_file not in discovered_names:
                reasons = (", ".join(f"{k}: {v}" for k, v in
                                     sorted(disc.skipped_reasons.items()))
                           or "no reason recorded")
                print(f"\n⚠ The file you targeted ({self._only_file}) is not in the scan "
                      f"scope, so there is nothing to analyze.\n"
                      f"  Discovery skipped {disc.skipped} file(s) in {self.root} "
                      f"({reasons}).\n"
                      f"  Common causes: its language is not in scope.include_languages, "
                      f"it is larger than scope.max_file_kb, or its extension is treated "
                      f"as binary.")
                self.audit.event("targeted_file_not_in_scope", file=self._only_file,
                                 skipped=disc.skipped, reasons=disc.skipped_reasons)
                self._write_empty_reports(
                    f"targeted file {self._only_file} was skipped by discovery")
                return EXIT_FATAL
            print(f"Single-file scope: hunting only {self._only_file}. The rest of "
                  f"{self.root.name}/ is still indexed and readable, so cross-file "
                  f"navigation and reachability still work.")
            self.audit.event("single_file_scope", file=self._only_file,
                             indexed_files=len(disc.files))


        # 2) Index + attack surface (+ rules) over file texts
        # DELIBERATELY over every discovered file, even in single-file scope: the symbol
        # index, call graph and attack surface are what make find_definition and
        # reachability work, and narrowing them to one file would make every cross-file
        # lookup miss.
        file_texts: list[tuple[str, str, str]] = []
        for df in disc.files:
            try:
                file_texts.append((df.rel_path, df.language,
                                   fileio.read_text_meta(df.path).text))
            except (OSError, ValueError):
                continue
        index = indexing.build_index(file_texts, backend=s.indexing.backend)
        surface = attack_surface.analyze([(r, t) for (r, _l, t) in file_texts])
        self._surface = surface  # reused by trace (reachability) during passes
        self._index = index      # reused by finalize (sample-request builder)
        # file_rel → text, so finalize can build deterministic sample requests.
        self._file_text: dict[str, str] = {r: t for (r, _l, t) in file_texts}
        # file_rel → language, so finalize can give scoring.enrich() the language it
        # needs for the memory-unsafe-language FP penalty. Without this the
        # `quality.language_aware_fp` switch was configurable but inert: enrich()
        # accepted a `language` argument that no caller ever supplied.
        self._file_lang: dict[str, str] = {r: l for (r, l, _t) in file_texts}


        trust_files = surface.trust_boundary_files()

        # In single-file scope, findings from OTHER files must not reach the report: you
        # asked about one file, and a rule hit or an imported SARIF row in a sibling would
        # gate CI on something you did not request. The sibling is still indexed and
        # readable for context — it just isn't a subject of this scan.
        def _in_scope(file_rel: str) -> bool:
            return self._only_file is None or file_rel == self._only_file

        # Deterministic local rules → seed findings (independent of the LLM).

        if s.rules.enabled:
            try:
                org = rules.load_org_rules(s.rules.org_patterns)
            except Exception as exc:  # noqa: BLE001
                # The deterministic rule pass is a bonus layer; a malformed org-rules
                # file must never abort the whole scan (which, uncaught, exited 1 —
                # the code documented as "vulnerabilities found").
                self.audit.event("org_rules_error", error=str(exc))
                print(f"⚠ Could not load org rules from {s.rules.org_patterns}: {exc}. "
                      f"Continuing with the built-in rules only.")
                org = []
            for hit in rules.run_rules([(r, t) for (r, _l, t) in file_texts], org_rules=org):
                if not _in_scope(hit.file_rel):
                    continue
                self.findings.append(Finding(
                    file_rel=hit.file_rel, line_start=hit.line, line_end=hit.line,
                    category="misconfig" if "config" in hit.rule_id.lower() else "secrets"
                    if "secret" in hit.rule_id.lower() else "logic",
                    title=hit.rule_id, description=hit.description, severity=hit.severity,
                    confidence=0.9, cwe=hit.cwe, cited_snippet=hit.snippet, source="rule",
                    # Same identity stamping as LLM findings, so a rule hit and an LLM
                    # hit on the same bug dedupe against each other.
                    symbol=resolve_symbol(index, hit.file_rel, hit.line),
                ))

        # 2b) Optional hybrid seeding: import Fortify/SARIF findings to triage.
        try:
            from .seeding import load_seeds
            seeds = [s_ for s_ in load_seeds(s) if _in_scope(s_.file_rel)]
            if seeds:
                # Stamp the enclosing symbol here (load_seeds has no index). Without
                # it an imported SARIF finding can never dedupe against the LLM's
                # finding for the SAME bug — which is the entire point of hybrid
                # seeding — because identity keys on file+category+symbol.
                for seed in seeds:
                    if not seed.symbol:
                        seed.symbol = resolve_symbol(index, seed.file_rel,
                                                     seed.line_start)
                self.findings.extend(seeds)
                self.audit.event("seeds_imported", count=len(seeds), mode=s.seeding.mode)
        except Exception as exc:  # noqa: BLE001
            self.audit.event("seed_import_error", error=str(exc))

        # 3) Seed the coverage ledger.
        # `resume` decides whether existing cell progress survives. Without it a
        # repeat scan on the same out_dir kept a completed ledger while restoring NO
        # findings (those only reload under --resume), so it hunted nothing and
        # reported a clean 100%.
        if not self.resume:
            self._warn_if_reusing_ledger()
        # Cells are seeded for the SUBJECT of the scan, which in single-file scope is one
        # file — everything else was indexed for context only and must not be hunted.
        cell_files = ([df for df in disc.files if df.rel_path == self._only_file]
                      if self._only_file is not None else disc.files)
        # `prune_missing`: drop the cells of files that are in the ledger but no longer
        # discovered (deleted/renamed/out of scope). Without it a reused out_dir keeps
        # claiming — and paying full conversations for — cells whose file does not exist,
        # and their never-ANALYZED state pins coverage below 1.0 forever.
        #
        # Disabled in single-file scope: the list passed here is a deliberate subset, so
        # pruning against it would delete the ledger for every OTHER file in the folder —
        # the same reason it is not driven by --limit. (seed_ledger's majority guard would
        # usually catch that, but "usually" is not a safety property.)
        stats = coverage.seed_ledger(self.db, s, cell_files, index, trust_files,
                                     resume=self.resume,
                                     prune_missing=self._only_file is None)
        if stats.files_pruned:
            print(f"» Pruned {stats.files_pruned} file(s) that no longer exist in the "
                  f"scan scope from the ledger (their cells are no longer hunted).")
            self.audit.event("ledger_pruned_missing_files", files=stats.files_pruned)

        classes = coverage.resolve_attack_classes(s)
        open_ended_on = coverage.OPEN_ENDED_CLASS in classes
        groups = list(getattr(s.hunting, "attack_class_groups", None) or [])
        self.audit.event("ledger_seeded", files=stats.files, cells=stats.cells,
                         functions=stats.functions, rule_findings=len(self.findings),
                         attack_classes=classes, open_ended=open_ended_on)
        print(f"Seeded {stats.cells} cells across {stats.files} files "
              f"({stats.functions} functions). Trust-boundary files: {len(trust_files)}.")
        directed = [c for c in classes if c != coverage.OPEN_ENDED_CLASS]
        if groups:
            # One cell per (chunk, GROUP): full called-out coverage at (groups)x
            # cost instead of (classes)x. Spell the groups out — the cell labels in
            # the ledger are comma-joined and unreadable without this legend.
            n_classes = sum(len(g) for g in groups)
            open_note = (" + 1 OPEN-ENDED (business-logic / novel issues)"
                         if open_ended_on else "")
            print(f"  Attack classes: {len(groups)} directed group(s) covering "
                  f"{n_classes} class(es){open_note} = {len(classes)} cell(s) "
                  f"per chunk:")
            for i, g in enumerate(groups, 1):
                print(f"    [{i}] {', '.join(g)}")
            if not open_ended_on:
                # The flat path's advice ("add 'all' to attack_classes") is WRONG in
                # group mode — 'all' cannot join a group; the flag is the switch.
                print(f"  Open-ended hunt OFF (set hunting.open_ended: true to add "
                      f"one open-ended cell per chunk alongside the groups).")
        elif open_ended_on:
            # Say so explicitly. This bucket used to be silently discarded, so a user who
            # had asked for it in `attack_classes` was quietly not getting it — and now
            # that it works, its cost should not be a surprise either.
            print(f"  Attack classes: {len(directed)} directed + 1 OPEN-ENDED "
                  f"(business-logic / novel issues) = {len(classes)} cells per chunk.")
        else:
            print(f"  Attack classes: {len(directed)} directed, open-ended hunt OFF "
                  f"(add '{coverage.OPEN_ENDED_CLASS}' to hunting.attack_classes to "
                  f"enable it).")

        # 4) Build the shared context
        ctx = ScanContext(
            settings=s, root=self.root, db=self.db, audit=self.audit, index=index,
            sanitizer=Sanitizer(mode=s.sanitization.mode,
                                 entropy_threshold=s.sanitization.entropy_threshold,
                                 quarantine_extensions=tuple(s.sanitization.quarantine_extensions),
                                 audit=self.audit),
            system_prompt=build_system_prompt_for(s),
        )

        # 5a) Preflight: one small call per configured model so a bad ARN / gateway / key
        # fails in seconds — not after hunting thousands of cells.
        #
        # This runs BEFORE recon and before the cost prompt, and that ordering is the
        # point. Recon is a tool-using agent with its own turn budget (max_turns: 120 in
        # the shipped config), so running it first meant a misconfigured model was paid
        # for at full agent price and only THEN refused — and worse, declining the cost
        # prompt below printed "no cost incurred" after recon had already spent. Nothing
        # here needs recon's output: it consumes `ctx`, which is built above, and recon's
        # weight boosts are consumed by the pass loop further down.
        if getattr(s, "preflight", None) and s.preflight.enabled:
            # One tiny call per distinct model — normally seconds, but a hung or
            # rate-limited endpoint would otherwise stall here with no output.
            print("» Preflight: probing each configured model (one tiny call per "
                  "distinct model)…")
            t_preflight = time.time()
            ok, msg = self._preflight(ctx)
            if not ok:
                print(f"\n⚠ Preflight failed — not starting the scan:\n    {msg}\n"
                      f"Fix the model/config (often a bad ARN or gateway) and re-run.")
                self.audit.event("preflight_failed", error=msg)
                return EXIT_FATAL
            print(f"Preflight OK — all configured models responded "
                  f"({int(time.time() - t_preflight)}s).")

        # 5c) Up-front cost estimate + confirmation (skipped with --yes / --ci).
        # Prices the ledger we ALREADY built. Calling estimate() here re-ran discovery,
        # re-read every file, rebuilt the index and re-chunked everything just to
        # approximate `stats.cells`, which we know exactly.
        projected = None
        try:
            from .estimate import price_known_cells
            est = price_known_cells(
                s, cells=stats.cells, files=stats.files, skipped=disc.skipped,
                total_lines=sum(df.line_count for df in disc.files), quiet=True)
            projected = est.get("est_cost_usd")
        except Exception as exc:  # noqa: BLE001
            # Swallowing this into 0.0 printed "Estimated cost: ~$0.0" and then asked the
            # user to confirm against a number we had just failed to compute.
            self.audit.event("estimate_failed", error=str(exc))
        cap = s.budget.max_cost_usd
        if projected is None:
            print(f"Estimated cost: UNAVAILABLE (projection failed — see the audit log). "
                  f"The hard ceiling of ${cap} still applies.")
        else:
            print(f"Estimated cost: ~${projected} (hard ceiling ${cap}).")
        if not self.assume_yes:
            try:
                ans = input(f"Proceed with the scan (hard stop at ${cap})? [y/N] ").strip().lower()
            except EOFError:
                ans = "n"
            if ans not in ("y", "yes"):
                # True again now that preflight is the only model call before this point:
                # its handful of probe tokens are the entire spend, and recon — the
                # expensive part — has not run yet.
                spent = self.audit.cost.snapshot()
                print(f"Aborted at the cost prompt. Spend so far: "
                      f"${spent['cost_usd']} (preflight probes only).")
                self._write_empty_reports("aborted at the cost-estimate prompt")
                return EXIT_OK

        # 5d) Recon (best-effort; non-fatal). Deliberately AFTER preflight and after the
        # cost confirmation: it is the first expensive model call of a run, and it must not
        # happen until we know the models work and the operator has agreed to the bill.
        # One tool-using agent with a real turn budget: on a slow upstream (or in retry
        # backoff) it can run for minutes with nothing on screen. Announce it BEFORE
        # the call, or that wait looks exactly like a hang.
        print("» Recon: asking the model to map the codebase + entry points…")
        t_recon = time.time()
        try:
            recon = recon_stage.run_recon(ctx, surface)
            recon_elapsed = int(time.time() - t_recon)
            self.audit.event("recon_done", focus_areas=len(recon.focus_areas),
                             error=recon.error)
            if recon.error:
                # Previously audit-only, so a recon failure was invisible on the console
                # even though we had just paid for the attempt.
                print(f"⚠ Recon did not complete ({_classify_error(recon.error)}, "
                      f"after {recon_elapsed}s); continuing with deterministic "
                      f"weights only.")
            # CONSUME the result. Cells are claimed in weight DESC order, so boosting the
            # files recon flagged makes the hunt actually start where recon said to look.
            prefixes = _recon_scopes(recon.focus_areas)
            if prefixes:
                boosted = self.db.boost_cell_weights(prefixes, delta=1.0)
                print(f"» Recon: {len(recon.focus_areas)} focus area(s) → prioritized "
                      f"{boosted} cell(s) ({recon_elapsed}s).")
                self.audit.event("recon_weights_applied", prefixes=prefixes[:25],
                                 cells_boosted=boosted)
            elif not recon.error:
                # The start line must ALWAYS get a matching completion — an empty
                # answer is not the call still running.
                print(f"» Recon: no focus areas returned ({recon_elapsed}s); "
                      f"hunting in ledger order.")
        except BudgetExceeded:
            raise  # cost ceiling is never "best-effort" — stop the scan
        except Exception as exc:  # noqa: BLE001
            self.audit.event("recon_failed", error=str(exc))

        # 5b) Reload persisted findings so the report describes the whole ledger, not
        # just the cells this invocation happened to hunt.
        #
        # Two situations need this, and only the first used to be handled:
        #   * ``--resume`` — continue a crashed run without re-judging what we already
        #     confirmed. Cell states already persist, so hunting naturally skips
        #     ANALYZED cells; this restores the findings side.
        #   * ``scope.mode: incremental`` — ``seed_ledger`` skips files whose sha256 is
        #     unchanged, which leaves their cells ANALYZED. With the findings NOT
        #     reloaded, a repeat scan of an unchanged repo hunted nothing, restored
        #     nothing, reported ``coverage_pct: 1.0`` / ``coverage_complete: true``,
        #     OVERWROTE the previous good report.json with an empty one, and exited 0.
        #     That is the exact false-clean failure the fresh-scan ledger reset exists
        #     to prevent — it just never covered the incremental path.
        if getattr(stats, "files_changed_on_resume", 0):
            print(f"↻ {stats.files_changed_on_resume} file(s) changed since the run being "
                  f"resumed; their cells were reset so the new code is actually analyzed.")
            self.audit.event("resume_files_changed",
                             count=stats.files_changed_on_resume)
        incremental = s.scope.mode == "incremental"
        if self.resume:
            self._resume_from_checkpoint(s)
        elif incremental:
            restored = self._reload_persisted_findings()
            print(f"Incremental mode: restored {restored} finding(s) from previous "
                  f"runs (unchanged files keep their analysis).")
            self.audit.event("incremental_findings_reloaded", restored=restored)

        # 6) Pass loop
        self._t_start = time.time()

        print(f"Starting analysis: {stats.cells} cells, {s.concurrency.hunters} parallel "
              f"hunter(s). This can take a while — live events also stream to "
              f"{Path(s.output.out_dir) / 'audit' / 'events.jsonl'}.")
        budget = Budget(s, self.audit)
        # Make the wall-clock limit enforceable IN FLIGHT, not just between passes: every
        # provider call now checks it through the same gate as the cost ceiling.
        self._deadline = budget.start + s.budget.max_time_minutes * 60.0
        self.audit.set_deadline(self._deadline)
        converged_passes = 0
        while True:
            if self._interrupted:
                print("⏹  Stopping at a clean checkpoint (interrupted). "
                      "Re-run with --resume to continue where this left off.")
                self.audit.event("interrupt_checkpoint", pass_no=budget.passes)
                break
            reason = budget.exceeded()
            if reason:
                self.audit.event("budget_ceiling", reason=reason)
                if reason == "max_cost":
                    print(f"Cost ceiling reached (${self.settings.budget.max_cost_usd}); "
                          f"checkpointing.")
                elif reason == "max_passes":
                    print(f"Pass limit reached (max_passes={self.settings.budget.max_passes}); "
                          f"stopping — NOT a cost limit. Raise budget.max_passes to sweep more.")
                elif reason == "max_time":
                    print(f"Time limit reached (max_time_minutes="
                          f"{self.settings.budget.max_time_minutes}); checkpointing.")
                else:
                    print(f"Safety ceiling reached ({reason}); checkpointing.")
                break

            budget.passes += 1
            self._pass_no = budget.passes
            self.db.renew_run_lock(self._run_id)
            self.db.set_checkpoint("hunting", pass_no=budget.passes,
                                   config_hash=s.config_hash())
            pr = self._run_pass(ctx, budget.passes)
            # Persist findings after EVERY pass so a late crash never loses the
            # expensive confirmed findings + judge votes (crash-resume, item 7).
            self._persist_findings()
            self.db.set_checkpoint("validated", pass_no=budget.passes,
                                   config_hash=s.config_hash())
            cov = self.db.coverage_pct()
            self.audit.event("pass_complete", pass_no=budget.passes,
                             coverage_pct=round(cov, 4), new_confirmed=pr.confirmed,
                             analyzed=pr.analyzed, thin=pr.thin, shallow=pr.shallow,
                             failed=pr.failed)

            # Systemic failure → stop immediately (error already printed).
            if pr.systemic_failure:
                self.audit.event("systemic_failure", error=pr.last_error)
                code = self._finalize(ctx)
                # Same rule as the other abort paths: a report that lists reportable
                # findings must exit 1, not the abort's own code.
                return EXIT_FATAL if code in (None, EXIT_OK) else code

            # Per-pass summary — surface failures loudly instead of a silent "0".
            thin_note = f" ({pr.thin} thin)" if pr.thin else ""
            line = (f"Pass {budget.passes}: coverage {cov*100:.1f}% · "
                    f"{pr.analyzed} analyzed{thin_note}, {pr.shallow} shallow, "
                    f"{pr.failed} failed · {pr.confirmed} new confirmed finding(s).")
            if pr.failed and pr.analyzed == 0:
                line += f"\n  ⚠ ALL cells failed this pass — last error: {pr.last_error}"
            print(line)

            if pr.confirmed == 0:
                converged_passes += 1
            else:
                converged_passes = 0
            coverage_complete = cov >= 1.0
            converged = converged_passes >= s.budget.convergence_passes
            if coverage_complete and converged:
                self.audit.event("converged", passes=budget.passes)
                break
            # If nothing is left to claim (no PENDING/SHALLOW cells), stop too.
            if pr.total == 0:
                if budget.passes == 1 and self.resume:
                    # A resume of a COMPLETED scan. Re-finalizing would re-run the
                    # whole-set semantic dedupe and per-finding LLM route resolution —
                    # hundreds of dollars to reproduce a byte-identical report.
                    # Report what is already known and stop.
                    print("Nothing left to hunt — this scan is already complete. "
                          "Reporting the stored findings without re-running any model.")
                    self.audit.event("resume_already_complete",
                                     findings=len(self.findings))
                    self._no_model_calls_left = True
                break

        return self._finalize(ctx)


    # ── one pass ──────────────────────────────────────────────────────────────
    def _run_pass(self, ctx: ScanContext, pass_no: int) -> "PassResult":
        """Claim and hunt all available cells in bounded parallel; validate + trace
        new findings. Returns a PassResult (confirmed count + failure stats so the
        loop can surface errors and abort early on systemic failure)."""
        s = self.settings
        cells = self._claim_all_cells(ctx)
        result = PassResult(total=len(cells))
        if not cells:
            return result

        import threading as _threading
        done_lock = _threading.Lock()
        progress = {"done": 0, "findings": 0}
        total = len(cells)
        heartbeat_every = max(1, min(25, total // 20 or 1))

        pass_findings: list[Finding] = []

        def _heartbeat(label: str, file_rel: str, n_find: int, detail: str = "") -> None:
            with done_lock:
                progress["done"] += 1
                progress["findings"] += n_find
                done = progress["done"]
            # Per-cell line: status + (for failures) the categorized reason so the
            # CLI explains WHY a cell didn't complete, not just "Error".
            extra = f" ({n_find} finding(s))" if n_find else ""
            if detail:
                extra += f" — {_classify_error(detail)}"
            print(f"  [hunt {done}/{total}] {file_rel} {label}{extra}")

            if done % heartbeat_every == 0 or done == total:
                cost = self.audit.cost.snapshot()
                elapsed = int(time.time() - self._t_start)
                print(f"    … {done}/{total} cells "
                      f"({done/total*100:.1f}%) · {progress['findings']} raw finding(s) "
                      f"· ~${cost['cost_usd']} · {elapsed//60}m{elapsed%60:02d}s")
                # Renew the out_dir's advisory lock HERE, not just at the pass boundary.
                # Its TTL is 30 minutes (db._RUN_LOCK_TTL_SECONDS) but one pass over a
                # large ledger routinely runs far longer — budget.max_time_minutes alone
                # allows 120 — so for most of a long scan the out_dir was effectively
                # unlocked. A CI retry or a second operator then passes acquire_run_lock,
                # and its seed_ledger(resume=False) resets THIS run's live IN_PROGRESS
                # cells to PENDING with lease_owner = NULL; the per-cell owner guard
                # cannot help because the reset clears the owner it would compare
                # against, so every in-flight cell is paid for twice. Piggy-backing on
                # the throttled heartbeat keeps this to one extra write per
                # `heartbeat_every` cells, and renew_run_lock swallows its own errors, so
                # it can never affect a hunt.
                self.db.renew_run_lock(self._run_id)

        budget_exceeded: BudgetExceeded | None = None
        hard_interrupt: BaseException | None = None
        handled: set = set()   # futures already folded into result/pass_findings
        interrupted_mid_pass = False
        # Failure classification → count, so the systemic-abort heuristic can check
        # that failures really do share a cause instead of merely counting them.
        error_kinds: dict[str, int] = {}
        mixed_failures_warned = False
        # An explicit pool (not `with`) because ThreadPoolExecutor.__exit__ calls
        # shutdown(wait=True) WITHOUT cancel_futures, i.e. it DRAINS the work queue.
        # On a hard interrupt that meant the "exiting immediately" message was a lie:
        # every remaining cell in the pass was still hunted, at full token cost, after
        # the user had asked twice to stop. We must own the shutdown to pass
        # cancel_futures=True.
        pool = ThreadPoolExecutor(max_workers=s.concurrency.hunters)
        futures: dict = {}
        try:
            # Inside the try so that a failure while SUBMITTING (e.g. the OS refuses a
            # thread) still reaches the shutdown in the `finally` — the `with` form
            # used to guarantee that.
            futures = {pool.submit(self._hunt_one, ctx, cell): cell for cell in cells}
            # KeyboardInterrupt is caught around the WHOLE loop, not just around
            # fut.result(): as_completed() is where the main thread actually blocks
            # (result() on an already-completed future returns immediately), so a
            # per-future handler almost never sees the signal.
            try:
                for fut in as_completed(futures):
                    cell = futures[fut]
                    file_rel = self._file_rel_for(cell["file_id"])
                    handled.add(fut)
                    try:
                        outcome = fut.result()
                    except BudgetExceeded as exc:
                        # HARD COST CEILING: never treat this as an ordinary cell failure.
                        # Release the cell (so a resume re-hunts it), cancel everything not
                        # yet started, and re-raise after the pool drains so run() can
                        # checkpoint + write partial reports.
                        self.db.set_cell_state(cell["id"], "PENDING")
                        budget_exceeded = exc
                        cancelled = self._cancel_pending(futures)
                        self.audit.event("budget_kill_switch_abort", stage="hunt",
                                         cancelled_cells=cancelled)
                        print(f"\n🛑 COST CEILING hit mid-hunt — cancelled {cancelled} "
                              f"not-yet-started cell(s). Up to {s.concurrency.hunters - 1} "
                              f"already-running hunter(s) will finish first, so a little "
                              f"more spend is still committed past the ceiling.")
                        break
                    except KeyboardInterrupt:
                        # The signal landed inside result(): this cell's outcome is lost,
                        # so release it for a resume. ALL other salvage is done once, by
                        # the outer handler below, after the pool has joined.
                        self.db.set_cell_state(cell["id"], "PENDING")
                        raise
                    except CancelledError:
                        # WE cancelled this cell (budget ceiling / systemic abort /
                        # interrupt). It was never attempted, so put it straight back to
                        # PENDING rather than FAILED — PENDING costs it no attempt budget,
                        # whereas FAILED is now retryable but capped at max_passes tries.
                        self.db.set_cell_state(cell["id"], "PENDING")
                        continue
                    except Exception as exc:  # noqa: BLE001
                        self.db.set_cell_state(cell["id"], "FAILED", last_error=str(exc))
                        result.failed += 1
                        result.last_error = str(exc)
                        kind = _error_kind(str(exc))
                        error_kinds[kind] = error_kinds.get(kind, 0) + 1
                        _heartbeat("✗ FAILED", file_rel, 0, detail=str(exc))
                        # Check the systemic-abort heuristic HERE too. It used to live only
                        # after the success path, below a `continue` — so when every cell
                        # failed (exactly the case it exists for) it never ran at all.
                        if self._check_systemic_failure(result, error_kinds, pass_no,
                                                        mixed_warned=mixed_failures_warned):
                            mixed_failures_warned = True
                            if result.systemic_failure:
                                self._cancel_pending(futures)
                                break
                        continue

                    # Cell state transition based on thoroughness / errors.
                    if outcome.error and not outcome.analyzed:
                        state = "SHALLOW" if outcome.thoroughness > 0 else "FAILED"
                        self.db.set_cell_state(cell["id"], state,
                                               thoroughness=outcome.thoroughness,
                                               last_error=outcome.error)
                        if state == "FAILED":
                            result.failed += 1
                            result.last_error = outcome.error
                        else:
                            result.shallow += 1
                        _heartbeat(f"⚠ {state}", file_rel, len(outcome.findings))
                    elif outcome.analyzed:
                        # ATOMIC: the findings land in the same transaction as the
                        # ANALYZED flip, so a crash can never leave a durably-analyzed
                        # cell whose findings were lost (resume preserves ANALYZED, so
                        # they would never have been re-hunted).
                        self._settle_owned(
                            cell, "ANALYZED", thoroughness=outcome.thoroughness,
                            also=self._cell_findings_writer(outcome.findings))
                        coverage.mark_functions_visited(self.db, cell["file_id"])
                        result.analyzed += 1
                        _heartbeat("✓ analyzed", file_rel, len(outcome.findings))
                    elif outcome.findings:
                        # Below the thoroughness bar but it produced VERIFIED findings
                        # (citations already checked in hunt) → count as analyzed so a
                        # capable-but-terse model doesn't tank coverage to 0%.
                        self._settle_owned(
                            cell, "ANALYZED", thoroughness=outcome.thoroughness,
                            also=self._cell_findings_writer(outcome.findings))
                        coverage.mark_functions_visited(self.db, cell["file_id"])
                        result.analyzed += 1
                        _heartbeat("✓ analyzed (findings)", file_rel, len(outcome.findings))
                    elif self._is_shallow_plateau(cell, outcome):
                        # PLATEAU: this cell has now been hunted `max_shallow_attempts`
                        # times without its thoroughness improving, and it DID read code
                        # (score > 0) — it just found nothing. Settle it as analyzed so
                        # (a) we stop re-paying for the same answer and (b) coverage can
                        # reach 1.0, which is the only way the pass loop can converge. Its
                        # low thoroughness stays on the row, so the run summary and
                        # report still declare it as thin coverage rather than pretending
                        # it got a full look.
                        self._settle_owned(
                            cell, "ANALYZED", thoroughness=outcome.thoroughness,
                            also=self._cell_findings_writer(outcome.findings))
                        coverage.mark_functions_visited(self.db, cell["file_id"])
                        result.analyzed += 1
                        result.thin += 1
                        self.audit.event("cell_thin_settled", cell_id=cell["id"],
                                         attempts=_row_int(cell, "attempts"),
                                         thoroughness=outcome.thoroughness)
                        _heartbeat("✓ analyzed (thin)", file_rel,
                                   len(outcome.findings))
                    else:
                        self.db.set_cell_state(cell["id"], "SHALLOW",
                                               thoroughness=outcome.thoroughness)
                        result.shallow += 1
                        _heartbeat("⚠ SHALLOW", file_rel, len(outcome.findings))
                    pass_findings.extend(outcome.findings)

                    # Early-abort check also runs on the success path, so a scan that is
                    # mostly-but-not-entirely failing still trips it.
                    if self._check_systemic_failure(result, error_kinds, pass_no,
                                                    mixed_warned=mixed_failures_warned):
                        mixed_failures_warned = True
                        if result.systemic_failure:
                            self._cancel_pending(futures)
                            break

                    # Wall-clock ceiling reached mid-pass: stop scheduling new cells and
                    # let the in-flight ones land. Checking only between passes meant a
                    # single pass could overrun `max_time_minutes` by hours.
                    deadline = getattr(self, "_deadline", None)
                    if (not interrupted_mid_pass and deadline is not None
                            and time.time() >= deadline):
                        interrupted_mid_pass = True
                        cancelled = self._cancel_pending(futures)
                        self.audit.event("time_ceiling_mid_pass",
                                         cancelled_cells=cancelled)
                        print(f"⏱  Wall-clock limit reached mid-pass "
                              f"(budget.max_time_minutes): cancelled {cancelled} "
                              f"not-yet-started cell(s); finishing in-flight hunters, "
                              f"then reporting.")

                    # Interrupted (first Ctrl-C): stop scheduling NEW cells but let the
                    # in-flight ones finish, so the pass still validates + reports what it
                    # already paid for.
                    if self._interrupted and not interrupted_mid_pass:
                        interrupted_mid_pass = True
                        cancelled = self._cancel_pending(futures)
                        self.audit.event("interrupt_mid_pass", cancelled_cells=cancelled)
                        print(f"⏸  Interrupt: cancelled {cancelled} not-yet-started cell(s); "
                              f"finishing in-flight hunters, then validating what we have.")
            except KeyboardInterrupt as exc:
                # HARD ABORT. Cancel every not-yet-started cell BEFORE the shutdown in
                # the `finally` below, so it has nothing left to drain. Salvage happens
                # after the join (see `hard_interrupt` below) because only then is
                # `fut.cancelled()`/`fut.done()` a stable answer and no hunter is still
                # writing to the cell rows we are about to settle.
                hard_interrupt = exc
                cancelled = self._cancel_pending(futures)
                self.audit.event("interrupt_hard_abort", cancelled_cells=cancelled)
                print(f"\n⏹  Hard interrupt: cancelled {cancelled} not-yet-started "
                      f"cell(s); waiting for up to {s.concurrency.hunters} in-flight "
                      f"hunter(s) already paid for, then writing a partial report.")
        finally:
            # cancel_futures=True is the whole point of owning the pool: without it
            # shutdown() executes every queued work item before returning.
            pool.shutdown(wait=True, cancel_futures=True)

        # Past the join: every hunter has stopped, so cancelled()/done() are stable.
        if hard_interrupt is not None:
            pass_findings.extend(self._drain_completed(futures, handled))
            self._release_unfinished(futures)
            self.findings.extend(pass_findings)
            raise hard_interrupt

        # Cost ceiling reached inside a worker. We are past the pool's
        # shutdown(wait=True) here, so every in-flight hunter has finished — their
        # spend was already committed and cannot be recalled, which is why the ceiling
        # is an "stop scheduling" boundary rather than a hard instant cutoff. Salvage
        # everything we paid for, then propagate so no FURTHER work is scheduled.
        # run()'s handler turns these into a partial report.
        if budget_exceeded is not None:
            pass_findings.extend(self._drain_completed(futures, handled))
            self._release_unfinished(futures)
            self.findings.extend(pass_findings)
            raise budget_exceeded

        if result.systemic_failure:
            # Salvage here too. Cells that FINISHED but whose futures we never
            # processed (we broke out of as_completed) previously had their results
            # discarded AND their rows left IN_PROGRESS until the 10-minute lease
            # expired — paid-for work thrown away twice over.
            pass_findings.extend(self._drain_completed(futures, handled))
            self._release_unfinished(futures)
            self.findings.extend(pass_findings)
            return result

        # ── NOISE CONTROL: collapse duplicates BEFORE the expensive validation ──
        # This is the fix for "6000 raw findings": 20 near-identical detections on
        # the same endpoint/attack class merge into one before we spend judge calls.
        from .findings import merge_by_location
        raw_n = len(pass_findings)
        if self.settings.dedupe.merge_same_location:
            pass_findings = merge_by_location(pass_findings)
        else:
            pass_findings = merge_findings(pass_findings)
        heuristic_n = len(pass_findings)
        print(f"» Dedupe: {raw_n} raw → {heuristic_n} after heuristic merge", end="")

        # ── AI SEMANTIC DEDUPE (BEFORE the judges) ─────────────────────────────
        # Catch same-root-cause duplicates the heuristic misses so we never spend
        # judge tokens on them. Fail-open (returns input unchanged on any error).
        if self.settings.dedupe.semantic:
            pass_findings, removed = dedupe_stage.semantic_dedupe(
                ctx, pass_findings, role=self.settings.dedupe.semantic_role)
            print(f" → {len(pass_findings)} after AI dedupe (−{removed})")
        else:
            print()

        # Severity gate: skip the multi-judge quorum for low-value candidates.
        to_validate, skipped = self._apply_severity_gate(pass_findings)
        print(f"» Severity gate: {len(to_validate)} to validate, "
              f"{len(skipped)} below gate (kept as-is)")

        # ── REACHABILITY GATE (BEFORE the expensive judges) ────────────────────
        # Deterministically walk the call graph: findings that can't be reached
        # from any external entry point are demoted to informational and kept OUT
        # of the multi-judge queue, so we never spend judge money on a bug an
        # attacker can't trigger.
        demoted: list[Finding] = []
        if self.settings.reachability.enabled:
            to_validate, demoted = trace_stage.gate_unreachable(
                ctx, to_validate, self._surface)
            print(f"» Reachability gate: {len(demoted)} demoted to informational "
                  f"(not judged), {len(to_validate)} proceed to judges")
        if raw_n:
            print(f"» Validating {len(to_validate)} unique finding(s) with the judge "
                  f"panel (deduped from {raw_n} raw)…")

        # Validate (multi-judge quorum, mixed vendors) + trace (reachability),
        # in parallel with a live progress heartbeat. Trace uses the REAL attack
        # surface computed at scan start.
        #
        # A cost-ceiling abort inside the judge pool must still hand back everything
        # we found (_validate_and_trace leaves `to_validate` fully populated), so the
        # collection below happens whether or not it raised.
        try:
            confirmed = self._validate_and_trace(ctx, to_validate)
        finally:
            self.findings.extend(to_validate)
            self.findings.extend(skipped)
            self.findings.extend(demoted)
            # Dedupe across everything found so far (fingerprint-level). Inside the
            # `finally` so an aborted pass still reports a deduped set.
            self.findings = dedupe_stage.dedupe(self.findings)

        result.confirmed = confirmed
        return result

    def _apply_severity_gate(self, findings: list[Finding]):
        """Split findings into (validate, skip) based on skip_validation_below."""
        gate = self.settings.validation.skip_validation_below
        order = {"none": 0, "low": 1, "medium": 2, "high": 3}
        bar = order.get(gate, 1)
        # 'informational' must rank BELOW low. It was missing from this map, so
        # `.get(sev, 2)` scored it as medium and it sailed into the 3-judge panel —
        # paying full judge price for findings that passes_threshold() then drops
        # from the report entirely.
        sev_rank = {"informational": 0, "low": 1, "medium": 2, "high": 3, "critical": 4}
        to_validate, skipped = [], []
        for f in findings:
            # rule/seed findings are already high-confidence deterministic → validate.
            if bar and f.source == "llm" and sev_rank.get(f.severity, 2) < bar:
                skipped.append(f)
            else:
                to_validate.append(f)
        return to_validate, skipped

    def _validate_and_trace(self, ctx: ScanContext, findings: list[Finding]) -> int:
        """Run multi-judge validation + reachability trace, in parallel, with a
        live progress heartbeat + ETA + running cost (kills the old silent wait)."""
        import threading as _threading
        surface = self._surface
        total = len(findings)
        if total == 0:
            return 0
        confirmed = 0
        done = 0
        lock = _threading.Lock()
        t0 = time.time()

        def _one(f: Finding) -> Finding:
            f = validate_stage.validate_finding_quorum(ctx, f)
            if f.status == "confirmed":
                f = trace_stage.trace_reachability(ctx, f, surface)
            return f

        def _tick(f: Finding) -> None:
            nonlocal done, confirmed
            with lock:
                done += 1
                if f.status == "confirmed":
                    confirmed += 1
                d = done
            elapsed = time.time() - t0
            rate = d / elapsed if elapsed > 0 else 0
            eta = int((total - d) / rate) if rate > 0 else 0
            cost = self.audit.cost.snapshot()
            print(f"  [validate {d}/{total}] {f.file_rel} — {f.status} "
                  f"(votes {f.votes}, {f.reachable}) · ~${cost['cost_usd']} "
                  f"· ETA ~{eta//60}m{eta%60:02d}s")

        workers = self.settings.concurrency.hunters if self.settings.validation.parallel else 1
        results: list[Finding] = []
        budget_exceeded: BudgetExceeded | None = None
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            futs = {pool.submit(_one, f): f for f in findings}
            for fut in as_completed(futs):
                orig = futs[fut]
                try:
                    out = fut.result()
                except BudgetExceeded as exc:
                    # Cost ceiling during judging: stop scheduling new judge calls and
                    # keep the un-judged finding as-is (fail-open, never drop).
                    budget_exceeded = exc
                    cancelled = self._cancel_pending(futs)
                    self.audit.event("budget_kill_switch_abort", stage="validate",
                                     cancelled_findings=cancelled)
                    print(f"\n🛑 COST CEILING hit mid-validation — cancelled {cancelled} "
                          f"pending judge task(s); finishing in-flight work.")
                    results.append(orig)
                    break
                except CancelledError:
                    # We cancelled this judge task; keep the finding un-judged rather
                    # than logging a spurious validation error.
                    results.append(orig)
                    continue
                except Exception as exc:  # noqa: BLE001
                    self.audit.event("validate_error", error=str(exc))
                    out = orig
                results.append(out)
                _tick(out)

        if budget_exceeded is not None:
            # Preserve every finding: those already validated, plus the ones whose
            # judging never ran (still carrying their pre-validation status).
            done_ids = {id(f) for f in results}
            results.extend(f for f in findings if id(f) not in done_ids)
            findings[:] = sort_canonical(results)
            raise budget_exceeded

        # Replace list contents in place so callers see validated objects, in a
        # deterministic order (as_completed order varies run-to-run).
        findings[:] = sort_canonical(results)
        return confirmed


    def _file_rel_for(self, file_id: int) -> str:
        def _q(cur):
            cur.execute("SELECT path FROM files WHERE id = ?;", (file_id,))
            row = cur.fetchone()
            return row["path"] if row else f"file#{file_id}"
        try:
            return self.db.read(_q)
        except Exception:  # noqa: BLE001
            return f"file#{file_id}"


    def _preflight(self, ctx: ScanContext) -> tuple[bool, str]:
        """One tiny call per DISTINCT model used by any role — fail fast on a bad
        ARN / gateway error instead of after hunting thousands of cells."""
        from .providers.base import Message
        s = self.settings
        # Refuse BEFORE spending anything if the hard cost ceiling cannot work. An
        # unpriced model contributes $0.00 to the ledger, so `cost >= max_cost_usd` is
        # never true and the documented "HARD spend stop" is simply absent — the run
        # would proceed unbounded until the time/pass limits.
        unpriced = s.unpriced_role_models()
        if unpriced:
            return False, (
                f"models {unpriced} are used by a role but have no pricing, so the "
                f"hard cost ceiling (${s.budget.max_cost_usd}) could never fire and the "
                f"scan would be effectively unbounded. Set price_per_1m_input / "
                f"price_per_1m_output on each (a rough figure is fine — it only has to "
                f"be non-zero).")
        for role_a, role_b, model in s.duplicate_judge_models():
            print(f"⚠ validation.judges '{role_a}' and '{role_b}' both resolve to model "
                  f"'{model}', so the quorum is NOT a cross-model vote — correlated "
                  f"errors from that one model will look like agreement.")
            self.audit.event("judges_share_a_model", roles=[role_a, role_b], model=model)
        # A `billing: free` model is exempt from the refusal above because $0 is the truth
        # for it, not missing data. Say so out loud: for those models the documented "HARD
        # spend stop" does not apply, and only budget.max_time_minutes bounds the run.
        free_models = s.free_billing_models()
        if free_models:
            print(f"ℹ models {free_models} are declared `billing: free`, so the cost "
                  f"ceiling (${s.budget.max_cost_usd}) cannot bound them.\n"
                  f"  Only budget.max_time_minutes "
                  f"({s.budget.max_time_minutes} min) limits work on those models.")
            self.audit.event("free_billing_models", models=free_models,
                             max_time_minutes=s.budget.max_time_minutes)
        # Record each configured HTTP destination and its transport protection.
        from .settings import endpoint_transport
        for name, mp in s.models.items():
            if not mp.base_url:
                continue
            transport = endpoint_transport(mp.base_url)
            self.audit.event("custom_model_endpoint", model=name,
                             provider=mp.provider, endpoint=mp.base_url,
                             transport=transport)
            if transport in ("loopback", "private"):
                # Permitted (the config validator refused public plaintext outright), but
                # it must not be silent: on `private` the request body — your source code —
                # crosses a network unencrypted, and anyone on that network can read it.
                where = ("this machine" if transport == "loopback"
                         else "your private network, UNENCRYPTED")
                print(f"ℹ model '{name}' talks plaintext http:// to {mp.base_url} "
                      f"({where}).")
                if transport == "private":
                    print(f"  Source code is sent in the clear to that address; anyone "
                          f"who can see that network segment can read it.\n"
                          f"  Use https:// if the endpoint can terminate TLS.")
        # A local endpoint is usually one model on one GPU. If far more hunters than
        # in-flight slots are configured, the surplus threads block inside the provider
        # without renewing their cell lease (renewal happens per COMPLETED turn), so a
        # slow queue can let a lease lapse and hand the cell to another hunter — work
        # done twice for no benefit.
        hunt_rc = s.roles.get("hunt")
        hunt_mp = s.models.get(hunt_rc.model) if hunt_rc else None
        cap = getattr(hunt_mp, "max_concurrent_requests", None) if hunt_mp else None
        if cap and s.concurrency.hunters > cap * 2:
            print(f"⚠ concurrency.hunters is {s.concurrency.hunters} but the hunt model "
                  f"allows only {cap} concurrent request(s).\n"
                  f"  {s.concurrency.hunters - cap} hunter(s) will sit blocked in the "
                  f"provider without renewing their cell lease, so a slow queue can let "
                  f"leases lapse and cells be hunted twice.\n"
                  f"  Set concurrency.hunters closer to {cap} for this model.")
            self.audit.event("hunters_exceed_endpoint_capacity",
                             hunters=s.concurrency.hunters, endpoint_capacity=cap)
        # A single request that can run longer than the cell lease is the same hazard from
        # the other direction: the lease is renewed per COMPLETED turn, so a call that
        # blocks past _CELL_LEASE_SECONDS lets another hunter reclaim the cell while this
        # one is still working on it. Local models are the usual reason someone raises this
        # (generation really is that slow), so warn rather than refuse — and say what the
        # cost is, which for a free model is duplicated wall-clock rather than money.
        timeout = getattr(hunt_mp, "request_timeout_seconds", 0) if hunt_mp else 0
        if timeout and timeout > _CELL_LEASE_SECONDS:
            print(f"⚠ the hunt model's request_timeout_seconds ({timeout}) exceeds the "
                  f"{_CELL_LEASE_SECONDS}s cell lease.\n"
                  f"  A single slow call can outlive the claim it is working under, so "
                  f"another hunter may reclaim and re-hunt that cell.\n"
                  f"  The owner guard still stops both results being committed, so this "
                  f"costs duplicated work rather than a wrong answer.")
            self.audit.event("request_timeout_exceeds_lease", timeout_seconds=timeout,
                             lease_seconds=_CELL_LEASE_SECONDS)
        seen: set[str] = set()
        for role in list(s.roles.keys()):
            try:
                _rc, mp = s.model_for_role(role)
            except Exception:  # noqa: BLE001
                continue
            key = f"{mp.provider}:{mp.model_id}"
            if key in seen:
                continue
            seen.add(key)
            try:
                provider = ctx.provider_for_role(role)
                # The hunt role is probed WITH a tool. Tool use is the one capability the
                # hunt cannot work without — a cell that makes no successful tool call
                # scores at most 0.4 thoroughness against a 0.7 bar — and it is exactly
                # what a local or brokered model is most likely to lack. Probing without
                # tools (as this did) meant an endpoint with no tool support sailed
                # through preflight and then produced thousands of thin cells before
                # anyone could tell why.
                probe_tools = self._preflight_tools(role, mp)
                # Enough room to ANSWER. This used to send max_tokens=16, which is fine
                # for a chat model and hostile to a reasoning one: GLM, DeepSeek-R1 and
                # Qwen3-thinking emit their chain of thought first, so a 16-token budget
                # was spent entirely on thinking and `content` was still empty when the
                # cap hit. Preflight then declared a perfectly healthy endpoint "empty"
                # and refused to start the scan. One probe per distinct model, so 512
                # tokens costs about a cent even on the priciest model.
                probe_budget = max(1, min(int(mp.max_tokens), 512))
                resp = provider.generate(
                    system="Connectivity test.",
                    messages=[Message.user_text("Reply with: OK")],
                    max_tokens=probe_budget, temperature=0.0, tools=probe_tools,
                )
                reasoning = (getattr(resp, "reasoning_text", None) or "").strip()
                self.audit.event("preflight_ok", role=role, model=mp.model_id,
                                 tools_probed=bool(probe_tools),
                                 input_tokens=resp.input_tokens,
                                 output_tokens=resp.output_tokens,
                                 stop_reason=resp.stop_reason,
                                 reasoning_detected=bool(reasoning),
                                 reported_cost_usd=getattr(resp, "reported_cost_usd",
                                                           None))
                answered = bool((resp.text or "").strip()) or bool(resp.tool_uses)
                if not answered:
                    # Distinguish "thought but ran out of room" from "said nothing at
                    # all". The first is a healthy reasoning model on too small a budget —
                    # failing the run for it is a false negative that blocks a working
                    # setup. The second is an endpoint that produced no output whatsoever,
                    # which no amount of budget explains.
                    produced_output = (resp.output_tokens > 0 or bool(reasoning))
                    if not produced_output:
                        return False, (
                            f"model '{mp.model_id}' (role {role}) returned no content, no "
                            f"tool call and no output tokens — the endpoint answered but "
                            f"produced nothing at all.")
                    excerpt = (reasoning[:80] + "…") if len(reasoning) > 80 else reasoning
                    print(f"ℹ model '{mp.model_id}' (role {role}) is a REASONING model: it "
                          f"used {resp.output_tokens} token(s) thinking and had not "
                          f"reached its answer within the {probe_budget}-token probe"
                          + (f'\n  (thinking began: "{excerpt}")' if excerpt else "")
                          + f"\n  Connectivity is fine. Make sure "
                            f"models.<name>.max_tokens ({mp.max_tokens}) leaves room for "
                            f"the reasoning AND a full <findings> block, or cells will be "
                            f"re-queued as truncated.")
                    self.audit.event("preflight_reasoning_model", role=role,
                                     model=mp.model_id, probe_budget=probe_budget,
                                     output_tokens=resp.output_tokens,
                                     configured_max_tokens=mp.max_tokens)
                # A METERED model that reports no token usage makes the cost ledger — and
                # therefore the hard kill-switch — structurally blind to its spend. That
                # is the same unbounded-spend hole the unpriced-model check closes, so it
                # fails the same way.
                if (getattr(mp, "billing", "metered") != "free"
                        and resp.input_tokens == 0 and resp.output_tokens == 0):
                    return False, (
                        f"model '{mp.model_id}' (role {role}) returned no token usage, so "
                        f"its spend cannot be tracked and the hard cost ceiling "
                        f"(${s.budget.max_cost_usd}) could never fire for it. Use an "
                        f"endpoint that reports `usage`, or declare `billing: free` if it "
                        f"genuinely costs nothing.")
            except BudgetExceeded:
                raise  # already at the ceiling before hunting — abort, don't mask it
            except Exception as exc:  # noqa: BLE001
                return False, (f"role '{role}' → '{mp.model_id}' ({mp.provider}): "
                               f"{_classify_error(str(exc))}")
        return True, "all models responded"

    @staticmethod
    def _preflight_tools(role: str, mp) -> list | None:
        """A single throwaway tool spec for the hunt role's connectivity probe.

        Only the hunt role needs tools (judges read a finding and vote), and only when the
        model is configured to use them — probing a ``tool_mode: react`` endpoint with a
        native ``tools`` array would test a path it will never take.
        """
        if role != "hunt" or getattr(mp, "tool_mode", "native") != "native":
            return None
        from .providers.base import ToolSpec
        return [ToolSpec(
            name="preflight_echo",
            description="Connectivity probe: not called during the scan.",
            input_schema={"type": "object",
                          "properties": {"value": {"type": "string"}}},
        )]

    def _hunt_one(self, ctx: ScanContext, cell):

        # Run directed hunt; if open-ended enabled, the orchestrator could also
        # schedule an open pass — here we alternate by attack_class 'all'.
        open_ended = (cell["attack_class"] == "all") and ctx.settings.hunting.open_ended
        return hunt_stage.run_hunt_cell(ctx, cell, open_ended=open_ended)

    def _check_systemic_failure(self, result: PassResult, error_kinds: dict[str, int],
                                pass_no: int, *, mixed_warned: bool) -> bool:
        """Decide whether the opening batch is failing for ONE shared reason.

        Sets ``result.systemic_failure`` when a single failure CLASS dominates and
        nothing has succeeded — that's a config/permissions problem, and grinding
        every remaining cell just burns money on a guaranteed failure.

        Verifying the errors genuinely match matters: the old message asserted "the
        same error" while only counting failures, sending people to hunt for a shared
        cause that might not exist. Scattered failures now say so instead.

        Returns True if it printed something (so the caller can avoid repeating the
        mixed-cause warning).
        """
        threshold = max(5, self.settings.concurrency.hunters)
        if not (pass_no == 1 and result.analyzed == 0
                and result.failed >= threshold and result.shallow == 0):
            return False

        kinds = dict(error_kinds)
        dominant, dominant_n = (max(kinds.items(), key=lambda kv: kv[1]) if kinds
                                else ("other", result.failed))
        # Require the dominant class to be both frequent AND actually dominant. A bare
        # count would eventually abort on scattered failures too, once any one class
        # happened to accumulate `threshold` hits across a long run.
        dominates = dominant_n >= threshold and dominant_n >= 0.7 * result.failed
        if dominates:
            result.systemic_failure = True
            print(f"\n⚠ ABORTING: {dominant_n} of the first {result.failed} cells "
                  f"failed the same way and none succeeded:\n"
                  f"    {dominant}\n"
                  f"    (last detail: {result.last_error})\n"
                  f"Fix the cause (usually a model/config/permissions issue) and "
                  f"re-run. Full detail in wildginger_output/audit/events.jsonl.")
            self.audit.event("systemic_failure", dominant_error=dominant,
                             count=dominant_n, total_failed=result.failed,
                             error_kinds=kinds)
            return True

        # Mixed causes: keep going, but say so rather than grinding silently.
        if not mixed_warned:
            top = sorted(kinds.items(), key=lambda kv: -kv[1])[:3]
            summary = ", ".join(f"{k} ×{v}" for k, v in top)
            print(f"\n⚠ {result.failed} cells failed with MIXED causes ({summary}). "
                  f"Continuing — this is not one systemic fault.")
            self.audit.event("mixed_failures", error_kinds=kinds,
                             total_failed=result.failed)
            return True
        return False

    @staticmethod
    def _cancel_pending(futures: dict) -> int:
        """Cancel every future that hasn't started yet. Returns the count cancelled.

        ``Future.cancel()`` only succeeds for work still queued — already-running
        threads can't be interrupted, so in-flight hunters finish (their spend is
        already committed). This is what stops the scan from burning the whole
        remaining ledger after a kill-switch / systemic-failure abort.
        """
        cancelled = 0
        for fut in futures:
            if fut.cancel():
                cancelled += 1
        return cancelled

    def _is_shallow_plateau(self, cell, outcome) -> bool:
        """True if re-hunting this cell can no longer be expected to help.

        Requires ALL of:
          * it did real work — thoroughness > 0, i.e. at least one successful tool call.
            A cell that read nothing is a genuine coverage gap and must stay SHALLOW;
          * it has been attempted at least ``quality.max_shallow_attempts`` times;
          * this attempt was no better than the last one.
        """
        if outcome.thoroughness <= 0:
            return False
        limit = getattr(self.settings.quality, "max_shallow_attempts", 2)
        attempts = _row_int(cell, "attempts")
        if attempts < max(1, int(limit)):
            return False
        previous = _row_float(cell, "thoroughness")
        return outcome.thoroughness <= previous

    @staticmethod
    def _owner_of(cell) -> str | None:
        """The lease owner stamped on a claimed cell row, if the row carries one."""
        try:
            return cell["lease_owner"]
        except (IndexError, KeyError, TypeError):
            return None

    def _settle_owned(self, cell, state: str, **kw) -> bool:
        """Terminal transition CONDITIONAL on still owning the cell's lease.

        Returns False when the claim was taken over while we were working, in which case
        the other holder's result stands and ours is discarded — the alternative is two
        hunters overwriting each other and one paid result vanishing silently.
        """
        owner = self._owner_of(cell)
        won = self.db.set_cell_state(cell["id"], state, owner=owner, **kw)
        if not won:
            self.audit.event("cell_lease_lost", cell_id=cell["id"], state=state,
                             owner=owner)
        return won

    def _cell_findings_writer(self, findings):
        """A same-transaction callback that persists one cell's findings, or None.

        Returned rather than inlined so the ANALYZED transition and the findings share
        one atomic writer job (see ``Database.set_cell_state(also=...)``).
        """
        if not findings:
            return None
        return lambda cur: self._write_findings(cur, findings)

    def _settle_cell(self, cell, state: str, findings=None, **kw) -> None:
        """Write a terminal cell state, never raising.

        Used by the abort-salvage paths, where leaving a cell without a state means it
        sits IN_PROGRESS until its lease expires and a resume can't reclaim it.
        ``findings`` (when given) is persisted in the SAME transaction, so salvaged
        work is as crash-safe as work settled by the main loop.
        """
        try:
            self.db.set_cell_state(cell["id"], state,
                                   also=self._cell_findings_writer(findings), **kw)
        except Exception as exc:  # noqa: BLE001 — best-effort cleanup
            self.audit.event("cell_state_error", cell_id=cell["id"], state=state,
                             error=str(exc))

    def _drain_completed(self, futures: dict, handled: set) -> list[Finding]:
        """Collect findings from futures that finished but weren't processed yet.

        When we break out of the ``as_completed`` loop on a budget abort, some hunters
        have already finished (and their tokens are already paid for) but their results
        were still queued behind us. Salvaging them means an aborted run reports every
        finding it actually paid to produce. Their cells are also marked ANALYZED so a
        resume doesn't re-hunt work that's already done.
        """
        salvaged: list[Finding] = []
        for fut, cell in futures.items():
            if fut in handled or fut.cancelled() or not fut.done():
                continue
            try:
                outcome = fut.result()
            except BudgetExceeded:
                # The kill-switch is never swallowed by a blanket handler. The primary
                # BudgetExceeded is already captured and re-raised by the caller, so
                # there is nothing to salvage from this cell — but keep the exception
                # type explicit so this stays a deliberate skip, not an accident.
                self._settle_cell(cell, "PENDING")
                continue
            except Exception as exc:  # noqa: BLE001 — nothing to salvage from a failure
                # It still needs a TERMINAL state. A future that finished with an
                # exception but was never processed used to fall through BOTH helpers
                # (not cancelled, so not released; raised, so not salvaged) and its row
                # sat IN_PROGRESS until the 10-minute lease expired.
                self._settle_cell(cell, "FAILED", last_error=str(exc))
                continue
            if outcome.findings:
                salvaged.extend(outcome.findings)
            if outcome.analyzed or outcome.findings:
                self._settle_cell(cell, "ANALYZED", findings=outcome.findings,
                                  thoroughness=outcome.thoroughness)
            else:
                # Ran, but produced nothing worth keeping — SHALLOW is reclaimable, so
                # a resume can re-hunt it rather than leaving it stuck IN_PROGRESS.
                self._settle_cell(cell, "SHALLOW", thoroughness=outcome.thoroughness)
        if salvaged:
            self.audit.event("budget_abort_salvaged_findings", count=len(salvaged))
        return salvaged

    def _release_unfinished(self, futures: dict) -> None:
        """Return never-run cells to PENDING after an aborted pass.

        A cancelled cell still holds an IN_PROGRESS row with a 10-minute lease;
        without this a resume would have to wait the lease out before the cell became
        claimable again.

        Only ``cancelled()`` futures are released. Releasing merely-``not done()``
        ones raced with the worker still running them: we would write PENDING while a
        live hunter was about to write ANALYZED, and a concurrent ``--resume`` in that
        window could double-claim the cell. Every caller now runs AFTER the pool's
        ``shutdown(wait=True, cancel_futures=True)``, so nothing is still running and
        the cancelled/done split is stable; futures that finished but were never
        processed are settled by :meth:`_drain_completed`, which this complements.
        """
        for fut, cell in futures.items():
            if fut.cancelled():
                self._settle_cell(cell, "PENDING")
    def _warn_if_reusing_ledger(self) -> None:
        """Tell the user loudly when a fresh scan discards a previous run's progress.

        Reusing an ``out_dir`` without ``--resume`` is legitimate — it is how you
        re-scan after a code change — but it resets the ledger and does NOT reload the
        prior run's findings. Silence about that is exactly what let the false-clean
        bug hide, so we surface it. Deliberately a warning rather than a refusal: a
        hard error would break both the ordinary "just scan it again" workflow and any
        CI job that reuses a workspace.
        """
        try:
            def _q(cur):
                cur.execute("SELECT COUNT(*) AS n FROM cells WHERE state = 'ANALYZED';")
                return cur.fetchone()["n"]
            analyzed = self.db.read(_q)
        except Exception:  # noqa: BLE001 — advisory only, never break a scan
            return
        if not analyzed:
            return
        if self.settings.scope.mode == "incremental":
            # Incremental mode deliberately KEEPS the ledger: unchanged files stay
            # ANALYZED and are not re-hunted. The fresh-scan wording below is the exact
            # opposite of what happens, so printing it here actively misled the user
            # about whether their code had been re-analyzed.
            print(f"\nℹ Incremental mode: {analyzed} cell(s) from a previous run stay "
                  f"analyzed and will NOT be re-hunted.\n"
                  f"  Their findings are reloaded from the ledger so the report still "
                  f"covers them.\n"
                  f"  Use `scope.mode: full` to re-analyze the whole scope.\n")
            self.audit.event("ledger_reused_incremental", analyzed_cells=analyzed)
            return
        print(f"\n⚠ This output directory already holds {analyzed} analyzed cell(s) from "
              f"a previous run.\n"
              f"  Without --resume that progress is RESET and the whole scope is "
              f"re-hunted (and the\n"
              f"  previous run's findings are not reloaded). Use --resume to continue "
              f"that run instead.\n")
        self.audit.event("ledger_reset", analyzed_cells_discarded=analyzed)



    def _claim_all_cells(self, ctx: ScanContext) -> list:
        """Claim every currently-available cell for this pass.

        The loop is guarded against claiming the SAME cell twice. Every claim stamps
        ``lease_expires = now + 600`` using *that call's* clock, so on a large ledger
        the loop can outlive the lease it handed to its own first cell (measured
        full-drain cost: ~28s for 20k cells, ~700s for 100k — the claim is one writer
        round-trip plus a per-claim sort of the candidate set). That cell then matches
        the expired-lease branch of ``claim_next_cell`` and — because it is the
        highest-weight row, returned first by ``ORDER BY weight DESC`` — comes straight
        back. Without this guard it was submitted to the pool again, so the same cell
        ran a full hunt concurrently with itself (double token spend, racing terminal
        writes) and burned a second ``attempts`` slot against a cap that is only
        ``max_passes`` deep, which can strand the cell for the rest of the run.

        A repeat therefore means "the lease clock has lapped", i.e. everything
        claimable has already been claimed — so we stop. Nothing else mutates the
        ledger mid-loop, so there is no other way to see the same id twice.
        """
        cells = []
        seen: set[int] = set()
        # A UNIQUE owner per claim. `f"pass-{id(self)}"` was the same string for every
        # cell in the pass AND `id()` is a heap address, so it is not unique across
        # processes either — meaning there was no identity to guard a transition WITH
        # (see Database.set_cell_state(owner=...)). uuid4 + the cell id gives every claim
        # a token nothing else can forge.
        import uuid as _uuid
        run_token = _uuid.uuid4().hex[:12]
        # One claim per cell per pass, so the per-cell attempt cap is the pass
        # budget. This is what lets a FAILED cell be retried on a later pass
        # (transient throttling/timeouts) without ever looping forever.
        max_attempts = self.settings.budget.max_passes
        while True:
            owner = f"{run_token}:{len(cells)}"
            row = self.db.claim_next_cell(owner, lease_seconds=_CELL_LEASE_SECONDS,
                                          max_attempts=max_attempts)
            if row is None:
                break
            cell_id = row["id"]
            if cell_id in seen:
                # The re-claim already refreshed this cell's lease and bumped its
                # attempts, which is harmless (it is about to be hunted anyway) — but
                # it must NOT be queued a second time.
                self.audit.event("claim_loop_lapped", cell_id=cell_id,
                                 claimed=len(cells))
                print(f"⚠ Claim loop outlived its own 600s cell lease after "
                      f"{len(cells)} cell(s); stopping this pass's claiming here. "
                      f"Remaining cells are picked up by the next pass.")
                break
            seen.add(cell_id)
            cells.append(row)
            # --limit N: cap cells per pass for a cheap smoke test.
            if self.limit and len(cells) >= self.limit:
                break
            # Safety: avoid infinite loop if a cell stays IN_PROGRESS.
            if len(cells) > 100000:
                break
        return cells

    def _persist_findings(self) -> None:
        """Upsert findings into the DB so `scan mark` + audits can reference them.

        Keyed on ``dedup_key`` (UNIQUE). This replaces the old
        ``DELETE FROM findings`` + re-INSERT, which re-issued a NEW row id for the same
        finding on every pass — orphaning any ``verifications`` row a human had
        recorded with ``scan mark``, and making report ids unstable mid-run.

        Human lifecycle verdicts are preserved: a status set by ``scan mark``
        (false_positive / accepted_risk / fixed) is NOT overwritten by a later pass.
        """
        def _w(cur):
            self._write_findings(cur, self.findings)
        try:
            self.db.write(_w)
        except Exception as exc:  # noqa: BLE001
            self.audit.event("persist_findings_error", error=str(exc))

    # Human lifecycle verdicts a later pass must never overwrite.
    _KEEP_STATUS = ("false_positive", "accepted_risk", "fixed", "suppressed")

    def _write_findings(self, cur, findings) -> None:
        """Upsert ``findings`` using an ALREADY-OPEN writer cursor.

        Split out of :meth:`_persist_findings` so a cell's findings can be written in
        the SAME transaction that marks the cell ANALYZED (see
        ``Database.set_cell_state(also=...)``) — the atomicity that stops an ungraceful
        crash from leaving a durably-analyzed cell whose findings were never stored.

        Also reaps rows this finding has SUPERSEDED (see ``Finding.superseded_keys``):
        ``fingerprint()`` includes the CWE, ``enrich`` fills a missing CWE at finalize,
        and dedupe merges findings away — all of which strand old rows that no report
        contains but that ``--resume`` faithfully reloads as duplicates.
        """
        import time as _t

        keep_status = self._KEEP_STATUS

        now = _t.time()
        for f in findings:
            key = f.dedup_key()
            stale = getattr(f, "superseded_keys", set()) - {key}
            if stale:
                marks = ",".join("?" * len(stale))
                # finding_evidence has no ON DELETE CASCADE, so clear it first or
                # the judge verdicts of a deleted finding become orphan rows.
                cur.execute(
                    f"DELETE FROM finding_evidence WHERE finding_id IN "
                    f"(SELECT id FROM findings WHERE dedup_key IN ({marks}));",
                    tuple(stale))
                cur.execute(
                    f"DELETE FROM findings WHERE dedup_key IN ({marks});",
                    tuple(stale))
                self.audit.event("superseded_rows_reaped", count=len(stale),
                                 fingerprint=key)
            f.superseded_keys = {key}
            cur.execute(
                "INSERT INTO findings(file_id, line_start, line_end, category, cwe, "
                "severity, confidence, votes, hedged, reachable, status, dedup_key, "
                "symbol, cited_snippet, base_severity, base_confidence, "
                "sarif_rule_id, source, title, description, "
                "remediation, created_at) "
                "VALUES ((SELECT id FROM files WHERE path=?),?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?) "
                "ON CONFLICT(dedup_key) DO UPDATE SET "
                "  line_start=excluded.line_start, line_end=excluded.line_end, "
                "  category=excluded.category, cwe=excluded.cwe, "
                "  severity=excluded.severity, confidence=excluded.confidence, "
                "  votes=excluded.votes, hedged=excluded.hedged, "
                "  reachable=excluded.reachable, "
                # A human verdict outranks anything a later pass computes.
                "  status=CASE WHEN findings.status IN "
                f"    ({','.join('?' * len(keep_status))}) "
                "    THEN findings.status ELSE excluded.status END, "
                "  symbol=excluded.symbol, cited_snippet=excluded.cited_snippet, "
                # NEVER overwrite an existing latch with NULL: the stored baseline is
                # the model's original claim, and losing it re-arms the severity
                # decay this column exists to prevent.
                "  base_severity=COALESCE(excluded.base_severity, findings.base_severity), "
                "  base_confidence=COALESCE(excluded.base_confidence, findings.base_confidence), "
                "  sarif_rule_id=excluded.sarif_rule_id, source=excluded.source, "
                "  title=excluded.title, description=excluded.description, "
                "  remediation=excluded.remediation;",
                (f.file_rel, f.line_start, f.line_end, f.category, f.cwe, f.severity,
                 f.confidence, f.votes, int(f.hedged), f.reachable, f.status,
                 f.dedup_key(), f.symbol, f.cited_snippet,
                 f.base_severity, f.base_confidence, f.sarif_rule_id,
                 f.source, f.title, f.description, f.remediation, now,
                 *keep_status),
            )
            # Adopt the row id so the in-memory finding and the DB agree (this is
            # what `scan mark <id>` refers to).
            if f.id is None:
                cur.execute("SELECT id FROM findings WHERE dedup_key = ?;",
                            (f.dedup_key(),))
                row = cur.fetchone()
                if row is not None:
                    f.id = row["id"]
            # Drain the judge panel's verdicts now that we have a finding_id.
            # Replace rather than append so re-persisting the same finding on a
            # later pass doesn't duplicate its evidence rows.
            if f.id is not None and f.judge_verdicts:
                cur.execute("DELETE FROM finding_evidence WHERE finding_id = ?;",
                            (f.id,))
                for v in f.judge_verdicts:
                    cur.execute(
                        "INSERT INTO finding_evidence(finding_id, pass_no, model, "
                        "evidence, raw_response_ref) VALUES (?,?,?,?,?);",
                        (f.id, self._pass_no, v.get("role"),
                         f"{v.get('verdict')}: {v.get('evidence', '')}"[:2000],
                         f.dedup_key()),
                    )

    def _resume_from_checkpoint(self, s: Settings) -> None:
        """Reload persisted findings from a prior run into ``self.findings``.

        Guarded by config_hash: if the config changed since the checkpoint we refuse
        to resume (unless --resume-force) so a different model/config never gets mixed
        into a repo's partial output.
        """
        cp = self.db.get_checkpoint()
        if cp is None:
            print("Resume requested but no checkpoint found — starting fresh.")
            return
        cp_hash = cp["config_hash"]
        if cp_hash and cp_hash != s.config_hash() and not self.resume_force:
            print(f"⚠ Resume refused: config changed since checkpoint "
                  f"(was {cp_hash}, now {s.config_hash()}). "
                  f"Re-run with --resume-force to override, or drop --resume for a "
                  f"fresh scan.")
            raise SystemExit(EXIT_FATAL)

        restored = self._reload_persisted_findings()
        print(f"Resumed from checkpoint (phase='{cp['phase']}', pass={cp['pass_no']}): "
              f"restored {restored} finding(s). Hunting will skip already-analyzed cells.")
        self.audit.event("resume", phase=cp["phase"], pass_no=cp["pass_no"],
                         restored=restored)

    def _reload_persisted_findings(self) -> int:
        """Rebuild in-memory Findings from the ``findings`` table. Returns the count.

        Shared by ``--resume`` and ``scope.mode: incremental`` — both need the findings
        side of the ledger back, or the report describes only the cells this invocation
        re-hunted while coverage still counts the ones it skipped.
        """
        rows = self.db.load_findings()
        restored = 0
        skipped = 0
        existing = {f.dedup_key() for f in self.findings}
        for r in rows:
            try:
                f = Finding(
                    file_rel=r.get("file_rel") or "",
                    line_start=r.get("line_start") or 0,
                    line_end=r.get("line_end") or 0,
                    category=r.get("category") or "novel/other",
                    title=r.get("title") or "",
                    description=r.get("description") or "",
                    severity=r.get("severity") or "medium",
                    # `or` would rewrite legitimately-stored zeros: a dead judge panel
                    # persists votes=0, and 0.0 is a real confidence. Coerce only NULL.
                    confidence=(0.5 if r.get("confidence") is None
                                else float(r["confidence"])),
                    cwe=r.get("cwe"),
                    votes=(1 if r.get("votes") is None else int(r["votes"])),
                    hedged=bool(r.get("hedged")),
                    reachable=r.get("reachable") or "unclear",
                    status=r.get("status") or "new",
                    source=r.get("source") or "llm",
                    remediation=r.get("remediation"),
                    sarif_rule_id=r.get("sarif_rule_id"),
                    # Both must round-trip: `symbol` is half of the fingerprint (a
                    # resumed finding would otherwise get a NEW identity and duplicate
                    # itself), and the snippet is the reviewer's evidence.
                    symbol=r.get("symbol") or "",
                    cited_snippet=r.get("cited_snippet") or "",
                    # enrich()'s idempotency latch (schema v5). Restored EXACTLY as
                    # stored — including None, which means "never latched" — so a
                    # resumed finding is re-scored from the model's original claim
                    # instead of from its own already-discounted output. `or` would be
                    # wrong here: a legitimately stored 0.0 confidence must survive.
                    base_severity=r.get("base_severity"),
                    base_confidence=(None if r.get("base_confidence") is None
                                     else float(r["base_confidence"])),
                    id=r.get("id"),
                )
            except Exception as exc:  # noqa: BLE001
                # Silently dropping rows here meant "restored N" could under-report
                # with no hint that anything was lost.
                skipped += 1
                self.audit.event("finding_restore_failed", error=str(exc),
                                 dedup_key=str(r.get("dedup_key"))[:32])
                continue
            if f.dedup_key() in existing:
                continue
            existing.add(f.dedup_key())
            # Remember the key the row is ACTUALLY stored under, not the recomputed
            # one. They can differ (e.g. a NULL file_id makes file_rel come back empty,
            # which changes the fingerprint), and when they do, the next persist would
            # INSERT a second row instead of updating this one. Carrying the stored key
            # makes the persist step reap the old row, so identity drift self-heals
            # instead of accumulating duplicates that a later resume resurrects.
            stored_key = r.get("dedup_key")
            f.superseded_keys.add(str(stored_key) if stored_key else f.dedup_key())
            self.findings.append(f)
            restored += 1
        if skipped:
            print(f"⚠ {skipped} stored finding(s) could not be restored (see the audit "
                  f"log); they are not in this run's report.")
        return restored

    def _attach_sample_requests(self, ctx=None, findings=None) -> None:
        """Attach a malicious ``sample_request`` (curl) to each reachable endpoint
        finding. Deterministic first (parsed route/params); when ``ctx`` is provided
        and the deterministic walk fails, an LLM route-resolution fallback runs
        (verified against source — see endpoints.build_sample_request).

        ``findings`` MUST be the reportable set. The LLM fallback is a tool-using agent
        run PER FINDING, and running it over ``self.findings`` meant paying for route
        resolution on findings the report then discarded (below threshold, baselined,
        suppressed, reachability-demoted) — hundreds of LLM calls after the scan is
        logically over, for output nobody ever sees.
        """
        from . import endpoints
        index = getattr(self, "_index", None)
        file_text = getattr(self, "_file_text", {}) or {}
        if index is None:
            return
        # Only endpoint-reachable, non-false-positive findings get a sample. Filter
        # up front so the per-finding heartbeat below knows the total — the LLM
        # route-resolution fallback can spend minutes on a single finding, and a
        # silent loop here used to look exactly like a hung process.
        candidates = [
            f for f in (self.findings if findings is None else findings)
            if f.status != "false_positive" and f.reachable != "unreachable"
            and file_text.get(f.file_rel)
        ]
        total = len(candidates)
        attached = 0
        for i, f in enumerate(candidates, 1):
            text = file_text[f.file_rel]
            try:
                sr = endpoints.build_sample_request(f, index, text, file_text,
                                                    ctx=ctx,
                                                    surface=self._surface)
            except BudgetExceeded:
                # The LLM route-resolution fallback can hit the ceiling. Stop trying to
                # enrich further findings, but keep the ones already attached.
                self.audit.event("sample_request_budget_stop", file=f.file_rel)
                print(f"  [sample {i}/{total}] {f.file_rel} — cost ceiling reached; "
                      f"stopping with {attached} sample request(s) attached")
                return
            except Exception as exc:  # noqa: BLE001 — never break finalize
                self.audit.event("sample_request_error", file=f.file_rel, error=str(exc))
                print(f"  [sample {i}/{total}] {f.file_rel} — error (see audit log)")
                continue
            if sr:
                f.sample_request = sr
                attached += 1
                print(f"  [sample {i}/{total}] {f.file_rel} — "
                      f"attached ({sr.get('derivation', 'derived')})")
            else:
                print(f"  [sample {i}/{total}] {f.file_rel} — no route resolved")

    def _write_empty_reports(self, reason: str) -> None:
        """Write a valid, empty report set so a stale one is never mistaken for current."""
        try:
            from .reporter import write_reports
            out_dir = Path(self.settings.output.out_dir)
            write_reports(self.settings, [], coverage_pct=0.0, out_dir=out_dir,
                          run_stats={"note": f"no scan performed: {reason}"})
            print(f"» Wrote an empty report set to {out_dir} "
                  f"(so a previous run's report is not mistaken for this one).")
        except Exception as exc:  # noqa: BLE001 — advisory
            self.audit.event("empty_report_write_failed", error=str(exc))

    # ── gating + exit code (single source of truth) ───────────────────────────
    def _enrich_and_settle(self) -> None:
        """Score every finding, then RE-MERGE. Shared by both finalize paths.

        The re-merge is not cosmetic. ``fingerprint()`` includes the CWE and
        ``scoring.enrich`` fills a missing one from the category map, so enrichment can
        give two findings that were distinct before it the SAME identity — most often
        the pass-time copy of a finding and the copy reloaded from disk by ``--resume``.
        The only ``dedupe()`` call used to run inside ``_run_pass``, i.e. BEFORE
        enrichment (and not at all when a resume pass claims zero cells), so those
        collapsed pairs were reported twice, with their votes split, under one
        ``short_id``. Merging after enrich is what actually closes it.
        """
        s = self.settings
        for f in self.findings:
            scoring.enrich(f, hedge_penalty=s.quality.hedge_penalty,
                           language_aware_fp=s.quality.language_aware_fp,
                           language=self._file_lang.get(f.file_rel, ""))
        before = len(self.findings)
        self.findings = merge_findings(self.findings)
        if len(self.findings) != before:
            self.audit.event("post_enrich_merge", before=before,
                             after=len(self.findings))
        self.findings = sort_canonical(self.findings)

    def _gate(self):
        """Evaluate baseline/suppressions/threshold ONCE. Fails CLOSED."""
        from .baseline import GateResult, apply_baseline_and_suppressions
        try:
            return apply_baseline_and_suppressions(self.settings, self.findings)
        except Exception as exc:  # noqa: BLE001 — gating math must never kill a run
            # Fail CLOSED on the findings dimension: if we cannot evaluate the
            # baseline we must not silently report "clean".
            self.audit.event("baseline_eval_error", error=str(exc))
            print(f"⚠ Could not evaluate baseline/suppressions ({exc}); reporting "
                  f"every finding at/above the threshold instead of trusting them.")
            return GateResult(
                reportable=[f for f in self.findings
                            if f.status not in ("false_positive", "suppressed")
                            and scoring.passes_threshold(
                                f, self.settings.output.severity_threshold)],
                informational=[], suppressed=0, baselined=0)

    def _exit_code_for(self, reportable, cov: float) -> int:
        """The documented exit-code contract, in one place.

        Findings outrank coverage: a critical vulnerability is the more urgent signal
        and CI keys on "1 == findings". Checking coverage first meant any scan with an
        incomplete ledger (including every ``--limit`` smoke test) reported 2 and
        masked real criticals. The abort paths used to hardcode 3/2 while still writing
        a report that listed criticals, which broke the same contract from the other
        side — they now come through here too.
        """
        if reportable:
            return EXIT_FINDINGS
        # --limit deliberately scans a subset, so incomplete coverage is expected
        # and must not be reported as a failure.
        if (cov < 1.0 and self.settings.output.fail_on_incomplete_coverage
                and not self.limit):
            return EXIT_COVERAGE_INCOMPLETE
        return EXIT_OK

    def _finalize_partial(self) -> int | None:
        """Finalize an aborted run (cost/time kill-switch): enrich, attach malicious
        sample requests, generate PoCs, write reports — so the partial output still
        carries curls + reproduction sketches for what we found. Never raises.

        Idempotent: if the full :meth:`_finalize` already wrote reports (i.e. the
        ceiling was hit during reporting, after the findings were complete) this is a
        no-op so we never overwrite the better, complete report with a partial one.
        """
        if self._finalized:
            return self._partial_exit_code
        self._finalized = True
        s = self.settings
        cov = 0.0
        try:
            # Enrich + RE-MERGE (enrichment can collapse two identities into one), in
            # the same canonical order a full finalize uses so a partial report stays
            # diffable against a complete one.
            self._enrich_and_settle()
            # Gate ONCE, before anything that costs money or writes files, so the
            # per-finding LLM route resolver and the PoC writer only ever run on
            # findings that will actually be in the report.
            gate = self._gate()
            if s.poc.include_sample_request:
                self._attach_sample_requests(findings=gate.reportable)
            self._persist_findings()
            out_dir = Path(s.output.out_dir)
            if s.poc.generate:
                try:
                    from .poc import generate_pocs
                    generate_pocs(gate.reportable, out_dir)
                except Exception as exc:  # noqa: BLE001
                    self.audit.event("poc_error", error=str(exc))
            run_stats = self._build_run_stats()
            cov = self.db.coverage_pct()
            from .reporter import write_reports
            write_reports(s, self.findings, coverage_pct=cov,
                          out_dir=out_dir, run_stats=run_stats, gate=gate)
            self._print_run_summary(run_stats)
            # An aborted run still returns the DOCUMENTED code for what it found. The
            # abort paths used to hardcode 2 ("coverage incomplete") or 3 ("fatal")
            # while writing a report.json that listed criticals — so a CI job treating
            # 2 as a soft warning shipped a real vulnerability.
            self._partial_exit_code = self._exit_code_for(gate.reportable, cov)
        except Exception as exc:  # noqa: BLE001 — a partial report must never re-raise
            # Silence here is what made a failed partial-finalize unexplainable.
            self.audit.event("finalize_partial_error", error=str(exc))
            print(f"⚠ Could not write the partial report: {exc}")
        return self._partial_exit_code

    # ── run-stats + end-of-run summary ─────────────────────────────────────────
    def _build_run_stats(self) -> dict:
        from .runstats import build_run_stats
        return build_run_stats(
            self.settings, self.audit.cost.snapshot(),
            time.time() - self._scan_start,
            price_mismatches=getattr(self.audit, "price_mismatches", 0))

    def _print_run_summary(self, stats: dict) -> None:
        from .runstats import summary_lines
        print("\n" + "=" * 60)
        print("SCAN SUMMARY")
        print("=" * 60)
        for line in summary_lines(stats):
            print(line)
        print("=" * 60)

    # ── finalize ─────────────────────────────────────────────────────────────
    def _finalize(self, ctx: ScanContext) -> int:


        s = self.settings
        # Whole-set AI dedupe: the per-pass dedupe only compares a pass's OWN new
        # findings, so a pass-2 finding that duplicates a pass-1 finding slips
        # through. Run one semantic pass over the ENTIRE set here to catch those
        # cross-pass duplicates before we report. Fail-open (returns input on error).
        if s.dedupe.semantic and len(self.findings) >= 3 and not self._no_model_calls_left:
            before = len(self.findings)
            print(f"» Final dedupe: asking the model to group {before} finding(s)…")
            t_dedupe = time.time()
            try:
                self.findings, removed = dedupe_stage.semantic_dedupe(
                    ctx, self.findings, role=s.dedupe.semantic_role)
                print(f"» Final dedupe: {before} → {len(self.findings)} "
                      f"(−{removed} cross-pass duplicate(s)) "
                      f"({int(time.time() - t_dedupe)}s)")
            except BudgetExceeded as exc:
                # We are already in the reporting phase — the remaining work is local
                # (enrich → PoCs → reports) and costs nothing. Skip the optional dedupe
                # and still produce the full report rather than losing it to the ceiling.
                self.audit.event("final_dedupe_budget_skipped", error=str(exc))
                print(f"» Final dedupe skipped — cost ceiling reached ({exc}). "
                      f"Continuing to reports (no further model calls).")

        # Score + enrich, then RE-MERGE (enrich fills the CWE, which is part of the
        # fingerprint, so it can collapse two identities into one) and sort.
        print("» Enriching findings…")
        self._enrich_and_settle()

        cov = self.db.coverage_pct()
        out_dir = Path(s.output.out_dir)

        # ── GATE ONCE ──────────────────────────────────────────────────────────
        # Baseline / suppressions / severity threshold, evaluated a single time. This
        # has to happen BEFORE the three consumers below, because each of them used to
        # run over the unfiltered set:
        #   * sample requests  — a tool-using LLM call PER FINDING, including findings
        #     the report then dropped (below threshold, baselined, suppressed);
        #   * PoC generation   — wrote pocs/*.md, complete with a runnable attack curl,
        #     for findings deliberately excluded from the report;
        #   * _persist_findings — ran before `status='suppressed'` was even assigned, so
        #     that status never reached the DB despite being in `keep_status`.
        gate = self._gate()
        print(f"» Gate: {len(gate.reportable)} reportable, "
              f"{len(gate.informational)} informational (not gating), "
              f"{gate.suppressed} suppressed, {gate.baselined} baselined")

        # Attach a MALICIOUS sample request (curl) to reachable endpoint findings.
        # Deterministic route parsing first; ctx enables the verified LLM fallback
        # for findings whose route the call-graph walk couldn't resolve.
        if s.poc.include_sample_request:
            print("» Attaching sample requests…")
            # `ctx=None` disables the per-finding LLM route-resolution fallback; the
            # deterministic walk still runs. Nothing new was hunted, so paying for route
            # resolution again would buy an identical report.
            self._attach_sample_requests(
                ctx=None if self._no_model_calls_left else ctx,
                findings=gate.reportable)

        # From here on the work is local (no model calls), so this IS the definitive
        # report — mark it so a late kill-switch can't replace it with a partial one.
        self._finalized = True

        # Persist findings to the DB (for verification ingest + audit). ALL findings,
        # not just the reportable ones: the DB is the audit record and `scan mark`
        # needs to be able to reference anything the run saw.
        self._persist_findings()

        # Generate inert, human-run PoC artifacts for the findings we are reporting.
        if s.poc.generate:
            try:
                from .poc import generate_pocs
                n = generate_pocs(gate.reportable, out_dir)
                print(f"» Generating PoCs… {n} written")
                self.audit.event("pocs_generated", count=n)
            except Exception as exc:  # noqa: BLE001
                self.audit.event("poc_error", error=str(exc))

        # Reports (with the end-of-run metrics block at the top).
        thin = self.db.thin_cell_count(s.quality.thoroughness_min)
        run_stats = self._build_run_stats()
        print("» Writing reports…")
        try:
            from .reporter import write_reports
            write_reports(s, self.findings, coverage_pct=cov, out_dir=out_dir,
                          run_stats={**(run_stats or {}), "thin_cells": thin},
                          gate=gate)
        except ImportError:
            self.audit.event("reporter_unavailable")
        except OSError as exc:
            # Disk full / read-only out_dir / permissions. This used to escape run()
            # entirely and surface as a traceback with exit 1 — the code CI reads as
            # "vulnerabilities found".
            self.audit.event("report_write_failed", error=str(exc))
            print(f"⚠ Could not write reports to {out_dir}: {exc}")


        # Exit code is derived from the SAME gate the reporter used, so
        # the code can never disagree with report.json.
        reportable = gate.reportable
        self.audit.event("scan_end", coverage_pct=round(cov, 4), thin_cells=thin,
                         findings=len(self.findings), reportable=len(reportable),
                         informational=len(gate.informational),
                         suppressed=gate.suppressed, baselined=gate.baselined,
                         cost=self.audit.cost.snapshot())
        self.db.set_checkpoint("done", config_hash=s.config_hash())
        thin_note = ""
        if thin:
            thin_note = (f" {thin} cell(s) counted as covered on a THIN look (the model "
                         f"read the code and reported nothing on repeated attempts).")
        print(f"Done. Coverage {cov*100:.1f}%.{thin_note} "
              f"{len(reportable)} finding(s) >= {s.output.severity_threshold}.")
        self._print_run_summary(run_stats)
        return self._exit_code_for(reportable, cov)
