# Copyright 2026
# SPDX-License-Identifier: Apache-2.0
"""Command-line entrypoint.

Subcommands:
  scan       run a scan against a target
  validate   load + validate config.yaml and print the effective config hash
  dry-run    discovery + indexing + ledger seeding, print projected work (no LLM)
  mark       human-verification ingest: mark a finding confirmed/false_positive
  purge      delete scan state, audit logs, reports, and PoCs

Exit codes:
  0 complete & clean · 1 findings >= threshold · 2 coverage incomplete · 3 fatal
"""
from __future__ import annotations

import argparse
import signal
import sys
from pathlib import Path

from . import __version__
from .settings import (
    CONFIG_ENV_VAR,
    Settings,
    endpoint_transport,
    load_settings,
    resolve_config_path,
)

EXIT_OK = 0
EXIT_FINDINGS = 1
EXIT_COVERAGE_INCOMPLETE = 2
EXIT_FATAL = 3
# Conventional 128+SIGINT. Only used by the short-lived subcommands: `scan` installs
# its own handler so an interrupt checkpoints and still writes reports.
EXIT_INTERRUPTED = 130


def _install_signal_handlers() -> None:
    """Trap Ctrl-C so short-lived commands exit cleanly (no traceback).

    This is only the fallback for the non-scanning subcommands (validate / dry-run /
    mark / purge), which hold no expensive state. ``scan`` is different: the
    orchestrator REPLACES this handler for the duration of the run
    (``Orchestrator._install_signal_handlers``) so an interrupt stops at a clean
    checkpoint and still writes reports, then restores whatever it found here.

    That ordering matters — this handler raises SystemExit, which would skip the
    orchestrator's finalize path and discard every finding.
    """
    def _handler(signum, _frame):
        print(f"\n[signal {signum}] interrupted — exiting.", file=sys.stderr)
        sys.exit(130)

    try:
        signal.signal(signal.SIGINT, _handler)
    except (ValueError, OSError, AttributeError):
        pass
    try:
        signal.signal(signal.SIGTERM, _handler)
    except (ValueError, OSError, AttributeError):
        # SIGTERM may be unavailable on some Windows contexts; ignore.
        pass


def _load(args) -> Settings:
    # Resolve WHICH config file before loading it, so `--config`, $WILDGINGER_CONFIG
    # and the fallbacks all follow one documented precedence
    # (see settings.resolve_config_path).
    config_path = resolve_config_path(getattr(args, "config", None))
    settings = load_settings(config_path)
    # CLI overrides beat the config file when present (handy for CI / quick runs).
    if getattr(args, "path", None):
        settings.scope.path = args.path
    if getattr(args, "output", None):
        settings.output.out_dir = args.output
    if getattr(args, "hunters", None):
        settings.concurrency.hunters = args.hunters
    if getattr(args, "max_cost", None) is not None:
        settings.budget.max_cost_usd = args.max_cost
    if getattr(args, "experiment", None):
        # Explicit --experiment wins over per-target auto-naming + the config
        # default; MLflow auto-creates the experiment on first use.
        settings.monitoring.mlflow.experiment = args.experiment
        settings.monitoring.mlflow.experiment_per_target = False
    return settings



def cmd_validate(args) -> int:

    settings = _load(args)
    print(f"WildGinger v{__version__}")
    print(f"config OK: {settings.config_source}")
    print(f"  config_hash:      {settings.config_hash()}")
    print(f"  region/profile:   {settings.aws.region} / "
          f"{settings.aws.profile or 'default credential chain'}")
    print(f"  models:           {', '.join(sorted(settings.models))}")
    print(f"  roles:            {', '.join(sorted(settings.roles))}")
    print(f"  target path:      {settings.scope.path}")
    print(f"  scan mode:        {settings.scope.mode}")
    print(f"  seeding mode:     {settings.seeding.mode}")
    print(f"  sanitization:     {settings.sanitization.mode}")
    print(f"  output formats:   {', '.join(settings.output.formats)}")
    # WHERE CODE GOES. With `openai_compat`/`openrouter` the destination is a config
    # value rather than a compiled-in endpoint, so list it here — this is the screen
    # people check before a run, and "which third parties see our source" is the first
    # question an auditor asks.
    print("  model endpoints:")
    for name in sorted(settings.models):
        mp = settings.models[name]
        where = mp.base_url or {
            "bedrock": "AWS Bedrock Runtime (SigV4, compiled in)",
            "openai_gpt": "not configured (set base_url or GPT_BASE_URL)",
        }.get(mp.provider, "provider default")
        extras = []
        if mp.base_url:
            # Say plainly whether code leaves the machine in the clear. This screen is
            # what people check before a run, and "is it encrypted" is not inferable from
            # a URL at a glance.
            transport = endpoint_transport(mp.base_url)
            if transport == "loopback":
                extras.append("plaintext, stays on this machine")
            elif transport == "private":
                extras.append("PLAINTEXT over your private network")
        if mp.billing == "free":
            extras.append("billing: free")
        if mp.tool_mode != "native":
            extras.append(f"tool_mode: {mp.tool_mode}")
        if mp.api_key_env:
            extras.append(f"key from ${mp.api_key_env}")
        suffix = f"  [{', '.join(extras)}]" if extras else ""
        print(f"    {name:<16} {mp.provider:<14} {where}{suffix}")
    allowed = settings.governance.allowed_model_endpoints
    print(f"  egress allowlist: "
          f"{', '.join(allowed) if allowed else 'empty (no HTTP API endpoints allowed)'}")
    # Config-level problems that `scan` refuses on but that are worth surfacing here,
    # where people actually look for them.
    unpriced = settings.unpriced_role_models()
    if unpriced:
        print(f"\n⚠ models {unpriced} are used by a role but have NO pricing. "
              f"`scan` will refuse to start:\n"
              f"  with a $0.00 price the hard cost ceiling "
              f"(budget.max_cost_usd = {settings.budget.max_cost_usd}) can never fire.")
    free_models = settings.free_billing_models()
    if free_models:
        print(f"\nℹ models {free_models} are declared `billing: free`, so the hard cost "
              f"ceiling does not apply to them.\n"
              f"  Only budget.max_time_minutes "
              f"({settings.budget.max_time_minutes} min) bounds work on those models.")
    for role_a, role_b, model in settings.duplicate_judge_models():
        print(f"\n⚠ validation.judges '{role_a}' and '{role_b}' both resolve to model "
              f"'{model}' — the quorum is not a cross-model vote.")
    return EXIT_OK


def cmd_dry_run(args) -> int:
    settings = _load(args)
    try:
        from .estimate import estimate
    except ImportError:
        print("Could not import the estimator; check the installation and dependencies.",
              file=sys.stderr)
        return EXIT_FATAL
    estimate(settings)
    return EXIT_OK


def cmd_baseline(args) -> int:
    """Write baseline.json from an existing scan's report.json.

    Without this, ``baseline.mode: new_only`` — the shipped default — could never be
    bootstrapped: ``baseline.write_baseline`` existed but had no caller, so the file
    it expects never came into being and every finding looked new forever.

    Reads the fingerprints from the last run's report.json rather than re-scanning,
    so bootstrapping a baseline costs nothing.
    """
    settings = _load(args)
    import json as _json
    from .baseline import FINGERPRINT_SCHEME, write_baseline_fingerprints

    out_dir = Path(settings.output.out_dir)
    report = Path(args.report) if args.report else out_dir / "report.json"
    if not report.exists():
        print(f"error: no report at {report} — run a scan first, or pass --report.",
              file=sys.stderr)
        return EXIT_FATAL
    try:
        doc = _json.loads(report.read_text(encoding="utf-8"))
    except (OSError, _json.JSONDecodeError) as exc:
        print(f"error: could not read {report}: {exc}", file=sys.stderr)
        return EXIT_FATAL

    fps = sorted({f["fingerprint"] for f in doc.get("findings", [])
                  if f.get("fingerprint")})
    if args.out:
        target = Path(args.out)
    elif settings.baseline.file:
        target = Path(settings.baseline.file)
    else:
        # Writing ./baseline.json here and printing "these will no longer gate CI" was
        # a lie: `scan` reads `baseline.file`, which defaults to null, so
        # load_baseline(None) returns an empty set and the file is NEVER read. The
        # documented recipe silently did nothing. Refuse instead of pretending.
        print("error: baseline.file is not set in your config, so `scan` would never "
              "read the file this command writes.\n"
              "  Set it (e.g. `baseline: {file: ./baseline.json}`) and re-run, or pass "
              "an explicit --out PATH\n"
              "  together with that config key.", file=sys.stderr)
        return EXIT_FATAL
    target.parent.mkdir(parents=True, exist_ok=True)
    # MERGE, never overwrite. report.json only contains findings that were not already
    # baselined (that is what new_only means), so a blind overwrite on the second run
    # replaced the whole baseline with just the newest batch and every previously
    # accepted finding came back as net-new.
    added, retained = write_baseline_fingerprints(str(target), fps)
    total = added + retained
    print(f"Baseline {target}: +{added} new, {retained} retained → {total} "
          f"fingerprint(s) (scheme v{FINGERPRINT_SCHEME}).")
    if not added and fps:
        print("  (everything in this report was already baselined)")
    print("Findings in this baseline will no longer gate CI while "
          f"baseline.mode is 'new_only' (currently: {settings.baseline.mode}).")
    return EXIT_OK


def cmd_scan(args) -> int:
    settings = _load(args)
    # Name the config on stdout. With several files in play (config.yaml vs
    # config_api.yaml) "which one did this run use?" is the first question asked when two
    # runs disagree, and the hash alone cannot answer it.
    print(f"Config: {settings.config_source} (hash {settings.config_hash()})")
    try:
        from .orchestrator import Orchestrator
    except ImportError:
        print("Could not import the orchestrator; check the installation and dependencies.",
              file=sys.stderr)
        print(f"config validated OK (hash {settings.config_hash()}).", file=sys.stderr)
        return EXIT_FATAL
    orch = Orchestrator(settings, ci=args.ci,
                        limit=getattr(args, "limit", None),
                        assume_yes=getattr(args, "yes", False) or args.ci,
                        resume=getattr(args, "resume", False),
                        resume_force=getattr(args, "resume_force", False))
    return orch.run()


def cmd_mark(args) -> int:
    settings = _load(args)
    try:
        from .verification import mark_finding
    except ImportError:
        print("Could not import verification; check the installation and dependencies.",
              file=sys.stderr)
        return EXIT_FATAL
    # ValueError (unknown/ambiguous id, missing DB) is turned into a clean message by
    # main(); catch sqlite errors here so a schema/constraint problem also reports as
    # a normal CLI error rather than a raw traceback.
    import sqlite3
    try:
        mark_finding(settings, args.finding_id, result=args.result, notes=args.notes)
    except sqlite3.Error as exc:
        print(f"error: could not record verification: {exc}", file=sys.stderr)
        return EXIT_FATAL
    return EXIT_OK


def cmd_purge(args) -> int:
    settings = _load(args)
    out_dir = Path(settings.output.out_dir)
    state_db = out_dir / "scan_state.db"
    audit_dir = out_dir / "audit"
    removed = []
    if args.confirm:
        for p in (state_db, state_db.with_suffix(".db-wal"), state_db.with_suffix(".db-shm")):
            if p.exists():
                p.unlink()
                removed.append(str(p))
        if audit_dir.exists():
            import shutil
            for f in audit_dir.glob("*"):
                # `f.unlink()` raised IsADirectoryError on any subdirectory under audit/,
                # which escaped as a traceback (and therefore exit 1).
                try:
                    if f.is_dir():
                        shutil.rmtree(f, ignore_errors=True)
                    else:
                        f.unlink()
                except OSError as exc:
                    print(f"  (could not remove {f}: {exc})", file=sys.stderr)
            removed.append(str(audit_dir) + "/*")
        # purge is a RETENTION command, so leaving the actual findings on disk was the
        # opposite of what it promises.
        for name in ("report.json", "report.sarif", "report.md",
                     "executive_summary.md"):
            fp = out_dir / name
            if fp.exists():
                try:
                    fp.unlink()
                    removed.append(str(fp))
                except OSError as exc:
                    print(f"  (could not remove {fp}: {exc})", file=sys.stderr)
        pocs = out_dir / "pocs"
        if pocs.exists():
            import shutil as _sh
            _sh.rmtree(pocs, ignore_errors=True)
            removed.append(str(pocs))
        print("purged:", *removed, sep="\n  ")
    else:
        print("DRY RUN — would remove:")
        for p in (state_db, audit_dir, out_dir / "report.json", out_dir / "report.sarif",
                  out_dir / "report.md", out_dir / "executive_summary.md",
                  out_dir / "pocs"):
            print(f"  {p}")
        print("Re-run with --confirm to actually delete.")
    return EXIT_OK


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="WildGinger",
                                description="LLM-powered code vulnerability scanning harness")
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    def add_config(sp):
        # Default None, not "config.yaml": the resolver has to be able to tell an
        # EXPLICIT choice from a fallback. An explicit path that does not exist is a hard
        # error, never quietly replaced by another file — silently scanning with a
        # different config than the one you named would change every model, price,
        # threshold and egress rule while the report still looked normal.
        sp.add_argument("--config", default=None, metavar="PATH",
                        help=(f"config file to load. Keep several side by side "
                              f"(config.yaml, config_local.yaml) and "
                              f"pick one per run. Precedence: --config > "
                              f"${CONFIG_ENV_VAR} > ./config.yaml > the config shipped "
                              f"with the package."))

    def add_overrides(sp):
        """Optional CLI overrides that beat config.yaml when provided."""
        sp.add_argument("--path", default=None,
                        help="target folder to scan (overrides scope.path)")
        sp.add_argument("--output", default=None,
                        help="output directory (overrides output.out_dir)")
        sp.add_argument("--hunters", type=int, default=None,
                        help="parallel hunters (overrides concurrency.hunters)")
        sp.add_argument("--max-cost", type=float, default=None, dest="max_cost",
                        help="hard USD cost ceiling (overrides budget.max_cost_usd)")
        sp.add_argument("--limit", type=int, default=None,
                        help="scan only the first N cells (cheap smoke test)")
        sp.add_argument("--yes", action="store_true",
                        help="skip the cost-estimate confirmation prompt")
        sp.add_argument("--experiment", default=None,
                        help="MLflow experiment for this scan (e.g. the repo name); "
                             "auto-created in MLflow. Overrides monitoring.mlflow.experiment.")


    sp = sub.add_parser("validate", help="load + validate config")
    add_config(sp)
    sp.set_defaults(func=cmd_validate)

    sp = sub.add_parser("dry-run", help="project cells/tokens/cost/time without LLM calls")
    add_config(sp)
    add_overrides(sp)
    sp.set_defaults(func=cmd_dry_run)

    sp = sub.add_parser("scan", help="run a scan")
    add_config(sp)
    add_overrides(sp)
    sp.add_argument("--ci", action="store_true", help="non-interactive CI mode")
    sp.add_argument("--resume", action="store_true",
                    help="resume a previous (interrupted) scan from its checkpoint: "
                         "reload confirmed findings and skip already-analyzed cells")
    sp.add_argument("--resume-force", action="store_true", dest="resume_force",
                    help="allow --resume even if config.yaml changed since the "
                         "checkpoint (otherwise a config change refuses to resume)")
    sp.set_defaults(func=cmd_scan)


    sp = sub.add_parser("baseline",
                        help="write baseline.json from an existing report.json "
                             "(bootstraps baseline.mode: new_only)")
    add_config(sp)
    sp.add_argument("--output", default=None,
                    help="output directory holding report.json (overrides output.out_dir)")
    sp.add_argument("--report", default=None,
                    help="path to a specific report.json to harvest fingerprints from")
    sp.add_argument("--out", default=None,
                    help="where to write the baseline (default: baseline.file from config)")
    sp.set_defaults(func=cmd_baseline)

    sp = sub.add_parser("mark", help="record a human verification result for a finding")
    add_config(sp)
    sp.add_argument("finding_id")
    sp.add_argument("--result", required=True,
                    choices=["confirmed", "false_positive", "accepted_risk", "fixed"])
    sp.add_argument("--notes", default="")
    sp.set_defaults(func=cmd_mark)

    sp = sub.add_parser("purge", help="delete state/audit (retention)")
    add_config(sp)
    sp.add_argument("--confirm", action="store_true", help="actually delete")
    sp.set_defaults(func=cmd_purge)

    return p


def main(argv: list[str] | None = None) -> int:
    """Dispatch a subcommand and map every failure onto the documented exit codes.

    The handler below is deliberately broad. It used to catch only
    ``(FileNotFoundError, ValueError)``, which left a whole family of ordinary
    operator mistakes escaping as tracebacks — and an uncaught traceback exits **1**,
    the code documented as "findings at/above threshold". So a malformed config.yaml
    (``yaml.YAMLError``), a file passed to ``--path`` (``NotADirectoryError``), a
    directory passed to ``--config`` (``IsADirectoryError``), a read-only output dir
    (``PermissionError``) and a sqlite fault all looked exactly like "we found
    vulnerabilities" to CI. Everything unexpected is now a clean 3.
    """
    import yaml

    _install_signal_handlers()
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except SystemExit as exc:
        # A subcommand that already explained itself and chose a code.
        return int(exc.code) if isinstance(exc.code, int) else EXIT_FATAL
    except KeyboardInterrupt:
        print("\ninterrupted.", file=sys.stderr)
        return EXIT_INTERRUPTED
    except (OSError, ValueError, yaml.YAMLError) as exc:
        # OSError covers FileNotFoundError / NotADirectoryError / IsADirectoryError /
        # PermissionError / disk-full. ValueError covers pydantic ValidationError.
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_FATAL
    except Exception as exc:  # noqa: BLE001 — never exit 1 for a harness bug
        import traceback
        print(f"internal error: {type(exc).__name__}: {exc}", file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)
        print("This is a harness bug, not a scan result (exit 3).", file=sys.stderr)
        return EXIT_FATAL


if __name__ == "__main__":
    raise SystemExit(main())
