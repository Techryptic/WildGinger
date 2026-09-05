# Copyright 2026
# SPDX-License-Identifier: Apache-2.0
"""Human verification ingest — `scan mark`.

The no-execution tradeoff means a finding is a model judgment until a human runs
the PoC. This records the human verdict into the ``verifications`` table so it can
feed future dedupe/baseline/lifecycle decisions (a confirmed finding boosts
confidence on its pattern; a false positive suppresses re-nagging).
"""
from __future__ import annotations

import time
from pathlib import Path

from .db import Database
from .settings import Settings

_VALID = ("confirmed", "false_positive", "accepted_risk", "fixed")


def mark_finding(settings: Settings, finding_id: str, *, result: str, notes: str = "") -> None:
    """Record a human verdict for ``finding_id``.

    ``finding_id`` accepts any of the identifiers a user can actually see:

      * the short display id printed in reports and PoC filenames (``F-ab12``),
      * the full fingerprint from ``report.json``,
      * the raw database row id.

    Previously only a raw row id worked — and that is the one identifier never
    shown anywhere — so every realistic invocation died with an IntegrityError
    traceback (``finding_id`` is ``INTEGER NOT NULL REFERENCES findings(id)``).
    We now resolve the token to a real row before inserting, and raise a clear
    ValueError when it can't be resolved.
    """
    if result not in _VALID:
        raise ValueError(f"result must be one of {_VALID}")
    db_path = Path(settings.output.out_dir) / "scan_state.db"
    if not db_path.exists():
        raise ValueError(
            f"no scan database at {db_path} — run a scan first (or point --config at "
            f"the config whose output.out_dir holds this scan)."
        )
    db = Database(db_path)
    try:
        db.initialize()
        row_id = _resolve_finding_id(db, finding_id)
        if row_id is None:
            raise ValueError(
                f"no finding matches {finding_id!r}. Use the id shown in the report "
                f"(e.g. F-ab12), the full fingerprint from report.json, or the "
                f"numeric row id."
            )

        def _ins(cur):
            cur.execute(
                "INSERT INTO verifications(finding_id, human_result, notes, ts) "
                "VALUES (?,?,?,?);",
                (row_id, result, notes, time.time()),
            )
            cur.execute("UPDATE findings SET status = ? WHERE id = ?;", (result, row_id))
        db.write(_ins)
        print(f"Recorded verification: {finding_id} (row {row_id}) → {result}")
    finally:
        db.close()


def _resolve_finding_id(db: Database, token: str) -> int | None:
    """Map a user-supplied identifier to a ``findings.id``, or None.

    Order matters: an all-digits token is treated as a row id first, but we still
    verify it EXISTS (an unknown number used to blow up on the foreign key).
    """
    token = token.strip()

    def _q(cur):
        if token.isdigit():
            cur.execute("SELECT id FROM findings WHERE id = ?;", (int(token),))
            row = cur.fetchone()
            if row:
                return row["id"]
        # Full fingerprint (dedup_key) — exact match.
        cur.execute("SELECT id FROM findings WHERE dedup_key = ?;", (token,))
        row = cur.fetchone()
        if row:
            return row["id"]
        # Short display id: 'F-abcd' → dedup_key LIKE 'abcd%'. Also accept a bare
        # prefix so 'abcd' works the same way.
        prefix = token[2:] if token.lower().startswith("f-") else token
        if prefix:
            # Escape LIKE wildcards: `_` and `%` are metacharacters, so an id containing
            # them silently matched things it should not (or reported a spurious
            # ambiguity). Fingerprints are hex today, but the resolver also accepts
            # arbitrary user input.
            safe = (prefix.replace("\\", "\\\\")
                          .replace("%", "\\%").replace("_", "\\_"))
            cur.execute(
                "SELECT id FROM findings WHERE dedup_key LIKE ? ESCAPE '\\' "
                "ORDER BY id LIMIT 2;",
                (safe + "%",),
            )
            rows = cur.fetchall()
            if len(rows) == 1:
                return rows[0]["id"]
            if len(rows) > 1:
                raise ValueError(
                    f"{token!r} is ambiguous (matches {len(rows)}+ findings) — use the "
                    f"full fingerprint from report.json."
                )
        return None
    return db.read(_q)
