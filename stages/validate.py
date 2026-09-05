# Copyright 2026
# SPDX-License-Identifier: Apache-2.0
"""Validate stage — adversarial, blind to hunter reasoning.

A DIFFERENT model (role 'validate') gets ONLY the finding claim + location — never
the hunter's transcript — and tries to disprove it with its own fresh navigation.
Fail-open-never-drop: an unparseable/UNCERTAIN verdict keeps the finding for human
review (flagged low confidence) rather than silently discarding it.
"""
from __future__ import annotations

from ..agent import find_last_json
from ..context import ScanContext
from ..findings import Finding
from ..prompts.templates import VALIDATE_PROMPT


def validate_finding_quorum(ctx: ScanContext, finding: Finding) -> Finding:
    """Multi-judge validation. Run several DIFFERENT judge models and
    require a quorum to CONFIRM. ``votes`` = number of judges that voted CONFIRMED
    (a real, cross-model vote). Falls back to single-judge if no judges configured.
    """
    v = ctx.settings.validation
    judge_roles = [r for r in v.judges if r in ctx.settings.roles]
    if not judge_roles:
        return validate_finding(ctx, finding)  # legacy single 'validate' role

    # A quorum bigger than the ACTUAL panel can never be met, so every finding would
    # silently stay 'new'. Settings validates this for the configured list, but a judge
    # role can also be dropped here for not existing in `roles` — clamp and say so.
    quorum = v.quorum
    if quorum > len(judge_roles):
        ctx.audit.event("quorum_clamped", configured=v.quorum,
                        available_judges=len(judge_roles), judges=judge_roles)
        quorum = len(judge_roles)

    prior_confidence = finding.confidence
    prior_reachable = finding.reachable
    confirmed_votes = 0
    rejected_votes = 0
    reach_votes: dict[str, int] = {}
    unavailable = 0
    verdicts: list[dict[str, str]] = []
    # Deliberately NO early exit once quorum is reached:
    #   * `votes` is the CONFIRMED count and feeds confidence (0.5 + 0.15 * votes), so
    #     stopping at 2 of 3 reports 0.80 for a finding all three judges would have
    #     confirmed at 0.95 — a worse number, silently, for a stronger finding;
    #   * the reachability tally becomes order-dependent (an unpolled judge cannot vote),
    #     so the same finding gets different verdicts depending on judge order;
    #   * the `judge_panel` audit event exists to show WHICH judge disagreed, and an
    #     unpolled judge has no opinion to show.
    for role in judge_roles:
        verdict, reachable, evidence, voted = _run_one_judge(ctx, role, finding)
        verdicts.append({"role": role, "verdict": verdict, "reachable": reachable,
                         "evidence": evidence, "voted": str(voted)})
        if evidence:
            finding.evidence.append(f"{role}: {evidence}")
        if not voted:
            unavailable += 1
            # A judge that never answered abstains from EVERY tally, including
            # reachability. Counting its placeholder verdict meant two failed calls
            # could out-vote the one judge that actually replied.
            continue
        if verdict == "CONFIRMED":
            confirmed_votes += 1
        elif verdict == "REJECTED":
            rejected_votes += 1
        if reachable in ("reachable", "unreachable", "unclear"):
            reach_votes[reachable] = reach_votes.get(reachable, 0) + 1

    finding.votes = confirmed_votes  # real cross-model vote count
    # Reachability = majority vote among judges that ACTUALLY VOTED (best available
    # signal pre-trace). With no votes at all we keep what we came in with rather than
    # inventing a verdict — the deterministic gate's value is better than nothing.
    # Sort the keys so a tie resolves deterministically instead of by dict order.
    if reach_votes:
        # Only judges that EXPLICITLY stated a reachability are in `reach_votes` (see
        # _run_one_judge: a missing field is "" and never tallied). Judges that actually
        # answer and disagree with the deterministic walk are allowed to move the
        # verdict — the call graph is name-based and over-approximates, so an
        # independent read of the code is real evidence. What must not happen is a
        # DEFAULTED "unclear" out-voting a proof, which is what the sentinel fixes.
        finding.reachable = max(sorted(reach_votes), key=reach_votes.get)
    else:
        finding.reachable = prior_reachable
    if unavailable:
        # Make a degraded panel visible in the report: a finding judged by 1 of 3 is
        # not the same evidence as one judged by 3 of 3, and the old code left that
        # indistinguishable.
        finding.evidence.append(
            f"validation: {len(judge_roles) - unavailable}/{len(judge_roles)} judge(s) "
            f"responded; {unavailable} unavailable (their verdicts were not counted)."
        )

    if confirmed_votes >= quorum:
        finding.status = "confirmed"
        # Judges AGREEING must never LOWER confidence. The old formula was absolute,
        # so a deterministic rule finding at 0.9 got knocked down to 0.8 by a 2-of-3
        # confirmation — being proven right made it look weaker.
        finding.confidence = max(prior_confidence,
                                 min(1.0, 0.5 + 0.15 * confirmed_votes))
    elif rejected_votes >= quorum:
        finding.status = "false_positive"
        finding.confidence = max(0.0, prior_confidence - 0.4)
    elif len(judge_roles) - unavailable < quorum:
        # The panel could not REACH quorum because too few judges answered — a
        # transport problem, not a disagreement. Capping confidence here punished the
        # finding for our own outage: every finding in a throttle window was reported at
        # 0.20, including deterministic `rule` findings that carry 0.9 by construction,
        # and because enrich() latches base_confidence from this value, 0.20 is what
        # landed in report.json. "The panel was down" and "the panel couldn't agree" are
        # different facts and must not look the same.
        finding.status = "new"
        finding.confidence = prior_confidence
        finding.evidence.append(
            "validation: not enough judges responded to reach quorum — confidence is "
            "the pre-validation value, NOT a panel verdict.")
        ctx.audit.event("validation_unavailable", fingerprint=finding.fingerprint(),
                        responded=len(judge_roles) - unavailable,
                        judges=len(judge_roles), quorum=quorum)
    else:
        # A real no-quorum: enough judges voted, they just did not agree. Keep the
        # finding for human review (fail-open, never drop), but cap confidence so it
        # sorts below anything the panel actually agreed on.
        finding.status = "new"
        finding.confidence = max(0.1, min(prior_confidence,
                                          0.2 + 0.1 * confirmed_votes))

    # Record the panel's individual verdicts — otherwise a confirmation is an
    # unauditable number and there's no way to see WHICH judge disagreed.
    ctx.audit.event("judge_panel", fingerprint=finding.fingerprint(),
                    file_rel=finding.file_rel, line_start=finding.line_start,
                    category=finding.category, status=finding.status,
                    quorum=quorum, confirmed=confirmed_votes,
                    rejected=rejected_votes, unavailable=unavailable,
                    reachable=finding.reachable, verdicts=verdicts)
    # Carry the per-judge verdicts on the finding so _persist_findings can write them
    # to `finding_evidence` once the row (and therefore its id) exists. Writing them
    # here is impossible: validation runs BEFORE findings are persisted, and
    # finding_evidence.finding_id is NOT NULL.
    finding.judge_verdicts = verdicts
    return finding


def _run_one_judge(ctx: ScanContext, role: str, finding: Finding
                   ) -> tuple[str, str, str, bool]:
    """Run a single judge model.

    Returns ``(verdict, reachable, evidence, voted)``. ``voted`` is False when the
    judge never actually produced an opinion (transport failure, exhausted retries,
    unparseable reply). Callers MUST NOT count a non-voting judge's ``reachable``
    value: a failed call is the *absence* of evidence, not evidence of un-reachability
    — and counting it let two dead calls out-vote the one judge that answered,
    demoting a reachable finding to "unclear" and costing 30% of its severity.
    """
    agent_id = f"{role}-{finding.fingerprint()}"
    # Judges receive the claim, not the hunter's transcript, and use navigation
    # tools to independently check the code.
    agent = ctx.make_agent(role, agent_id, with_tools=True)
    prompt = VALIDATE_PROMPT.format(
        file_rel=finding.file_rel,
        line_start=finding.line_start,
        line_end=finding.line_end,
        category=finding.category,
        title=finding.title,
        description=finding.description,
        cited_snippet=finding.cited_snippet,
    )
    result = agent.run(prompt)
    if result.error:
        # Record a CLASSIFIED reason, not the raw exception. The raw string was
        # appended to finding.evidence and serialized into report.json, leaking model
        # ARNs / endpoint hosts / request ids into an artifact that gets shared. The
        # full detail is still in the audit log, which is 0600 and stays local.
        ctx.audit.event("judge_error", role=role, agent_id=agent_id,
                        error=result.error, stop_reason=result.stop_reason)
        return ("UNCERTAIN", "",
                f"({role} unavailable: {_judge_error_kind(result)})", False)
    parsed = find_last_json(result.tagged("verdict") or result.final_text)
    if isinstance(parsed, dict):
        # A MISSING `reachable` key is not an opinion. Defaulting it to "unclear" cast a
        # real vote, and three judges that simply omitted the field could overrule the
        # deterministic call-graph proof — turning `reachable` into `unclear` and costing
        # the finding a whole severity tier (_REACH_MULT 1.0 -> 0.7). Use "" for "did not
        # say" and let the caller ignore it.
        raw_reach = parsed.get("reachable")
        reach = str(raw_reach).lower() if raw_reach is not None else ""
        return (
            str(parsed.get("verdict", "UNCERTAIN")).upper(),
            reach,
            str(parsed.get("evidence", "")),
            True,
        )
    # The model replied but we couldn't parse a verdict out of it — it expressed no
    # usable opinion, so it must not vote on reachability either.
    ctx.audit.event("judge_unparseable", role=role, agent_id=agent_id,
                    stop_reason=result.stop_reason)
    return "UNCERTAIN", "", "", False


def _judge_error_kind(result) -> str:
    """Short, non-sensitive description of why a judge produced no verdict.

    Deliberately a small fixed vocabulary: anything derived from the provider's
    message could carry account/endpoint identifiers into the report.
    """
    if getattr(result, "stop_reason", "") == "retries_exhausted":
        return "transport retries exhausted"
    if getattr(result, "stop_reason", "") == "max_turns":
        return "turn budget exhausted"
    err = (getattr(result, "error", "") or "").lower()
    if "throttl" in err or "toomanyrequests" in err or "rate" in err:
        return "rate limited"
    if "timeout" in err or "timed out" in err:
        return "timed out"
    if "accessdenied" in err or "unauthor" in err or "forbidden" in err:
        return "access denied"
    if "validation" in err:
        return "request rejected by the model"
    return "model call failed"


def validate_finding(ctx: ScanContext, finding: Finding) -> Finding:
    """Return the finding with status/confidence updated by adversarial review."""
    agent_id = f"validate-{finding.fingerprint()}"
    agent = ctx.make_agent("validate", agent_id, with_tools=True)
    prompt = VALIDATE_PROMPT.format(

        file_rel=finding.file_rel,
        line_start=finding.line_start,
        line_end=finding.line_end,
        category=finding.category,
        title=finding.title,
        description=finding.description,
        cited_snippet=finding.cited_snippet,
    )
    result = agent.run(prompt)

    prior_confidence = finding.confidence
    verdict = "UNCERTAIN"
    reachable = finding.reachable
    evidence = ""
    unavailable = bool(result.error)
    if not unavailable:
        parsed = find_last_json(result.tagged("verdict") or result.final_text)
        if isinstance(parsed, dict):
            verdict = str(parsed.get("verdict", "UNCERTAIN")).upper()
            # Only an EXPLICIT value counts (see _run_one_judge): defaulting a missing
            # key to the incoming value is fine, but a garbage string must not silently
            # become "unclear" and cost a severity tier.
            raw_reach = parsed.get("reachable")
            if raw_reach is not None:
                reachable = str(raw_reach).lower()
            evidence = str(parsed.get("evidence", ""))
        else:
            unavailable = True   # replied, but with nothing usable

    finding.reachable = (reachable if reachable in ("reachable", "unreachable", "unclear")
                         else finding.reachable)
    if evidence:
        finding.evidence.append(f"validator: {evidence}")

    if verdict == "CONFIRMED":
        finding.status = "confirmed"
        # Bounded by the same rule the quorum path uses: agreement must never LOWER
        # confidence, and repeat calls must not keep inflating it.
        finding.confidence = max(prior_confidence, min(1.0, 0.65))
    elif verdict == "REJECTED":
        finding.status = "false_positive"
        finding.confidence = max(0.0, prior_confidence - 0.4)
    elif unavailable:
        # The judge never produced an opinion (transport failure / unusable reply). That
        # is the ABSENCE of evidence: docking 0.1 punished the finding for our outage,
        # and repeated resumes compounded it.
        finding.status = "new"
        finding.confidence = prior_confidence
        finding.evidence.append(
            "validation: the validator was unavailable — confidence is the "
            "pre-validation value, NOT a verdict.")
    else:  # a real UNCERTAIN verdict → fail open, keep for human review
        finding.status = "new"
        finding.confidence = max(0.0, prior_confidence - 0.1)
    return finding
