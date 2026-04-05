"""
Contribution Health Skill (v2)

Answers: Is marketing going to hit the pipeline number this quarter?
And if not, what can realistically be done about it?

Core shift from v1: Replaces point-in-time pacing with velocity trajectory,
projects quarter-end outcome, assesses what's closeable from current pipeline,
and evaluates scalability of recommended actions.

Reads from: snapshot["contribution"], snapshot["pipeline_trajectory"],
            snapshot["channel_economics"], snapshot["sdr_capacity"], snapshot["targets"]
"""


def analyze(snapshot: dict) -> list[dict]:
    """Analyze pipeline contribution with trajectory and actionability."""
    contrib = snapshot["contribution"]
    traj = snapshot["pipeline_trajectory"]
    econ = snapshot["channel_economics"]
    sdr = snapshot.get("sdr_capacity", {})
    targets = snapshot["targets"]
    findings = []

    # ── Check 1: Pipeline trajectory + projection ─────────────────────────
    # The most important finding: not "where are we" but "where will we land"
    actual = traj["actual_qtd"]
    target = traj["target"]
    pacing = traj["pacing_pct"]
    trend = traj["velocity_trend"]
    proj_trailing = traj["projection_trailing_avg"]
    proj_linear = traj["projection_linear"]
    trailing_avg = traj["trailing_3wk_avg"]
    weeks_remaining = traj["weeks_remaining"]
    slope = traj["velocity_slope_per_week"]

    # Use the more conservative projection
    projection = min(proj_trailing, proj_linear)
    projected_pct = projection / target if target > 0 else 0

    gap = target - actual
    weekly_needed = gap / max(1, weeks_remaining)

    if projected_pct < 0.90:
        severity = "critical" if projected_pct < 0.80 else "warning"

        trend_context = ""
        if trend == "decelerating":
            trend_context = (
                f" Pipeline velocity is decelerating (${abs(slope):,.0f}/week decline). "
                f"Trailing 3-week average is ${trailing_avg:,.0f}/week, "
                f"down from the Q4 average of ${actual / max(1, traj['weeks_elapsed']):,.0f}/week."
            )
        elif trend == "accelerating":
            trend_context = (
                f" Pipeline velocity is accelerating (${slope:,.0f}/week increase), "
                f"which improves the outlook."
            )

        findings.append({
            "severity": severity,
            "domain": "contribution",
            "metric": "Pipeline trajectory",
            "current_value": actual,
            "target_value": target,
            "prior_value": None,
            "change_pct": pacing - 1.0,
            "finding": (
                f"Q4 pipeline is at ${actual:,.0f} of ${target:,.0f} target ({pacing:.0%}). "
                f"Projected to finish at ${projection:,.0f} ({projected_pct:.0%} of target) "
                f"based on recent velocity.{trend_context} "
                f"Gap: ${gap:,.0f}. Need ${weekly_needed:,.0f}/week for remaining "
                f"{weeks_remaining} weeks to close the gap — "
                f"{'achievable' if weekly_needed < trailing_avg * 1.3 else 'unlikely'} "
                f"at current trajectory."
            ),
            "action": (
                f"Focus on high-velocity pipeline sources that can close within Q4. "
                f"New pipeline created this late in the quarter won't close — "
                f"the action is accelerating existing open deals, not generating new ones."
            ),
        })
    else:
        findings.append({
            "severity": "info",
            "domain": "contribution",
            "metric": "Pipeline trajectory",
            "current_value": actual,
            "target_value": target,
            "prior_value": None,
            "change_pct": pacing - 1.0,
            "finding": (
                f"Q4 pipeline is at ${actual:,.0f} ({pacing:.0%} of target). "
                f"Projected to finish at ${projection:,.0f} ({projected_pct:.0%}). "
                f"Velocity is {trend}."
            ),
            "action": "On track. Focus on pipeline quality and conversion rather than volume.",
        })

    # ── Check 2: Closeable pipeline — what can actually convert? ──────────
    open_pipe = traj["open_pipeline"]
    closeable = open_pipe["closeable_usd"]
    closeable_pct = open_pipe["closeable_pct"]
    total_open = open_pipe["total_usd"]

    won_qtd = contrib["won_acv_qtd"]
    won_target = targets["pipeline_won"]
    won_gap = won_target - won_qtd

    if won_gap > 0:
        can_close_gap = closeable >= won_gap

        findings.append({
            "severity": "info" if can_close_gap else "warning",
            "domain": "contribution",
            "metric": "Closeable pipeline vs won target",
            "current_value": closeable,
            "target_value": won_gap,
            "prior_value": None,
            "change_pct": None,
            "finding": (
                f"Won revenue QTD: ${won_qtd:,.0f} of ${won_target:,.0f} target "
                f"(${won_gap:,.0f} gap). "
                f"${closeable:,.0f} in late-stage pipeline (Proposal/Negotiation, "
                f"Security Review, Procurement) — {closeable_pct:.0%} of ${total_open:,.0f} open. "
                f"{'Enough late-stage pipeline exists to close the gap if conversion holds.' if can_close_gap else 'Not enough late-stage pipeline to close the won revenue gap this quarter.'}"
            ),
            "action": (
                f"Prioritize the {sum(s['opps'] for s in open_pipe['by_stage'] if s['closeable_this_quarter'])} "
                f"deals in Proposal+ stages (${closeable:,.0f} total). "
                f"Identify which need executive sponsorship, competitive intel, or pricing flexibility "
                f"to close before quarter-end."
            ),
        })

    # ── Check 3: Pipeline by stage — is the funnel balanced? ──────────────
    stages = open_pipe.get("by_stage", [])
    if stages:
        early = sum(s["pipeline_usd"] for s in stages if not s["closeable_this_quarter"])
        late = sum(s["pipeline_usd"] for s in stages if s["closeable_this_quarter"])

        if early > 0 and late > 0:
            early_pct = early / (early + late)
            # If >70% is in early stages, most won't close this quarter
            if early_pct > 0.60:
                findings.append({
                    "severity": "warning",
                    "domain": "contribution",
                    "metric": "Pipeline stage distribution",
                    "current_value": early_pct,
                    "target_value": None,
                    "prior_value": None,
                    "change_pct": None,
                    "finding": (
                        f"{early_pct:.0%} of open pipeline (${early:,.0f}) is in "
                        f"Discovery/Qualification/Demo stages — unlikely to close this quarter. "
                        f"Only ${late:,.0f} is in late stages."
                    ),
                    "action": (
                        "Focus sales resources on converting mid-stage deals (Demo/Evaluation) "
                        "to Proposal stage. Each deal that advances increases closeable pipeline."
                    ),
                })

    # ── Check 4: Pipeline by source + scalability assessment ──────────────
    by_source = contrib.get("pipeline_by_source", [])
    if by_source:
        total_pipeline = sum(s["pipeline_usd"] for s in by_source)

        # Find the best-performing motion
        sorted_sources = sorted(by_source, key=lambda s: s["pipeline_usd"], reverse=True)

        # Outbound scalability check using SDR capacity
        outbound = next((s for s in by_source if s["source"] == "Outbound"), None)
        if outbound and sdr:
            utilization = sdr.get("utilization_pct", 0)
            high_intent = sdr.get("tier1_high_surging_intent", 0)
            active_sdrs = sdr.get("active_sdrs", 0)

            if utilization < 0.75 and high_intent > 50:
                spare_capacity = (0.75 - utilization) * active_sdrs
                potential_additional_meetings = spare_capacity * 20  # business days remaining

                findings.append({
                    "severity": "info",
                    "domain": "contribution",
                    "metric": "Outbound scalability",
                    "current_value": outbound["pipeline_usd"],
                    "target_value": None,
                    "prior_value": None,
                    "change_pct": None,
                    "finding": (
                        f"Outbound has sourced ${outbound['pipeline_usd']:,.0f} QTD "
                        f"({outbound['pipeline_usd'] / total_pipeline:.0%} of total). "
                        f"SDR team is at {utilization:.0%} utilization ({active_sdrs} active SDRs, "
                        f"{sdr.get('meetings_per_sdr_per_day', 0)} meetings/day vs ~3.0 capacity). "
                        f"{high_intent} Tier 1 accounts show High/Surging intent signals this quarter."
                    ),
                    "action": (
                        f"Activate SDR blitz on the {high_intent} high-intent Tier 1 accounts. "
                        f"At current utilization there's capacity for ~{potential_additional_meetings:.0f} "
                        f"additional meetings before quarter-end. "
                        f"Outbound pipeline won't close this Q but seeds Q1."
                    ),
                })

    # ── Check 5: Channel contribution — pipeline ROI leaders ──────────────
    paid_econ = [c for c in econ if c["is_paid"] and c["pipeline_roi"] is not None]
    if paid_econ:
        by_roi = sorted(paid_econ, key=lambda c: c["pipeline_roi"], reverse=True)
        top = by_roi[0]
        total_paid_pipeline = sum(c["pipeline_usd"] for c in paid_econ)

        findings.append({
            "severity": "info",
            "domain": "contribution",
            "metric": "Channel pipeline efficiency",
            "current_value": top["pipeline_roi"],
            "target_value": None,
            "prior_value": None,
            "change_pct": None,
            "finding": (
                f"{top['channel']} leads paid channels in pipeline ROI at {top['pipeline_roi']}x "
                f"(${top['pipeline_usd']:,.0f} from ${top['spend_usd']:,.0f} spend, "
                f"avg ACV ${top['avg_opp_acv']:,.0f}). "
                f"Total paid pipeline: ${total_paid_pipeline:,.0f}."
            ),
            "action": (
                f"Protect and expand {top['channel']} investment. "
                f"At {top['pipeline_roi']}x ROI, every additional dollar spent here "
                f"generates ${top['pipeline_roi']:.0f} in pipeline."
            ),
        })

    # Sort by severity
    severity_order = {"critical": 0, "warning": 1, "info": 2}
    findings.sort(key=lambda f: severity_order.get(f["severity"], 3))

    return findings[:6]