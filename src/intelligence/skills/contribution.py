"""
Contribution Health Skill

Answers: Is marketing going to hit the pipeline number this quarter?
Reads from snapshot["contribution"] and snapshot["targets"].
"""


def analyze(snapshot: dict) -> list[dict]:
    """Analyze pipeline contribution health and return prioritized findings."""
    contrib = snapshot["contribution"]
    targets = snapshot["targets"]
    findings = []

    report_date = snapshot["report_date"]
    quarter = snapshot["quarter"]

    # ── Check 1: Pipeline pacing vs quarterly target ──────────────────────
    # The most important contribution finding
    pipeline_created = contrib["pipeline_created_qtd"]
    pipeline_target = targets["pipeline_created"]
    pacing = contrib["pipeline_pacing_pct"]

    if pacing < 0.90:  # Behind pace
        gap_usd = pipeline_target - pipeline_created
        severity = "critical" if pacing < 0.75 else "warning"

        # Calculate what's needed per week to close the gap
        # Rough: assume ~13 weeks in a quarter, figure out weeks remaining
        from datetime import datetime
        rd = datetime.strptime(report_date, "%Y-%m-%d")
        quarter_num = (rd.month - 1) // 3 + 1
        if quarter_num < 4:
            quarter_end_month = quarter_num * 3
            quarter_end = datetime(rd.year, quarter_end_month + 1, 1) if quarter_end_month < 12 else datetime(rd.year + 1, 1, 1)
        else:
            quarter_end = datetime(rd.year + 1, 1, 1)
        weeks_remaining = max(1, (quarter_end - rd).days / 7)
        weekly_needed = gap_usd / weeks_remaining

        findings.append({
            "severity": severity,
            "domain": "contribution",
            "metric": "Pipeline pacing",
            "current_value": pipeline_created,
            "target_value": pipeline_target,
            "prior_value": None,
            "change_pct": pacing - 1.0,
            "finding": (
                f"Marketing has created ${pipeline_created:,.0f} of ${pipeline_target:,.0f} "
                f"pipeline target ({pacing:.0%}). ${gap_usd:,.0f} gap remaining with "
                f"{weeks_remaining:.0f} weeks left in {quarter}. "
                f"Need ${weekly_needed:,.0f}/week to close the gap."
            ),
            "action": (
                f"Prioritize high-velocity pipeline sources. "
                f"Outbound and event-driven motions close faster than content-sourced pipeline. "
                f"Consider an SDR blitz targeting open Tier 1 accounts."
            ),
        })
    elif pacing >= 1.0:
        findings.append({
            "severity": "info",
            "domain": "contribution",
            "metric": "Pipeline pacing",
            "current_value": pipeline_created,
            "target_value": pipeline_target,
            "prior_value": None,
            "change_pct": pacing - 1.0,
            "finding": (
                f"Pipeline creation is at ${pipeline_created:,.0f}, "
                f"{pacing:.0%} of the ${pipeline_target:,.0f} quarterly target. On or ahead of pace."
            ),
            "action": "Maintain current trajectory. Focus on pipeline quality and conversion rather than additional volume.",
        })

    # ── Check 2: Won revenue vs target ────────────────────────────────────
    won_target = targets["pipeline_won"]
    won_actual = contrib["won_acv_qtd"]
    won_pacing = won_actual / won_target if won_target > 0 else 0

    if won_pacing < 0.80:
        gap_usd = won_target - won_actual
        severity = "critical" if won_pacing < 0.60 else "warning"
        findings.append({
            "severity": severity,
            "domain": "contribution",
            "metric": "Won revenue QTD",
            "current_value": won_actual,
            "target_value": won_target,
            "prior_value": None,
            "change_pct": won_pacing - 1.0,
            "finding": (
                f"Marketing-sourced closed-won is ${won_actual:,.0f} of "
                f"${won_target:,.0f} target ({won_pacing:.0%}). "
                f"${gap_usd:,.0f} gap. This is a harder number to recover than pipeline — "
                f"it requires deals already in late stage to close."
            ),
            "action": (
                f"Review open pipeline in Proposal/Negotiation and Procurement stages. "
                f"Identify deals that can be accelerated with executive sponsorship, "
                f"competitive intel, or pricing flexibility."
            ),
        })

    # ── Check 3: Pipeline by deal source ──────────────────────────────────
    by_source = contrib.get("pipeline_by_source", [])
    if by_source:
        total_pipeline = sum(s["pipeline_usd"] for s in by_source)
        if total_pipeline > 0:
            # Find the dominant source and any underperformers
            sorted_sources = sorted(by_source, key=lambda s: s["pipeline_usd"], reverse=True)
            top_source = sorted_sources[0]
            top_pct = top_source["pipeline_usd"] / total_pipeline

            # Check if Inbound is too dominant (>70%) — concentration risk
            inbound = next((s for s in by_source if s["source"] == "Inbound"), None)
            if inbound and (inbound["pipeline_usd"] / total_pipeline) > 0.70:
                findings.append({
                    "severity": "warning",
                    "domain": "contribution",
                    "metric": "Pipeline source concentration",
                    "current_value": inbound["pipeline_usd"] / total_pipeline,
                    "target_value": targets.get("marketing_sourced_pipeline_pct"),
                    "prior_value": None,
                    "change_pct": None,
                    "finding": (
                        f"Inbound sources account for {inbound['pipeline_usd'] / total_pipeline:.0%} "
                        f"of QTD pipeline (${inbound['pipeline_usd']:,.0f} of ${total_pipeline:,.0f}). "
                        f"Over-reliance on one motion creates risk if inbound volume softens."
                    ),
                    "action": (
                        "Diversify pipeline sources. Increase outbound coverage on Tier 1 accounts. "
                        "Activate partner co-sell motions. Consider event-driven pipeline sprints."
                    ),
                })

            # Report the source mix as an info finding
            source_summary = ", ".join(
                f"{s['source']}: ${s['pipeline_usd']:,.0f} ({s['pipeline_usd']/total_pipeline:.0%})"
                for s in sorted_sources[:4]
            )
            findings.append({
                "severity": "info",
                "domain": "contribution",
                "metric": "Pipeline by source",
                "current_value": total_pipeline,
                "target_value": None,
                "prior_value": None,
                "change_pct": None,
                "finding": f"QTD pipeline by source: {source_summary}.",
                "action": "No action required — informational breakdown for pipeline review.",
            })

    # ── Check 4: Pipeline by first-touch channel ─────────────────────────
    by_channel = contrib.get("pipeline_by_channel_first_touch", [])
    if len(by_channel) >= 3:
        # Top 3 channels driving pipeline
        top_3 = by_channel[:3]
        bottom_3 = by_channel[-3:] if len(by_channel) >= 6 else by_channel[-2:]

        top_summary = ", ".join(
            f"{c['channel']} (${c['pipeline_usd']:,.0f}, {c['opp_count']} opps)"
            for c in top_3
        )
        findings.append({
            "severity": "info",
            "domain": "contribution",
            "metric": "Top pipeline channels (first-touch)",
            "current_value": sum(c["pipeline_usd"] for c in top_3),
            "target_value": None,
            "prior_value": None,
            "change_pct": None,
            "finding": (
                f"Top pipeline-generating channels by first-touch attribution: {top_summary}. "
                f"These channels are sourcing the most opportunities regardless of where the lead later engaged."
            ),
            "action": "Protect budget for top-performing channels. Consider increasing investment if CPL is within target.",
        })

    # ── Check 5: Open pipeline health ─────────────────────────────────────
    open_pipeline = contrib["open_pipeline"]
    if pipeline_created > 0:
        open_ratio = open_pipeline / pipeline_created
        if open_ratio > 0.75:
            # Most pipeline is still open — lots of unresolved deals
            findings.append({
                "severity": "info",
                "domain": "contribution",
                "metric": "Open pipeline ratio",
                "current_value": open_pipeline,
                "target_value": None,
                "prior_value": None,
                "change_pct": None,
                "finding": (
                    f"${open_pipeline:,.0f} of ${pipeline_created:,.0f} QTD pipeline "
                    f"is still open ({open_ratio:.0%}). "
                    f"${won_actual:,.0f} has closed-won so far."
                ),
                "action": (
                    "Review stage distribution of open pipeline. "
                    "Deals in Discovery/Qualification may not close this quarter. "
                    "Focus sales support on deals in Proposal/Negotiation and later stages."
                ),
            })

    # Sort by severity
    severity_order = {"critical": 0, "warning": 1, "info": 2}
    findings.sort(key=lambda f: severity_order.get(f["severity"], 3))

    return findings[:5]
