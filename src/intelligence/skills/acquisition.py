"""
Acquisition Health Skill

Answers: Are we spending effectively and driving enough top-of-funnel activity?
Reads from snapshot["acquisition"] and snapshot["targets"].
"""


def analyze(snapshot: dict) -> list[dict]:
    """Analyze acquisition health and return prioritized findings."""
    acq = snapshot["acquisition"]
    targets = snapshot["targets"]
    findings = []

    # ── Check 1: CPL vs target by channel ─────────────────────────────────
    # The most actionable acquisition finding — where is money being wasted?
    worst_cpl_gap = None
    for ch in acq["cpl_by_channel"]:
        if ch["current"] is None or ch["target"] is None:
            continue
        gap_pct = (ch["current"] - ch["target"]) / ch["target"]
        if gap_pct > 0.10:  # Only flag if >10% above target
            severity = "critical" if gap_pct > 0.30 else "warning"
            finding = {
                "severity": severity,
                "domain": "acquisition",
                "metric": f"{ch['channel']} CPL",
                "current_value": ch["current"],
                "target_value": ch["target"],
                "prior_value": ch["prior"],
                "change_pct": ch["change_pct"],
                "finding": (
                    f"{ch['channel']} CPL is ${ch['current']:.0f}, "
                    f"{gap_pct:.0%} above the ${ch['target']:.0f} target."
                    + (f" Up from ${ch['prior']:.0f} last week." if ch['prior'] else "")
                ),
                "action": (
                    f"Review {ch['channel']} campaigns for underperformers. "
                    f"Pause bottom-performing ad sets and reallocate "
                    f"${(ch['current'] - ch['target']) * 10:.0f}+ weekly to channels at or below target."
                ),
            }
            findings.append(finding)
            if worst_cpl_gap is None or gap_pct > worst_cpl_gap:
                worst_cpl_gap = gap_pct

    # ── Check 2: Budget pacing ────────────────────────────────────────────
    # Overspend = burning budget early. Underspend = pipeline risk later.
    for bp in acq["budget_pacing"]:
        if bp["pacing_pct"] is None or bp["expected_pct"] is None:
            continue
        pace_delta = bp["pacing_pct"] - bp["expected_pct"]

        if abs(pace_delta) > 0.15:
            if pace_delta > 0:
                severity = "critical" if pace_delta > 0.25 else "warning"
                direction = "ahead of"
                risk = "Budget will exhaust before month-end at current rate."
                action = f"Reduce {bp['channel']} daily spend by {pace_delta:.0%} to align with monthly budget of ${bp['budget']:,.0f}."
            else:
                severity = "critical" if pace_delta < -0.25 else "warning"
                direction = "behind"
                risk = "Underspend risks missing pipeline targets for the quarter."
                action = f"Increase {bp['channel']} daily spend to utilize remaining ${bp['budget'] - bp['spent_mtd']:,.0f} budget."

            findings.append({
                "severity": severity,
                "domain": "acquisition",
                "metric": f"{bp['channel']} budget pacing",
                "current_value": bp["pacing_pct"],
                "target_value": bp["expected_pct"],
                "prior_value": None,
                "change_pct": None,
                "finding": (
                    f"{bp['channel']} has spent ${bp['spent_mtd']:,.0f} of ${bp['budget']:,.0f} monthly budget "
                    f"({bp['pacing_pct']:.0%} spent, expected {bp['expected_pct']:.0%}). "
                    f"{pace_delta:+.0%} {direction} pace. {risk}"
                ),
                "action": action,
            })

    # ── Check 3: Session trend WoW ────────────────────────────────────────
    # Total sessions drop
    total_change = acq["sessions"]["change_pct"]
    if total_change is not None and total_change < -0.10:
        findings.append({
            "severity": "warning" if total_change > -0.20 else "critical",
            "domain": "acquisition",
            "metric": "Total sessions WoW",
            "current_value": acq["sessions"]["current"],
            "target_value": None,
            "prior_value": acq["sessions"]["prior"],
            "change_pct": total_change,
            "finding": (
                f"Total web sessions dropped {abs(total_change):.0%} WoW "
                f"({acq['sessions']['prior']:,} → {acq['sessions']['current']:,})."
            ),
            "action": "Identify which channels drove the decline (see channel breakdown below) and assess whether this is seasonal or structural.",
        })

    # Individual channel drops >15%
    channel_drops = []
    for ch in acq["sessions_by_channel"]:
        if ch["change_pct"] is not None and ch["change_pct"] < -0.15 and ch["prior"] > 100:
            channel_drops.append(ch)

    if channel_drops:
        # Report the worst drop, not all of them
        worst = min(channel_drops, key=lambda x: x["change_pct"])
        findings.append({
            "severity": "warning",
            "domain": "acquisition",
            "metric": f"{worst['channel']} sessions WoW",
            "current_value": worst["current"],
            "target_value": None,
            "prior_value": worst["prior"],
            "change_pct": worst["change_pct"],
            "finding": (
                f"{worst['channel']} sessions dropped {abs(worst['change_pct']):.0%} WoW "
                f"({worst['prior']:,} → {worst['current']:,}). "
                f"{len(channel_drops)} channel{'s' if len(channel_drops) > 1 else ''} "
                f"declined >15% this week."
            ),
            "action": (
                f"Check {worst['channel']} for campaign pauses, bid changes, or landing page issues. "
                f"{'Other declining channels: ' + ', '.join(c['channel'] for c in channel_drops if c != worst) + '.' if len(channel_drops) > 1 else ''}"
            ),
        })

    # ── Check 4: Spend efficiency trend ───────────────────────────────────
    # Spend up + sessions down = deteriorating efficiency
    spend_change = acq["spend"]["change_pct"]
    if (spend_change is not None and total_change is not None
            and spend_change > 0.05 and total_change < -0.05):
        findings.append({
            "severity": "critical",
            "domain": "acquisition",
            "metric": "Spend efficiency",
            "current_value": acq["spend"]["current"],
            "target_value": None,
            "prior_value": acq["spend"]["prior"],
            "change_pct": spend_change,
            "finding": (
                f"Paid spend increased {spend_change:.0%} WoW "
                f"(${acq['spend']['prior']:,.0f} → ${acq['spend']['current']:,.0f}) "
                f"while sessions dropped {abs(total_change):.0%}. "
                f"Efficiency is deteriorating — spending more, getting less."
            ),
            "action": "Audit paid channel performance immediately. Likely CPC inflation or audience saturation. Consider shifting budget to organic/outbound.",
        })

    # Sort by severity: critical first, then warning, then info
    severity_order = {"critical": 0, "warning": 1, "info": 2}
    findings.sort(key=lambda f: severity_order.get(f["severity"], 3))

    # Cap at 6 findings — more than that and the briefing is noise
    return findings[:6]
