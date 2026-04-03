"""
Conversion Health Skill

Answers: Are leads converting at the rates we need, and where is the funnel leaking?
Reads from snapshot["conversion"] and snapshot["targets"].
"""


def analyze(snapshot: dict) -> list[dict]:
    """Analyze conversion health and return prioritized findings."""
    conv = snapshot["conversion"]
    targets = snapshot["targets"]
    findings = []

    # ── Check 1: MQL rate vs target ───────────────────────────────────────
    mql_rate = conv["mql_rate"]
    if mql_rate["current"] is not None and mql_rate["target"] is not None:
        gap = mql_rate["target"] - mql_rate["current"]  # positive = below target
        gap_points = gap * 100  # in percentage points

        if gap_points > 2:  # More than 2pp below target
            severity = "critical" if gap_points > 5 else "warning"
            findings.append({
                "severity": severity,
                "domain": "conversion",
                "metric": "MQL rate",
                "current_value": mql_rate["current"],
                "target_value": mql_rate["target"],
                "prior_value": mql_rate["prior"],
                "change_pct": mql_rate["change_pct"],
                "finding": (
                    f"MQL rate is {mql_rate['current']:.1%}, "
                    f"{gap_points:.1f} points below the {mql_rate['target']:.0%} target. "
                    + (f"Down from {mql_rate['prior']:.1%} last week." if mql_rate['prior'] else "")
                ),
                "action": (
                    "Investigate lead quality by source — are we attracting the wrong audience, "
                    "or has the scoring model drifted? Check if recent campaign changes "
                    "shifted the lead source mix toward lower-converting channels."
                ),
            })
        elif gap_points < -2:  # Significantly above target
            findings.append({
                "severity": "info",
                "domain": "conversion",
                "metric": "MQL rate",
                "current_value": mql_rate["current"],
                "target_value": mql_rate["target"],
                "prior_value": mql_rate["prior"],
                "change_pct": mql_rate["change_pct"],
                "finding": (
                    f"MQL rate is {mql_rate['current']:.1%}, "
                    f"{abs(gap_points):.1f} points above target. "
                    f"Verify this reflects genuine quality improvement, not loosened scoring criteria."
                ),
                "action": "Spot-check recent MQLs for quality. If scoring thresholds were changed, validate against SQL conversion downstream.",
            })

    # ── Check 2: MQL→SQL rate vs target ───────────────────────────────────
    sql_rate = conv["mql_to_sql_rate"]
    if sql_rate["current"] is not None and sql_rate["target"] is not None:
        gap = sql_rate["target"] - sql_rate["current"]
        gap_points = gap * 100

        if gap_points > 2:
            severity = "critical" if gap_points > 5 else "warning"
            findings.append({
                "severity": severity,
                "domain": "conversion",
                "metric": "MQL→SQL rate",
                "current_value": sql_rate["current"],
                "target_value": sql_rate["target"],
                "prior_value": sql_rate["prior"],
                "change_pct": sql_rate["change_pct"],
                "finding": (
                    f"MQL→SQL conversion is {sql_rate['current']:.1%}, "
                    f"{gap_points:.1f} points below the {sql_rate['target']:.0%} target. "
                    f"This means {100 - sql_rate['current'] * 100:.0f}% of MQLs are not being accepted by sales."
                ),
                "action": (
                    "Review sales rejection reasons. Common causes: leads don't match ICP, "
                    "timing is wrong, or SDRs aren't following up fast enough. "
                    "Check MQL→SQL rate by source below to isolate the problem."
                ),
            })

    # ── Check 3: MQL→SQL by source — find the outliers ────────────────────
    by_source = conv.get("mql_to_sql_by_source", [])
    if len(by_source) >= 2:
        # Filter to sources with enough volume to matter
        meaningful = [s for s in by_source if s["mqls"] >= 10]
        if len(meaningful) >= 2:
            best = max(meaningful, key=lambda s: s["rate"])
            worst = min(meaningful, key=lambda s: s["rate"])
            spread = best["rate"] - worst["rate"]

            if spread > 0.10:  # >10pp spread between best and worst
                findings.append({
                    "severity": "warning" if spread > 0.15 else "info",
                    "domain": "conversion",
                    "metric": "MQL→SQL source spread",
                    "current_value": spread,
                    "target_value": None,
                    "prior_value": None,
                    "change_pct": None,
                    "finding": (
                        f"{best['source']} converts MQLs to SQLs at {best['rate']:.0%} "
                        f"({best['sqls']}/{best['mqls']}), while {worst['source']} "
                        f"converts at {worst['rate']:.0%} ({worst['sqls']}/{worst['mqls']}). "
                        f"That's a {spread:.0%} gap."
                    ),
                    "action": (
                        f"Investigate why {worst['source']} leads convert at half the rate. "
                        f"Is it lead quality (wrong ICP), timing (too early in buying cycle), "
                        f"or sales deprioritization? If quality, reduce spend on that source. "
                        f"If deprioritization, align with sales on SLAs."
                    ),
                })

    # ── Check 4: Lead volume trend WoW ────────────────────────────────────
    leads = conv["leads"]
    mqls = conv["mqls"]
    if leads["change_pct"] is not None and leads["change_pct"] < -0.15:
        # Did MQL volume drop proportionally, or did the rate change?
        mql_drop = mqls["change_pct"] if mqls["change_pct"] is not None else 0
        lead_drop = leads["change_pct"]

        if mql_drop < lead_drop - 0.05:
            # MQLs dropped more than leads — rate is declining (quality problem)
            diagnosis = "MQL volume dropped faster than lead volume, suggesting a lead quality shift — not just fewer leads."
            action_detail = "Check if the lead source mix shifted toward lower-quality channels."
        else:
            # Proportional drop — acquisition problem
            diagnosis = "MQL decline matches lead decline — this is a volume problem, not a quality problem."
            action_detail = "Check acquisition channels for the root cause (campaign pauses, budget cuts, or seasonal effects)."

        findings.append({
            "severity": "warning" if leads["change_pct"] > -0.25 else "critical",
            "domain": "conversion",
            "metric": "Lead volume WoW",
            "current_value": leads["current"],
            "target_value": None,
            "prior_value": leads["prior"],
            "change_pct": leads["change_pct"],
            "finding": (
                f"Lead volume dropped {abs(leads['change_pct']):.0%} WoW "
                f"({leads['prior']:,} → {leads['current']:,}). "
                f"MQLs went from {mqls['prior']:,} to {mqls['current']:,} "
                f"({abs(mql_drop):.0%} drop). {diagnosis}"
            ),
            "action": action_detail,
        })

    # ── Check 5: Lead score shift ─────────────────────────────────────────
    avg_score = conv.get("avg_lead_score", {})
    if (avg_score.get("current") is not None and avg_score.get("prior") is not None):
        score_change = avg_score["current"] - avg_score["prior"]

        # Score dropped while MQL rate also dropped — consistent quality decline
        mql_declining = (mql_rate.get("change_pct") is not None
                         and mql_rate["change_pct"] < -0.05)

        if score_change < -3:  # More than 3 points decline
            findings.append({
                "severity": "warning",
                "domain": "conversion",
                "metric": "Avg lead score",
                "current_value": avg_score["current"],
                "target_value": None,
                "prior_value": avg_score["prior"],
                "change_pct": avg_score.get("change_pct"),
                "finding": (
                    f"Average lead composite score dropped from {avg_score['prior']:.1f} "
                    f"to {avg_score['current']:.1f} ({score_change:+.1f} points). "
                    f"{'This aligns with the declining MQL rate — lower quality leads are entering the funnel.' if mql_declining else 'MQL rate has not yet been affected, but watch for downstream impact.'}"
                ),
                "action": "Review lead source mix changes this week. Score decline usually indicates a channel mix shift toward lower-fit audiences.",
            })
        elif score_change > 3 and mql_declining:
            # Scores going up but MQLs going down — scoring model misalignment
            findings.append({
                "severity": "warning",
                "domain": "conversion",
                "metric": "Lead score vs MQL rate divergence",
                "current_value": avg_score["current"],
                "target_value": None,
                "prior_value": avg_score["prior"],
                "change_pct": avg_score.get("change_pct"),
                "finding": (
                    f"Average lead score improved ({avg_score['prior']:.1f} → {avg_score['current']:.1f}) "
                    f"but MQL rate declined. This suggests the scoring model is not aligned "
                    f"with what sales actually accepts as qualified."
                ),
                "action": "Audit MQL→SQL feedback loop. If high-scored leads are being rejected by sales, recalibrate scoring weights.",
            })

    # Sort by severity
    severity_order = {"critical": 0, "warning": 1, "info": 2}
    findings.sort(key=lambda f: severity_order.get(f["severity"], 3))

    return findings[:5]
