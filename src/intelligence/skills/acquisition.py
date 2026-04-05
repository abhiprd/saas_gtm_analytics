"""
Acquisition Health Skill (v2)

Answers: Where is our marketing spend generating the most pipeline per dollar,
and where is it being wasted?

Core shift from v1: Ranks by pipeline ROI and dollar impact, not CPL deviation.
A channel with high CPL but high pipeline ROI is a GOOD channel.
A channel with low CPL but no pipeline is the real problem.

Reads from: snapshot["channel_economics"], snapshot["acquisition"], snapshot["targets"]
"""


def analyze(snapshot: dict) -> list[dict]:
    """Analyze acquisition health through the lens of full-funnel economics."""
    econ = snapshot["channel_economics"]
    acq = snapshot["acquisition"]
    targets = snapshot["targets"]
    findings = []

    # Separate paid channels (have spend data) from organic
    paid = [c for c in econ if c["is_paid"] and c["spend_usd"] > 0]
    organic = [c for c in econ if not c["is_paid"] and c["pipeline_usd"] > 0]

    # ── Check 1: Pipeline ROI ranking — the real efficiency story ─────────
    # This is the headline: which channels generate the most pipeline per dollar?
    if len(paid) >= 2:
        paid_by_roi = sorted(
            [c for c in paid if c["pipeline_roi"] is not None],
            key=lambda c: c["pipeline_roi"],
            reverse=True,
        )

        if len(paid_by_roi) >= 2:
            best = paid_by_roi[0]
            worst = paid_by_roi[-1]
            roi_spread = best["pipeline_roi"] - worst["pipeline_roi"]

            # Build the reallocation story
            total_paid_spend = sum(c["spend_usd"] for c in paid)
            worst_spend_pct = worst["spend_usd"] / total_paid_spend if total_paid_spend > 0 else 0

            # Is the worst ROI channel also the biggest spender? That's the real problem.
            paid_by_spend = sorted(paid, key=lambda c: c["spend_usd"], reverse=True)
            biggest_spender = paid_by_spend[0]

            if worst["channel"] == biggest_spender["channel"]:
                severity = "critical"
                finding_text = (
                    f"{worst['channel']} is the largest paid channel at ${worst['spend_usd']:,.0f} QTD "
                    f"but has the lowest pipeline ROI at {worst['pipeline_roi']}x "
                    f"(${worst['pipeline_usd']:,.0f} pipeline from ${worst['spend_usd']:,.0f} spend). "
                    f"Meanwhile {best['channel']} generates {best['pipeline_roi']}x ROI — "
                    f"${best['pipeline_usd']:,.0f} pipeline from just ${best['spend_usd']:,.0f} spend."
                )
            else:
                severity = "warning" if roi_spread > 10 else "info"
                finding_text = (
                    f"Pipeline ROI ranges from {best['pipeline_roi']}x ({best['channel']}) "
                    f"to {worst['pipeline_roi']}x ({worst['channel']}). "
                    f"{best['channel']} generates ${best['pipeline_usd']:,.0f} pipeline "
                    f"from ${best['spend_usd']:,.0f} spend. "
                    f"{worst['channel']} generates ${worst['pipeline_usd']:,.0f} from ${worst['spend_usd']:,.0f}."
                )

            findings.append({
                "severity": severity,
                "domain": "acquisition",
                "metric": "Channel pipeline ROI",
                "current_value": worst["pipeline_roi"],
                "target_value": best["pipeline_roi"],
                "prior_value": None,
                "change_pct": None,
                "finding": finding_text,
                "action": (
                    f"Shift ${worst['spend_usd'] * 0.2:,.0f} (20%) from {worst['channel']} "
                    f"to {best['channel']} and {paid_by_roi[1]['channel'] if len(paid_by_roi) > 1 else 'Events'}. "
                    f"At {best['channel']}'s current ROI of {best['pipeline_roi']}x, "
                    f"this could generate an additional ${worst['spend_usd'] * 0.2 * best['pipeline_roi']:,.0f} "
                    f"in pipeline."
                ),
            })

    # ── Check 2: Cost per opportunity — the real acquisition cost ─────────
    # CPL is a vanity metric. Cost per opp is what matters.
    paid_with_opps = [c for c in paid if c["cost_per_opp"] is not None and c["opps"] > 0]
    if len(paid_with_opps) >= 2:
        by_cpo = sorted(paid_with_opps, key=lambda c: c["cost_per_opp"])
        cheapest = by_cpo[0]
        most_expensive = by_cpo[-1]
        cpo_spread = most_expensive["cost_per_opp"] - cheapest["cost_per_opp"]

        if cpo_spread > 10000:  # >$10K spread is significant
            # But check if the expensive channel creates bigger deals
            expensive_justified = (
                most_expensive["avg_opp_acv"] is not None
                and cheapest["avg_opp_acv"] is not None
                and most_expensive["avg_opp_acv"] > cheapest["avg_opp_acv"] * 1.5
            )

            if expensive_justified:
                findings.append({
                    "severity": "info",
                    "domain": "acquisition",
                    "metric": "Cost per opportunity",
                    "current_value": most_expensive["cost_per_opp"],
                    "target_value": cheapest["cost_per_opp"],
                    "prior_value": None,
                    "change_pct": None,
                    "finding": (
                        f"{most_expensive['channel']} costs ${most_expensive['cost_per_opp']:,.0f}/opp "
                        f"vs ${cheapest['cost_per_opp']:,.0f}/opp for {cheapest['channel']}. "
                        f"However, {most_expensive['channel']} produces ${most_expensive['avg_opp_acv']:,.0f} "
                        f"avg ACV deals vs ${cheapest['avg_opp_acv']:,.0f} for {cheapest['channel']} — "
                        f"the higher acquisition cost is justified by deal size."
                    ),
                    "action": (
                        f"Maintain {most_expensive['channel']} spend for Enterprise/Strategic deals. "
                        f"Monitor cost-per-pipeline-dollar (currently ${most_expensive['cost_per_pipeline_dollar']:.2f}) "
                        f"as the true efficiency measure."
                    ),
                })
            else:
                findings.append({
                    "severity": "warning",
                    "domain": "acquisition",
                    "metric": "Cost per opportunity",
                    "current_value": most_expensive["cost_per_opp"],
                    "target_value": cheapest["cost_per_opp"],
                    "prior_value": None,
                    "change_pct": None,
                    "finding": (
                        f"{most_expensive['channel']} costs ${most_expensive['cost_per_opp']:,.0f}/opp — "
                        f"{most_expensive['cost_per_opp'] / cheapest['cost_per_opp']:.1f}x more than "
                        f"{cheapest['channel']} (${cheapest['cost_per_opp']:,.0f}/opp) — "
                        f"without proportionally larger deal sizes "
                        f"(${most_expensive['avg_opp_acv']:,.0f} vs ${cheapest['avg_opp_acv']:,.0f} avg ACV)."
                    ),
                    "action": (
                        f"Audit {most_expensive['channel']} campaigns for underperformers. "
                        f"At ${most_expensive['cost_per_opp']:,.0f}/opp with ${most_expensive['avg_opp_acv']:,.0f} "
                        f"avg ACV, the payback math is thin."
                    ),
                })

    # ── Check 3: Segment mix — who are channels actually reaching? ────────
    # Surface channels that pull Enterprise/Strategic (high-value) vs only SMB
    for ch in paid:
        if not ch["segment_mix"]:
            continue
        enterprise_plus = sum(
            s["pct_of_channel_pipeline"]
            for s in ch["segment_mix"]
            if s["segment"] in ("Enterprise", "Strategic")
        )
        if enterprise_plus > 0.60 and ch["pipeline_usd"] > 500000:
            findings.append({
                "severity": "info",
                "domain": "acquisition",
                "metric": f"{ch['channel']} segment mix",
                "current_value": enterprise_plus,
                "target_value": None,
                "prior_value": None,
                "change_pct": None,
                "finding": (
                    f"{ch['channel']} sources {enterprise_plus:.0%} of its pipeline "
                    f"from Enterprise/Strategic deals (avg ACV ${ch['avg_opp_acv']:,.0f}). "
                    f"This channel is your primary Enterprise pipeline engine."
                ),
                "action": (
                    f"Protect {ch['channel']} budget even if CPL appears high — "
                    f"the ${ch['avg_opp_acv']:,.0f} avg deal size justifies the acquisition cost. "
                    f"Consider increasing investment to scale Enterprise pipeline."
                ),
            })
            break  # Only report the most notable one

    # ── Check 4: Organic pipeline contribution — free channels matter ─────
    if organic:
        total_pipeline = sum(c["pipeline_usd"] for c in econ if c["pipeline_usd"] > 0)
        organic_pipeline = sum(c["pipeline_usd"] for c in organic)
        organic_pct = organic_pipeline / total_pipeline if total_pipeline > 0 else 0

        top_organic = sorted(organic, key=lambda c: c["pipeline_usd"], reverse=True)[:3]
        organic_summary = ", ".join(
            f"{c['channel']} (${c['pipeline_usd']:,.0f}, {c['opps']} opps)"
            for c in top_organic
        )
        findings.append({
            "severity": "info",
            "domain": "acquisition",
            "metric": "Organic pipeline contribution",
            "current_value": organic_pct,
            "target_value": None,
            "prior_value": None,
            "change_pct": None,
            "finding": (
                f"Organic/unpaid channels contribute {organic_pct:.0%} of total QTD pipeline "
                f"(${organic_pipeline:,.0f}). Top organic sources: {organic_summary}."
            ),
            "action": "Organic pipeline is the highest-margin revenue. Invest in content, SEO, and partner programs to grow this share.",
        })

    # ── Check 5: Weekly traffic + spend efficiency trend ──────────────────
    total_change = acq["sessions"]["change_pct"]
    spend_change = acq["spend"]["change_pct"]

    if (spend_change is not None and total_change is not None
            and spend_change > 0.05 and total_change < -0.05):
        findings.append({
            "severity": "critical",
            "domain": "acquisition",
            "metric": "Spend efficiency trend",
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
            "action": "Audit paid channel performance. Likely CPC inflation or audience saturation in highest-spend channels.",
        })

    # Sort by severity
    severity_order = {"critical": 0, "warning": 1, "info": 2}
    findings.sort(key=lambda f: severity_order.get(f["severity"], 3))

    return findings[:6]