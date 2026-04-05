"""
Conversion Health Skill (v2)

Answers: Are leads converting at the rates we need, and where is the funnel
leaking VALUE (not just volume)?

Core shift from v1: Connects conversion rates to pipeline value. A source
converting at 27% but producing $168K ACV opps is more valuable than one
converting at 43% producing $62K opps.

Reads from: snapshot["conversion"], snapshot["channel_economics"],
            snapshot["targets"], snapshot["sdr_capacity"]
"""


def analyze(snapshot: dict) -> list[dict]:
    """Analyze conversion health connecting rates to pipeline value."""
    conv = snapshot["conversion"]
    econ = snapshot["channel_economics"]
    targets = snapshot["targets"]
    sdr = snapshot.get("sdr_capacity", {})
    findings = []

    # ── Check 1: MQL rate vs target with context ─────────────────────────
    mql_rate = conv["mql_rate"]
    if mql_rate["current"] is not None and mql_rate["target"] is not None:
        gap_points = (mql_rate["target"] - mql_rate["current"]) * 100

        if gap_points > 2:
            severity = "critical" if gap_points > 5 else "warning"

            # Diagnose WHY: is it volume (fewer leads) or quality (source mix shift)?
            leads_change = conv["leads"].get("change_pct")
            mqls_change = conv["mqls"].get("change_pct")

            if leads_change is not None and mqls_change is not None:
                if mqls_change < leads_change - 0.05:
                    diagnosis = (
                        "MQL volume dropped faster than lead volume, suggesting a lead quality shift — "
                        "the source mix may have shifted toward lower-converting channels."
                    )
                else:
                    diagnosis = (
                        "MQL decline tracks lead decline — this is a volume problem, not a quality problem. "
                        "Fixing acquisition volume will fix MQL volume."
                    )
            else:
                diagnosis = ""

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
                    + (f"Down from {mql_rate['prior']:.1%} last week. " if mql_rate['prior'] else "")
                    + f"{diagnosis}"
                ),
                "action": (
                    "Investigate source mix — see MQL→SQL by source below for which channels "
                    "are dragging the blended rate. Consider whether scoring thresholds need recalibration."
                ),
            })

    # ── Check 2: MQL→SQL by source with pipeline value context ───────────
    # Don't just show rates — connect to pipeline value per source
    by_source = conv.get("mql_to_sql_by_source", [])
    if len(by_source) >= 2:
        meaningful = [s for s in by_source if s["mqls"] >= 10]
        if len(meaningful) >= 2:
            best_rate = max(meaningful, key=lambda s: s["rate"])
            worst_rate = min(meaningful, key=lambda s: s["rate"])
            spread = best_rate["rate"] - worst_rate["rate"]

            # Now check: does the worst-converting source produce high-value opps?
            # Map source names to channel economics where possible
            # (This is an approximation — source != channel, but directionally useful)
            source_to_channel_hint = {
                "Outbound - SDR": "Outbound SDR",
                "Inbound - Demo Request": "Paid Search",  # approximation
                "Partner Referral": "Partner/Channel",
                "Event": "Events/Conferences",
            }

            worst_pipeline_context = ""
            for src_name, ch_name in source_to_channel_hint.items():
                if worst_rate["source"] == src_name:
                    ch_econ = next((c for c in econ if c["channel"] == ch_name), None)
                    if ch_econ and ch_econ["avg_opp_acv"]:
                        worst_pipeline_context = (
                            f" However, {ch_name}-sourced opps average ${ch_econ['avg_opp_acv']:,.0f} ACV"
                            f"{' — the low conversion rate may be acceptable given deal size.' if ch_econ['avg_opp_acv'] > 100000 else '.'}"
                        )
                    break

            if spread > 0.10:
                findings.append({
                    "severity": "warning" if spread > 0.15 else "info",
                    "domain": "conversion",
                    "metric": "MQL→SQL conversion spread",
                    "current_value": worst_rate["rate"],
                    "target_value": best_rate["rate"],
                    "prior_value": None,
                    "change_pct": None,
                    "finding": (
                        f"{best_rate['source']} converts MQLs to SQLs at {best_rate['rate']:.0%} "
                        f"({best_rate['sqls']}/{best_rate['mqls']}), while {worst_rate['source']} "
                        f"converts at {worst_rate['rate']:.0%} ({worst_rate['sqls']}/{worst_rate['mqls']}). "
                        f"That's a {spread:.0%} gap QTD."
                        f"{worst_pipeline_context}"
                    ),
                    "action": (
                        f"Investigate {worst_rate['source']} rejection reasons — is it lead quality (wrong ICP), "
                        f"timing (too early), or sales deprioritization? "
                        f"If the deals are high-value, the issue may be SDR follow-up speed, not lead quality."
                    ),
                })

    # ── Check 3: MQL→SQL rate vs target with handoff framing ─────────────
    sql_rate = conv["mql_to_sql_rate"]
    if sql_rate["current"] is not None and sql_rate["target"] is not None:
        gap_points = (sql_rate["target"] - sql_rate["current"]) * 100

        if gap_points > 2:
            # Calculate the pipeline impact of the gap
            qtd_mqls = sum(s["mqls"] for s in by_source)
            missed_sqls = int(qtd_mqls * (gap_points / 100))

            # Estimate missed pipeline using avg opp ACV across channels
            avg_acvs = [c["avg_opp_acv"] for c in econ if c["avg_opp_acv"] and c["avg_opp_acv"] > 0]
            avg_acv = sum(avg_acvs) / len(avg_acvs) if avg_acvs else 80000
            missed_pipeline = missed_sqls * 0.70 * avg_acv  # 70% SQL→opp rate

            severity = "critical" if missed_pipeline > 2000000 else "warning"

            findings.append({
                "severity": severity,
                "domain": "conversion",
                "metric": "MQL→SQL handoff rate",
                "current_value": sql_rate["current"],
                "target_value": sql_rate["target"],
                "prior_value": sql_rate["prior"],
                "change_pct": sql_rate["change_pct"],
                "finding": (
                    f"MQL→SQL conversion is {sql_rate['current']:.1%} vs {sql_rate['target']:.0%} target. "
                    f"The {gap_points:.1f}-point gap represents ~{missed_sqls} missed SQLs QTD, "
                    f"which at a 70% SQL→opp rate and ${avg_acv:,.0f} avg ACV translates to "
                    f"~${missed_pipeline:,.0f} in unrealized pipeline."
                ),
                "action": (
                    "This is a revenue problem, not a process problem. "
                    "Review sales rejection reasons — if 'not ICP' is the top reason, "
                    "the issue is upstream lead quality. If 'timing' or 'no response', "
                    "the issue is SDR follow-up SLA."
                ),
            })

    # ── Check 4: Lead volume trend + quality diagnosis ────────────────────
    leads = conv["leads"]
    mqls = conv["mqls"]

    if leads.get("change_pct") is not None and leads["change_pct"] < -0.15:
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
                f"MQLs: {mqls['prior']:,} → {mqls['current']:,}."
            ),
            "action": "Check acquisition channels for campaign pauses, budget cuts, or seasonal effects.",
        })

    # ── Check 5: Lead score divergence ────────────────────────────────────
    avg_score = conv.get("avg_lead_score", {})
    if avg_score.get("current") and avg_score.get("prior"):
        score_change = avg_score["current"] - avg_score["prior"]
        mql_declining = mql_rate.get("change_pct") is not None and mql_rate["change_pct"] < -0.05

        if score_change > 3 and mql_declining:
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
                    f"but MQL rate declined. The scoring model appears misaligned "
                    f"with what sales actually accepts as qualified."
                ),
                "action": "Audit the MQL→SQL feedback loop. Recalibrate scoring weights based on which scored leads actually convert.",
            })

    # Sort by severity
    severity_order = {"critical": 0, "warning": 1, "info": 2}
    findings.sort(key=lambda f: severity_order.get(f["severity"], 3))

    return findings[:5]