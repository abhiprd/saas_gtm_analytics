"""
Monday Marketing Pulse Agent

Composes the three skill outputs (Acquisition, Conversion, Contribution)
into a single narrative briefing delivered as Markdown.

The briefing is structured for a VP of Growth Marketing who has 90 seconds
on Monday morning to understand where the funnel stands, what's broken,
and what to do about it.
"""

from datetime import datetime


def _severity_icon(severity: str) -> str:
    return {"critical": "🔴", "warning": "🟡", "info": "🟢"}.get(severity, "⚪")


def _format_finding(finding: dict) -> str:
    """Format a single finding as a briefing line."""
    icon = _severity_icon(finding["severity"])
    return f"{icon} **{finding['metric']}** — {finding['finding']}\n   → *{finding['action']}*"


def _headline(acq_findings: list, conv_findings: list, contrib_findings: list) -> str:
    """Generate the one-sentence headline: the single most important thing this Monday."""
    all_findings = acq_findings + conv_findings + contrib_findings
    criticals = [f for f in all_findings if f["severity"] == "critical"]
    warnings = [f for f in all_findings if f["severity"] == "warning"]

    if not criticals and not warnings:
        return "All systems nominal. No critical or warning-level findings this week."

    # Lead with the most impactful critical finding
    if criticals:
        # Prioritize: contribution > conversion > acquisition
        # (pipeline pacing is the #1 thing a Growth leader cares about)
        domain_priority = {"contribution": 0, "conversion": 1, "acquisition": 2}
        criticals.sort(key=lambda f: domain_priority.get(f["domain"], 3))
        lead = criticals[0]

        other_count = len(criticals) - 1 + len(warnings)
        suffix = f" {other_count} additional issue{'s' if other_count != 1 else ''} flagged below." if other_count > 0 else ""

        return f"{lead['finding']}{suffix}"

    # No criticals, lead with the most impactful warning
    domain_priority = {"contribution": 0, "conversion": 1, "acquisition": 2}
    warnings.sort(key=lambda f: domain_priority.get(f["domain"], 3))
    lead = warnings[0]
    other_count = len(warnings) - 1
    suffix = f" {other_count} additional warning{'s' if other_count != 1 else ''} below." if other_count > 0 else ""

    return f"{lead['finding']}{suffix}"


def compose_briefing(
    snapshot: dict,
    acq_findings: list,
    conv_findings: list,
    contrib_findings: list,
) -> str:
    """Compose skill findings into a Monday Marketing Pulse briefing.

    Args:
        snapshot: The weekly snapshot dict
        acq_findings: Output from acquisition.analyze()
        conv_findings: Output from conversion.analyze()
        contrib_findings: Output from contribution.analyze()

    Returns:
        Markdown string of the complete briefing
    """
    report_date = snapshot["report_date"]
    period = snapshot["period"]
    quarter = snapshot["quarter"]

    # Parse for display
    rd = datetime.strptime(report_date, "%Y-%m-%d")
    date_display = rd.strftime("%B %d, %Y")
    period_display = f"{period['start']} to {period['end']}"

    # Count severities across all findings
    all_findings = acq_findings + conv_findings + contrib_findings
    n_critical = sum(1 for f in all_findings if f["severity"] == "critical")
    n_warning = sum(1 for f in all_findings if f["severity"] == "warning")
    n_info = sum(1 for f in all_findings if f["severity"] == "info")

    headline = _headline(acq_findings, conv_findings, contrib_findings)

    # ── Build the briefing ────────────────────────────────────────────────

    sections = []

    # Header
    sections.append(f"# Monday Marketing Pulse — {date_display}")
    sections.append(f"*Reporting period: {period_display} | {quarter}*")
    sections.append("")

    # Status bar
    status_parts = []
    if n_critical:
        status_parts.append(f"🔴 {n_critical} critical")
    if n_warning:
        status_parts.append(f"🟡 {n_warning} warning")
    if n_info:
        status_parts.append(f"🟢 {n_info} on track")
    sections.append(f"**Status:** {' · '.join(status_parts)}")
    sections.append("")

    # Headline
    sections.append("## The One Thing")
    sections.append(headline)
    sections.append("")

    # ── Acquisition ───────────────────────────────────────────────────────
    sections.append("---")
    sections.append("## Acquisition — Is the top of funnel healthy?")
    sections.append("")

    # Quick stats line
    acq = snapshot["acquisition"]
    sections.append(
        f"**Sessions:** {acq['sessions']['current']:,} "
        f"({acq['sessions']['change_pct']:+.0%} WoW) · "
        f"**Paid spend:** ${acq['spend']['current']:,.0f} "
        f"({acq['spend']['change_pct']:+.0%} WoW)"
    )
    sections.append("")

    if acq_findings:
        for finding in acq_findings:
            sections.append(_format_finding(finding))
            sections.append("")
    else:
        sections.append("🟢 No issues flagged. Acquisition metrics within targets.")
        sections.append("")

    # ── Conversion ────────────────────────────────────────────────────────
    sections.append("---")
    sections.append("## Conversion — Are leads becoming pipeline?")
    sections.append("")

    conv_data = snapshot["conversion"]
    sections.append(
        f"**Leads:** {conv_data['leads']['current']:,} "
        f"({conv_data['leads']['change_pct']:+.0%} WoW) · "
        f"**MQLs:** {conv_data['mqls']['current']:,} · "
        f"**MQL rate:** {conv_data['mql_rate']['current']:.1%} "
        f"(target: {conv_data['mql_rate']['target']:.0%})"
    )
    sections.append("")

    if conv_findings:
        for finding in conv_findings:
            sections.append(_format_finding(finding))
            sections.append("")
    else:
        sections.append("🟢 No issues flagged. Conversion metrics within targets.")
        sections.append("")

    # ── Contribution ──────────────────────────────────────────────────────
    sections.append("---")
    sections.append("## Contribution — Will marketing hit the number?")
    sections.append("")

    contrib_data = snapshot["contribution"]
    sections.append(
        f"**Pipeline QTD:** ${contrib_data['pipeline_created_qtd']:,.0f} "
        f"of ${snapshot['targets']['pipeline_created']:,.0f} target "
        f"({contrib_data['pipeline_pacing_pct']:.0%}) · "
        f"**Won QTD:** ${contrib_data['won_acv_qtd']:,.0f}"
    )
    sections.append("")

    if contrib_findings:
        for finding in contrib_findings:
            sections.append(_format_finding(finding))
            sections.append("")
    else:
        sections.append("🟢 No issues flagged. Pipeline contribution on track.")
        sections.append("")

    # ── Actions summary ───────────────────────────────────────────────────
    critical_and_warning = [f for f in all_findings if f["severity"] in ("critical", "warning")]
    if critical_and_warning:
        sections.append("---")
        sections.append("## This Week's Priority Actions")
        sections.append("")
        for i, finding in enumerate(critical_and_warning[:5], 1):
            sections.append(f"{i}. **{finding['metric']}:** {finding['action']}")
        sections.append("")

    # ── Footer ────────────────────────────────────────────────────────────
    sections.append("---")
    sections.append(
        f"*Generated by Vantage Marketing Intelligence Platform · "
        f"{len(all_findings)} findings from {len(acq_findings)} acquisition, "
        f"{len(conv_findings)} conversion, {len(contrib_findings)} contribution checks · "
        f"Data period: {period_display}*"
    )

    return "\n".join(sections)
