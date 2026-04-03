"""
Weekly snapshot generator for the Marketing Intelligence Platform.

Reads parquet files + quarterly targets and produces a JSON snapshot
for a given week. Every computed metric includes _query provenance metadata
for traceability.

Snapshot sections:
  - acquisition: sessions, spend, CPL, budget pacing (weekly view)
  - conversion: leads, MQLs, SQLs, rates, source breakdown (weekly + QTD)
  - contribution: pipeline created/won, deal source breakdown (QTD)
  - channel_economics: full-funnel spend→CPL→opps→pipeline→ROI by channel (QTD)
  - pipeline_trajectory: weekly velocity, trend, projection, stage breakdown (QTD)
  - targets: quarterly operational targets

Usage:
  python -m src.intelligence.snapshot --date 2025-11-03
"""

import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from shared.constants import DATA_DIR, OUTPUT_DIR
from shared.targets import get_targets


# ── Helpers ──────────────────────────────────────────────────────────────────

def _load(table: str) -> pd.DataFrame:
    return pd.read_parquet(DATA_DIR / f"{table}.parquet")


def _query_meta(table: str, filters: dict, aggregation: str, row_count: int) -> dict:
    """Provenance metadata for a computed metric."""
    return {
        "table": table,
        "filters": filters,
        "aggregation": aggregation,
        "row_count": row_count,
    }


def _pct_change(current, prior):
    if prior == 0 or prior is None:
        return None
    return round((current - prior) / prior, 4)


def _quarter_for_date(dt: datetime) -> str:
    q = (dt.month - 1) // 3 + 1
    return f"{dt.year}-Q{q}"


def _quarter_start(dt: datetime) -> pd.Timestamp:
    q = (dt.month - 1) // 3
    return pd.Timestamp(dt.year, q * 3 + 1, 1)


def _quarter_end(dt: datetime) -> pd.Timestamp:
    q = (dt.month - 1) // 3 + 1
    if q < 4:
        return pd.Timestamp(dt.year, q * 3 + 1, 1) - pd.Timedelta(days=1)
    else:
        return pd.Timestamp(dt.year, 12, 31)


# ── Snapshot Builder ─────────────────────────────────────────────────────────

def generate_snapshot(report_date: str) -> dict:
    """Generate the weekly snapshot JSON for the Monday briefing.

    Args:
        report_date: ISO date string for the Monday (e.g. "2025-11-03").
                     The reporting period is the prior Mon–Sun week.
    """
    rd = datetime.strptime(report_date, "%Y-%m-%d")
    week_end = rd - timedelta(days=1)          # Sunday
    week_start = week_end - timedelta(days=6)  # Monday
    prior_end = week_start - timedelta(days=1)
    prior_start = prior_end - timedelta(days=6)

    ws = pd.Timestamp(week_start)
    we = pd.Timestamp(week_end)
    ps = pd.Timestamp(prior_start)
    pe = pd.Timestamp(prior_end)

    quarter = _quarter_for_date(rd)
    qtd_start = _quarter_start(rd)
    qtr_end = _quarter_end(rd)
    month_start = pd.Timestamp(rd.year, rd.month, 1)

    targets = get_targets(quarter)

    # Load tables
    traffic = _load("fct_daily_web_traffic")
    spend = _load("fct_daily_ad_spend")
    contacts = _load("dim_contacts")
    opps = _load("dim_opportunities")
    channels = _load("dim_channels")
    attribution = _load("fct_multi_touch_attribution")
    reps = _load("dim_sales_reps")
    sales_activity = _load("fct_sales_activity")
    intent = _load("fct_account_intent_signals")
    accounts = _load("dim_accounts")

    ch_map = dict(zip(channels["channel_id"], channels["channel_name"]))
    paid_channel_ids = set(spend["channel_id"].unique())

    # ── ACQUISITION (weekly view) ────────────────────────────────────────

    # Sessions
    wk_traffic = traffic[(traffic["date"] >= ws) & (traffic["date"] <= we)]
    pr_traffic = traffic[(traffic["date"] >= ps) & (traffic["date"] <= pe)]

    sessions_current = int(wk_traffic["sessions"].sum())
    sessions_prior = int(pr_traffic["sessions"].sum())

    # Sessions by channel
    sessions_by_channel = []
    for ch_id in sorted(ch_map.keys()):
        cur = wk_traffic[wk_traffic["channel_id"] == ch_id]
        pri = pr_traffic[pr_traffic["channel_id"] == ch_id]
        cur_val = int(cur["sessions"].sum())
        pri_val = int(pri["sessions"].sum())
        sessions_by_channel.append({
            "channel": ch_map[ch_id],
            "channel_id": ch_id,
            "current": cur_val,
            "prior": pri_val,
            "change_pct": _pct_change(cur_val, pri_val),
            "_query": _query_meta("fct_daily_web_traffic",
                                  {"date_gte": str(ws.date()), "date_lte": str(we.date()),
                                   "channel_id": ch_id},
                                  "sum(sessions)", len(cur)),
        })

    # Spend
    wk_spend = spend[(spend["date"] >= ws) & (spend["date"] <= we)]
    pr_spend = spend[(spend["date"] >= ps) & (spend["date"] <= pe)]
    spend_current = round(float(wk_spend["spend_usd"].sum()), 2)
    spend_prior = round(float(pr_spend["spend_usd"].sum()), 2)

    # Spend by channel
    spend_by_channel = []
    for ch_id in sorted(paid_channel_ids):
        cur = wk_spend[wk_spend["channel_id"] == ch_id]
        pri = pr_spend[pr_spend["channel_id"] == ch_id]
        cur_val = round(float(cur["spend_usd"].sum()), 2)
        pri_val = round(float(pri["spend_usd"].sum()), 2)
        spend_by_channel.append({
            "channel": ch_map[ch_id],
            "channel_id": ch_id,
            "current": cur_val,
            "prior": pri_val,
            "change_pct": _pct_change(cur_val, pri_val),
            "_query": _query_meta("fct_daily_ad_spend",
                                  {"date_gte": str(ws.date()), "date_lte": str(we.date()),
                                   "channel_id": ch_id},
                                  "sum(spend_usd)", len(cur)),
        })

    # CPL by channel (spend / conversions for the week)
    cpl_by_channel = []
    for ch_id in sorted(paid_channel_ids):
        cur = wk_spend[wk_spend["channel_id"] == ch_id]
        pri = pr_spend[pr_spend["channel_id"] == ch_id]
        cur_spend = float(cur["spend_usd"].sum())
        cur_conv = int(cur["conversions"].sum())
        pri_spend = float(pri["spend_usd"].sum())
        pri_conv = int(pri["conversions"].sum())
        cur_cpl = round(cur_spend / cur_conv, 2) if cur_conv > 0 else None
        pri_cpl = round(pri_spend / pri_conv, 2) if pri_conv > 0 else None
        target_cpl = targets["cpl_targets"].get(ch_map.get(ch_id))
        cpl_by_channel.append({
            "channel": ch_map[ch_id],
            "channel_id": ch_id,
            "current": cur_cpl,
            "prior": pri_cpl,
            "target": target_cpl,
            "change_pct": _pct_change(cur_cpl, pri_cpl) if cur_cpl and pri_cpl else None,
            "_query": _query_meta("fct_daily_ad_spend",
                                  {"date_gte": str(ws.date()), "date_lte": str(we.date()),
                                   "channel_id": ch_id},
                                  "sum(spend_usd)/sum(conversions)", len(cur)),
        })

    # Budget pacing (MTD spend vs monthly budget)
    mtd_spend_data = spend[(spend["date"] >= month_start) & (spend["date"] <= we)]
    next_month = month_start + pd.DateOffset(months=1)
    days_in_month = (next_month - month_start).days
    days_elapsed = (we - month_start).days + 1

    budget_pacing = []
    for ch_id in sorted(paid_channel_ids):
        ch_name = ch_map.get(ch_id)
        ch_mtd = mtd_spend_data[mtd_spend_data["channel_id"] == ch_id]
        spent_mtd = round(float(ch_mtd["spend_usd"].sum()), 2)
        budget = targets["channel_budgets_monthly"].get(ch_name, 0)
        expected_pct = days_elapsed / days_in_month if days_in_month > 0 else 0
        pacing_pct = round(spent_mtd / budget, 4) if budget > 0 else None
        budget_pacing.append({
            "channel": ch_name,
            "channel_id": ch_id,
            "spent_mtd": spent_mtd,
            "budget": budget,
            "expected_pct": round(expected_pct, 4),
            "pacing_pct": pacing_pct,
            "_query": _query_meta("fct_daily_ad_spend",
                                  {"date_gte": str(month_start.date()),
                                   "date_lte": str(we.date()),
                                   "channel_id": ch_id},
                                  "sum(spend_usd)", len(ch_mtd)),
        })

    acquisition = {
        "sessions": {
            "current": sessions_current,
            "prior": sessions_prior,
            "change_pct": _pct_change(sessions_current, sessions_prior),
            "_query": _query_meta("fct_daily_web_traffic",
                                  {"date_gte": str(ws.date()), "date_lte": str(we.date())},
                                  "sum(sessions)", len(wk_traffic)),
        },
        "sessions_by_channel": sessions_by_channel,
        "spend": {
            "current": spend_current,
            "prior": spend_prior,
            "change_pct": _pct_change(spend_current, spend_prior),
            "_query": _query_meta("fct_daily_ad_spend",
                                  {"date_gte": str(ws.date()), "date_lte": str(we.date())},
                                  "sum(spend_usd)", len(wk_spend)),
        },
        "spend_by_channel": spend_by_channel,
        "cpl_by_channel": cpl_by_channel,
        "budget_pacing": budget_pacing,
    }

    # ── CONVERSION (weekly + QTD) ────────────────────────────────────────

    wk_leads = contacts[(contacts["created_date"] >= ws) & (contacts["created_date"] <= we)]
    pr_leads = contacts[(contacts["created_date"] >= ps) & (contacts["created_date"] <= pe)]

    leads_current = len(wk_leads)
    leads_prior = len(pr_leads)
    mqls_current = len(wk_leads[wk_leads["mql_date"].notna()])
    mqls_prior = len(pr_leads[pr_leads["mql_date"].notna()])
    sqls_current = len(wk_leads[wk_leads["sql_date"].notna()])
    sqls_prior = len(pr_leads[pr_leads["sql_date"].notna()])

    mql_rate_current = round(mqls_current / leads_current, 4) if leads_current > 0 else 0
    mql_rate_prior = round(mqls_prior / leads_prior, 4) if leads_prior > 0 else 0

    mql_to_sql_current = round(sqls_current / mqls_current, 4) if mqls_current > 0 else 0
    mql_to_sql_prior = round(sqls_prior / mqls_prior, 4) if mqls_prior > 0 else 0

    # MQL→SQL by source (QTD)
    qtd_contacts = contacts[contacts["created_date"] >= qtd_start]
    qtd_mqls = qtd_contacts[qtd_contacts["mql_date"].notna()]

    mql_to_sql_by_source = []
    for source in sorted(qtd_mqls["lead_source"].unique()):
        src_mqls = qtd_mqls[qtd_mqls["lead_source"] == source]
        src_sqls = src_mqls[src_mqls["sql_date"].notna()]
        rate = round(len(src_sqls) / len(src_mqls), 3) if len(src_mqls) > 0 else 0
        mql_to_sql_by_source.append({
            "source": source,
            "mqls": len(src_mqls),
            "sqls": len(src_sqls),
            "rate": rate,
            "_query": _query_meta("dim_contacts",
                                  {"created_date_gte": str(qtd_start.date()),
                                   "mql_date": "notna", "lead_source": source},
                                  "count(sql_date notna)/count(mql_date notna)",
                                  len(src_mqls)),
        })

    # Average lead score
    score_col = "lead_score_composite"
    wk_scores = wk_leads[score_col] if score_col in wk_leads.columns else None
    pr_scores = pr_leads[score_col] if score_col in pr_leads.columns else None
    avg_score_current = round(float(wk_scores.mean()), 1) if wk_scores is not None and len(wk_scores) > 0 else None
    avg_score_prior = round(float(pr_scores.mean()), 1) if pr_scores is not None and len(pr_scores) > 0 else None

    conversion = {
        "leads": {
            "current": leads_current,
            "prior": leads_prior,
            "change_pct": _pct_change(leads_current, leads_prior),
            "_query": _query_meta("dim_contacts",
                                  {"created_date_gte": str(ws.date()),
                                   "created_date_lte": str(we.date())},
                                  "count(*)", leads_current),
        },
        "mqls": {
            "current": mqls_current,
            "prior": mqls_prior,
            "change_pct": _pct_change(mqls_current, mqls_prior),
            "_query": _query_meta("dim_contacts",
                                  {"created_date_gte": str(ws.date()),
                                   "created_date_lte": str(we.date()),
                                   "mql_date": "notna"},
                                  "count(*)", mqls_current),
        },
        "mql_rate": {
            "current": mql_rate_current,
            "prior": mql_rate_prior,
            "target": targets["mql_rate"],
            "change_pct": _pct_change(mql_rate_current, mql_rate_prior),
        },
        "mql_to_sql_rate": {
            "current": mql_to_sql_current,
            "prior": mql_to_sql_prior,
            "target": targets["mql_to_sql_rate"],
            "change_pct": _pct_change(mql_to_sql_current, mql_to_sql_prior),
        },
        "mql_to_sql_by_source": mql_to_sql_by_source,
        "avg_lead_score": {
            "current": avg_score_current,
            "prior": avg_score_prior,
            "change_pct": _pct_change(avg_score_current, avg_score_prior) if avg_score_current and avg_score_prior else None,
        },
    }

    # ── CONTRIBUTION (QTD) ───────────────────────────────────────────────

    qtd_opps = opps[(opps["created_date"] >= qtd_start) & (opps["created_date"] <= we)]
    pipeline_created_qtd = round(float(qtd_opps["acv_usd"].sum()), 2)
    won_qtd = qtd_opps[qtd_opps["is_won"]]
    won_acv_qtd = round(float(won_qtd["acv_usd"].sum()), 2)
    open_qtd = qtd_opps[~qtd_opps["stage"].str.startswith("Closed")]
    open_pipeline = round(float(open_qtd["acv_usd"].sum()), 2)

    # Pipeline pacing
    days_in_quarter = (qtr_end - qtd_start).days + 1
    days_into_quarter = (we - qtd_start).days + 1
    pipeline_target_qtd = round(targets["pipeline_created"] * (days_into_quarter / days_in_quarter), 2)
    pipeline_pacing_pct = round(pipeline_created_qtd / targets["pipeline_created"], 4)

    # Pipeline by deal source
    pipeline_by_source = []
    for source in sorted(qtd_opps["deal_source"].unique()):
        src_opps = qtd_opps[qtd_opps["deal_source"] == source]
        pipeline_by_source.append({
            "source": source,
            "pipeline_usd": round(float(src_opps["acv_usd"].sum()), 2),
            "opp_count": len(src_opps),
            "_query": _query_meta("dim_opportunities",
                                  {"created_date_gte": str(qtd_start.date()),
                                   "created_date_lte": str(we.date()),
                                   "deal_source": source},
                                  "sum(acv_usd)", len(src_opps)),
        })

    # Pipeline by first-touch channel
    qtd_opp_ids = set(qtd_opps["opp_id"])
    qtd_attr = attribution[attribution["opp_id"].isin(qtd_opp_ids)]
    ft_touches = qtd_attr[qtd_attr["is_first_touch"]]
    ft_with_acv = ft_touches.merge(opps[["opp_id", "acv_usd"]], on="opp_id")

    pipeline_by_channel_ft = []
    for ch_id in sorted(ft_with_acv["channel_id"].unique()):
        ch_data = ft_with_acv[ft_with_acv["channel_id"] == ch_id]
        pipeline_by_channel_ft.append({
            "channel": ch_map.get(ch_id, ch_id),
            "channel_id": ch_id,
            "pipeline_usd": round(float(ch_data["acv_usd"].sum()), 2),
            "opp_count": len(ch_data),
            "_query": _query_meta("fct_multi_touch_attribution + dim_opportunities",
                                  {"opp_created_gte": str(qtd_start.date()),
                                   "is_first_touch": True, "channel_id": ch_id},
                                  "sum(acv_usd)", len(ch_data)),
        })
    pipeline_by_channel_ft.sort(key=lambda x: x["pipeline_usd"], reverse=True)

    contribution = {
        "pipeline_created_qtd": pipeline_created_qtd,
        "pipeline_target_qtd": pipeline_target_qtd,
        "pipeline_pacing_pct": pipeline_pacing_pct,
        "pipeline_by_source": pipeline_by_source,
        "pipeline_by_channel_first_touch": pipeline_by_channel_ft,
        "won_acv_qtd": won_acv_qtd,
        "open_pipeline": open_pipeline,
        "_query": _query_meta("dim_opportunities",
                              {"created_date_gte": str(qtd_start.date()),
                               "created_date_lte": str(we.date())},
                              "sum(acv_usd)", len(qtd_opps)),
    }

    # ── CHANNEL ECONOMICS (full funnel, QTD) ─────────────────────────────

    channel_economics = _compute_channel_economics(
        spend, traffic, attribution, opps, channels,
        qtd_start, we, ch_map
    )

    # ── PIPELINE TRAJECTORY (velocity + projection) ──────────────────────

    pipeline_trajectory = _compute_pipeline_trajectory(
        opps, qtd_start, we, qtr_end, targets["pipeline_created"]
    )

    # ── SDR CAPACITY (for scalability assessment) ────────────────────────

    sdr_capacity = _compute_sdr_capacity(
        reps, sales_activity, intent, accounts, qtd_start, we
    )

    # ── ASSEMBLE ─────────────────────────────────────────────────────────

    snapshot = {
        "report_date": report_date,
        "period": {
            "start": str(ws.date()),
            "end": str(we.date()),
        },
        "prior_period": {
            "start": str(ps.date()),
            "end": str(pe.date()),
        },
        "quarter": quarter,
        "acquisition": acquisition,
        "conversion": conversion,
        "contribution": contribution,
        "channel_economics": channel_economics,
        "pipeline_trajectory": pipeline_trajectory,
        "sdr_capacity": sdr_capacity,
        "targets": targets,
    }

    return snapshot


# ── Channel Economics ────────────────────────────────────────────────────────

def _compute_channel_economics(
    spend_df, traffic_df, attribution_df, opps_df, channels_df,
    qtd_start, we, ch_map
) -> list:
    """Full-funnel economics: spend → CPL → opps → pipeline → ROI by channel.

    This is the core view that replaces CPL-only analysis. A channel with
    high CPL but high pipeline ROI is a good channel. A channel with low CPL
    but no pipeline contribution is a bad channel.
    """
    qtd_opps = opps_df[
        (opps_df["created_date"] >= qtd_start) & (opps_df["created_date"] <= we)
    ]

    # First-touch attribution to channels
    ft = attribution_df[
        attribution_df["is_first_touch"]
        & attribution_df["opp_id"].isin(qtd_opps["opp_id"])
    ]
    ft_merged = ft.merge(
        qtd_opps[["opp_id", "acv_usd", "is_won", "segment"]], on="opp_id"
    )

    ft_by_ch = ft_merged.groupby("channel_id").agg(
        opps=("opp_id", "nunique"),
        pipeline_usd=("acv_usd", "sum"),
        won_count=("is_won", "sum"),
    ).reset_index()

    won_acv_by_ch = (
        ft_merged[ft_merged["is_won"]]
        .groupby("channel_id")["acv_usd"].sum()
    )

    # Segment mix by channel
    seg_mix = (
        ft_merged.groupby(["channel_id", "segment"])
        .agg(opps=("opp_id", "nunique"), pipeline=("acv_usd", "sum"))
        .reset_index()
    )

    # Spend QTD
    qtd_spend = spend_df[
        (spend_df["date"] >= qtd_start) & (spend_df["date"] <= we)
    ]
    ch_spend = qtd_spend.groupby("channel_id").agg(
        spend_usd=("spend_usd", "sum"),
        ad_conversions=("conversions", "sum"),
    ).reset_index()

    # Traffic QTD
    qtd_traffic = traffic_df[
        (traffic_df["date"] >= qtd_start) & (traffic_df["date"] <= we)
    ]
    ch_traffic = qtd_traffic.groupby("channel_id").agg(
        sessions=("sessions", "sum"),
    ).reset_index()

    economics = []
    for ch_id in sorted(ch_map.keys()):
        ch_name = ch_map[ch_id]

        sp = ch_spend[ch_spend["channel_id"] == ch_id]
        spend_val = float(sp["spend_usd"].sum()) if len(sp) > 0 else 0
        ad_conv = int(sp["ad_conversions"].sum()) if len(sp) > 0 else 0

        tr = ch_traffic[ch_traffic["channel_id"] == ch_id]
        sessions = int(tr["sessions"].sum()) if len(tr) > 0 else 0

        ft_row = ft_by_ch[ft_by_ch["channel_id"] == ch_id]
        opp_count = int(ft_row["opps"].sum()) if len(ft_row) > 0 else 0
        pipeline = float(ft_row["pipeline_usd"].sum()) if len(ft_row) > 0 else 0
        won = int(ft_row["won_count"].sum()) if len(ft_row) > 0 else 0
        won_acv = float(won_acv_by_ch.get(ch_id, 0))

        is_paid = spend_val > 0
        cpl = round(spend_val / ad_conv, 2) if ad_conv > 0 else None
        cost_per_opp = round(spend_val / opp_count, 2) if is_paid and opp_count > 0 else None
        pipeline_roi = round(pipeline / spend_val, 2) if spend_val > 0 else None
        avg_acv = round(pipeline / opp_count, 2) if opp_count > 0 else None
        cost_per_pipeline_dollar = round(spend_val / pipeline, 4) if spend_val > 0 and pipeline > 0 else None
        session_to_opp_rate = round(opp_count / sessions, 6) if sessions > 0 else None

        # Segment breakdown
        ch_seg = seg_mix[seg_mix["channel_id"] == ch_id].sort_values(
            "pipeline", ascending=False
        )
        segments = []
        for _, s in ch_seg.iterrows():
            seg_pct = s["pipeline"] / pipeline if pipeline > 0 else 0
            segments.append({
                "segment": s["segment"],
                "opps": int(s["opps"]),
                "pipeline_usd": round(float(s["pipeline"]), 2),
                "pct_of_channel_pipeline": round(seg_pct, 3),
            })

        economics.append({
            "channel": ch_name,
            "channel_id": ch_id,
            "is_paid": is_paid,
            "spend_usd": round(spend_val, 2),
            "sessions": sessions,
            "ad_conversions": ad_conv,
            "cpl": cpl,
            "opps": opp_count,
            "cost_per_opp": cost_per_opp,
            "pipeline_usd": round(pipeline, 2),
            "pipeline_roi": pipeline_roi,
            "avg_opp_acv": avg_acv,
            "cost_per_pipeline_dollar": cost_per_pipeline_dollar,
            "session_to_opp_rate": session_to_opp_rate,
            "won_count": won,
            "won_acv_usd": round(won_acv, 2),
            "segment_mix": segments,
            "_query": _query_meta(
                "fct_daily_ad_spend + fct_daily_web_traffic + fct_multi_touch_attribution + dim_opportunities",
                {"qtd_start": str(qtd_start.date()), "qtd_end": str(we.date()),
                 "channel_id": ch_id},
                "full funnel: spend, sessions, opps, pipeline, ROI",
                opp_count,
            ),
        })

    economics.sort(key=lambda x: x["pipeline_usd"], reverse=True)
    return economics


# ── Pipeline Trajectory ──────────────────────────────────────────────────────

def _compute_pipeline_trajectory(opps_df, qtd_start, we, qtr_end, target_pipeline) -> dict:
    """Weekly velocity trend, projection, and open pipeline stage breakdown."""

    qtd_opps = opps_df[
        (opps_df["created_date"] >= qtd_start) & (opps_df["created_date"] <= we)
    ]

    # Weekly velocity
    q_copy = qtd_opps.copy()
    q_copy["week_start"] = q_copy["created_date"].dt.to_period("W").apply(
        lambda p: p.start_time
    )
    weekly = (
        q_copy.groupby("week_start")
        .agg(opps=("opp_id", "count"), pipeline_usd=("acv_usd", "sum"))
        .reset_index()
        .sort_values("week_start")
    )

    weeks_data = []
    trailing_values = weekly["pipeline_usd"].tolist()
    for i, (_, r) in enumerate(weekly.iterrows()):
        t3 = round(sum(trailing_values[max(0, i - 2): i + 1]) / min(3, i + 1), 2) if i >= 2 else None
        weeks_data.append({
            "week_start": str(r["week_start"].date()),
            "opps": int(r["opps"]),
            "pipeline_usd": round(float(r["pipeline_usd"]), 2),
            "trailing_3wk_avg": t3,
        })

    actual_total = float(weekly["pipeline_usd"].sum())
    weeks_elapsed = len(weekly)
    weeks_in_quarter = max(1, round((qtr_end - qtd_start).days / 7))
    weeks_remaining = max(0, weeks_in_quarter - weeks_elapsed)

    # Trailing average projection
    recent_avg = float(weekly.tail(3)["pipeline_usd"].mean()) if len(weekly) >= 3 else float(weekly["pipeline_usd"].mean())
    projection_trailing = actual_total + (recent_avg * weeks_remaining)

    # Linear regression projection
    if len(weekly) >= 2:
        x = np.arange(len(weekly))
        y = weekly["pipeline_usd"].values.astype(float)
        slope, intercept = np.polyfit(x, y, 1)
        reg_remaining = sum(
            max(0, slope * (len(weekly) + i) + intercept)
            for i in range(weeks_remaining)
        )
        projection_linear = actual_total + reg_remaining
    else:
        slope = 0.0
        projection_linear = projection_trailing

    # Velocity trend classification
    if slope > 50000:
        trend = "accelerating"
    elif slope < -50000:
        trend = "decelerating"
    else:
        trend = "stable"

    # Open pipeline by stage
    open_opps = qtd_opps[~qtd_opps["stage"].str.startswith("Closed")]
    stage_order = [
        "Discovery", "Qualification", "Demo/Evaluation",
        "Proposal/Negotiation", "Security Review", "Procurement/Legal",
    ]
    late_stages = ["Proposal/Negotiation", "Security Review", "Procurement/Legal"]

    stages = []
    for stage in stage_order:
        s_opps = open_opps[open_opps["stage"] == stage]
        if len(s_opps) > 0:
            stages.append({
                "stage": stage,
                "opps": len(s_opps),
                "pipeline_usd": round(float(s_opps["acv_usd"].sum()), 2),
                "closeable_this_quarter": stage in late_stages,
            })

    closeable = float(open_opps[open_opps["stage"].isin(late_stages)]["acv_usd"].sum())
    total_open = float(open_opps["acv_usd"].sum())

    return {
        "weekly_velocity": weeks_data,
        "actual_qtd": round(actual_total, 2),
        "target": target_pipeline,
        "pacing_pct": round(actual_total / target_pipeline, 4) if target_pipeline > 0 else None,
        "weeks_elapsed": weeks_elapsed,
        "weeks_remaining": weeks_remaining,
        "trailing_3wk_avg": round(recent_avg, 2),
        "velocity_slope_per_week": round(float(slope), 2),
        "velocity_trend": trend,
        "projection_trailing_avg": round(projection_trailing, 2),
        "projection_linear": round(projection_linear, 2),
        "open_pipeline": {
            "total_usd": round(total_open, 2),
            "closeable_usd": round(closeable, 2),
            "closeable_pct": round(closeable / total_open, 3) if total_open > 0 else 0,
            "by_stage": stages,
        },
        "_query": _query_meta(
            "dim_opportunities",
            {"created_date_gte": str(qtd_start.date()),
             "created_date_lte": str(we.date())},
            "weekly pipeline velocity + stage breakdown",
            len(qtd_opps),
        ),
    }


# ── SDR Capacity ─────────────────────────────────────────────────────────────

def _compute_sdr_capacity(reps_df, activity_df, intent_df, accounts_df, qtd_start, we) -> dict:
    """SDR team capacity and intent signal coverage for scalability assessment."""

    active_sdrs = reps_df[
        reps_df["role"].isin(["SDR", "BDR"]) & reps_df["is_active"]
    ]
    sdr_count = len(active_sdrs)

    # Activity for the current month
    month_start = pd.Timestamp(we.year, we.month, 1)
    month_activity = activity_df[
        (activity_df["date"] >= month_start) & (activity_df["date"] <= we)
    ]
    sdr_activity = month_activity[
        month_activity["rep_id"].isin(active_sdrs["rep_id"])
    ]

    business_days = len(pd.bdate_range(month_start, we))
    meetings_booked = int(sdr_activity["meetings_booked"].sum())
    meetings_held = int(sdr_activity["meetings_held"].sum())
    meetings_per_sdr_per_day = round(
        meetings_booked / sdr_count / max(1, business_days), 2
    ) if sdr_count > 0 else 0

    # Intent signals — Tier 1 accounts
    tier1 = accounts_df[accounts_df["icp_tier"] == "Tier 1"]
    recent_intent = intent_df[
        (intent_df["signal_date"] >= qtd_start)
        & (intent_df["account_id"].isin(tier1["account_id"]))
    ]
    tier1_with_intent = recent_intent["account_id"].nunique()
    high_intent = recent_intent[
        recent_intent["signal_strength"].isin(["High", "Surging"])
    ]["account_id"].nunique()

    return {
        "active_sdrs": sdr_count,
        "meetings_booked_mtd": meetings_booked,
        "meetings_held_mtd": meetings_held,
        "meetings_per_sdr_per_day": meetings_per_sdr_per_day,
        "estimated_daily_capacity": 3.0,  # industry benchmark for B2B SDR
        "utilization_pct": round(meetings_per_sdr_per_day / 3.0, 3),
        "tier1_accounts_total": len(tier1),
        "tier1_with_intent_qtd": tier1_with_intent,
        "tier1_high_surging_intent": high_intent,
        "_query": _query_meta(
            "dim_sales_reps + fct_sales_activity + fct_account_intent_signals + dim_accounts",
            {"qtd_start": str(qtd_start.date()), "sdr_roles": ["SDR", "BDR"],
             "icp_tier": "Tier 1"},
            "SDR capacity + intent coverage",
            sdr_count,
        ),
    }


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate weekly snapshot JSON")
    parser.add_argument(
        "--date", default="2025-11-03",
        help="Report date (Monday), default: 2025-11-03",
    )
    args = parser.parse_args()

    snapshot = generate_snapshot(args.date)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "snapshot.json"
    with open(out_path, "w") as f:
        json.dump(snapshot, f, indent=2)

    # Summary
    print(f"Snapshot generated for {args.date}")
    print(f"Period: {snapshot['period']['start']} to {snapshot['period']['end']}")
    print(f"Output: {out_path}")
    print(f"\nAcquisition: {snapshot['acquisition']['sessions']['current']:,} sessions, "
          f"${snapshot['acquisition']['spend']['current']:,.0f} spend")
    print(f"Conversion: {snapshot['conversion']['leads']['current']} leads, "
          f"{snapshot['conversion']['mqls']['current']} MQLs "
          f"({snapshot['conversion']['mql_rate']['current']:.1%} rate)")
    print(f"Contribution: ${snapshot['contribution']['pipeline_created_qtd']:,.0f} pipeline QTD, "
          f"${snapshot['contribution']['won_acv_qtd']:,.0f} won")

    # Channel economics highlights
    print(f"\nChannel Economics (top 3 by pipeline ROI):")
    paid = [c for c in snapshot["channel_economics"] if c["is_paid"] and c["pipeline_roi"]]
    paid.sort(key=lambda c: c["pipeline_roi"], reverse=True)
    for c in paid[:3]:
        print(f"  {c['channel']}: {c['pipeline_roi']}x ROI, "
              f"${c['pipeline_usd']:,.0f} pipeline, "
              f"${c['cost_per_opp']:,.0f}/opp, "
              f"avg ACV ${c['avg_opp_acv']:,.0f}")

    # Trajectory
    traj = snapshot["pipeline_trajectory"]
    print(f"\nPipeline Trajectory: {traj['velocity_trend']}")
    print(f"  Trailing 3wk avg: ${traj['trailing_3wk_avg']:,.0f}/week")
    print(f"  Projection (trailing): ${traj['projection_trailing_avg']:,.0f} "
          f"({traj['projection_trailing_avg']/traj['target']:.0%} of target)")
    print(f"  Closeable this Q: ${traj['open_pipeline']['closeable_usd']:,.0f} "
          f"({traj['open_pipeline']['closeable_pct']:.0%} of open)")

    # SDR capacity
    sdr = snapshot["sdr_capacity"]
    print(f"\nSDR Capacity: {sdr['active_sdrs']} SDRs, "
          f"{sdr['meetings_per_sdr_per_day']} meetings/day "
          f"({sdr['utilization_pct']:.0%} utilization)")
    print(f"  Tier 1 with high/surging intent: {sdr['tier1_high_surging_intent']}")


if __name__ == "__main__":
    main()