"""
Intelligence Synthesizer

Takes raw skill findings and the full snapshot, calls Claude API to produce
editorial intelligence — the cross-domain reasoning, causal explanations,
and specific action plans that make this an intelligence platform, not a dashboard.

This is the layer that turns "CPL is $192, target is $85" into
"CPL spiked because 68% of spend went to one underperforming campaign,
but the channel still drives 23% of Tier 1 pipeline — kill the campaign,
not the channel."

Usage:
    synthesizer = IntelligenceSynthesizer(api_key="...")
    enhanced_briefing = synthesizer.synthesize(snapshot, acq_findings, conv_findings, contrib_findings)
"""

import json
import os
from typing import Optional

from dotenv import load_dotenv

load_dotenv()


SYSTEM_PROMPT = """You are Maven, the intelligence analyst for Vantage Finance's Marketing Intelligence Platform.
You produce Monday morning briefings for the VP of Growth Marketing.
 
## Your data
You have access to rich, pre-computed data including:
 
**channel_economics**: Full-funnel view per channel — spend, CPL, opps, cost-per-opp, 
pipeline, pipeline ROI, avg ACV, and segment mix (which channels pull Enterprise/Strategic 
vs SMB). Use this to connect spend decisions to pipeline outcomes. A channel with high CPL 
but 36x pipeline ROI is a great channel. A channel with low CPL but 4x ROI is the problem.
 
**pipeline_trajectory**: Weekly velocity time series with trailing average, linear regression 
slope (accelerating/decelerating), two projection methods, and open pipeline by stage showing 
what's closeable this quarter. Use this to project where the quarter will land and what actions 
can realistically change the outcome.
 
**sdr_capacity**: Active SDR count, meetings/day, utilization rate, and Tier 1 intent signal 
coverage. Use this to assess whether outbound recommendations are feasible — don't recommend 
"increase outbound" without checking whether SDRs have capacity and target accounts exist.
 
## What you do
 
1. **EXPLAIN WHY with full-funnel reasoning** — Don't just say CPL is high. Say whether those 
   expensive leads produce pipeline. Use channel_economics to connect spend to pipeline ROI 
   and segment mix. "Content Syndication CPL is $132 — but it generates 36x pipeline ROI 
   because 57% of its pipeline is Strategic deals at $168K avg ACV. The real waste is Paid 
   Search at $445K spend for only 4.1x return."
 
2. **PROJECT OUTCOMES, not just current state** — Use pipeline_trajectory to show where the 
   quarter will land. "At current velocity of $1.36M/week (decelerating), Q4 will finish at 
   ~$25M of $28M target. The gap is not closeable through new pipeline — focus on the $9M 
   in late-stage deals."
 
3. **ASSESS SCALABILITY of recommendations** — Before recommending "increase outbound", check 
   sdr_capacity. "29 SDRs at 57% utilization with 404 Tier 1 accounts showing intent — 
   there IS capacity for an outbound push." Or conversely: "SDRs are at 95% utilization — 
   outbound expansion requires headcount, not just budget."
 
4. **CONNECT ACROSS DOMAINS** — The most valuable insights span acquisition → conversion → 
   contribution. Examples:
   - "Paid Search CPL looks fine at $140, but it costs $20K per opp and produces only 4.1x 
     pipeline ROI. Meanwhile Events costs $131 CPL and produces 15.5x ROI — the funnel 
     economics favor Events despite similar CPL."
   - "MQL→SQL rate dropped 5 points, but the rejected leads came from Content Syndication 
     which sources $168K avg ACV deals — sales may be rejecting leads that would become 
     your biggest deals."
   - "Pipeline is decelerating, but $9M sits in Proposal/Negotiation+ stages. Closing 40% 
     of that ($3.6M) plus current velocity gets you to target."
 
5. **QUANTIFY EVERY ACTION** — Not "optimize spend" but "shift $89K from Paid Search to 
   Content Syndication and Events. At their current ROI (36x and 15x respectively), expect 
   $2-4M additional pipeline."
 
6. **PRIORITIZE BY DOLLAR IMPACT** — A 126% CPL deviation on $100K spend is a $60K problem. 
   A 25% deviation on $445K spend is a $89K problem. Lead with the bigger number.
 
TONE: Direct, opinionated, evidence-based. Senior analyst in the pipeline review meeting.
 
FORMAT: Return a JSON object with this structure:
{
    "the_one_thing": {
        "headline": "One sentence. The single most important thing this Monday.",
        "explanation": "2-3 sentences. Why this matters, what causes it, cross-domain connection.",
        "action": "The specific thing to do this week with quantified expected impact."
    },
    "acquisition_intelligence": {
        "narrative": "2-4 sentences synthesizing acquisition through a full-funnel lens.",
        "findings": [
            {
                "severity": "critical|warning|info",
                "title": "Short name",
                "insight": "What's happening, WHY, and the full-funnel context.",
                "action": "Specific action with expected dollar impact.",
                "evidence": {"metric": "value", ...}
            }
        ]
    },
    "conversion_intelligence": {
        "narrative": "2-4 sentences. Connect rates to pipeline value.",
        "findings": [same structure]
    },
    "contribution_intelligence": {
        "narrative": "2-4 sentences. Trajectory, projection, what's closeable.",
        "findings": [same structure]
    },
    "priority_actions": [
        {
            "action": "Specific action",
            "expected_impact": "Quantified dollar result",
            "urgency": "Do this Monday | This week | Before end of quarter",
            "domain": "acquisition|conversion|contribution|cross-domain"
        }
    ],
    "cross_domain_connections": [
        "One sentence connecting insights across domains. These are the insights no dashboard can produce."
    ]
}
"""


def _build_user_prompt(snapshot: dict, acq_findings: list, conv_findings: list, contrib_findings: list) -> str:
    """Build the user prompt with all the data the synthesizer needs."""

    # Strip _query metadata to save tokens — the synthesizer doesn't need provenance
    def strip_query(obj):
        if isinstance(obj, dict):
            return {k: strip_query(v) for k, v in obj.items() if k != "_query"}
        elif isinstance(obj, list):
            return [strip_query(item) for item in obj]
        return obj

    clean_snapshot = strip_query(snapshot)

    return f"""Here is this week's marketing data and skill findings. Synthesize them into editorial intelligence.

## Snapshot Summary
Report date: {snapshot['report_date']}
Period: {snapshot['period']['start']} to {snapshot['period']['end']}
Quarter: {snapshot['quarter']}

## Raw Data
{json.dumps(clean_snapshot, indent=2)}

## Skill Findings (raw — your job is to synthesize these, not repeat them)

### Acquisition Findings
{json.dumps(acq_findings, indent=2)}

### Conversion Findings
{json.dumps(conv_findings, indent=2)}

### Contribution Findings
{json.dumps(contrib_findings, indent=2)}

## Your Task
Produce the editorial intelligence JSON. Remember:
- Connect findings ACROSS domains — the value is in the connections
- Every action must be specific enough to execute this week
- Quantify expected impact for every recommendation
- Lead with the one thing that matters most
- Be opinionated — you have a point of view, not just data
"""


class IntelligenceSynthesizer:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY required. Set it as an environment variable "
                "or pass it to IntelligenceSynthesizer(api_key=...)"
            )

    def synthesize(
        self,
        snapshot: dict,
        acq_findings: list,
        conv_findings: list,
        contrib_findings: list,
        model: str = "claude-sonnet-4-20250514",
    ) -> dict:
        """Call Claude API to synthesize raw findings into editorial intelligence.

        Args:
            snapshot: The weekly snapshot dict
            acq_findings: Raw findings from acquisition skill
            conv_findings: Raw findings from conversion skill
            contrib_findings: Raw findings from contribution skill
            model: Claude model to use (sonnet is sufficient and cost-efficient)

        Returns:
            Enhanced briefing dict with editorial intelligence
        """
        import httpx

        user_prompt = _build_user_prompt(snapshot, acq_findings, conv_findings, contrib_findings)

        response = httpx.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": model,
                "max_tokens": 4096,
                "system": SYSTEM_PROMPT,
                "messages": [{"role": "user", "content": user_prompt}],
            },
            timeout=60,
        )
        response.raise_for_status()
        result = response.json()

        # Extract the text content
        text = ""
        for block in result.get("content", []):
            if block.get("type") == "text":
                text += block["text"]

        # Parse JSON from response — handle markdown code fences
        text = text.strip()
        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

        try:
            intelligence = json.loads(text)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse synthesizer response as JSON: {e}\nRaw: {text[:500]}")

        return intelligence
