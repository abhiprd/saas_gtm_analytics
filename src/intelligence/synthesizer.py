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

Your job is NOT to repeat metrics. Any dashboard can show numbers. Your job is to:

1. EXPLAIN WHY — Connect cause and effect across domains. If CPL is high, explain whether 
   those expensive leads are actually producing pipeline. If MQL rate dropped, identify 
   whether it's a volume problem (acquisition) or a quality problem (scoring/source mix).

2. CONNECT THE DOTS — The most valuable insight is always the one that spans domains:
   - Acquisition spend → Conversion quality → Pipeline contribution
   - Channel mix shift → Lead score change → MQL rate decline
   - Pipeline pacing gap → Which deals can realistically close this quarter → What to do NOW

3. BE SPECIFIC ABOUT ACTIONS — Not "optimize spend" but "pause the AP Automation whitepaper 
   campaign on Content Syndication ($X spend, 0.3% conversion) and reallocate to Spend 
   Management campaigns (1.8% conversion). Expected CPL impact: $192 → ~$110."

4. QUANTIFY OUTCOMES — Every recommended action should include an expected impact. 
   "If you do X, expect Y." This is what separates intelligence from observation.

5. PRIORITIZE RUTHLESSLY — The VP has 90 seconds. Lead with the one thing that matters most 
   this week. Everything else is supporting context.

TONE: Direct, opinionated, evidence-based. You're a senior analyst who's been in pipeline 
review meetings, not a reporting tool. You have a point of view and you back it with data.

FORMAT: Return a JSON object with this structure:
{
    "the_one_thing": {
        "headline": "One sentence. The single most important thing this Monday.",
        "explanation": "2-3 sentences explaining why this matters and what causes it.",
        "action": "The specific thing to do this week with expected impact."
    },
    "acquisition_intelligence": {
        "narrative": "2-4 sentences synthesizing acquisition findings. Connect spend to quality to pipeline.",
        "findings": [
            {
                "severity": "critical|warning|info",
                "title": "Short metric name",
                "insight": "What's happening AND why. Not just the number.",
                "action": "Specific action with expected impact.",
                "evidence": {"current": X, "target": Y, "prior": Z}
            }
        ]
    },
    "conversion_intelligence": {
        "narrative": "2-4 sentences synthesizing conversion findings. Connect rates to sources to quality.",
        "findings": [same structure]
    },
    "contribution_intelligence": {
        "narrative": "2-4 sentences synthesizing contribution findings. Connect pipeline to forecast to actions.",
        "findings": [same structure]
    },
    "priority_actions": [
        {
            "action": "Specific action",
            "expected_impact": "Quantified expected result",
            "urgency": "Do this Monday | This week | Before end of quarter",
            "domain": "acquisition|conversion|contribution"
        }
    ],
    "cross_domain_connections": [
        "One sentence connecting insights across two or more domains. These are the insights no single skill can produce."
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
