"""
Marketing Intelligence Briefing — CLI entrypoint.

Runs the full pipeline:
  snapshot → skills → Claude API synthesis → enhanced briefing JSON + Markdown

Usage:
  python -m src.intelligence.briefing --date 2025-10-20
  python -m src.intelligence.briefing --date 2025-10-20 --no-synthesize  (skip API call, use raw findings only)
"""

import argparse
import json
from pathlib import Path

from shared.constants import OUTPUT_DIR
from src.intelligence.snapshot import generate_snapshot
from src.intelligence.skills.acquisition import analyze as analyze_acquisition
from src.intelligence.skills.conversion import analyze as analyze_conversion
from src.intelligence.skills.contribution import analyze as analyze_contribution
from src.intelligence.agents.monday_pulse import compose_briefing


def main():
    parser = argparse.ArgumentParser(description="Generate Monday Marketing Pulse briefing")
    parser.add_argument("--date", default="2025-10-20",
                        help="Report date (Monday), default: 2025-10-20")
    parser.add_argument("--save-snapshot", action="store_true",
                        help="Also save the snapshot JSON to outputs/")
    parser.add_argument("--no-synthesize", action="store_true",
                        help="Skip Claude API synthesis, use raw skill findings only")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Generate snapshot
    print(f"Generating snapshot for {args.date}...")
    snapshot = generate_snapshot(args.date)

    if args.save_snapshot:
        snapshot_path = OUTPUT_DIR / "snapshot.json"
        with open(snapshot_path, "w") as f:
            json.dump(snapshot, f, indent=2)
        print(f"  Snapshot saved to {snapshot_path}")

    # Step 2: Run skills
    print("Running skills...")
    acq_findings = analyze_acquisition(snapshot)
    conv_findings = analyze_conversion(snapshot)
    contrib_findings = analyze_contribution(snapshot)

    print(f"  Acquisition: {len(acq_findings)} findings")
    print(f"  Conversion: {len(conv_findings)} findings")
    print(f"  Contribution: {len(contrib_findings)} findings")

    # Step 3: Synthesize with Claude API (or skip)
    intelligence = None
    if not args.no_synthesize:
        print("Synthesizing intelligence via Claude API...")
        try:
            from src.intelligence.synthesizer import IntelligenceSynthesizer
            synth = IntelligenceSynthesizer()
            intelligence = synth.synthesize(snapshot, acq_findings, conv_findings, contrib_findings)
            print("  Synthesis complete.")

            # Save the enhanced intelligence JSON — this is what the frontend renders
            intel_path = OUTPUT_DIR / "intelligence.json"
            with open(intel_path, "w") as f:
                json.dump(intelligence, f, indent=2)
            print(f"  Intelligence saved to {intel_path}")

        except Exception as e:
            print(f"  Synthesis failed: {e}")
            print("  Falling back to raw findings.")
            intelligence = None

    # Step 4: Compose briefing markdown (from raw findings — always available)
    print("Composing markdown briefing...")
    briefing = compose_briefing(snapshot, acq_findings, conv_findings, contrib_findings)

    briefings_dir = OUTPUT_DIR / "briefings"
    briefings_dir.mkdir(parents=True, exist_ok=True)
    out_path = briefings_dir / f"monday_pulse_{args.date}.md"
    with open(out_path, "w") as f:
        f.write(briefing)
    print(f"  Markdown briefing saved to {out_path}")

    # Step 5: Save the complete frontend data bundle
    # This is everything the React frontend needs to render
    frontend_data = {
        "snapshot": snapshot,
        "raw_findings": {
            "acquisition": acq_findings,
            "conversion": conv_findings,
            "contribution": contrib_findings,
        },
        "intelligence": intelligence,  # None if synthesis was skipped
        "briefing_markdown": briefing,
        "metadata": {
            "report_date": args.date,
            "synthesized": intelligence is not None,
            "period": snapshot["period"],
            "quarter": snapshot["quarter"],
        },
    }

    frontend_path = OUTPUT_DIR / "briefing-data.json"
    with open(frontend_path, "w") as f:
        json.dump(frontend_data, f, indent=2)
    print(f"  Frontend data bundle saved to {frontend_path}")

    # Print summary
    print(f"\n{'='*60}")
    if intelligence:
        print(f"\nTHE ONE THING:")
        print(f"  {intelligence['the_one_thing']['headline']}")
        print(f"\n  {intelligence['the_one_thing']['explanation']}")
        print(f"\n  Action: {intelligence['the_one_thing']['action']}")
        if intelligence.get("cross_domain_connections"):
            print(f"\nCROSS-DOMAIN INSIGHTS:")
            for conn in intelligence["cross_domain_connections"]:
                print(f"  • {conn}")
        if intelligence.get("priority_actions"):
            print(f"\nPRIORITY ACTIONS:")
            for i, pa in enumerate(intelligence["priority_actions"][:5], 1):
                print(f"  {i}. [{pa['urgency']}] {pa['action']}")
                print(f"     Expected: {pa['expected_impact']}")
    else:
        print(briefing[:500])


if __name__ == "__main__":
    main()
