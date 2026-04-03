"""
Marketing Intelligence Briefing — CLI entrypoint.

Runs the full pipeline: snapshot → skills → Monday Marketing Pulse briefing.

Usage:
  python -m src.intelligence.briefing --date 2025-10-20
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
    args = parser.parse_args()

    # Step 1: Generate snapshot
    print(f"Generating snapshot for {args.date}...")
    snapshot = generate_snapshot(args.date)

    if args.save_snapshot:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        snapshot_path = OUTPUT_DIR / "snapshot.json"
        with open(snapshot_path, "w") as f:
            json.dump(snapshot, f, indent=2)
        print(f"  Snapshot saved to {snapshot_path}")

    # Step 2: Run skills
    print("Running skills...")
    acq_findings = analyze_acquisition(snapshot)
    conv_findings = analyze_conversion(snapshot)
    contrib_findings = analyze_contribution(snapshot)

    print(f"  Acquisition: {len(acq_findings)} findings "
          f"({sum(1 for f in acq_findings if f['severity'] == 'critical')} critical, "
          f"{sum(1 for f in acq_findings if f['severity'] == 'warning')} warning)")
    print(f"  Conversion: {len(conv_findings)} findings "
          f"({sum(1 for f in conv_findings if f['severity'] == 'critical')} critical, "
          f"{sum(1 for f in conv_findings if f['severity'] == 'warning')} warning)")
    print(f"  Contribution: {len(contrib_findings)} findings "
          f"({sum(1 for f in contrib_findings if f['severity'] == 'critical')} critical, "
          f"{sum(1 for f in contrib_findings if f['severity'] == 'warning')} warning)")

    # Step 3: Compose briefing
    print("Composing briefing...")
    briefing = compose_briefing(snapshot, acq_findings, conv_findings, contrib_findings)

    # Save
    briefings_dir = OUTPUT_DIR / "briefings"
    briefings_dir.mkdir(parents=True, exist_ok=True)
    out_path = briefings_dir / f"monday_pulse_{args.date}.md"
    with open(out_path, "w") as f:
        f.write(briefing)

    print(f"\nBriefing saved to {out_path}")
    print(f"\n{'='*60}")
    print(briefing)


if __name__ == "__main__":
    main()
