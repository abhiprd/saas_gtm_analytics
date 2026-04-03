/**
 * Types and metadata for the Monday Marketing Pulse briefing.
 *
 * The intelligence data comes from intelligence.json (Claude API synthesis).
 * This file provides TypeScript types matching that schema plus static metadata.
 */

// ── Intelligence JSON types (from synthesizer.py output) ────────────────────

export type Severity = "critical" | "warning" | "info";

export interface TheOneThing {
  headline: string;
  explanation: string;
  action: string;
}

export interface IntelFinding {
  severity: Severity;
  title: string;
  insight: string;
  action: string;
  evidence: {
    current: number | null;
    target: number | null;
    prior: number | null;
  };
}

export interface DomainIntelligence {
  narrative: string;
  findings: IntelFinding[];
}

export interface PriorityAction {
  action: string;
  expected_impact: string;
  urgency: string;
  domain: string;
}

export interface Intelligence {
  the_one_thing: TheOneThing;
  acquisition_intelligence: DomainIntelligence;
  conversion_intelligence: DomainIntelligence;
  contribution_intelligence: DomainIntelligence;
  priority_actions: PriorityAction[];
  cross_domain_connections: string[];
}

// ── Domain display config ───────────────────────────────────────────────────

export interface DomainConfig {
  id: "acquisition" | "conversion" | "contribution";
  key: "acquisition_intelligence" | "conversion_intelligence" | "contribution_intelligence";
  title: string;
  question: string;
}

export const domains: DomainConfig[] = [
  { id: "acquisition", key: "acquisition_intelligence", title: "Acquisition", question: "Is the top of funnel healthy?" },
  { id: "conversion", key: "conversion_intelligence", title: "Conversion", question: "Are leads becoming pipeline?" },
  { id: "contribution", key: "contribution_intelligence", title: "Contribution", question: "Will marketing hit the number?" },
];

// ── Briefing metadata ───────────────────────────────────────────────────────

export const meta = {
  date: "2025-10-20",
  periodStart: "2025-10-13",
  periodEnd: "2025-10-19",
  quarter: "2025-Q4",
};
