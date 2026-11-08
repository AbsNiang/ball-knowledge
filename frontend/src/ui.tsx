import type { LucideIcon } from 'lucide-react';
import {
  Ban, Check, Circle, CircleAlert, CircleCheck, CircleX, ClipboardList, Cog, FileText, FlaskConical,
  Gavel, Globe, Hourglass, Lightbulb, LoaderCircle, MessageCircleQuestion, MessageSquareText, Pause,
  Scale, StickyNote, Target, TriangleAlert,
} from 'lucide-react';
import type { CheckRecord } from '../../contracts/generated';
import { GATE_NAMES, STATE_LABELS, sentence, type NodeState, type NodeType, type Tone } from './describe';

const TYPE_ICONS: Record<NodeType, LucideIcon> = {
  question: MessageCircleQuestion,
  premise_check: Target,
  hypothesis: Lightbulb,
  test_plan: ClipboardList,
  test_result: FlaskConical,
  interpretation: MessageSquareText,
  mechanism_prediction: Cog,
  verdict: Gavel,
  human_note: StickyNote,
  source: Globe,
  judgment: Scale,
  report: FileText,
};

const STATE_ICONS: Record<NodeState, LucideIcon> = {
  running: LoaderCircle,
  paused: Pause,
  checking: Hourglass,
  awaiting: Hourglass,
  passed: CircleCheck,
  flagged: CircleAlert,
  failed: CircleX,
  tainted: TriangleAlert,
  pruned: Ban,
  pending: Circle,
  complete: Check,
};

export function TypeIcon({ type, size = 13 }: { type: NodeType; size?: number }) {
  const Icon = TYPE_ICONS[type] ?? Circle;
  return <Icon size={size} aria-hidden="true" />;
}

export function StatusPill({ state }: { state: NodeState }) {
  const Icon = STATE_ICONS[state];
  return (
    <span className={`status-pill ${state}`}>
      <Icon size={11} aria-hidden="true" />
      {STATE_LABELS[state]}
    </span>
  );
}

export function ToneBadge({ tone, children }: { tone?: Tone; children: string }) {
  return <span className={`tone-badge ${tone ?? ''}`}>{children}</span>;
}

/** The latest check per gate (G4 is re-run before the report) or per non-gate checker. */
export function latestChecks(checks: CheckRecord[] = []): CheckRecord[] {
  const latest = new Map<string, CheckRecord>();
  for (const check of checks) latest.set(check.gate_id ?? check.checker, check);
  return [...latest.values()].sort((a, b) => (a.gate_id ?? `~${a.checker}`).localeCompare(b.gate_id ?? `~${b.checker}`));
}

export const checkName = (check: CheckRecord) =>
  check.gate_id ? `${check.gate_id} · ${GATE_NAMES[check.gate_id] ?? 'Gate'}` : `${sentence(check.checker)} review`;

export function CheckChips({ checks }: { checks?: CheckRecord[] }) {
  const shown = latestChecks(checks);
  if (!shown.length) return null;
  return (
    <span className="check-chips">
      {shown.map(check => (
        <span key={check.id} className={`check-chip ${check.status}`} title={`${checkName(check)}: ${check.status}`}>
          {check.gate_id ?? sentence(check.checker)}
        </span>
      ))}
    </span>
  );
}
