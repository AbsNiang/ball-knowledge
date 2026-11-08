/**
 * Presentation of research nodes: labels, untruncated titles and the facts each card shows.
 *
 * Node titles are capped at 80 characters by the backend, so full text is recovered from the
 * structured LLM and tool records wherever they carry it. Nothing here judges evidence; values
 * are only formatted for display.
 */
import type { Edge, Node as ResearchNode } from '../../contracts/generated';
import type { Session } from './state';

export type NodeType = ResearchNode['type'];
export type Tone = 'good' | 'warn' | 'bad';
export interface Fact { label: string; value: string; tone?: Tone; wide?: boolean }
export interface NodeView { typeLabel: string; title: string; facts: Fact[] }
export interface GraphContext {
  question: string;
  nodes: Record<string, ResearchNode>;
  edges: Edge[];
  /** Structural parents: `derives_from` points child -> parent, `revises` new -> old. */
  parents: Map<string, string[]>;
  children: Map<string, string[]>;
}
type Rec = Record<string, unknown>;

export const STRUCTURAL_EDGES = new Set<Edge['type']>(['derives_from', 'revises']);

export const asRecord = (value: unknown): Rec | undefined =>
  value !== null && typeof value === 'object' && !Array.isArray(value) ? (value as Rec) : undefined;
export const asText = (value: unknown): string | undefined =>
  typeof value === 'string' && value.trim() ? value.trim() : undefined;
export const asNumber = (value: unknown): number | undefined =>
  typeof value === 'number' && Number.isFinite(value) ? value : undefined;
export const asList = (value: unknown): unknown[] => (Array.isArray(value) ? value : []);

export const TYPE_LABELS: Record<NodeType, string> = {
  question: 'Question',
  premise_check: 'Premise',
  hypothesis: 'Hypothesis',
  test_plan: 'Test plan',
  test_result: 'Test result',
  interpretation: 'Interpretation',
  mechanism_prediction: 'Mechanism',
  verdict: 'Verdict',
  human_note: 'Note',
  source: 'Web source',
  judgment: 'Judge',
  report: 'Answer',
};

export const GATE_NAMES: Record<string, string> = {
  G1: 'Sample size and power',
  G2: 'Confidence interval excludes zero',
  G3: 'Statistical significance',
  G4: 'Multiple-testing correction',
  G5: 'Out-of-sample replication',
  G6: 'Market check',
  G7: 'Holdout confirmation',
};

const TOOL_LABELS: Record<string, string> = {
  compare_groups: 'Group comparison',
  regress: 'Regression',
  persistence: 'Persistence test',
  out_of_sample: 'Out-of-sample replication',
  placebo: 'Placebo test',
  market_test: 'Market test',
  market_check: 'Market check',
  holdout_confirm: 'Holdout confirmation',
  power: 'Power analysis',
  define_slice: 'Slice definition',
  describe_dataset: 'Dataset description',
  declare_primary_test: 'Declare primary test',
  run_primary_test: 'Run primary test',
  replicate_primary: 'Replicate primary test',
  web_search: 'Web search',
  demo_comparison: 'Comparison',
};

const METRIC_LABELS: Record<string, string> = {
  ats_margin: 'ATS margin',
  ats_cover: 'ATS cover rate',
  margin: 'Point margin',
  win: 'Win rate',
  total_over: 'Over rate',
  roi: 'ROI',
};

const DIRECTIONS: Record<string, string> = {
  a_lt_b: 'Group A lower than B',
  a_gt_b: 'Group A higher than B',
  two_sided: 'Two-sided',
};

const OPERATORS: Record<string, string> = { eq: '=', ne: '≠', gte: '≥', lte: '≤', in: 'in', between: 'between' };

export function sentence(value: string): string {
  const text = value.replaceAll('_', ' ').trim();
  return text.charAt(0).toUpperCase() + text.slice(1);
}
export const toolLabel = (tool: string) => TOOL_LABELS[tool] ?? sentence(tool);
export const metricLabel = (metric: string) => METRIC_LABELS[metric] ?? metric;
export const directionLabel = (direction: string) => DIRECTIONS[direction] ?? sentence(direction);

export const signed = (value: number, digits = 2) =>
  `${value > 0 ? '+' : value < 0 ? '−' : ''}${Math.abs(value).toFixed(digits)}`;
export const count = (value: number) => value.toLocaleString('en-US');
export const pValue = (p: number) =>
  p < 0.0001 ? '< 0.0001' : p < 0.001 ? p.toFixed(4) : String(Number(p.toPrecision(2)));
export const interval = (low: number, high: number) => `[${signed(low)}, ${signed(high)}]`;

/** `{split, filters}` as text, e.g. "explore · b2b = true, travel_km ≥ 1000". */
export function sliceLabel(slice: unknown): string {
  if (slice === 'complement') return 'All other games (complement)';
  const spec = asRecord(slice);
  if (!spec) return typeof slice === 'string' ? slice : JSON.stringify(slice);
  const filters = asList(spec.filters).map(item => {
    const f = asRecord(item) ?? {};
    const value = Array.isArray(f.value) ? `[${f.value.join(', ')}]` : String(f.value);
    return `${String(f.col)} ${OPERATORS[String(f.op)] ?? String(f.op)} ${value}`;
  });
  const split = asText(spec.split);
  return [split && `${split} seasons`, filters.join(', ') || 'no filters'].filter(Boolean).join(' · ');
}

export function verdictTone(label: string): Tone {
  if (label.startsWith('Real') || label === 'Chance refuted') return 'good';
  if (label === 'Luck' || label === 'Premise false' || label === 'Chance supported') return 'bad';
  return 'warn';
}

const PLAIN_VERDICTS: Record<string, string> = {
  'Real and beats market (provisional)': 'Holds up, and the betting market has not caught on yet',
  'Real but priced in': 'Holds up, but the betting market already accounts for it',
  'Luck': 'Does not hold up; most likely chance',
  'Inconclusive': 'Not enough evidence either way',
  'Inconclusive (not replicated)': 'Promising, but not yet confirmed in other seasons',
  'Inconclusive (not tested)': 'Not tested',
  'Inconclusive (budget)': 'Not finished; the research budget ran out',
  'Inconclusive (review)': 'Needs a human review before drawing a conclusion',
  'Premise false': 'The pattern in the question does not show up in the data',
  'Untestable': 'Cannot be tested with the available data',
  'Chance supported': 'The gap looks like random noise',
  'Chance refuted': 'The gap is not explained by chance',
  'Chance undetermined': 'Unclear whether the gap is just chance',
};

/** A verdict label as an everyday sentence. */
export const plainVerdict = (label: string) => PLAIN_VERDICTS[label] ?? label;

const PLAIN_TERMS: Record<string, string> = {
  ...Object.fromEntries(Object.entries(METRIC_LABELS).map(([key, label]) => [key, label.toLowerCase().replace('ats', 'ATS')])),
  ...Object.fromEntries(Object.entries(TOOL_LABELS).map(([key, label]) => [key, label.toLowerCase()])),
  b2b: 'back-to-back',
  opp_b2b: 'opponent on a back-to-back',
  travel_km: 'travel distance',
  altitude_m: 'altitude',
  rest_days: 'rest days',
  opp_rest_days: "opponent's rest days",
  tz_shift: 'time-zone change',
  is_home: 'home games',
  is_playoffs: 'playoff games',
  is_bubble: 'bubble games',
  games_into_season: 'point in the season',
  rolling_ats_10: 'recent form against the spread',
  rolling_margin_10: 'recent scoring margin',
};

/** A clause carrying statistics or gate ids: "effect -1.21", "p=0.03", "n = 4,810", "95% CI", "G6". */
const STATISTICAL =
  /\bG[1-7]\b|\b(?:effect|p|n|n_a|n_b|mde|sesoi|alpha|q|roi)\s*(?:=|<|>|≤|≥|of)?\s*[+\-−]?\d|\b95%|\bCI\b|\bp-values?\b|\bconfidence intervals?\b|\b(?:ats_|n_)?[a-z]+_[a-z_]+\s*(?:=|>=|<=|≥|≤|>|<)\s*\d/i;

/** One line of prose without its statistical clauses (a whole line of statistics is dropped). */
function plainLine(line: string): string {
  let text = line.replace(/`/g, '');
  // asides that carry numbers or gate ids, innermost first: "(effect -1.2, 95% CI [-1.7, -0.7])"
  for (let before = ''; before !== text;) {
    before = text;
    text = text.replace(/\s*[([][^()[\]]*(?:\d|\bG[1-7]\b)[^()[\]]*[)\]]/g, '').replace(/\s*\(\s*\)/g, '');
  }
  const sentences = text.split(/(?<=[.!?])\s+/).map(sentence => {
    const ending = /[.!?]$/.test(sentence) ? sentence.slice(-1) : '';
    const clauses = sentence.split(/(?<=[,;:])\s+/);
    // without its opening clause the rest is a fragment ("…, showing no difference"), so it goes
    if (STATISTICAL.test(clauses[0])) return '';
    const kept = clauses.filter(clause => !STATISTICAL.test(clause)).join(' ').replace(/[\s,;:.!?]+$/, '');
    return kept && kept + ending;
  });
  return sentences.filter(Boolean).join(' ');
}

/**
 * Text for a lay reader: clauses with statistics (effects, intervals, p-values, sample sizes,
 * gate ids) are dropped, and code names like `ats_margin` become words. The full report keeps
 * the numbers.
 */
export function plainText(text: string): string {
  return text
    .split('\n')
    // blank lines are paragraph breaks and stay; a line that was all statistics goes
    .flatMap(line => {
      if (!line.trim()) return [line];
      const plain = plainLine(line);
      return plain ? [plain] : [];
    })
    .join('\n')
    .replace(/\b[a-z][a-z0-9]*(?:_[a-z0-9]+)+\b/g, word => PLAIN_TERMS[word] ?? word.replaceAll('_', ' '))
    .replace(/\bb2b\b/gi, 'back-to-back')
    .replace(/[ \t]{2,}/g, ' ')
    .trim();
}

const TONES: Record<string, Tone> = {
  sound: 'good', concerns: 'warn', flawed: 'bad',
  supported: 'good', refuted: 'bad', mixed: 'warn',
};

export function graphContext(session?: Session): GraphContext {
  const nodes = session?.nodes ?? {};
  const edges = Object.values(session?.edges ?? {}).filter(e => nodes[e.source] && nodes[e.target]);
  const parents = new Map<string, string[]>();
  const children = new Map<string, string[]>();
  for (const edge of edges) {
    if (!STRUCTURAL_EDGES.has(edge.type)) continue;
    parents.set(edge.source, [...(parents.get(edge.source) ?? []), edge.target]);
    children.set(edge.target, [...(children.get(edge.target) ?? []), edge.source]);
  }
  return { question: session?.question ?? '', nodes, edges, parents, children };
}

/** The closest structural ancestor of one of `types`, breadth first. */
export function nearestAncestor(ctx: GraphContext, id: string, types: NodeType[]): ResearchNode | undefined {
  const seen = new Set([id]);
  let frontier = [id];
  while (frontier.length) {
    const next: string[] = [];
    for (const current of frontier) {
      for (const parent of ctx.parents.get(current) ?? []) {
        if (seen.has(parent)) continue;
        seen.add(parent);
        const node = ctx.nodes[parent];
        if (node && types.includes(node.type)) return node;
        next.push(parent);
      }
    }
    frontier = next;
  }
  return undefined;
}

/** A backend title split at its first ": " (titles are `${prefix}: ${detail}`). */
function splitTitle(title: string): [string, string] | undefined {
  const at = title.indexOf(': ');
  return at > 0 ? [title.slice(0, at), title.slice(at + 2)] : undefined;
}

/** Prefer structured full text; fall back to the (possibly truncated) title. */
const full = (candidate: unknown, fallback: string) => asText(candidate) ?? fallback;

/** `text` when `title` is a cut-off prefix of it (with or without a trailing ellipsis). */
function extend(title: string, text: string): string {
  const stem = title.replace(/…$/, '');
  return text.length > stem.length && text.startsWith(stem) ? text : title;
}

export const parsedOutput = (n: ResearchNode): Rec => asRecord(n.llm?.parsed_output) ?? {};
export const toolOutput = (n: ResearchNode): Rec => asRecord(n.tool?.output) ?? {};

/** The verdict label and the statement it rules on, untruncated where the graph allows. */
export function verdictParts(n: ResearchNode, ctx: GraphContext): { label: string; statement: string } {
  const parts = splitTitle(n.title);
  const label = parts?.[0] ?? n.verdict?.label ?? 'Verdict';
  let statement = parts?.[1] ?? n.title;
  if (statement.endsWith('…')) {
    const subject = nearestAncestor(ctx, n.id, ['hypothesis', 'premise_check']);
    if (subject) statement = describe(subject, ctx).title;
  }
  return { label, statement };
}

function seedFor(n: ResearchNode): Rec | undefined {
  const stem = n.title.endsWith('…') ? n.title.slice(0, -1) : n.title;
  return asList(parsedOutput(n).seeds)
    .map(asRecord)
    .find(seed => {
      const statement = asText(seed?.statement);
      return statement && (statement === n.title || (n.title.endsWith('…') && statement.startsWith(stem)));
    });
}

function testResultFacts(n: ResearchNode): Fact[] {
  const out = toolOutput(n);
  if (out.status === 'error') {
    return [
      { label: 'Status', value: 'Did not run', tone: 'bad' },
      { label: 'Reason', value: asText(out.reason) ?? 'Unknown error', wide: true },
    ];
  }
  const facts: Fact[] = [];
  const effect = asNumber(out.effect);
  const low = asNumber(out.ci_low);
  const high = asNumber(out.ci_high);
  const p = asNumber(out.p_value);
  const total = asNumber(out.n);
  const extra = asRecord(out.extra) ?? {};
  if (effect !== undefined) facts.push({ label: 'Effect', value: signed(effect) });
  if (low !== undefined && high !== undefined) facts.push({ label: '95% CI', value: interval(low, high) });
  if (effect === undefined) {
    const a = asNumber(extra.n_a);
    const b = asNumber(extra.n_b);
    if (a !== undefined) facts.push({ label: 'Group A', value: `n = ${count(a)}` });
    if (b !== undefined) facts.push({ label: 'Group B', value: `n = ${count(b)}` });
  }
  if (total !== undefined && (effect !== undefined || !facts.length)) facts.push({ label: 'Sample', value: `n = ${count(total)}` });
  if (p !== undefined) facts.push({ label: 'p-value', value: pValue(p) });
  return facts;
}

export function describe(n: ResearchNode, ctx: GraphContext): NodeView {
  const typeLabel = TYPE_LABELS[n.type] ?? sentence(n.type);
  const parsed = parsedOutput(n);
  const facts: Fact[] = [];
  if (n.verdict && n.type !== 'verdict') {
    facts.push({ label: 'Verdict', value: n.verdict.label, tone: verdictTone(n.verdict.label), wide: true });
  }

  switch (n.type) {
    case 'question': {
      return { typeLabel, title: extend(n.title, ctx.question), facts };
    }
    case 'premise_check': {
      const claim = asRecord(parsed.claim);
      const missing = asList(parsed.missing_data).map(String);
      if (claim) {
        if (asText(claim.metric)) facts.push({ label: 'Metric', value: metricLabel(String(claim.metric)) });
        if (asText(claim.primary_test)) facts.push({ label: 'Test', value: toolLabel(String(claim.primary_test)) });
      } else if (missing.length) {
        facts.push({ label: 'Testable', value: 'No', tone: 'bad' }, { label: 'Missing data', value: missing.join(', '), wide: true });
      }
      // the scripted demo titles its premise with the question cut to 80 characters
      return { typeLabel, title: full(claim?.statement, extend(n.title, ctx.question)), facts };
    }
    case 'hypothesis': {
      const seed = seedFor(n);
      if (seed) {
        if (asText(seed.kind)) facts.push({ label: 'Kind', value: sentence(String(seed.kind)) });
        if (typeof seed.testable_with_data === 'boolean') {
          facts.push({ label: 'Testable', value: seed.testable_with_data ? 'With data' : 'No', tone: seed.testable_with_data ? undefined : 'warn' });
        }
      }
      return { typeLabel, title: full(seed?.statement, n.title), facts };
    }
    case 'test_plan': {
      const tool = asText(parsed.tool) ?? splitTitle(n.title)?.[0];
      if (tool) facts.push({ label: 'Tool', value: toolLabel(tool) });
      const metric = asText(asRecord(asRecord(parsed.args)?.test)?.metric);
      if (metric) facts.push({ label: 'Metric', value: metricLabel(metric) });
      const reason = asText(parsed.reason) ?? (tool ? n.title.replace(`${tool}: `, '') : n.title);
      return { typeLabel, title: reason, facts };
    }
    case 'test_result': {
      const out = toolOutput(n);
      const tool = n.tool?.tool_name ?? splitTitle(n.title)?.[0] ?? 'Test';
      const spec = asRecord(asRecord(n.tool?.args)?.spec);
      facts.push(...testResultFacts(n));
      if (tool === 'declare_primary_test') {
        const declared = asRecord(asRecord(out.extra)?.spec) ?? spec;
        return { typeLabel: 'Primary test', title: full(declared?.statement, n.title.replace(/^Declared: /, '')), facts };
      }
      const metric = asText(spec?.metric);
      const title = toolLabel(tool) + (metric ? ` on ${metricLabel(metric)}` : '');
      return { typeLabel, title, facts };
    }
    case 'source': {
      const query = asText(asRecord(n.tool?.args)?.query) ?? n.title.replace(/^Search: /, '');
      const out = toolOutput(n);
      if (out.status === 'error') facts.push({ label: 'Status', value: 'Search failed', tone: 'bad' });
      else facts.push({ label: 'Sources', value: String(asList(out.sources).length) });
      return { typeLabel, title: query, facts };
    }
    case 'judgment': {
      const onPremise = n.title.startsWith('Judge on the premise');
      for (const [key, label] of [['stance', 'Stance'], ['confidence', 'Confidence'], ['step_review', 'Step review']] as const) {
        const value = asText(parsed[key]);
        if (value) facts.push({ label, value: sentence(value), tone: TONES[value] });
      }
      return {
        typeLabel: onPremise ? 'Judge · premise' : typeLabel,
        title: full(parsed.step_rationale, n.title),
        facts,
      };
    }
    case 'verdict': {
      const { label, statement } = verdictParts(n, ctx);
      facts.push({ label: 'Verdict', value: label, tone: verdictTone(label), wide: true });
      if (n.verdict?.holdout_failed) facts.push({ label: 'Holdout', value: 'Failed', tone: 'bad' });
      if (n.verdict?.human_overridden) facts.push({ label: 'Override', value: 'Human', tone: 'warn' });
      return { typeLabel, title: statement, facts };
    }
    case 'report': {
      const verdicts = (ctx.parents.get(n.id) ?? []).filter(id => ctx.nodes[id]?.type === 'verdict').length;
      if (verdicts) facts.push({ label: 'Verdicts', value: String(verdicts) });
      return { typeLabel, title: full(parsed.answer, n.title), facts };
    }
    default:
      return { typeLabel, title: n.title, facts };
  }
}

export type NodeState =
  | 'running' | 'paused' | 'checking' | 'awaiting' | 'passed' | 'flagged' | 'failed'
  | 'tainted' | 'pruned' | 'pending' | 'complete';

export const STATE_LABELS: Record<NodeState, string> = {
  running: 'Researching',
  paused: 'Paused',
  checking: 'Checking',
  awaiting: 'Awaiting approval',
  passed: 'Passed',
  flagged: 'Flagged',
  failed: 'Failed',
  tainted: 'Tainted',
  pruned: 'Pruned',
  pending: 'Pending',
  complete: 'Complete',
};

/** One display state per node. A "passed" outcome with no checks is shown as complete. */
export function nodeState(n: ResearchNode, paused: boolean): NodeState {
  const flags = n.flags ?? [];
  if (flags.includes('pruned')) return 'pruned';
  if (flags.includes('tainted')) return 'tainted';
  if (flags.includes('awaiting_approval')) return 'awaiting';
  if (n.lifecycle === 'running') return paused ? 'paused' : 'running';
  const outcome = n.check_outcome ?? 'pending';
  if (outcome === 'pending') return n.lifecycle === 'checking' ? 'checking' : 'pending';
  if (outcome === 'passed' && !n.checks?.length) return 'complete';
  return outcome;
}

/** A short handle for a node id: `test_result-13` -> `#13`. */
export function shortId(id: string): string {
  const match = /-(\d+)$/.exec(id);
  return match ? `#${match[1]}` : id.length > 10 ? id.slice(0, 8) : id;
}

// -- card geometry (kept in sync with `.research-node` in styles.css) ----------------------

export const CARD_WIDTH = 300;
const CARD_BASE = 138;
const FACT_ROW = 30;
const FACT_GAP = 8;
const FACTS_MARGIN = 12;
export const CARD_FACT_ROWS = 2;

export function factRows(facts: Fact[]): number {
  let rows = 0;
  let column = 0;
  for (const fact of facts) {
    if (fact.wide) {
      rows += column ? 2 : 1;
      column = 0;
    } else if (++column === 2) {
      rows += 1;
      column = 0;
    }
  }
  return rows + (column ? 1 : 0);
}

/** The facts that fit on a card, in order. */
export function cardFacts(facts: Fact[]): Fact[] {
  const shown: Fact[] = [];
  for (const fact of facts) {
    if (factRows([...shown, fact]) > CARD_FACT_ROWS) break;
    shown.push(fact);
  }
  return shown;
}

export function cardHeight(facts: Fact[]): number {
  const rows = factRows(facts);
  return CARD_BASE + (rows ? FACTS_MARGIN + rows * FACT_ROW + (rows - 1) * FACT_GAP : 0);
}
