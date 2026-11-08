import { useState, type ReactNode } from 'react';
import Markdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { Bar, BarChart, CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import { ExternalLink, X } from 'lucide-react';
import type { CheckRecord, Node as ResearchNode, Verdict } from '../../contracts/generated';
import {
  asList, asNumber, asRecord, asText, count, describe, directionLabel, interval, metricLabel,
  nodeState, parsedOutput, pValue, sentence, signed, sliceLabel, toolLabel, toolOutput, verdictParts,
  verdictTone, type Fact, type GraphContext, type NodeView, type Tone,
} from './describe';
import { StatusPill, ToneBadge, TypeIcon, checkName, latestChecks } from './ui';

type Tab = 'Overview' | 'Evidence' | 'Checks' | 'Model';
type Select = (id: string) => void;

// -- building blocks -----------------------------------------------------------------------

function Section({ title, children }: { title: string; children: ReactNode }) {
  return (
    <section className="inspector-section">
      <h3>{title}</h3>
      {children}
    </section>
  );
}

function Prose({ children }: { children: string }) {
  return (
    <div className="markdown">
      <Markdown
        remarkPlugins={[remarkGfm]}
        components={{ a: ({ node: _node, ...props }) => <a {...props} target="_blank" rel="noreferrer" /> }}
      >
        {children}
      </Markdown>
    </div>
  );
}

interface Row { label: string; value: ReactNode; tone?: Tone; mono?: boolean }

function Rows({ rows }: { rows: Row[] }) {
  if (!rows.length) return null;
  return (
    <dl className="kv">
      {rows.map(row => (
        <div key={row.label}>
          <dt>{row.label}</dt>
          <dd className={[row.tone, row.mono ? 'mono' : ''].filter(Boolean).join(' ') || undefined}>{row.value}</dd>
        </div>
      ))}
    </dl>
  );
}

const factRows = (facts: Fact[]): Row[] => facts.map(f => ({ label: f.label, value: f.value, tone: f.tone }));

function Json({ value }: { value: unknown }) {
  return <pre className="json">{typeof value === 'string' ? value : JSON.stringify(value, null, 2)}</pre>;
}

function NodeLink({ id, ctx, onSelect }: { id: string; ctx: GraphContext; onSelect: Select }) {
  const n = ctx.nodes[id];
  if (!n) return <code className="chip">{id}</code>;
  const view = describe(n, ctx);
  return (
    <button className="node-link" onClick={() => onSelect(id)} title={view.title}>
      <TypeIcon type={n.type} size={12} />
      <span className="node-link-type">{view.typeLabel}</span>
      <span className="node-link-title">{view.title}</span>
    </button>
  );
}

function List({ items }: { items: string[] }) {
  return <ul className="plain-list">{items.map((item, i) => <li key={i}>{item}</li>)}</ul>;
}

/** Scalars as text; numbers keep their precision but get thousands separators. */
function scalar(value: unknown): string {
  if (typeof value === 'boolean') return value ? 'Yes' : 'No';
  if (typeof value === 'number') return Number.isInteger(value) ? count(value) : String(Number(value.toPrecision(4)));
  if (value === null || value === undefined || value === '') return '—';
  return String(value);
}

function specRows(spec: Record<string, unknown>): Row[] {
  const rows: Row[] = [];
  if (asText(spec.statement)) rows.push({ label: 'Statement', value: String(spec.statement) });
  if (asText(spec.kind)) rows.push({ label: 'Kind', value: sentence(String(spec.kind)) });
  if (asText(spec.metric)) rows.push({ label: 'Metric', value: `${metricLabel(String(spec.metric))} (${String(spec.metric)})` });
  if (spec.group_a !== undefined) rows.push({ label: 'Group A', value: sliceLabel(spec.group_a), mono: true });
  if (spec.group_b !== undefined) rows.push({ label: 'Group B', value: sliceLabel(spec.group_b), mono: true });
  if (asText(spec.direction)) rows.push({ label: 'Direction', value: directionLabel(String(spec.direction)) });
  if (asText(spec.primary_test)) rows.push({ label: 'Primary test', value: toolLabel(String(spec.primary_test)) });
  if (Array.isArray(spec.controls)) rows.push({ label: 'Controls', value: spec.controls.length ? spec.controls.join(', ') : 'None', mono: spec.controls.length > 0 });
  return rows;
}

/** Tool or call arguments: specs and slices are formatted, anything else is shown as JSON. */
function ArgumentRows({ args }: { args: Record<string, unknown> }) {
  const rows: Row[] = [];
  const nested: [string, unknown][] = [];
  for (const [key, value] of Object.entries(args)) {
    const spec = asRecord(value);
    if ((key === 'spec' || key === 'test') && spec) continue;
    if (spec && 'filters' in spec) rows.push({ label: sentence(key), value: sliceLabel(spec), mono: true });
    else if (spec || Array.isArray(value)) nested.push([key, value]);
    else rows.push({ label: sentence(key), value: scalar(value), mono: typeof value === 'string' && /^[\w-]+$/.test(value) });
  }
  const spec = asRecord(args.spec) ?? asRecord(args.test);
  if (!rows.length && !nested.length && !spec) return <p className="muted">No arguments.</p>;
  return (
    <>
      {spec && <Rows rows={specRows(spec)} />}
      <Rows rows={rows} />
      {nested.map(([key, value]) => (
        <div key={key} className="nested">
          <span className="nested-label">{sentence(key)}</span>
          <Json value={value} />
        </div>
      ))}
    </>
  );
}

function VerdictRows({ verdict, label }: { verdict: Verdict; label?: string }) {
  const shown = label ?? verdict.label;
  return (
    <Rows
      rows={[
        { label: 'Verdict', value: <ToneBadge tone={verdictTone(shown)}>{shown}</ToneBadge> },
        ...(label && label !== verdict.label ? [{ label: 'Contract label', value: verdict.label }] : []),
        { label: 'Mechanism', value: verdict.explanation_status ?? 'Mechanism plausible but untested' },
        { label: 'Holdout', value: verdict.holdout_failed ? 'Failed' : 'Not failed', tone: verdict.holdout_failed ? 'bad' : undefined },
        { label: 'Human override', value: verdict.human_overridden ? 'Yes' : 'No', tone: verdict.human_overridden ? 'warn' : undefined },
      ]}
    />
  );
}

// -- overview: type-specific detail -------------------------------------------------------

function Details({ node, ctx, view, onSelect }: { node: ResearchNode; ctx: GraphContext; view: NodeView; onSelect: Select }) {
  const parsed = parsedOutput(node);
  const rationale = asText(node.llm?.rationale);

  switch (node.type) {
    case 'premise_check': {
      const claim = asRecord(parsed.claim);
      const missing = asList(parsed.missing_data).map(String);
      return (
        <>
          {claim && <Section title="Measurable claim"><Rows rows={specRows({ ...claim, statement: undefined })} /></Section>}
          {missing.length > 0 && <Section title="Missing data"><List items={missing} /></Section>}
          {rationale && <Section title="Reasoning"><Prose>{rationale}</Prose></Section>}
        </>
      );
    }
    case 'test_plan': {
      const args = asRecord(parsed.args);
      return (
        <>
          {args && <Section title="Planned call"><ArgumentRows args={args} /></Section>}
          {rationale && rationale !== view.title && <Section title="Reasoning"><Prose>{rationale}</Prose></Section>}
        </>
      );
    }
    case 'test_result': {
      const spec = asRecord(asRecord(node.tool?.args)?.spec);
      const out = toolOutput(node);
      return (
        <>
          {spec && view.title !== spec.statement && asText(spec.statement) && (
            <Section title="Claim under test"><p className="lead">{String(spec.statement)}</p></Section>
          )}
          {out.status === 'error' && asText(out.reason) && (
            <Section title="Why it did not run"><p className="lead bad">{String(out.reason)}</p></Section>
          )}
        </>
      );
    }
    case 'judgment': {
      if (!asText(parsed.step_review)) return rationale ? <Section title="Judgment"><Prose>{rationale}</Prose></Section> : null;
      const claims = asList(parsed.claims).map(asRecord).filter((c): c is Record<string, unknown> => !!c);
      const open = asList(parsed.open_questions).map(String).filter(Boolean);
      const audit = asList(parsed.audit);
      return (
        <>
          <Section title="Synthesis so far"><Prose>{String(parsed.synthesis ?? '')}</Prose></Section>
          {claims.length > 0 && (
            <Section title="Claims and evidence">
              <ul className="claims">
                {claims.map((claim, i) => (
                  <li key={i}>
                    <p>{String(claim.text ?? '')}</p>
                    <div className="link-row">
                      {asList(claim.evidence).map(String).map(id => <NodeLink key={id} id={id} ctx={ctx} onSelect={onSelect} />)}
                    </div>
                  </li>
                ))}
              </ul>
            </Section>
          )}
          {asText(parsed.self_check) && <Section title="Self-check"><Prose>{String(parsed.self_check)}</Prose></Section>}
          {open.length > 0 && <Section title="Open questions"><List items={open} /></Section>}
          {asText(parsed.suggested_next) && <Section title="Suggested next step"><Prose>{String(parsed.suggested_next)}</Prose></Section>}
          {audit.length > 0 && <Section title="Audit notes"><List items={audit.map(a => (typeof a === 'string' ? a : JSON.stringify(a)))} /></Section>}
        </>
      );
    }
    case 'source': {
      const summary = asText(parsed.summary) ?? asText(toolOutput(node).summary);
      const sources = asList(parsed.sources ?? toolOutput(node).sources).map(asRecord).filter((s): s is Record<string, unknown> => !!s);
      const queries = asList(parsed.search_queries).map(String).filter(Boolean);
      if (!summary && !sources.length) return rationale ? <Section title="Search"><Prose>{rationale}</Prose></Section> : null;
      return (
        <>
          {summary && <Section title="What the sources say"><Prose>{summary}</Prose></Section>}
          {sources.length > 0 && <Section title={`Sources (${sources.length})`}><SourceList sources={sources} /></Section>}
          {queries.length > 0 && <Section title="Queries the search ran"><List items={queries} /></Section>}
        </>
      );
    }
    case 'verdict': {
      const { label } = verdictParts(node, ctx);
      return (
        <>
          <Section title="Ruling">
            {node.verdict ? <VerdictRows verdict={node.verdict} label={label} /> : <Rows rows={[{ label: 'Verdict', value: <ToneBadge tone={verdictTone(label)}>{label}</ToneBadge> }]} />}
          </Section>
          {rationale && <Section title="Why the research concluded"><Prose>{rationale}</Prose></Section>}
        </>
      );
    }
    case 'report': {
      const answer = asText(parsed.answer);
      const verdicts = (ctx.parents.get(node.id) ?? []).filter(id => ctx.nodes[id]?.type === 'verdict');
      const summaries = asList(parsed.seeds).map(asRecord).filter((s): s is Record<string, unknown> => !!s && !!asText(s.summary));
      if (!answer) return rationale ? <Section title="Report"><Prose>{rationale}</Prose></Section> : null;
      return (
        <>
          <Section title="Answer"><Prose>{answer}</Prose></Section>
          {verdicts.length > 0 && (
            <Section title="Verdicts">
              <ul className="verdict-list">
                {verdicts.map(id => {
                  const parts = verdictParts(ctx.nodes[id], ctx);
                  return (
                    <li key={id}>
                      <button onClick={() => onSelect(id)}>
                        <ToneBadge tone={verdictTone(parts.label)}>{parts.label}</ToneBadge>
                        <span>{parts.statement}</span>
                      </button>
                    </li>
                  );
                })}
              </ul>
            </Section>
          )}
          {summaries.length > 0 && (
            <Section title="Per-hypothesis summary">
              <Rows rows={summaries.map(s => ({ label: sentence(String(s.seed_id ?? 'Seed')), value: String(s.summary) }))} />
            </Section>
          )}
          {asText(parsed.rationale) && <Section title="How the answer was assembled"><Prose>{String(parsed.rationale)}</Prose></Section>}
        </>
      );
    }
    default:
      return rationale ? <Section title="Reasoning"><Prose>{rationale}</Prose></Section> : null;
  }
}

function SourceList({ sources }: { sources: Record<string, unknown>[] }) {
  return (
    <ul className="source-list">
      {sources.map((source, i) => {
        const url = asText(source.url);
        let host = url ?? '';
        try { host = url ? new URL(url).host : ''; } catch { /* keep the raw text */ }
        return (
          <li key={url ?? i}>
            {url ? (
              <a href={url} target="_blank" rel="noreferrer">
                <span>{asText(source.title) ?? url}</span>
                <small>{host}<ExternalLink size={11} aria-hidden="true" /></small>
              </a>
            ) : <span>{asText(source.title) ?? JSON.stringify(source)}</span>}
          </li>
        );
      })}
    </ul>
  );
}

const RELATIONS: Record<string, [outgoing: string, incoming: string]> = {
  derives_from: ['Builds on', 'Leads to'],
  revises: ['Revises', 'Revised by'],
  supports: ['Supports', 'Supported by'],
  refutes: ['Refutes', 'Refuted by'],
  tests: ['Tests', 'Tested by'],
};

function Connections({ node, ctx, onSelect }: { node: ResearchNode; ctx: GraphContext; onSelect: Select }) {
  const groups = new Map<string, string[]>();
  for (const edge of ctx.edges) {
    const [outgoing, incoming] = RELATIONS[edge.type] ?? [sentence(edge.type), sentence(edge.type)];
    if (edge.source === node.id) groups.set(outgoing, [...(groups.get(outgoing) ?? []), edge.target]);
    if (edge.target === node.id) groups.set(incoming, [...(groups.get(incoming) ?? []), edge.source]);
  }
  if (!groups.size) return null;
  return (
    <Section title="Connections">
      <dl className="connections">
        {[...groups].map(([relation, ids]) => (
          <div key={relation}>
            <dt>{relation}</dt>
            <dd>{ids.map(id => <NodeLink key={id} id={id} ctx={ctx} onSelect={onSelect} />)}</dd>
          </div>
        ))}
      </dl>
    </Section>
  );
}

function Overview({ node, ctx, view, onSelect }: { node: ResearchNode; ctx: GraphContext; view: NodeView; onSelect: Select }) {
  // Verdicts get their own section (the ruling, for a verdict node); the summary would repeat them.
  const facts = node.type === 'verdict' ? [] : view.facts.filter(f => !(node.verdict && f.label === 'Verdict'));
  return (
    <>
      {facts.length > 0 && <Section title="Summary"><Rows rows={factRows(facts)} /></Section>}
      <Details node={node} ctx={ctx} view={view} onSelect={onSelect} />
      {node.verdict && node.type !== 'verdict' && <Section title="Verdict"><VerdictRows verdict={node.verdict} /></Section>}
      {!!node.human?.length && (
        <Section title="Reviewer notes">
          {node.human.map((note, i) => (
            <blockquote key={i} className="note">
              <p>{note.note}</p>
              <small>{note.actor_id} · {formatTime(note.ts)}</small>
            </blockquote>
          ))}
        </Section>
      )}
      <Connections node={node} ctx={ctx} onSelect={onSelect} />
    </>
  );
}

// -- evidence: the tool's output ----------------------------------------------------------

interface ChartSpec { type: 'bar' | 'line'; title: string; xKey: string; yKey: string; data: Record<string, string | number>[] }

function chartsFrom(output: Record<string, unknown>): ChartSpec[] {
  return asList(output.charts).filter((c): c is ChartSpec => {
    const chart = asRecord(c);
    return !!chart && ['bar', 'line'].includes(String(chart.type)) && typeof chart.title === 'string'
      && typeof chart.xKey === 'string' && typeof chart.yKey === 'string' && Array.isArray(chart.data)
      && chart.data.every(row => {
        const r = asRecord(row);
        return !!r && ['string', 'number'].includes(typeof r[chart.xKey as string]) && asNumber(r[chart.yKey as string]) !== undefined;
      });
  });
}

function Chart({ chart }: { chart: ChartSpec }) {
  return (
    <figure className="chart">
      <figcaption>{chart.title}</figcaption>
      <ResponsiveContainer width="100%" height={220}>
        {chart.type === 'bar' ? (
          <BarChart data={chart.data}>
            <CartesianGrid stroke="#303d40" vertical={false} />
            <XAxis dataKey={chart.xKey} />
            <YAxis />
            <Tooltip contentStyle={{ background: '#202b2e', border: '1px solid #435459' }} />
            <Bar dataKey={chart.yKey} fill="#94d6b9" radius={[3, 3, 0, 0]} />
          </BarChart>
        ) : (
          <LineChart data={chart.data}>
            <XAxis dataKey={chart.xKey} />
            <YAxis />
            <Tooltip contentStyle={{ background: '#202b2e', border: '1px solid #435459' }} />
            <Line dataKey={chart.yKey} stroke="#94d6b9" />
          </LineChart>
        )}
      </ResponsiveContainer>
    </figure>
  );
}

const RESULT_KEYS = new Set(['tool', 'args', 'status', 'reason', 'args_hash', 'n', 'effect', 'ci_low', 'ci_high', 'p_value', 'slice_hash', 'extra', 'charts', 'sources', 'summary', 'query', 'search_queries', 'model_id']);

function Evidence({ node }: { node: ResearchNode }) {
  const tool = node.tool!;
  const out = toolOutput(node);
  const extra = asRecord(out.extra) ?? {};
  const effect = asNumber(out.effect);
  const low = asNumber(out.ci_low);
  const high = asNumber(out.ci_high);
  const p = asNumber(out.p_value);
  const rows: Row[] = [];
  if (out.status) rows.push({ label: 'Status', value: out.status === 'ok' ? 'Completed' : 'Error', tone: out.status === 'ok' ? undefined : 'bad' });
  if (asText(out.reason)) rows.push({ label: 'Reason', value: String(out.reason), tone: 'bad' });
  if (effect !== undefined) rows.push({ label: 'Effect', value: signed(effect, 3) });
  if (low !== undefined && high !== undefined) rows.push({ label: '95% CI', value: interval(low, high) });
  if (p !== undefined) rows.push({ label: 'p-value', value: pValue(p) });
  if (asNumber(out.n) !== undefined) rows.push({ label: 'Sample (n)', value: count(out.n as number) });
  const nested: [string, unknown][] = [];
  for (const [key, value] of [...Object.entries(extra), ...Object.entries(out).filter(([key]) => !RESULT_KEYS.has(key))]) {
    if (key === 'spec') continue;
    if (value !== null && typeof value === 'object') nested.push([key, value]);
    else rows.push({ label: EXTRA_LABELS[key] ?? sentence(key), value: scalar(value), mono: key.endsWith('_id') });
  }
  const declared = asRecord(extra.spec);
  const charts = chartsFrom(out);
  const sources = asList(out.sources).map(asRecord).filter((s): s is Record<string, unknown> => !!s);
  const args = asRecord(tool.args) ?? {};

  return (
    <>
      {rows.length > 0 && <Section title="Result"><Rows rows={rows} /></Section>}
      {charts.map((chart, i) => <Chart key={i} chart={chart} />)}
      {sources.length > 0 && <Section title={`Sources (${sources.length})`}><SourceList sources={sources} /></Section>}
      {declared && <Section title="Declared specification"><Rows rows={specRows(declared)} /></Section>}
      {nested.map(([key, value]) => <Section key={key} title={sentence(key)}><Json value={value} /></Section>)}
      <Section title="Arguments"><ArgumentRows args={args} /></Section>
      <Section title="Provenance">
        <Rows
          rows={[
            { label: 'Tool', value: `${toolLabel(tool.tool_name)} (${tool.tool_name})` },
            { label: 'Arguments hash', value: tool.args_hash || '—', mono: true },
            { label: 'Data slice hash', value: tool.data_slice_hash || '—', mono: true },
            { label: 'Seed', value: String(tool.seed), mono: true },
            { label: 'Duration', value: `${tool.duration_ms < 10 ? tool.duration_ms.toFixed(2) : Math.round(tool.duration_ms)} ms` },
          ]}
        />
      </Section>
      <Section title="Full tool output"><Json value={out} /></Section>
    </>
  );
}

const EXTRA_LABELS: Record<string, string> = {
  n_a: 'Group A size',
  n_b: 'Group B size',
  mde: 'Minimum detectable effect',
  test_id: 'Test id',
  same_sign: 'Same sign as primary',
  retention: 'Effect retained',
  roi: 'ROI',
  breakeven: 'Break-even rate',
  side: 'Side',
};

// -- checks --------------------------------------------------------------------------------

const CHECK_TONES: Record<CheckRecord['status'], string> = { pass: 'passed', flag: 'flagged', fail: 'failed' };
const CHECKER_GROUPS: [CheckRecord['checker'], string][] = [['gate', 'Statistical gates'], ['judge', 'Judge'], ['critic', 'Critic'], ['human', 'Human review']];

function CheckCard({ check, ctx, onSelect, superseded }: { check: CheckRecord; ctx: GraphContext; onSelect: Select; superseded?: boolean }) {
  return (
    <article className={`check-card ${CHECK_TONES[check.status]}${superseded ? ' superseded' : ''}`}>
      <header>
        <span className="check-name">{checkName(check)}</span>
        <span className={`check-status ${CHECK_TONES[check.status]}`}>{check.status}</span>
      </header>
      <p>{check.rationale}</p>
      <footer>
        <span className="muted">Severity: {check.severity}{superseded ? ' · superseded by a later run' : ''}</span>
        {!!check.evidence?.length && (
          <div className="link-row">
            {check.evidence.map(item => (ctx.nodes[item]
              ? <NodeLink key={item} id={item} ctx={ctx} onSelect={onSelect} />
              : <code key={item} className="chip">{item}</code>))}
          </div>
        )}
      </footer>
    </article>
  );
}

function Checks({ node, ctx, onSelect }: { node: ResearchNode; ctx: GraphContext; onSelect: Select }) {
  const all = node.checks ?? [];
  const latest = latestChecks(all);
  const current = new Set(latest.map(c => c.id));
  const passed = latest.filter(c => c.status === 'pass').length;
  return (
    <>
      <p className="check-summary">{passed} of {latest.length} check{latest.length === 1 ? '' : 's'} passed</p>
      {CHECKER_GROUPS.map(([checker, title]) => {
        const group = latest.filter(c => c.checker === checker);
        return group.length ? (
          <Section key={checker} title={title}>
            {group.map(check => <CheckCard key={check.id} check={check} ctx={ctx} onSelect={onSelect} />)}
          </Section>
        ) : null;
      })}
      {all.length > latest.length && (
        <Section title="Earlier results">
          {all.filter(c => !current.has(c.id)).map(check => <CheckCard key={check.id} check={check} ctx={ctx} onSelect={onSelect} superseded />)}
        </Section>
      )}
    </>
  );
}

// -- model ---------------------------------------------------------------------------------

function parseJson(text: string): unknown {
  try { return JSON.parse(text); } catch { return undefined; }
}

function Model({ node }: { node: ResearchNode }) {
  const llm = node.llm!;
  const prompt = parseJson(llm.prompt);
  const decision = asText(asRecord(prompt)?.decision);
  const context = asRecord(prompt)?.context;
  const raw = parseJson(llm.raw_response);
  const rawDiffers = llm.raw_response.trim() !== '' && JSON.stringify(raw) !== JSON.stringify(llm.parsed_output);
  const hasOutput = Object.keys(llm.parsed_output ?? {}).length > 0;
  return (
    <>
      <Section title="Call">
        <Rows
          rows={[
            { label: 'Model', value: llm.model_id, mono: true },
            ...(decision ? [{ label: 'Role', value: sentence(decision) }] : []),
            { label: 'Cached response', value: llm.from_cache ? 'Yes' : 'No' },
          ]}
        />
      </Section>
      {hasOutput && <Section title="Structured output"><Json value={llm.parsed_output} /></Section>}
      {rawDiffers && <Section title="Raw response"><Json value={raw ?? llm.raw_response} /></Section>}
      <Section title={context !== undefined ? 'Prompt context' : 'Prompt'}><Json value={context ?? llm.prompt} /></Section>
    </>
  );
}

// -- panel ---------------------------------------------------------------------------------

function formatTime(value: string): string {
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? value : date.toLocaleString([], { dateStyle: 'medium', timeStyle: 'medium' });
}

export function Inspector({ node, ctx, paused, onClose, onSelect }: {
  node: ResearchNode;
  ctx: GraphContext;
  paused: boolean;
  onClose: () => void;
  onSelect: Select;
}) {
  const view = describe(node, ctx);
  const tabs: Tab[] = ['Overview'];
  if (node.tool) tabs.push('Evidence');
  if (node.checks?.length) tabs.push('Checks');
  if (node.llm) tabs.push('Model');
  const [chosen, setChosen] = useState<Tab>('Overview');
  const tab = tabs.includes(chosen) ? chosen : 'Overview';
  const version = node.version ?? 1;
  // A report's title is its whole answer, which the Overview shows; head it with the question.
  const heading = node.type === 'report' ? ctx.question || view.title : view.title;

  return (
    <aside className="inspector" aria-label="Step inspector">
      <div className="panel-heading">
        <span>STEP INSPECTOR</span>
        <button aria-label="Close inspector" onClick={onClose}><X size={18} /></button>
      </div>
      <div className="inspector-meta">
        <span className="node-type"><TypeIcon type={node.type} size={13} />{view.typeLabel}</span>
        <code>{node.id}{version > 1 ? ` · v${version}` : ''}</code>
      </div>
      <h2 className={heading.length > 140 ? 'long' : undefined}>{heading}</h2>
      <div className="inspector-status">
        <StatusPill state={nodeState(node, paused)} />
        <time dateTime={node.created_at}>{formatTime(node.created_at)}</time>
      </div>
      {tabs.length > 1 && (
        <div className="tabs" role="tablist">
          {tabs.map(t => (
            <button key={t} role="tab" aria-selected={tab === t} className={tab === t ? 'selected' : ''} onClick={() => setChosen(t)}>
              {t}
              {t === 'Checks' && <span className="tab-count">{latestChecks(node.checks).length}</span>}
            </button>
          ))}
        </div>
      )}
      <div className="inspector-content" role="tabpanel">
        {tab === 'Overview' && <Overview node={node} ctx={ctx} view={view} onSelect={onSelect} />}
        {tab === 'Evidence' && <Evidence node={node} />}
        {tab === 'Checks' && <Checks node={node} ctx={ctx} onSelect={onSelect} />}
        {tab === 'Model' && <Model node={node} />}
      </div>
    </aside>
  );
}
