import { useState } from 'react';
import Markdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { ChevronDown, CircleCheck, FileText } from 'lucide-react';
import {
  describe, nodeState, plainText, plainVerdict, verdictParts, verdictTone, type GraphContext, type Tone,
} from './describe';

interface Outcome { id: string; statement: string; finding: string; tone: Tone }

const NUMBERS = ['no', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten'];
const say = (n: number) => NUMBERS[n] ?? String(n);

/** "We looked at three explanations. One holds up, one does not, and one is still unclear." */
function headline(outcomes: Outcome[]): string {
  const counts = (tone: Tone) => outcomes.filter(o => o.tone === tone).length;
  const parts = [
    counts('good') && `${say(counts('good'))} ${counts('good') === 1 ? 'holds' : 'hold'} up`,
    counts('bad') && `${say(counts('bad'))} ${counts('bad') === 1 ? 'does' : 'do'} not`,
    counts('warn') && `${say(counts('warn'))} ${counts('warn') === 1 ? 'is' : 'are'} still unclear`,
  ].filter((part): part is string => !!part);
  const list = parts.length > 1 ? `${parts.slice(0, -1).join(', ')} and ${parts.at(-1)}` : parts[0] ?? '';
  const n = outcomes.length;
  const intro = `We looked at ${say(n)} possible explanation${n === 1 ? '' : 's'}.`;
  return list ? `${intro} ${list.charAt(0).toUpperCase()}${list.slice(1)}.` : intro;
}

/**
 * The run's answer in plain language, with one line per idea that was tested. Statistics stay
 * in the full report. Runs without a report (the scripted demo) summarise each hypothesis.
 */
export function Conclusion({ ctx, onSelect }: { ctx: GraphContext; onSelect: (id: string) => void }) {
  const [open, setOpen] = useState(true);
  const nodes = Object.values(ctx.nodes);
  const report = nodes.filter(n => n.type === 'report').at(-1);
  const revised = new Set(ctx.edges.filter(e => e.type === 'revises').map(e => e.target));
  let outcomes: Outcome[] = nodes
    .filter(n => n.type === 'verdict' && !revised.has(n.id))
    .map(n => {
      const { label, statement } = verdictParts(n, ctx);
      return { id: n.id, statement: plainText(statement), finding: plainVerdict(label), tone: verdictTone(label) };
    });
  if (!outcomes.length) {
    outcomes = nodes
      .filter(n => n.type === 'hypothesis')
      .map(n => {
        const state = nodeState(n, false);
        const tone: Tone = state === 'passed' ? 'good' : state === 'failed' || state === 'tainted' ? 'bad' : 'warn';
        const finding = tone === 'good' ? 'Held up' : tone === 'bad' ? 'Did not hold up' : 'Unclear';
        return { id: n.id, statement: plainText(describe(n, ctx).title), finding, tone };
      });
  }
  const answer = report ? plainText(describe(report, ctx).title) : '';
  // The fake LLM (and a terse model) answers with the verdict list itself; the rows say that better.
  const restatesRows = outcomes.length > 0 && outcomes.every(o => answer.includes(o.statement.slice(0, 40)));
  const lead = answer && !restatesRows ? answer : headline(outcomes);

  return (
    <section className="conclusion" aria-label="Research conclusion">
      <header className="conclusion-header">
        <button
          type="button"
          className="conclusion-toggle"
          aria-expanded={open}
          title={open ? 'Hide the conclusion' : 'Show the conclusion'}
          onClick={() => setOpen(value => !value)}
        >
          <span className="conclusion-eyebrow"><CircleCheck size={14} aria-hidden="true" />Research complete</span>
          <ChevronDown className="conclusion-chevron" size={14} aria-hidden="true" />
        </button>
        {report && (
          <button onClick={() => onSelect(report.id)}><FileText size={14} aria-hidden="true" />Full report</button>
        )}
      </header>
      {open && (
        <div className="conclusion-body">
          <div className="markdown conclusion-answer"><Markdown remarkPlugins={[remarkGfm]}>{lead}</Markdown></div>
          {outcomes.length > 0 && (
            <>
              <h3 className="conclusion-subhead">What we found</h3>
              <ul className="conclusion-findings">
                {outcomes.map(outcome => (
                  <li key={outcome.id}>
                    <button onClick={() => onSelect(outcome.id)} title="Show this step's evidence">
                      <span className={`finding-dot ${outcome.tone}`} aria-hidden="true" />
                      <span className="finding-text">
                        <span className="finding-statement">{outcome.statement}</span>
                        <span className={`finding-result ${outcome.tone}`}>{outcome.finding}</span>
                      </span>
                    </button>
                  </li>
                ))}
              </ul>
            </>
          )}
        </div>
      )}
    </section>
  );
}
