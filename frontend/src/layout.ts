/** The research graph's layout: dagre ranks and columns, plus orthogonal edge routes. Pure, no React. */
import dagre from '@dagrejs/dagre';
import { CARD_WIDTH, type GraphContext } from './describe';

/**
 * The run goes in phases: the premise is checked (claim, test, judge), then a web search seeds
 * the hypotheses. The search and the hypotheses hang directly off the premise, so on edges alone
 * they would sit level with its checks. These rank constraints keep each phase below the last.
 */
export function phaseConstraints(ctx: GraphContext): [string, string][] {
  const zone = new Set<string>();
  const stack = Object.values(ctx.nodes).filter(n => n.type === 'premise_check').map(n => n.id);
  while (stack.length) {
    const id = stack.pop()!;
    if (zone.has(id)) continue;
    zone.add(id);
    for (const child of ctx.children.get(id) ?? []) {
      if (ctx.nodes[child]?.type !== 'hypothesis') stack.push(child);
    }
  }
  const searches = [...zone].filter(id => ctx.nodes[id]?.type === 'source');
  const checks = new Set([...zone].filter(id => !searches.includes(id)));
  const lastChecks = [...checks].filter(id => !(ctx.children.get(id) ?? []).some(child => checks.has(child)));
  const hypotheses = Object.values(ctx.nodes).filter(n => n.type === 'hypothesis').map(n => n.id);
  const beforeHypotheses = searches.length ? searches : lastChecks;
  return [
    ...lastChecks.flatMap(check => searches.map((search): [string, string] => [check, search])),
    ...beforeHypotheses.flatMap(last => hypotheses.map((hypothesis): [string, string] => [last, hypothesis])),
  ];
}

export type Point = { x: number; y: number };
export const pairKey = (parent: string, child: string) => `${parent}\n${child}`;

export interface Link {
  parent: string;
  child: string;
  /** Also reachable by a longer path, e.g. premise -> hypothesis beside premise -> search -> hypothesis. */
  implied: boolean;
}

/** Every structural parent -> child link, once per pair. */
export function structuralLinks(ctx: GraphContext): Link[] {
  const links: Link[] = [];
  for (const [child, parents] of ctx.parents) {
    for (const parent of new Set(parents)) {
      const implied = parents.some(other => other !== parent && reaches(ctx, parent, other));
      links.push({ parent, child, implied });
    }
  }
  return links;
}

/** Whether `to` descends from `from` through structural links. */
function reaches(ctx: GraphContext, from: string, to: string): boolean {
  const seen = new Set<string>();
  const stack = [from];
  while (stack.length) {
    const id = stack.pop()!;
    if (id === to) return true;
    if (seen.has(id)) continue;
    seen.add(id);
    stack.push(...(ctx.children.get(id) ?? []));
  }
  return false;
}

export interface Layout {
  positions: Map<string, Point>;
  /** Orthogonal edge routes in flow coordinates, by `pairKey(parent, child)`. */
  routes: Map<string, Point[]>;
}

const RANK_SEP = 72;

export function layout(ctx: GraphContext, heights: Map<string, number>): Layout {
  const g = new dagre.graphlib.Graph()
    .setGraph({ rankdir: 'TB', nodesep: 48, ranksep: RANK_SEP, edgesep: 24 })
    .setDefaultEdgeLabel(() => ({}));
  for (const id of Object.keys(ctx.nodes)) g.setNode(id, { width: CARD_WIDTH, height: heights.get(id) ?? 0 });
  // Structural links only, parent above child. Evidence edges point back up the graph and
  // would otherwise form cycles that dagre resolves by moving whole branches.
  const links = structuralLinks(ctx);
  for (const { parent, child, implied } of links) {
    // Hypotheses are proposed together, so they pull hard to share one row under the search.
    // The report gathers verdicts from columns of different depths; letting those links pull
    // would drag shorter columns down to meet it, so it just sits below the last verdict.
    // An implied link only needs a lane to be drawn in, not a say in where nodes go.
    const type = ctx.nodes[child]?.type;
    const weight = implied || type === 'report' ? 0 : type === 'hypothesis' ? 8 : 1;
    g.setEdge(parent, child, { weight, minlen: 1 });
  }
  for (const [above, below] of phaseConstraints(ctx)) {
    // weight 0: a rank constraint only, ignored when shortening edges and ordering columns
    if (!g.hasEdge(above, below)) g.setEdge(above, below, { weight: 0, minlen: 1 });
  }
  dagre.layout(g);

  const positions = new Map<string, Point>();
  // Rows: dagre centres every node of a rank on one line; a row is as tall as its tallest card.
  const rows = new Map<number, number>();
  for (const id of Object.keys(ctx.nodes)) {
    const at = g.node(id);
    const half = (heights.get(id) ?? 0) / 2;
    positions.set(id, { x: at.x - CARD_WIDTH / 2, y: at.y - half });
    const centre = Math.round(at.y);
    rows.set(centre, Math.max(rows.get(centre) ?? 0, half));
  }
  const centres = [...rows.keys()].sort((a, b) => a - b);
  const rowOf = new Map(centres.map((centre, i) => [centre, i]));
  // the horizontal "bus" in the gap below row i, shared by every edge crossing that gap
  const bus = (i: number) => (centres[i] + rows.get(centres[i])! + centres[i + 1] - rows.get(centres[i + 1])!) / 2;

  const routes = new Map<string, Point[]>();
  // Edges from one parent share its lane through each row they both cross, and part only at
  // the bus above their own target's row. Every lane used is one dagre reserved, so none cuts a card.
  const sharedLanes = new Map<string, Map<number, number>>();
  // shortest first, so a parent's nearer targets set the lane the longer edges follow
  const ordered = [...links].sort((a, b) => g.node(a.child).y - g.node(b.child).y);
  for (const { parent, child } of ordered) {
    const p = g.node(parent);
    const c = g.node(child);
    const from = { x: p.x, y: p.y + (heights.get(parent) ?? 0) / 2 };
    const to = { x: c.x, y: c.y - (heights.get(child) ?? 0) / 2 };
    const top = rowOf.get(Math.round(p.y))!;
    const bottom = rowOf.get(Math.round(c.y))!;
    if (bottom <= top) {
      routes.set(pairKey(parent, child), [from, to]);
      continue;
    }
    // Edges spanning several rows follow the lanes dagre reserved beside the cards in between.
    const lanes = new Map<number, number>();
    for (const point of g.edge(parent, child)?.points ?? []) {
      const row = rowOf.get(Math.round(point.y));
      if (row !== undefined && row > top && row < bottom) lanes.set(row, point.x);
    }
    const shared = sharedLanes.get(parent) ?? new Map<number, number>();
    sharedLanes.set(parent, shared);
    for (const [row, x] of lanes) {
      if (shared.has(row)) lanes.set(row, shared.get(row)!);
      else shared.set(row, x);
    }
    const route: Point[] = [from, { x: from.x, y: bus(top) }];
    let x = from.x;
    for (let row = top + 1; row < bottom; row++) {
      x = lanes.get(row) ?? x;
      route.push({ x, y: bus(row - 1) }, { x, y: bus(row) });
    }
    route.push({ x: to.x, y: bus(bottom - 1) }, to);
    routes.set(pairKey(parent, child), simplify(route));
  }
  return { positions, routes };
}

/** Drops repeated points and points in the middle of a straight run. */
function simplify(points: Point[]): Point[] {
  const out: Point[] = [];
  for (const point of points) {
    const last = out.at(-1);
    if (last && Math.abs(last.x - point.x) < 0.5 && Math.abs(last.y - point.y) < 0.5) continue;
    const before = out.at(-2);
    if (last && before && ((before.x === last.x && last.x === point.x) || (before.y === last.y && last.y === point.y))) out.pop();
    out.push(point);
  }
  return out;
}

/** An orthogonal polyline with rounded corners. */
export function roundedPath(points: Point[], radius = 10): string {
  const at = (p: Point) => `${p.x.toFixed(1)} ${p.y.toFixed(1)}`;
  let d = `M ${at(points[0])}`;
  for (let i = 1; i < points.length - 1; i++) {
    const [a, b, c] = [points[i - 1], points[i], points[i + 1]];
    const r = Math.min(radius, Math.hypot(b.x - a.x, b.y - a.y) / 2, Math.hypot(c.x - b.x, c.y - b.y) / 2);
    const toward = (q: Point, length: number) => {
      const distance = Math.hypot(q.x - b.x, q.y - b.y) || 1;
      return { x: b.x + ((q.x - b.x) / distance) * length, y: b.y + ((q.y - b.y) / distance) * length };
    };
    d += ` L ${at(toward(a, r))} Q ${at(b)} ${at(toward(c, r))}`;
  }
  return `${d} L ${at(points[points.length - 1])}`;
}
