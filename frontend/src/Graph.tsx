import { useEffect, useMemo, useState } from 'react';
import {
  Background, BaseEdge, ControlButton, Controls, Handle, MarkerType, MiniMap, Position, ReactFlow, ReactFlowProvider,
  useReactFlow, useStore, type BuiltInEdge, type Edge as FlowEdge, type EdgeProps, type FitViewOptions,
  type Node as FlowNode, type NodeProps,
} from '@xyflow/react';
import { Locate } from 'lucide-react';
import type { Edge, Node as ResearchNode } from '../../contracts/generated';
import type { Session } from './state';
import {
  CARD_WIDTH, STRUCTURAL_EDGES, cardFacts, cardHeight, describe, nodeState, shortId,
  type Fact, type GraphContext, type NodeState, type NodeView,
} from './describe';
import { CheckChips, StatusPill, TypeIcon } from './ui';
import { layout, pairKey, roundedPath, type Point } from './layout';

interface CardData extends Record<string, unknown> {
  record: ResearchNode;
  view: NodeView;
  facts: Fact[];
  state: NodeState;
  height: number;
  selected: boolean;
}
type CardNode = FlowNode<CardData, 'research'>;

function ResearchCard({ data }: NodeProps<CardNode>) {
  const { record: n, view, facts, state, height, selected } = data;
  const version = n.version ?? 1;
  return (
    <article className={`research-node ${state}${selected ? ' selected' : ''}`} style={{ height }} title={view.title}>
      <Handle type="target" position={Position.Top} id="in" />
      <header className="node-header">
        <span className="node-type"><TypeIcon type={n.type} />{view.typeLabel}</span>
        <StatusPill state={state} />
      </header>
      <h3 className="node-title">{view.title}</h3>
      {facts.length > 0 && (
        <dl className="node-facts">
          {facts.map(fact => (
            <div key={fact.label} className={fact.wide ? 'wide' : undefined}>
              <dt>{fact.label}</dt>
              <dd className={fact.tone}>{fact.value}</dd>
            </div>
          ))}
        </dl>
      )}
      <footer className="node-footer">
        <CheckChips checks={n.checks} />
        <span className="node-id">{shortId(n.id)}{version > 1 ? ` · v${version}` : ''}</span>
      </footer>
      <Handle type="source" position={Position.Bottom} id="out" />
      {/* Evidence edges (supports/refutes) run up the card's side instead of through the chain. */}
      <Handle type="source" position={Position.Right} id="evidence-out" className="side-handle" />
      <Handle type="target" position={Position.Right} id="evidence-in" className="side-handle" />
    </article>
  );
}
const nodeTypes = { research: ResearchCard };

type RouteEdge = FlowEdge<{ points: Point[] }, 'route'>;

/** Draws the precomputed route, so edges leaving one card share its trunk and bus. */
function RouteEdgePath({ data, style }: EdgeProps<RouteEdge>) {
  return data ? <BaseEdge path={roundedPath(data.points)} style={style} /> : null;
}
const edgeTypes = { route: RouteEdgePath };

export const EDGE_COLORS: Record<Edge['type'], string> = {
  derives_from: '#4a5d61',
  tests: '#4a5d61',
  revises: '#a48d5c',
  supports: '#6fb38f',
  refutes: '#c9826e',
};

function flowEdges(ctx: GraphContext, routes: Map<string, Point[]>, selected: string | null): (BuiltInEdge | RouteEdge)[] {
  const edges: (BuiltInEdge | RouteEdge)[] = [];
  const drawn = new Set<string>();
  for (const e of ctx.edges) {
    const color = EDGE_COLORS[e.type] ?? EDGE_COLORS.derives_from;
    if (STRUCTURAL_EDGES.has(e.type)) {
      // derives_from and revises point child -> parent; draw them parent -> child, top-down
      const key = pairKey(e.target, e.source);
      const points = routes.get(key);
      if (!points || drawn.has(key)) continue; // a derives_from and a revises link drawn as one
      drawn.add(key);
      const active = selected === e.target || selected === e.source;
      edges.push({
        id: e.id,
        source: e.target,
        target: e.source,
        sourceHandle: 'out',
        targetHandle: 'in',
        type: 'route',
        data: { points },
        selectable: false,
        focusable: false,
        zIndex: active ? 1 : 0,
        style: {
          stroke: active ? '#a9cdb8' : color,
          strokeWidth: active ? 2.2 : 1.4,
          strokeDasharray: e.type === 'revises' ? '5 4' : undefined,
        },
      });
    } else {
      // Evidence links (a result supporting or refuting its hypothesis) run back up the graph
      // beside the column; they stay faint until one of their two steps is selected.
      const active = selected === e.source || selected === e.target;
      edges.push({
        id: e.id,
        source: e.source,
        target: e.target,
        sourceHandle: 'evidence-out',
        targetHandle: 'evidence-in',
        type: 'smoothstep',
        pathOptions: { borderRadius: 12, offset: 26 },
        selectable: false,
        focusable: false,
        zIndex: active ? 2 : 0,
        style: { stroke: color, strokeWidth: active ? 2 : 1.3, strokeDasharray: '5 4', opacity: active ? 1 : 0.55 },
        markerEnd: { type: MarkerType.ArrowClosed, color, width: 16, height: 16 },
      });
    }
  }
  return edges;
}

const MINIMAP_COLORS: Partial<Record<NodeState, string>> = {
  running: '#a3d7b8',
  passed: '#5f8f76',
  complete: '#4f6a68',
  failed: '#9a6556',
  flagged: '#a8925e',
  tainted: '#9a6556',
  paused: '#c4b082',
};

interface GraphProps {
  session: Session;
  ctx: GraphContext;
  selected: string | null;
  onSelect: (id: string) => void;
}

// Padding in px; the top clears the legend that overlays the canvas.
const PAD = { top: 64, side: 40, bottom: 32 };
const FIT: FitViewOptions = {
  padding: { top: `${PAD.top}px`, bottom: `${PAD.bottom}px`, x: `${PAD.side}px` },
  maxZoom: 1,
};
/** Below this zoom card text is unreadable, so following centres the newest step instead. */
const READABLE_ZOOM = 0.55;
const FOLLOW_ZOOM = 0.8;

export function Graph(props: GraphProps) {
  return <ReactFlowProvider><Flow {...props} /></ReactFlowProvider>;
}

function Flow({ session, ctx, selected, onSelect }: GraphProps) {
  const paused = session.status === 'paused';
  const { fitView, setCenter } = useReactFlow();
  const width = useStore(s => s.width);
  const height = useStore(s => s.height);
  // Following: show the whole graph while it fits at a readable zoom, else centre the newest
  // step. Any pan or zoom by the user stops it; the locate control resumes it.
  const [following, setFollowing] = useState(true);
  const stopFollowing = () => setFollowing(false);
  const cards = useMemo(() => {
    const out = new Map<string, { view: NodeView; facts: Fact[]; height: number }>();
    for (const n of Object.values(ctx.nodes)) {
      const view = describe(n, ctx);
      const facts = cardFacts(view.facts);
      out.set(n.id, { view, facts, height: cardHeight(facts) });
    }
    return out;
  }, [ctx]);
  const { positions, routes } = useMemo(
    () => layout(ctx, new Map([...cards].map(([id, card]) => [id, card.height]))),
    [ctx, cards],
  );
  const nodes = useMemo<CardNode[]>(
    () => Object.values(ctx.nodes).map(n => {
      const card = cards.get(n.id)!;
      return {
        id: n.id,
        type: 'research',
        position: positions.get(n.id)!,
        width: CARD_WIDTH,
        height: card.height,
        data: { record: n, ...card, state: nodeState(n, paused), selected: n.id === selected },
      };
    }),
    [ctx, cards, positions, paused, selected],
  );
  const edges = useMemo(() => flowEdges(ctx, routes, selected), [ctx, routes, selected]);
  const latest = Object.keys(ctx.nodes).at(-1);
  const bounds = useMemo(() => {
    let [left, top, right, bottom] = [Infinity, Infinity, -Infinity, -Infinity];
    for (const [id, at] of positions) {
      left = Math.min(left, at.x);
      top = Math.min(top, at.y);
      right = Math.max(right, at.x + CARD_WIDTH);
      bottom = Math.max(bottom, at.y + (cards.get(id)?.height ?? 0));
    }
    return { width: right - left, height: bottom - top };
  }, [positions, cards]);
  // The newest step's centre, as numbers so status-only events do not move the camera.
  const at = latest ? positions.get(latest) : undefined;
  const focusX = at ? at.x + CARD_WIDTH / 2 : 0;
  const focusY = at && latest ? at.y + (cards.get(latest)?.height ?? 0) / 2 : 0;
  useEffect(() => {
    if (!following || !latest || !width || !height) return;
    const wholeZoom = Math.min(
      (width - 2 * PAD.side) / bounds.width,
      (height - PAD.top - PAD.bottom) / bounds.height,
    );
    const frame = requestAnimationFrame(() => {
      if (wholeZoom >= READABLE_ZOOM) void fitView({ ...FIT, duration: 250 });
      // the step sits just below centre, so what it builds on stays in view above it
      else void setCenter(focusX, focusY - height * 0.18 / FOLLOW_ZOOM, { zoom: FOLLOW_ZOOM, duration: 250 });
    });
    return () => cancelAnimationFrame(frame);
  }, [following, latest, focusX, focusY, bounds.width, bounds.height, width, height, fitView, setCenter]);

  return (
    <ReactFlow
      nodes={nodes}
      edges={edges}
      nodeTypes={nodeTypes}
      edgeTypes={edgeTypes}
      onNodeClick={(_, node) => onSelect(node.id)}
      // a move with an input event is the user's; fitView's own moves have none
      onMoveStart={event => { if (event) stopFollowing(); }}
      minZoom={0.05}
      maxZoom={1.6}
      nodesDraggable={false}
      nodesConnectable={false}
      deleteKeyCode={null}
      colorMode="dark"
    >
      <Background color="#2c3a3e" gap={24} size={1} />
      <Controls
        showInteractive={false}
        fitViewOptions={FIT}
        onZoomIn={stopFollowing}
        onZoomOut={stopFollowing}
        onFitView={stopFollowing}
      >
        <ControlButton
          onClick={() => setFollowing(true)}
          className={following ? 'following' : undefined}
          title={following ? 'Following the latest step' : 'Follow the latest step'}
          aria-label="Follow the latest step"
          aria-pressed={following}
        >
          <Locate size={14} />
        </ControlButton>
      </Controls>
      <MiniMap
        pannable
        zoomable
        nodeColor={node => MINIMAP_COLORS[(node.data as CardData).state] ?? '#3f5254'}
        maskColor="#0c1219aa"
      />
    </ReactFlow>
  );
}
