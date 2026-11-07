import { useEffect, useState } from 'react';
import { Background, Controls, ReactFlow, type Edge, type Node } from '@xyflow/react';
import type { ResearchEvent } from '../../contracts/events';

export default function App() {
  const [events, setEvents] = useState<ResearchEvent[]>([]);
  const [error, setError] = useState('');

  useEffect(() => {
    const controller = new AbortController();
    fetch('/api/fixtures/events', { signal: controller.signal })
      .then((response) => {
        if (!response.ok) throw new Error(`API returned ${response.status}`);
        return response.json() as Promise<ResearchEvent[]>;
      })
      .then(setEvents)
      .catch((reason: unknown) => {
        if (!controller.signal.aborted) {
          setError(reason instanceof Error ? reason.message : 'Unable to load fixture');
        }
      });
    return () => controller.abort();
  }, []);

  const nodes: Node[] = events.flatMap((event, index) => {
    const payload = event.payload;
    if (payload.type !== 'node_created') return [];
    return [{
      id: payload.node.id,
      position: { x: 80, y: index * 170 },
      data: { label: payload.node.title },
    }];
  });
  const edges: Edge[] = events.flatMap((event) => {
    const payload = event.payload;
    if (payload.type !== 'edge_created') return [];
    return [{
      id: payload.edge.id,
      source: payload.edge.source,
      target: payload.edge.target,
    }];
  });

  return (
    <main>
      <header>
        <h1>Whack</h1>
        <p>NBA research, with an evidence trail.</p>
        <small>Mock replay · No statistical analysis or LLM calls performed.</small>
      </header>
      {error ? <p role="alert">{error}. Start the backend on port 8000 and refresh.</p> :
        events.length === 0 ? <p role="status">Loading fixture…</p> :
        <section className="graph" aria-label="Research graph">
          <ReactFlow nodes={nodes} edges={edges} fitView nodesDraggable={false}>
            <Background /><Controls />
          </ReactFlow>
        </section>}
      <footer>Research tool, not betting advice.</footer>
    </main>
  );
}
