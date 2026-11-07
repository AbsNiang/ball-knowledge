import type { Event, Node, Edge } from '../../contracts/generated';
export interface Session { id: string; question: string; demo: boolean; status: string; seq: number; nodes: Record<string, Node>; edges: Record<string, Edge>; error?: string }
export const newSession = (id: string, question: string, demo: boolean): Session => ({id, question, demo, status:'queued', seq:0, nodes:{}, edges:{}});
export function reduceEvent(s: Session, e: Event): Session {
  if (e.run_id !== s.id || e.seq <= s.seq) return s;
  if (e.seq !== s.seq + 1) throw new Error('The event stream has a gap. Retry the connection to reload missing events.');
  const next = {...s, seq:e.seq, nodes:{...s.nodes}, edges:{...s.edges}};
  const p = e.payload;
  switch(p.type) {
    case 'run_created': next.question = p.question; break;
    case 'run_status': next.status = p.status; break;
    case 'node_created': case 'node_updated': next.nodes[p.node.id] = p.node; break;
    case 'edge_created': next.edges[p.edge.id] = p.edge; break;
    case 'check_result': {
      const n = next.nodes[p.check.node_id];
      if(n) next.nodes[n.id] = {...n, checks:[...(n.checks ?? []).filter(c => c.id !== p.check.id), p.check]};
      break;
    }
    case 'verdict_computed': {const n=next.nodes[p.hypothesis_id]; if(n) next.nodes[n.id]={...n, verdict:p.verdict}; break;}
    case 'error': next.error = p.message; break;
  }
  return next;
}
export const isActive = (s: Session) => ['queued','running','awaiting_approval'].includes(s.status);
