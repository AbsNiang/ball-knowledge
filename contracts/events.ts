// Keep this envelope aligned with contracts/events.py.
export interface ResearchEvent {
  seq: number;
  run_id: string;
  ts: string;
  kind:
    | 'node_created'
    | 'node_updated'
    | 'edge_created'
    | 'check_result'
    | 'intervention'
    | 'run_status';
  data: Record<string, unknown>;
}

