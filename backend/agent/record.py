"""Writes research-graph events. Edges point from upstream (source) to downstream (target)."""

from typing import Any

from backend.events.projection import ResearchGraph
from backend.events.store import EventLog


class Recorder:
    def __init__(self, log: EventLog) -> None:
        self.log = log

    @property
    def graph(self) -> ResearchGraph:
        return self.log.graph

    def node(
        self,
        node_type: str,
        title: str,
        *,
        parents: list[str] | tuple[str, ...] = (),
        state: str = "checking",
        rationale: str | None = None,
        **fields: Any,
    ) -> str:
        node_id = f"{node_type}-{len(self.graph.nodes) + 1}"
        self.log.append(
            "node_created",
            {
                "id": node_id,
                "type": node_type,
                "title": title[:80],
                "state": state,
                "parent_ids": list(parents),
                "rationale": rationale,
                **fields,
            },
        )
        for parent in parents:
            self.edge(parent, node_id, "derives_from")
        return node_id

    def update(self, node_id: str, **fields: Any) -> None:
        self.log.append("node_updated", {"id": node_id, **fields})

    def edge(self, source: str, target: str, edge_type: str) -> None:
        edge_id = f"{source}-{target}-{edge_type}"
        if edge_id not in self.graph.edges:
            self.log.append(
                "edge_created",
                {"id": edge_id, "source": source, "target": target, "type": edge_type},
            )

    def status(self, **data: Any) -> None:
        self.log.append("run_status", data)

    def intervention(self, intervention_type: str, node_id: str | None, **payload: Any) -> None:
        self.log.append(
            "intervention", {"type": intervention_type, "node_id": node_id, "payload": payload}
        )
