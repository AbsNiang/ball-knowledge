"""Research-graph projection: the graph is derived from the event log, never written directly."""

import copy
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from contracts.events import Event

# Taint follows provenance only. Evidence edges (supports/refutes) and `revises` do not carry it.
LINEAGE_EDGES = frozenset({"derives_from"})


@dataclass
class ResearchGraph:
    nodes: dict[str, dict[str, Any]] = field(default_factory=dict)
    edges: dict[str, dict[str, Any]] = field(default_factory=dict)
    interventions: list[dict[str, Any]] = field(default_factory=list)
    status: dict[str, Any] = field(default_factory=dict)

    def apply(self, event: Event) -> None:
        data = copy.deepcopy(event.data)
        match event.kind:
            case "node_created":
                data.setdefault("checks", [])
                self.nodes[data["id"]] = data
            case "node_updated":
                self.nodes[data["id"]].update(data)
            case "edge_created":
                self.edges[data["id"]] = data
            case "check_result":
                self.nodes[data["node_id"]]["checks"].append(data)
            case "intervention":
                self.interventions.append(data)
            case "run_status":
                self.status.update(data)

    def descendants(self, node_id: str) -> list[str]:
        """Every node derived from `node_id`, breadth first."""
        children: dict[str, list[str]] = {}
        for edge in self.edges.values():
            if edge["type"] in LINEAGE_EDGES:
                children.setdefault(edge["source"], []).append(edge["target"])
        return _walk(node_id, children)

    def ancestors(self, node_id: str) -> list[str]:
        """The provenance path of `node_id`, nearest first (used for report paths)."""
        parents = {nid: node["parent_ids"] for nid, node in self.nodes.items()}
        return _walk(node_id, parents)


def _walk(start: str, neighbours: dict[str, list[str]]) -> list[str]:
    seen, order, queue = {start}, [], deque([start])
    while queue:
        for nxt in neighbours.get(queue.popleft(), []):
            if nxt not in seen:
                seen.add(nxt)
                order.append(nxt)
                queue.append(nxt)
    return order


def reduce(events: Iterable[Event]) -> ResearchGraph:
    graph = ResearchGraph()
    for event in events:
        graph.apply(event)
    return graph
