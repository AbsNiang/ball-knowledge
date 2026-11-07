"""Research-graph projection: the graph is derived from the event log, never written directly.

It applies events the way the frontend reducer does: nodes are full snapshots, a `check_result`
adds its check to the node, and `verdict_computed` sets the verdict on the hypothesis node.
Edges follow plan.md §5.3: `derives_from` points child -> parent and `revises` new -> old.
"""

from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass, field

from contracts.events import (
    BudgetUpdated,
    CheckResult,
    EdgeCreated,
    ErrorOccurred,
    Event,
    InterventionRecorded,
    ModeChanged,
    NodeCreated,
    NodeUpdated,
    RunCreated,
    StatusChanged,
    VerdictComputed,
)
from contracts.models import Edge, Mode, Node, RunStatus

# Taint follows provenance only. Evidence edges (supports/refutes) and `revises` do not carry it.
LINEAGE_EDGES = frozenset({"derives_from"})


@dataclass
class ResearchGraph:
    nodes: dict[str, Node] = field(default_factory=dict)
    edges: dict[str, Edge] = field(default_factory=dict)
    interventions: list[InterventionRecorded] = field(default_factory=list)
    question: str | None = None
    mode: Mode | None = None
    status: RunStatus | None = None
    budget: BudgetUpdated | None = None
    error: str | None = None

    def apply(self, event: Event) -> None:
        match event.payload:
            case NodeCreated(node=node) | NodeUpdated(node=node):
                self.nodes[node.id] = node.model_copy(deep=True)
            case EdgeCreated(edge=edge):
                self.edges[edge.id] = edge.model_copy()
            case CheckResult(check=check):
                node = self.nodes[check.node_id]
                checks = [c for c in node.checks if c.id != check.id]
                self.nodes[node.id] = node.model_copy(update={"checks": [*checks, check]})
            case VerdictComputed(hypothesis_id=node_id, verdict=verdict) if node_id in self.nodes:
                self.nodes[node_id] = self.nodes[node_id].model_copy(update={"verdict": verdict})
            case InterventionRecorded() as intervention:
                self.interventions.append(intervention)
            case RunCreated(question=question, mode=mode):
                self.question, self.mode = question, mode
            case ModeChanged(mode=mode):
                self.mode = mode
            case StatusChanged(status=status):
                self.status = status
            case BudgetUpdated() as budget:
                self.budget = budget
            case ErrorOccurred(message=message):
                self.error = message

    def parents(self, node_id: str) -> list[str]:
        """The nodes `node_id` was derived from, in the order they were linked."""
        return [
            e.target for e in self.edges.values() if e.type in LINEAGE_EDGES and e.source == node_id
        ]

    def descendants(self, node_id: str) -> list[str]:
        """Every node derived from `node_id`, breadth first."""
        children: dict[str, list[str]] = {}
        for edge in self.edges.values():
            if edge.type in LINEAGE_EDGES:
                children.setdefault(edge.target, []).append(edge.source)
        return _walk(node_id, children)

    def ancestors(self, node_id: str) -> list[str]:
        """The provenance path of `node_id`, nearest first (used for report paths)."""
        return _walk(node_id, {nid: self.parents(nid) for nid in self.nodes})

    def depth(self) -> int:
        """The longest `derives_from` chain in the graph (the question is depth 0)."""
        memo: dict[str, int] = {}

        def depth_of(node_id: str) -> int:
            if node_id not in memo:
                memo[node_id] = 0  # cycle guard; derives_from is acyclic by construction
                memo[node_id] = 1 + max(map(depth_of, self.parents(node_id)), default=-1)
            return memo[node_id]

        return max(map(depth_of, self.nodes), default=0)


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
