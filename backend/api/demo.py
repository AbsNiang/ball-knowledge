"""Scripted UI demonstration; all evidence is synthetic, never statistical analysis."""

import asyncio
from datetime import UTC, datetime
from uuid import uuid4

from contracts.events import EdgeCreated, NodeCreated, NodeUpdated, StatusChanged
from contracts.models import CheckRecord, Edge, LLMRecord, Node, ToolRecord


class DemoAgent:
    def __init__(self):
        self.runs = {}
        self.tasks = set()

    async def start(self, run_id, request, events):
        state = {
            "paused": False,
            "active": None,
            "nodes": {},
            "queue": [
                ("premise", request.question[:80], None, "Establish premise"),
                (
                    "fatigue",
                    "Does accumulated fatigue explain the effect?",
                    "premise",
                    "Test physical fatigue",
                ),
                (
                    "chance",
                    "Could the pattern be random variation?",
                    "premise",
                    "Challenge with a null model",
                ),
                (
                    "travel",
                    "Separate travel from rest disadvantage",
                    "premise",
                    "Control for travel",
                ),
                (
                    "replicate",
                    "Check whether the effect persists across seasons",
                    "fatigue",
                    "Validate out of sample",
                ),
            ],
        }
        self.runs[run_id] = state
        await events.emit("system", StatusChanged(status="running"))
        task = asyncio.create_task(self.work(run_id, events))
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)

    async def work(self, run_id, events):
        state = self.runs[run_id]
        while state["queue"] or state["active"]:
            await asyncio.sleep(0.1)
            if state["paused"]:
                continue
            if state["active"] is None:
                node_id, title, parent, label = state["queue"].pop(0)
                node = Node(
                    id=node_id,
                    run_id=run_id,
                    type="hypothesis" if parent else "premise_check",
                    created_at=datetime.now(UTC),
                    title=title,
                    lifecycle="running",
                    llm=LLMRecord(
                        model_id="scripted-demo",
                        prompt="Synthetic demonstration",
                        raw_response="",
                        parsed_output={},
                        from_cache=False,
                        rationale="## Research approach\n\nThis is **synthetic demonstration evidence**. No NBA analysis or LLM call was performed.\n\nWe compare the hypothesis against competing explanations, then inspect uncertainty.\n\n| Sample | Effect |\n| --- | ---: |\n| Explore | -2.4 |\n| Validate | -1.8 |\n\nA failed branch remains available for inspection.",
                    ),
                )
                state["nodes"][node_id] = node
                state["active"] = node_id
                state["ticks"] = 0
                await events.emit("system", NodeCreated(node=node), node_id)
                if parent:
                    await events.emit(
                        "system",
                        EdgeCreated(
                            edge=Edge(
                                id=str(uuid4()),
                                source=node_id,
                                target=parent,
                                type="derives_from",
                                label=label,
                            )
                        ),
                    )
            else:
                state["ticks"] += 1
                if state["ticks"] < 45:
                    continue
                node = state["nodes"][state["active"]].model_copy(deep=True)
                failed = node.id == "chance"
                node.lifecycle = "complete"
                node.check_outcome = "failed" if failed else "passed"
                node.checks = [
                    CheckRecord(
                        id=f"{node.id}-gate",
                        node_id=node.id,
                        checker="gate",
                        gate_id="G2",
                        status="fail" if failed else "pass",
                        severity="low",
                        rationale="Synthetic gate result for interface demonstration.",
                    )
                ]
                node.tool = ToolRecord(
                    tool_name="demo_comparison",
                    args={},
                    args_hash="demo",
                    data_slice_hash="synthetic",
                    seed=0,
                    duration_ms=0,
                    output={
                        "charts": [
                            {
                                "type": "bar",
                                "title": "Synthetic effect by sample",
                                "xKey": "sample",
                                "yKey": "effect",
                                "data": [
                                    {"sample": "Explore", "effect": -2.4},
                                    {"sample": "Validate", "effect": -1.8},
                                    {"sample": "Control", "effect": -0.2},
                                ],
                            }
                        ]
                    },
                )
                state["nodes"][node.id] = node
                state["active"] = None
                await events.emit("system", NodeUpdated(node=node), node.id)
        await events.emit("system", StatusChanged(status="completed"))

    async def intervene(self, run_id, intervention_id, request, events):
        state = self.runs[run_id]
        if not state["active"] and not state["queue"]:
            raise ValueError("This session has finished.")
        if request.type == "pause":
            state["paused"] = True
            await events.emit("system", StatusChanged(status="paused"))
        elif request.type == "resume":
            state["paused"] = False
            await events.emit("system", StatusChanged(status="running"))
        elif request.type == "redirect":
            if request.node_id != state["active"]:
                raise ValueError("Redirect must target the active node.")
            node = state["nodes"][state["active"]].model_copy(deep=True)
            node.lifecycle = "complete"
            node.flags.add("pruned")
            await events.emit("system", NodeUpdated(node=node), node.id)
            state["nodes"][node.id] = node
            state["queue"].insert(
                0,
                (str(uuid4()), request.payload.statement[:80], node.id, request.payload.statement),
            )
            state["active"] = None
            state["paused"] = False
            await events.emit("system", StatusChanged(status="running"))
        else:
            raise ValueError("Demo supports pause, resume, and redirect.")
