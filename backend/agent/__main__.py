"""Run the agent loop with the fake LLM and stub tools: `python -m backend.agent --help`."""

import argparse
import asyncio
from pathlib import Path

from backend.agent.graph import AgentRun
from backend.agent.ports import RegistryTools
from backend.agent.stubs import B2B_QUESTION, StubGate, StubTools, b2b_script
from backend.llm.fake import FakeLLM
from contracts.events import Event
from contracts.models import Node


def _state(node: Node) -> str:
    flags = "".join(f" +{flag}" for flag in sorted(node.flags))
    return f"{node.lifecycle}/{node.check_outcome}{flags}"


def _show(event: Event) -> None:
    p = event.payload
    match p.type:
        case "node_created":
            line = f"+ {p.node.id:<18} {p.node.title}"
        case "node_updated":
            line = f"~ {p.node.id:<18} {_state(p.node)}"
        case "check_result":
            gate = f" {p.check.gate_id}" if p.check.gate_id else ""
            line = f"  {p.check.node_id:<18} {p.check.checker}{gate}: {p.check.status}"
        case "verdict_computed":
            line = f"= {p.hypothesis_id:<18} {p.verdict.label}"
        case "run_status":
            line = f"# {p.status}"
        case "budget_update":
            line = f"# llm {p.llm_calls_used}, tests {p.tests_spent}, depth {p.depth}"
        case "mode_changed":
            line = f"! mode {p.mode}"
        case "intervention":
            line = f"! {p.request.type} {p.request.node_id or ''}"
        case "error":
            line = f"x {p.code}: {p.message}"
        case _:
            return
    print(f"{event.seq:>4} {line}")


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--question", default=B2B_QUESTION)
    parser.add_argument("--mode", choices=["live", "approval"], default="live")
    parser.add_argument("--yes", action="store_true", help="auto-approve in approval mode")
    parser.add_argument("--max-llm-calls", type=int, default=60)
    parser.add_argument(
        "--tools",
        choices=["stub", "mock"],
        default="stub",
        help="stub: canned B2B numbers; mock: Person A's mock tools via the registry",
    )
    parser.add_argument("--out", type=Path, help="write the event log as JSONL")
    args = parser.parse_args()

    run = AgentRun(
        args.question,
        llm=FakeLLM(b2b_script),
        tools=StubTools() if args.tools == "stub" else RegistryTools(),
        gate=StubGate(),
        mode=args.mode,
        budget={"max_llm_calls": args.max_llm_calls},
        run_id="fake-b2b",
    )
    run.log.subscribe(_show)
    review = await run.start()
    while review is not None:
        if args.yes:
            review = await run.resume("approve")
            continue
        answer = await asyncio.to_thread(input, f"approve {review['node_id']}? [Y/n] ")
        if answer.strip().lower().startswith("n"):
            note = await asyncio.to_thread(input, "why (10+ characters): ")
            review = await run.resume("reject", note)
        else:
            review = await run.resume("approve")

    report = run.state["report"]
    print(f"\n{report['answer']}\n")
    for row in report["verdicts"]:
        print(f"  {row['verdict']:<28} {row['statement']}")
    if args.out:
        args.out.write_text(run.log.to_jsonl(), encoding="utf-8")
        print(f"\nwrote {len(run.log.events())} events to {args.out}")


if __name__ == "__main__":
    asyncio.run(main())
