"""Run the agent loop with the fake LLM and stub tools: `python -m backend.agent --help`."""

import argparse
import asyncio
from pathlib import Path

from backend.agent.graph import AgentRun
from backend.agent.stubs import B2B_QUESTION, StubGate, StubTools, b2b_script
from backend.llm.fake import FakeLLM
from contracts.events import Event


def _show(event: Event) -> None:
    d = event.data
    match event.kind:
        case "node_created":
            line = f"+ {d['id']:<18} {d['title']}"
        case "node_updated":
            line = f"~ {d['id']:<18} {d.get('state')}"
        case "check_result":
            gate = f" {d['gate_id']}" if d.get("gate_id") else ""
            line = f"  {d['node_id']:<18} {d['checker']}{gate}: {d['status']}"
        case "run_status":
            budget = d.get("budget") or {}
            line = (
                f"# {d['status']}  llm {budget.get('llm_calls', 0)}, tests {budget.get('tests', 0)}"
            )
        case "intervention":
            line = f"! {d['type']} {d.get('node_id') or ''}"
        case _:
            return
    print(f"{event.seq:>4} {line}")


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--question", default=B2B_QUESTION)
    parser.add_argument("--mode", choices=["live", "approval"], default="live")
    parser.add_argument("--yes", action="store_true", help="auto-approve in approval mode")
    parser.add_argument("--max-llm-calls", type=int, default=60)
    parser.add_argument("--out", type=Path, help="write the event log as JSONL")
    args = parser.parse_args()

    run = AgentRun(
        args.question,
        llm=FakeLLM(b2b_script),
        tools=StubTools(),
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
