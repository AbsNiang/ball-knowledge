"""One bounded hosted research run; no credentials or model API calls from this client."""
import asyncio
import json
import sys
import time
from pathlib import Path

import httpx
import websockets


async def main(base: str):
    async with httpx.AsyncClient(base_url=base, timeout=30) as client:
        runtime = (await client.get("/api/runtime")).json()
        assert runtime["ready"], runtime
        print(json.dumps({"runtime": runtime}), flush=True)
        response = await client.post("/api/runs", json={
            "question": "Do NBA teams on back-to-back games have lower raw scoring margins than rested teams in the historical dataset, and is the effect already priced into closing spreads? Check with the real tools and use web sources as context only.",
            "mode": "live", "budget": {"max_llm_calls": 30, "max_tests": 8},
        })
        assert response.status_code == 202, response.text
        run_id = response.json()["run_id"]
        print(json.dumps({"run_id": run_id}), flush=True)
        socket_url = base.replace("https://", "wss://").replace("http://", "ws://") + f"/api/runs/{run_id}/events"
        async with websockets.connect(socket_url, open_timeout=30) as socket:
            first = json.loads(await asyncio.wait_for(socket.recv(), 30))
            assert first["seq"] == 1
            for action in ("pause", "resume"):
                result = await client.post(f"/api/runs/{run_id}/interventions", json={"type": action})
                assert result.status_code == 202, result.text
                if action == "pause":
                    deadline = time.monotonic() + 90
                    while time.monotonic() < deadline:
                        events = (await client.get(f"/api/runs/{run_id}/events")).json()
                        statuses = [e["payload"]["status"] for e in events if e["type"] == "run_status"]
                        if statuses[-1] == "paused":
                            break
                        await asyncio.sleep(2)
                    assert statuses[-1] == "paused", statuses
        async with websockets.connect(socket_url + "?from_seq=2", open_timeout=30) as socket:
            resumed = json.loads(await asyncio.wait_for(socket.recv(), 30))
            assert resumed["seq"] == 2
        print(json.dumps({"controls": "pause/resume passed", "stream": "reconnected at seq 2"}), flush=True)
        deadline = time.monotonic() + 1200
        previous = None
        while time.monotonic() < deadline:
            events = (await client.get(f"/api/runs/{run_id}/events")).json()
            statuses = [e["payload"]["status"] for e in events if e["type"] == "run_status"]
            budget_events = [e["payload"] for e in events if e["type"] == "budget_update"]
            progress = {"events": len(events), "status": statuses[-1], "budget": budget_events[-1] if budget_events else None}
            if progress != previous:
                print(json.dumps(progress), flush=True)
                previous = progress
            if statuses[-1] in ("completed", "failed", "cancelled"):
                break
            await asyncio.sleep(10)
        assert statuses[-1] == "completed", json.dumps(events[-8:])
        assert [e["seq"] for e in events] == list(range(1,len(events)+1))
        report_response = await client.get(f"/api/runs/{run_id}/report")
        assert report_response.status_code == 200, report_response.text
        report = report_response.json()
        nodes = [e["payload"]["node"] for e in events if e["type"] == "node_created"]
        tool_records = [n["tool"] for n in nodes if n.get("tool")]
        real = [t for t in tool_records if t["output"].get("extra",{}).get("mock") is False]
        assert real, "No real statistical provenance"
        assert not any(t["output"].get("extra",{}).get("mock") is True for t in tool_records)
        sources = [t["output"].get("sources",[]) for t in tool_records if t["tool_name"] == "web_search"]
        summary = {"run_id":run_id,"dataset_version":runtime["dataset_version"],"model":runtime["model_id"],"status":statuses[-1],"events":len(events),"real_tool_results":len(real),"sources":sources,"report":report,"controls":"passed","reconnect":"passed"}
        destination = Path("docs/live-run-result.json")
        destination.write_text(json.dumps(summary,indent=2)+"\n", encoding="utf-8")
        print(json.dumps(summary), flush=True)

if __name__ == "__main__":
    asyncio.run(main(sys.argv[1].rstrip("/")))
