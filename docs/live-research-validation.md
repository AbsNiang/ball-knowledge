# Live research validation

Normal API research requires Gemini, the registry and real tools. Scripted demo is opt-in.
Historical coverage is 2007–08 through 2021–22; 2021–22 is sealed. Web search cannot extend it.
The manifest records exact checksums, source provenance, coverage and per-season join rates.
Readiness hashes sealed bytes without reading their outcomes. Only holdout_confirm reads them.

## Independent public game checks

Checked 2026-11-08 against the NBA game pages and a contemporary Sports Insights odds report.
NBA pages: https://www.nba.com/game/0041500401 through https://www.nba.com/game/0041500407.
Odds source: https://www.sportsinsights.com/blog/2016-nba-finals-game-7-preview-cavs-at-warriors/

| Game | Home score | Away score | Archive home spread | ATS margin | Cover |
| --- | --- | --- | --- | --- | --- |
| 0041500401 | GSW 104 | CLE 89 | -6 | 9 | 1 |
| 0041500402 | GSW 110 | CLE 77 | -6 | 27 | 1 |
| 0041500404 | CLE 97 | GSW 108 | -2.5 | -13.5 | 0 |
| 0041500406 | CLE 115 | GSW 101 | -2 | 12 | 1 |
| 0041500407 | GSW 89 | CLE 93 | -4.5 | -8.5 | 0 |

The first four lines match the independent closing-line table. Game 7's -4.5 is the
dataset archive line; the contemporary report describes movement between -5 and -4.5,
rather than claiming one universal closing line. The NBA independently confirms 89–93.
Games 3 and 5 were excluded because the archive and that book's closing lines differ.
Push handling is covered separately by deterministic table tests.

## Hosted acceptance result

Run `5dc71a50-c9f4-42ed-b484-25e23d590739` completed 2026-11-08 on commit `30084d1`
(branch `feat/live-research`) with `gemini-3.5-flash-lite` and dataset `c20e8402ba6f6c85`.
The 168 events are sequence-contiguous; it used 28 of 60 model calls and 7 of 12 tests.
Statistical provenance was real (`mock=false`) for `compare_groups`, `define_slice`,
`out_of_sample`, `market_test` and `regress`; the report returned HTTP 200 and pause/resume
plus stream reconnect passed. Web search returned HTTP 429, so the run has no grounded
sources; statistical conclusions do not depend on them, and web is context-only.
Full artifact: `docs/live-run-result.json`.

## Rollout

Previous Railway deployment: 32cf6bd6-dbb1-4ba7-9576-eb6c4d8b2694, branch deploy/railway-demo.
Preserve Gemini key/model/RPM, one replica, one worker, 1 CPU and 1 GB memory.
Set LLM_MODE=gemini, WHACK_AGENT_TOOLS=registry, WHACK_TOOLS=real.
If hosted verification fails, reconnect deploy/railway-demo and restore LLM_MODE=fake;
remove/restore tool-selection variables as appropriate. Main remains unmerged.
Session storage is in memory and disappears on restart; durable sessions are separate work.
