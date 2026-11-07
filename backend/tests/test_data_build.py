"""End-to-end build on small fake raw files, plus the sealed-holdout guarantees."""

from pathlib import Path

import pandas as pd
import pytest

from backend.data.build import build, main
from backend.data.results import load_lauga_games, load_nba_api_gamelog
from backend.data.store import get_frame, get_games, read_sealed_holdout

BOS, MIA, OKC = 1610612738, 1610612748, 1610612760
REPO_BACKEND = Path(__file__).parent.parent


def lauga_row(game_id, day, home, away, home_pts=110, away_pts=100):
    return {
        "GAME_ID": game_id,
        "GAME_DATE_EST": day,
        "HOME_TEAM_ID": home,
        "VISITOR_TEAM_ID": away,
        "SEASON": 0,
        "PTS_home": home_pts,
        "PTS_away": away_pts,
    }


def write_odds(path: Path, *games: tuple) -> None:
    """Write an archive-shaped file from (mmdd, home, away, home_pts, away_pts) tuples."""
    rows = []
    for rot, (mmdd, home, away, home_pts, away_pts) in enumerate(games):
        rows.append([mmdd, 2 * rot + 1, "V", away, away_pts, 210, 170])
        rows.append([mmdd, 2 * rot + 2, "H", home, home_pts, 5, -200])
    frame = pd.DataFrame(rows, columns=["Date", "Rot", "VH", "Team", "Final", "Close", "ML"])
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".xlsx":
        frame.to_excel(path, index=False)
    else:
        frame.to_csv(path, index=False)


@pytest.fixture
def raw_dir(tmp_path: Path) -> Path:
    raw = tmp_path / "raw"
    (raw / "results").mkdir(parents=True)
    pd.DataFrame(
        [
            lauga_row(21000001, "2010-11-01", BOS, MIA),
            lauga_row(21000002, "2010-11-03", MIA, BOS),
            lauga_row(22100001, "2021-11-01", BOS, OKC),
        ]
    ).to_csv(raw / "results" / "games.csv", index=False)
    write_odds(
        raw / "odds" / "nba odds 2010-11.csv",
        (1101, "Boston", "Miami", 110, 100),
        (1103, "Miami", "Boston", 110, 100),
    )
    write_odds(raw / "odds" / "nba odds 2021-22.xlsx", (1101, "Boston", "OklahomaCity", 110, 100))
    return raw


def test_build_writes_tables_and_every_report(raw_dir, tmp_path):
    processed = tmp_path / "processed"
    report = build(raw_dir, processed)
    assert report.join_rate == 1.0
    assert report.team_games_rows == 6
    names = {path.name for path in (processed / "reports").iterdir()}
    assert names == {
        "join_rate_by_season.csv",
        "unmatched_games.csv",
        "unused_odds.csv",
        "score_mismatches.csv",
        "odds_issues.csv",
    }


def test_public_frames_hold_no_holdout_rows(raw_dir, tmp_path):
    processed = tmp_path / "processed"
    build(raw_dir, processed)
    public = pd.read_parquet(processed / "team_games.parquet")
    assert set(public["season"]) == {"2010-11"}
    assert set(pd.read_parquet(processed / "games.parquet")["season"]) == {"2010-11"}
    assert len(get_frame("explore", processed)) == 4
    assert len(get_games("explore", processed)) == 2


def test_sealed_store_holds_the_holdout_rows(raw_dir, tmp_path):
    processed = tmp_path / "processed"
    build(raw_dir, processed)
    assert set(read_sealed_holdout(processed)["season"]) == {"2021-22"}


@pytest.mark.parametrize("split", ["holdout", "HOLDOUT", "all", ""])
def test_get_frame_refuses_anything_but_public_splits(raw_dir, tmp_path, split):
    processed = tmp_path / "processed"
    build(raw_dir, processed)
    with pytest.raises(ValueError):
        get_frame(split, processed)
    with pytest.raises(ValueError):
        get_games(split, processed)


def test_no_module_other_than_store_reaches_the_sealed_rows():
    allowed = {REPO_BACKEND / "data" / "store.py"}  # add holdout_confirm's module when it exists
    for path in REPO_BACKEND.rglob("*.py"):
        if "tests" in path.parts or path in allowed:
            continue
        source = path.read_text(encoding="utf-8")
        assert "read_sealed_holdout" not in source, path
        assert "SEALED_DIR" not in source, path


def test_main_fails_below_the_join_threshold(raw_dir, tmp_path):
    write_odds(raw_dir / "odds" / "nba odds 2010-11.csv", (1101, "Boston", "Miami", 110, 100))
    args = ["--raw-dir", str(raw_dir), "--processed-dir", str(tmp_path / "processed")]
    assert main(args) == 1
    assert main([*args, "--allow-low-join"]) == 0


def test_lauga_loader_drops_preseason_old_seasons_and_duplicates(tmp_path):
    path = tmp_path / "games.csv"
    pd.DataFrame(
        [
            lauga_row(21000001, "2010-11-01", BOS, MIA),
            lauga_row(21000001, "2010-11-01", BOS, MIA),
            lauga_row(11000001, "2010-10-01", BOS, MIA),
            lauga_row(20600001, "2006-11-01", BOS, MIA),
            lauga_row(41000101, "2011-04-20", BOS, MIA),
            lauga_row(21000002, "2010-11-02", BOS, MIA, home_pts=float("nan")),
        ]
    ).to_csv(path, index=False)
    games = load_lauga_games(path)
    assert list(games["game_id"]) == ["0021000001", "0041000101"]
    assert list(games["season"]) == ["2010-11", "2010-11"]
    assert list(games["is_playoffs"]) == [False, True]


def test_cancelled_game_listed_zero_zero_is_dropped(tmp_path):
    path = tmp_path / "games.csv"
    pd.DataFrame(
        [
            lauga_row(21201214, "2013-04-16", BOS, MIA, home_pts=0, away_pts=0),
            lauga_row(21201215, "2013-04-16", MIA, BOS),
        ]
    ).to_csv(path, index=False)
    assert list(load_lauga_games(path)["game_id"]) == ["0021201215"]


def test_lauga_loader_keeps_the_seattle_franchise_under_okc(tmp_path):
    path = tmp_path / "games.csv"
    pd.DataFrame([lauga_row(20700001, "2007-11-01", OKC, BOS)]).to_csv(path, index=False)
    assert load_lauga_games(path).iloc[0]["home"] == "OKC"


def test_nba_api_loader_rebuilds_home_and_away_from_two_rows(tmp_path):
    path = tmp_path / "gamelog.csv"
    pd.DataFrame(
        [
            {
                "GAME_ID": "0021000001",
                "GAME_DATE": "2010-11-01",
                "TEAM_ID": BOS,
                "MATCHUP": "BOS vs. MIA",
                "PTS": 110,
            },
            {
                "GAME_ID": "0021000001",
                "GAME_DATE": "2010-11-01",
                "TEAM_ID": MIA,
                "MATCHUP": "MIA @ BOS",
                "PTS": 100,
            },
        ]
    ).to_csv(path, index=False)
    row = load_nba_api_gamelog(path).iloc[0]
    assert (row["home"], row["away"], row["home_pts"], row["away_pts"]) == ("BOS", "MIA", 110, 100)
    assert row["season"] == "2010-11"


def test_bubble_dates_are_flagged_from_results(tmp_path):
    path = tmp_path / "games.csv"
    pd.DataFrame(
        [
            lauga_row(21900001, "2020-03-01", BOS, MIA),
            lauga_row(21901260, "2020-08-05", BOS, MIA),
        ]
    ).to_csv(path, index=False)
    assert list(load_lauga_games(path)["is_bubble"]) == [False, True]
