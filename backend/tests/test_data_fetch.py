from pathlib import Path

import pytest

from backend.data.fetch import fetch_odds, odds_path, odds_url


def fake_download(content: bytes):
    calls = []

    def download(url: str, path: Path) -> None:
        calls.append(url)
        path.write_bytes(content)

    return download, calls


def test_odds_url_points_at_the_archived_season_file():
    assert odds_url(2007).endswith("/nba/nba%20odds%202007-08.xlsx")
    assert odds_url(2021).startswith("https://web.archive.org/web/2023if_/")


def test_fetch_saves_files_under_names_the_build_can_read(tmp_path):
    download, calls = fake_download(b"PK\x03\x04data")
    fetched = fetch_odds(range(2007, 2009), tmp_path, download)
    assert [path.name for path in fetched] == ["nba odds 2007-08.xlsx", "nba odds 2008-09.xlsx"]
    assert len(calls) == 2


def test_fetch_skips_files_already_present(tmp_path):
    odds_path(2007, tmp_path).write_bytes(b"PK existing")
    download, calls = fake_download(b"PK\x03\x04data")
    fetch_odds(range(2007, 2009), tmp_path, download)
    assert calls == [odds_url(2008)]
    assert odds_path(2007, tmp_path).read_bytes() == b"PK existing"


def test_fetch_rejects_and_removes_a_non_excel_response(tmp_path):
    download, _ = fake_download(b"<html>not found</html>")
    with pytest.raises(ValueError):
        fetch_odds(range(2007, 2008), tmp_path, download)
    assert not odds_path(2007, tmp_path).exists()
