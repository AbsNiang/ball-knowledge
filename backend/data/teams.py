"""Canonical team codes.

A team is keyed by franchise and uses its current code, so Seattle 2007-08 is OKC and the New
Jersey Nets are BKN. NBA team ids stay the same through a relocation, which makes them the
reliable key for results data. The renames this absorbs are Seattle->OKC (2008), New Jersey->
Brooklyn (2012), New Orleans Hornets->Pelicans (2013) and Charlotte Bobcats->Hornets (2014).
"""

import re

FRANCHISE_BY_TEAM_ID: dict[int, str] = {
    1610612737: "ATL",
    1610612738: "BOS",
    1610612739: "CLE",
    1610612740: "NOP",
    1610612741: "CHI",
    1610612742: "DAL",
    1610612743: "DEN",
    1610612744: "GSW",
    1610612745: "HOU",
    1610612746: "LAC",
    1610612747: "LAL",
    1610612748: "MIA",
    1610612749: "MIL",
    1610612750: "MIN",
    1610612751: "BKN",
    1610612752: "NYK",
    1610612753: "ORL",
    1610612754: "IND",
    1610612755: "PHI",
    1610612756: "PHX",
    1610612757: "POR",
    1610612758: "SAC",
    1610612759: "SAS",
    1610612760: "OKC",
    1610612761: "TOR",
    1610612762: "UTA",
    1610612763: "MEM",
    1610612764: "WAS",
    1610612765: "DET",
    1610612766: "CHA",
}

_ODDS_ALIASES: dict[str, str] = {
    "atlanta": "ATL",
    "boston": "BOS",
    "brooklyn": "BKN",
    "newjersey": "BKN",
    "charlotte": "CHA",
    "chicago": "CHI",
    "cleveland": "CLE",
    "dallas": "DAL",
    "denver": "DEN",
    "detroit": "DET",
    "goldenstate": "GSW",
    "houston": "HOU",
    "indiana": "IND",
    "laclippers": "LAC",
    "losangelesclippers": "LAC",
    "lalakers": "LAL",
    "losangeleslakers": "LAL",
    "memphis": "MEM",
    "miami": "MIA",
    "milwaukee": "MIL",
    "minnesota": "MIN",
    "neworleans": "NOP",
    "newyork": "NYK",
    "oklahomacity": "OKC",
    "oklacity": "OKC",
    "seattle": "OKC",
    "orlando": "ORL",
    "philadelphia": "PHI",
    "phoenix": "PHX",
    "portland": "POR",
    "sacramento": "SAC",
    "sanantonio": "SAS",
    "toronto": "TOR",
    "utah": "UTA",
    "washington": "WAS",
}


def franchise_from_team_id(team_id: int) -> str | None:
    """Return the franchise code for an NBA team id, or None for a non-NBA team."""
    return FRANCHISE_BY_TEAM_ID.get(int(team_id))


def franchise_from_odds_name(name: object) -> str | None:
    """Return the franchise code for an odds-file team name, or None if it isn't recognised."""
    key = re.sub(r"[^a-z]", "", str(name).lower())
    return _ODDS_ALIASES.get(key)
