"""Home arena by franchise and season, used for travel, time zone and altitude.

Coordinates and altitudes are approximate (about 1 km and 10 m), which is fine for travel and
altitude features. `utc_offset` is the standard-time offset, so `tz_shift` measures time zones
crossed and ignores daylight saving.

Known gap: one-off neutral regular-season games (Mexico City, London, Paris) aren't listed, so
they are placed at the designated home team's arena.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Arena:
    franchise: str
    first_start: int  # first season, as its starting calendar year
    last_start: int  # last season, inclusive
    name: str
    lat: float
    lon: float
    utc_offset: int
    altitude_m: int


_OPEN_ENDED = 9999

ARENAS: tuple[Arena, ...] = (
    Arena("ATL", 2007, _OPEN_ENDED, "State Farm Arena", 33.757, -84.396, -5, 320),
    Arena("BKN", 2007, 2009, "Izod Center", 40.812, -74.067, -5, 5),
    Arena("BKN", 2010, 2011, "Prudential Center", 40.734, -74.171, -5, 10),
    Arena("BKN", 2012, _OPEN_ENDED, "Barclays Center", 40.683, -73.975, -5, 10),
    Arena("BOS", 2007, _OPEN_ENDED, "TD Garden", 42.366, -71.062, -5, 5),
    Arena("CHA", 2007, _OPEN_ENDED, "Spectrum Center", 35.225, -80.839, -5, 230),
    Arena("CHI", 2007, _OPEN_ENDED, "United Center", 41.881, -87.674, -6, 180),
    Arena("CLE", 2007, _OPEN_ENDED, "Rocket Arena", 41.496, -81.688, -5, 200),
    Arena("DAL", 2007, _OPEN_ENDED, "American Airlines Center", 32.790, -96.810, -6, 130),
    Arena("DEN", 2007, _OPEN_ENDED, "Ball Arena", 39.749, -105.008, -7, 1600),
    Arena("DET", 2007, 2016, "The Palace of Auburn Hills", 42.697, -83.245, -5, 290),
    Arena("DET", 2017, _OPEN_ENDED, "Little Caesars Arena", 42.341, -83.055, -5, 190),
    Arena("GSW", 2007, 2018, "Oracle Arena", 37.750, -122.203, -8, 5),
    Arena("GSW", 2019, _OPEN_ENDED, "Chase Center", 37.768, -122.388, -8, 5),
    Arena("HOU", 2007, _OPEN_ENDED, "Toyota Center", 29.751, -95.362, -6, 15),
    Arena("IND", 2007, _OPEN_ENDED, "Gainbridge Fieldhouse", 39.764, -86.155, -5, 220),
    Arena("LAC", 2007, 2023, "Crypto.com Arena", 34.043, -118.267, -8, 90),
    Arena("LAC", 2024, _OPEN_ENDED, "Intuit Dome", 33.945, -118.342, -8, 30),
    Arena("LAL", 2007, _OPEN_ENDED, "Crypto.com Arena", 34.043, -118.267, -8, 90),
    Arena("MEM", 2007, _OPEN_ENDED, "FedExForum", 35.138, -90.051, -6, 80),
    Arena("MIA", 2007, _OPEN_ENDED, "Kaseya Center", 25.781, -80.188, -5, 2),
    Arena("MIL", 2007, _OPEN_ENDED, "Fiserv Forum", 43.045, -87.918, -6, 190),
    Arena("MIN", 2007, _OPEN_ENDED, "Target Center", 44.979, -93.276, -6, 255),
    Arena("NOP", 2007, _OPEN_ENDED, "Smoothie King Center", 29.949, -90.082, -6, 1),
    Arena("NYK", 2007, _OPEN_ENDED, "Madison Square Garden", 40.751, -73.994, -5, 10),
    Arena("OKC", 2007, 2007, "KeyArena (Seattle)", 47.622, -122.354, -8, 40),
    Arena("OKC", 2008, _OPEN_ENDED, "Paycom Center", 35.463, -97.515, -6, 365),
    Arena("ORL", 2007, _OPEN_ENDED, "Kia Center", 28.539, -81.384, -5, 30),
    Arena("PHI", 2007, _OPEN_ENDED, "Wells Fargo Center", 39.901, -75.172, -5, 5),
    Arena("PHX", 2007, _OPEN_ENDED, "Footprint Center", 33.446, -112.071, -7, 330),
    Arena("POR", 2007, _OPEN_ENDED, "Moda Center", 45.532, -122.667, -8, 15),
    Arena("SAC", 2007, 2015, "Sleep Train Arena", 38.649, -121.518, -8, 8),
    Arena("SAC", 2016, _OPEN_ENDED, "Golden 1 Center", 38.580, -121.500, -8, 8),
    Arena("SAS", 2007, _OPEN_ENDED, "Frost Bank Center", 29.427, -98.438, -6, 200),
    Arena("TOR", 2007, 2019, "Scotiabank Arena", 43.643, -79.379, -5, 80),
    Arena("TOR", 2020, 2020, "Amalie Arena (Tampa)", 27.943, -82.452, -5, 5),
    Arena("TOR", 2021, _OPEN_ENDED, "Scotiabank Arena", 43.643, -79.379, -5, 80),
    Arena("UTA", 2007, _OPEN_ENDED, "Delta Center", 40.768, -111.901, -7, 1290),
    Arena("WAS", 2007, _OPEN_ENDED, "Capital One Arena", 38.898, -77.021, -5, 10),
)

BUBBLE_ARENA = Arena("BUBBLE", 2019, 2019, "Walt Disney World", 28.337, -81.556, -5, 30)


def home_arena(franchise: str, start_year: int) -> Arena:
    """Return the arena a franchise played its home games in for a season."""
    for arena in ARENAS:
        if arena.franchise == franchise and arena.first_start <= start_year <= arena.last_start:
            return arena
    raise KeyError(f"no arena for {franchise} in season starting {start_year}")
