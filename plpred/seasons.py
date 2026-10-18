"""Season helpers. A season is named by its starting year: 2026 means 2026/27."""
from datetime import date


def current_season(today: date | None = None) -> int:
    today = today or date.today()
    # The Premier League starts in August; June and July belong to the next season's build-up.
    return today.year if today.month >= 7 else today.year - 1


# We only use recent form: the current season plus the two before it.
SEASONS_KEPT = 3


def window(today: date | None = None) -> list[int]:
    """Seasons we collect, oldest first, ending with the current one."""
    current = current_season(today)
    return list(range(current - SEASONS_KEPT + 1, current + 1))


def label(season: int) -> str:
    return f"{season}/{str(season + 1)[-2:]}"


def fd_code(season: int) -> str:
    """football-data.co.uk folder code, e.g. 2026 -> '2627'."""
    return f"{season % 100:02d}{(season + 1) % 100:02d}"
