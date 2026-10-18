"""One canonical name per club, so every source joins cleanly.

Canonical names follow football-data.co.uk, which holds the longest history.
"""
import logging

log = logging.getLogger(__name__)

ALIASES = {
    "Manchester United": "Man United",
    "Man Utd": "Man United",
    "Manchester City": "Man City",
    "Nottingham Forest": "Nott'm Forest",
    "Nottingham": "Nott'm Forest",
    "Wolverhampton Wanderers": "Wolves",
    "Wolverhampton": "Wolves",
    "Newcastle United": "Newcastle",
    "Tottenham Hotspur": "Tottenham",
    "Spurs": "Tottenham",
    "West Ham United": "West Ham",
    "Brighton & Hove Albion": "Brighton",
    "Brighton and Hove Albion": "Brighton",
    "Leicester City": "Leicester",
    "Leeds United": "Leeds",
    "Sheffield Utd": "Sheffield United",
    "Sheffield Weds": "Sheffield Weds",
    "Sheffield Wednesday": "Sheffield Weds",
    "West Bromwich Albion": "West Brom",
    "Ipswich Town": "Ipswich",
    "Luton Town": "Luton",
    "Norwich City": "Norwich",
    "Cardiff City": "Cardiff",
    "Swansea City": "Swansea",
    "Stoke City": "Stoke",
    "Hull City": "Hull",
    "Huddersfield Town": "Huddersfield",
    "AFC Bournemouth": "Bournemouth",
    "Queens Park Rangers": "QPR",
    "Wigan Athletic": "Wigan",
    "Birmingham City": "Birmingham",
    "Blackburn Rovers": "Blackburn",
    "Bolton Wanderers": "Bolton",
    "Coventry City": "Coventry",
    "Derby County": "Derby",
}


def canonical(name: str) -> str:
    name = name.strip()
    return ALIASES.get(name, name)


def check_known(names, known) -> list[str]:
    """Return names that don't match any known club, logging each once."""
    unknown = sorted(set(names) - set(known))
    for n in unknown:
        log.warning("Unrecognised team name %r: add it to plpred/teams.py ALIASES", n)
    return unknown
