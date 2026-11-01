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
    # La Liga: football-data.co.uk's short names are too terse for the site ("Ath Madrid",
    # "Sociedad"), so La Liga clubs go by their usual English names, without accents.
    **{alias: name for name, aliases in {
        "Alaves": ["Deportivo Alavés", "Deportivo Alaves", "Alavés"],
        "Almeria": ["UD Almería", "UD Almeria", "Almería"],
        "Athletic Club": ["Ath Bilbao", "Athletic Bilbao"],
        "Atletico Madrid": ["Ath Madrid", "Club Atlético de Madrid", "Atlético Madrid", "Atletico de Madrid"],
        "Barcelona": ["FC Barcelona"],
        "Cadiz": ["Cádiz CF", "Cadiz CF", "Cádiz"],
        "Celta Vigo": ["Celta", "RC Celta de Vigo", "RC Celta"],
        "Deportivo La Coruna": ["La Coruna", "RC Deportivo La Coruña", "Deportivo La Coruña", "Deportivo"],
        "Eibar": ["SD Eibar"],
        "Elche": ["Elche CF"],
        "Espanyol": ["Espanol", "RCD Espanyol de Barcelona", "RCD Espanyol"],
        "Getafe": ["Getafe CF"],
        "Girona": ["Girona FC"],
        "Granada": ["Granada CF"],
        "Las Palmas": ["UD Las Palmas"],
        "Leganes": ["CD Leganés", "CD Leganes", "Leganés"],
        "Levante": ["Levante UD"],
        "Malaga": ["Málaga CF", "Malaga CF", "Málaga"],
        "Mallorca": ["RCD Mallorca"],
        "Osasuna": ["CA Osasuna"],
        "Racing Santander": ["Santander", "Real Racing Club de Santander", "Racing", "Real Racing Club"],
        "Rayo Vallecano": ["Vallecano", "Rayo Vallecano de Madrid"],
        "Real Betis": ["Betis", "Real Betis Balompié"],
        "Real Madrid": ["Real Madrid CF"],
        "Real Oviedo": ["Oviedo"],
        "Real Sociedad": ["Sociedad", "Real Sociedad de Fútbol"],
        "Sevilla": ["Sevilla FC"],
        "Sporting Gijon": ["Sp Gijon", "Real Sporting de Gijón", "Sporting Gijón"],
        "Valencia": ["Valencia CF"],
        "Valladolid": ["Real Valladolid", "Real Valladolid CF"],
        "Villarreal": ["Villarreal CF"],
        "Zaragoza": ["Real Zaragoza"],
    }.items() for alias in aliases},
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
