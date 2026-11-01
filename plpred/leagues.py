"""The leagues we call: the Premier League (the main site) and La Liga (its own section, /laliga/).

Each league keeps its data in its own folder: data/ for the Premier League and data/laliga/
for La Liga. `plpred.update --league la-liga` writes a league.txt marker into its folder, so
every other step (predict, backtest, health, site) knows which league a data folder holds
from the folder alone.
"""
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class League:
    key: str                # as typed on the command line
    name: str               # as written on the site
    prefix: str             # where its pages live on the site, "" for the front of the site
    fd_code: str            # football-data.co.uk division
    understat: str          # Understat league name
    tag: str                # its hashtag, for X and Instagram
    round_tags: str         # the five Instagram tags on the round post
    fpl: bool = False       # Fantasy Premier League: team news, the FPL corner, Superbru
    openfootball: str = ""  # openfootball/football.json file code, for matchday numbers
    # Clubs whose games always get a carousel and a Reel (see featured); None means every game.
    big_clubs: frozenset | None = None
    codes: dict = field(default_factory=dict)      # three-letter codes, for monograms and match tags
    club_tags: dict = field(default_factory=dict)  # each club's hashtag
    short: dict = field(default_factory=dict)      # shorter names for cards

    @property
    def tagline(self) -> str:
        return f"Data-driven {self.name} predictions, with the reasons. New every Thursday."

    def view(self) -> dict:
        """What the templates and the posting kit need to know about the league."""
        return {"key": self.key, "name": self.name, "prefix": self.prefix, "fpl": self.fpl, "tag": self.tag,
                "tagline": self.tagline, "codes": self.codes, "main": not self.prefix,
                "round_tags": self.round_tags, "hashtags": f"{self.tag} #FootballAnalytics #xG"}


PREMIER_LEAGUE = League(
    key="premier-league", name="Premier League", prefix="", fd_code="E0", understat="EPL",
    tag="#PremierLeague", round_tags="#PremierLeague #PLPredictions #FootballAnalytics #xG #EPL", fpl=True,
    codes={"Arsenal": "ARS", "Aston Villa": "AVL", "Bournemouth": "BOU", "Brentford": "BRE", "Brighton": "BHA",
           "Burnley": "BUR", "Chelsea": "CHE", "Coventry": "COV", "Crystal Palace": "CRY", "Everton": "EVE",
           "Fulham": "FUL", "Hull": "HUL", "Ipswich": "IPS", "Leeds": "LEE", "Leicester": "LEI",
           "Liverpool": "LIV", "Luton": "LUT", "Man City": "MCI", "Man United": "MUN", "Newcastle": "NEW",
           "Nott'm Forest": "NFO", "Sheffield United": "SHU", "Southampton": "SOU", "Sunderland": "SUN",
           "Tottenham": "TOT", "West Ham": "WHU", "Wolves": "WOL"},
    club_tags={"Arsenal": "#Arsenal", "Aston Villa": "#AVFC", "Bournemouth": "#AFCB", "Brentford": "#BrentfordFC",
               "Brighton": "#BHAFC", "Burnley": "#BurnleyFC", "Chelsea": "#ChelseaFC", "Coventry": "#PUSB",
               "Crystal Palace": "#CPFC", "Everton": "#EFC", "Fulham": "#FulhamFC", "Hull": "#HullCity",
               "Ipswich": "#ITFC", "Leeds": "#LUFC", "Leicester": "#LCFC", "Liverpool": "#LFC",
               "Man City": "#MCFC", "Man United": "#MUFC", "Newcastle": "#NUFC", "Nott'm Forest": "#NFFC",
               "Southampton": "#SaintsFC", "Sunderland": "#SAFC", "Tottenham": "#THFC", "West Ham": "#WHUFC",
               "Wolves": "#Wolves"},
    short={"Man United": "Man Utd", "Nott'm Forest": "Forest", "Crystal Palace": "Palace",
           "Bournemouth": "B'mouth", "Aston Villa": "Villa", "Tottenham": "Spurs", "Brighton": "Brighton"},
)

LA_LIGA = League(
    key="la-liga", name="La Liga", prefix="laliga/", fd_code="SP1", understat="La_liga",
    tag="#LaLiga", round_tags="#LaLiga #LaLigaPredictions #FootballAnalytics #xG #FutbolEspanol",
    openfootball="es.1",
    # Gary's pick (2026-11-01): Real Madrid, Barcelona and Atletico, plus Betis for their recent
    # following. Top-four clashes are featured too.
    big_clubs=frozenset({"Real Madrid", "Barcelona", "Atletico Madrid", "Real Betis"}),
    codes={"Alaves": "ALA", "Almeria": "ALM", "Athletic Club": "ATH", "Atletico Madrid": "ATM",
           "Barcelona": "BAR", "Cadiz": "CAD", "Celta Vigo": "CEL", "Deportivo La Coruna": "DEP",
           "Eibar": "EIB", "Elche": "ELC", "Espanyol": "ESP", "Getafe": "GET", "Girona": "GIR",
           "Granada": "GRA", "Las Palmas": "LPA", "Leganes": "LEG", "Levante": "LEV", "Malaga": "MAL",
           "Mallorca": "MLL", "Osasuna": "OSA", "Racing Santander": "RAC", "Rayo Vallecano": "RAY",
           "Real Betis": "BET", "Real Madrid": "RMA", "Real Oviedo": "OVI", "Real Sociedad": "RSO",
           "Sevilla": "SEV", "Sporting Gijon": "SPO", "Valencia": "VAL", "Valladolid": "VLL",
           "Villarreal": "VIL", "Zaragoza": "ZAR"},
    club_tags={"Real Madrid": "#RealMadrid", "Barcelona": "#FCBarcelona", "Atletico Madrid": "#Atleti",
               "Real Betis": "#RealBetis", "Athletic Club": "#AthleticClub", "Real Sociedad": "#RealSociedad",
               "Villarreal": "#Villarreal", "Sevilla": "#SevillaFC", "Valencia": "#ValenciaCF",
               "Celta Vigo": "#RCCelta", "Osasuna": "#Osasuna", "Rayo Vallecano": "#RayoVallecano",
               "Espanyol": "#RCDEspanyol", "Getafe": "#GetafeCF", "Alaves": "#Alaves", "Girona": "#GironaFC",
               "Mallorca": "#RCDMallorca", "Levante": "#LevanteUD", "Elche": "#ElcheCF", "Malaga": "#MalagaCF",
               "Deportivo La Coruna": "#Depor", "Racing Santander": "#RealRacingClub", "Real Oviedo": "#RealOviedo"},
    short={"Atletico Madrid": "Atlético", "Athletic Club": "Athletic", "Rayo Vallecano": "Rayo",
           "Deportivo La Coruna": "Deportivo", "Racing Santander": "Racing", "Real Sociedad": "Sociedad",
           "Celta Vigo": "Celta"},
)

LEAGUES = {PREMIER_LEAGUE.key: PREMIER_LEAGUE, LA_LIGA.key: LA_LIGA}
MARKER = "league.txt"
# Where each league's data lives, relative to the main data folder.
FOLDERS = {PREMIER_LEAGUE.key: "", LA_LIGA.key: "laliga"}


def get(key: str) -> League:
    return LEAGUES[key]


def of(data: Path) -> League:
    """The league a data folder holds: its league.txt, or the Premier League without one."""
    marker = Path(data) / MARKER
    if marker.exists():
        return LEAGUES[marker.read_text().strip()]
    return PREMIER_LEAGUE


def mark(data: Path, league: League) -> None:
    if league is PREMIER_LEAGUE:
        return
    Path(data).mkdir(parents=True, exist_ok=True)
    (Path(data) / MARKER).write_text(league.key + "\n")


def featured(league: League, home: str, away: str, top_four: set[str]) -> bool:
    """Whether a game gets the full treatment (carousel, Reel, verdict post): every game in the
    Premier League; in La Liga, the big clubs' games and clashes between two of the top four."""
    if league.big_clubs is None:
        return True
    return bool({home, away} & league.big_clubs) or {home, away} <= top_four
