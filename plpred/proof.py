"""Proof that each week's calls were made before kick-off, from records we can't edit later.

    python -m plpred.proof                          # each calls file with its SHA-256 fingerprint
    python -m plpred.proof archive --calls FILE...  # save these calls files' GitHub copies to the Wayback Machine
    python -m plpred.proof archive --site           # save the front page and the newest matchweek page

Git history alone isn't proof, because history can be rewritten. So every calls file gets a
fingerprint (its SHA-256 hash), which goes in the Thursday X thread and on the record page, and
the Update data and Publish site workflows ask the Internet Archive's Wayback Machine to save the
calls and the pages that show them. Both are dated by someone else. Anyone can check a file by
downloading it from GitHub and running `sha256sum` on it: changing a single character changes
the fingerprint completely.
"""
import argparse
import hashlib
import json
import logging
from pathlib import Path

import pandas as pd

from plpred import http, seasons, social

log = logging.getLogger("plpred.proof")
DATA = Path(__file__).resolve().parent.parent / "data"
REPO = "Garygunn94/premier-league-predictions"
RAW = f"https://raw.githubusercontent.com/{REPO}/main/"
SAVE = "https://web.archive.org/save/"


def fingerprint(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def calls_files(data: Path) -> list[Path]:
    """Every real calls file (not replays), oldest first."""
    return [f for f in sorted((data / "predictions").glob("*.json")) if not f.name.startswith("replay-")]


def proofs(data: Path) -> list[dict]:
    """One row per calls file, newest first: its fingerprint and where to check it."""
    rows = []
    for f in calls_files(data):
        preds = json.loads(f.read_text())
        rel = f"data/predictions/{f.name}"
        rows.append({
            "file": f.name, "sha256": fingerprint(f), "short": fingerprint(f)[:16],
            "games": len(preds["games"]), "after_kickoff": f.stem.endswith("-after-kickoff"),
            "first_kickoff": pd.Timestamp(min(g["kickoff"] for g in preds["games"])).strftime("%d %B %Y"),
            "github": f"https://github.com/{REPO}/commits/main/{rel}",
            "raw": f"{RAW}{rel}",
            "archive": f"https://web.archive.org/web/*/{RAW}{rel}",
        })
    return rows[::-1]


def archive(url: str) -> bool:
    """Ask the Wayback Machine to save `url` now. It can take a minute, and it sometimes refuses
    when busy, so a failure is logged rather than raised: the calls are still published."""
    try:
        r = http.session().get(SAVE + url, timeout=180)
        r.raise_for_status()
    except Exception as e:  # noqa: BLE001 - any failure here just means no snapshot this time
        log.warning("The Wayback Machine didn't save %s: %s", url, e)
        return False
    log.info("Saved %s to the Wayback Machine", url)
    return True


def site_urls(data: Path) -> list[str]:
    """The front page and the newest matchweek's page."""
    from plpred import site  # site imports this module, so import it only when needed

    preds, _ = site.latest_predictions(data)
    matches = pd.read_csv(data / "matches.csv", parse_dates=["date"])
    season = int(preds["season"][:4])
    week = site.matchweek_of(preds, matches, season)
    return [social.SITE_URL, f"{social.SITE_URL}{seasons.label(season).replace('/', '-')}/matchweek/{week}/"]


def main(argv=None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", nargs="?", choices=["list", "archive"], default="list")
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--calls", nargs="*", default=[], metavar="FILE",
                        help="calls files to save (data/predictions/...); replays are skipped")
    parser.add_argument("--site", action="store_true", help="save the front page and the newest matchweek page")
    args = parser.parse_args(argv)

    if args.command == "list":
        for p in proofs(args.data):
            print(f"{p['sha256']}  {p['file']}")
        return
    urls = [f"{RAW}data/predictions/{Path(f).name}" for f in args.calls
            if f.endswith(".json") and not Path(f).name.startswith("replay-")]
    if args.site:
        urls += site_urls(args.data)
    saved = sum(archive(u) for u in urls)
    log.info("Saved %d of %d addresses", saved, len(urls))


if __name__ == "__main__":
    main()
