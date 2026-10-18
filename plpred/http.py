"""Shared HTTP session with retries and a polite user agent."""
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

USER_AGENT = "premier-league-predictions (+https://github.com/Garygunn94/premier-league-predictions)"


def session() -> requests.Session:
    s = requests.Session()
    retry = Retry(total=4, backoff_factor=2, status_forcelist=(429, 500, 502, 503, 504))
    s.mount("https://", HTTPAdapter(max_retries=retry))
    s.headers["User-Agent"] = USER_AGENT
    return s
