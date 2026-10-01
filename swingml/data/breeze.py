"""Breeze Connect session token -- storage and validation.

The distinction that drives this module
---------------------------------------
Breeze has **two** credentials and only one of them rotates:

* the **App Key** and **Secret Key** are static, created once in the portal;
* the **session token** (``apisession``) is generated daily by logging in at
  ``https://api.icicidirect.com/apiuser/login?api_key=<app key>``, which
  redirects to the registered Redirect URL carrying ``?apisession=...``.

ICICI's own position is that this daily generation is **manual, per SEBI
guidelines**, and that the token is "valid for 24 hours or until midnight". So
this module never tries to log in: it stores a token a human produced, and it
**refuses to use a token from a previous day**, because after midnight the API
will reject it and the failure would otherwise look like a data outage.

Secrets policy
--------------
Only the session token is written to disk, and only under ``data/`` (gitignored).
The App Key / Secret Key are read from the environment and never persisted.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import os
import re
from pathlib import Path
from urllib.parse import unquote

logger = logging.getLogger(__name__)

#: IST -- UTC+5:30, no daylight saving, so a fixed offset is exact. The token's
#: "until midnight" is IST midnight.
IST = dt.timezone(dt.timedelta(hours=5, minutes=30))

#: Default token location. Under ``data/`` deliberately -- that tree is
#: gitignored, and a session token is a bearer credential.
DEFAULT_SESSION_PATH = "data/breeze_session.json"

#: The two static credentials. Read from the environment (or a gitignored
#: ``.env``); never written to disk by this package.
API_KEY_ENV = "BREEZE_API_KEY"
SECRET_KEY_ENV = "BREEZE_SECRET_KEY"

#: Deliberately loose: real apisession values have been observed both as long
#: alphanumeric strings and as short (~8 char) numeric ids. This only catches a
#: pasted password or an obvious fragment before it reaches the API -- the API
#: itself (scripts/breeze_token.py --check) is the real validator.
_TOKEN_SHAPE = re.compile(r"^[A-Za-z0-9_-]{8,256}$")

#: Matches ``apisession=<value>`` in a URL query, a fragment, or bare text.
_APISESSION = re.compile(r"(?:^|[?&#])apisession=([^&#\s]+)", re.IGNORECASE)


class BreezeSessionError(RuntimeError):
    """The stored session token is missing, stale or shapeless."""


def load_env_file(path: str | Path | None = None, *, override: bool = False) -> int:
    """Load ``KEY=VALUE`` lines from a gitignored ``.env`` into the environment.

    The static credentials must not be committed, and exporting them by hand
    every morning is exactly the step that gets forgotten at 06:25. This makes
    ``.env`` the one place they live, for the CLI as well as for the scripts.

    Existing environment variables win unless ``override`` is set, so a shell
    export always beats the file. Looks at ``$CWD/.env`` then the project root's
    ``.env``. Returns how many variables were set; a missing file is not an error.
    """
    from swingml.config import PROJECT_ROOT

    candidates = [Path(path)] if path else [Path.cwd() / ".env", PROJECT_ROOT / ".env"]
    n = 0
    for candidate in candidates:
        if not candidate.exists():
            continue
        for raw in candidate.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            if not key or " " in key:
                continue
            if key in os.environ and not override:
                continue
            os.environ[key] = value.strip().strip('"').strip("'")
            n += 1
    return n


def ist_today(now: dt.datetime | None = None) -> dt.date:
    """Today's date in IST (the clock the token's midnight expiry runs on)."""
    clock = now or dt.datetime.now(IST)
    if clock.tzinfo is None:
        clock = clock.replace(tzinfo=IST)
    return clock.astimezone(IST).date()


def login_url(api_key: str) -> str:
    """The URL a human opens to mint a session. App Key must be URL-encoded."""
    from urllib.parse import quote

    if not api_key or not api_key.strip():
        raise ValueError("an App Key is required to build the login URL")
    return f"https://api.icicidirect.com/apiuser/login?api_key={quote(api_key.strip())}"


def parse_session_token(text: str) -> str:
    """Extract the session token from a pasted URL, query string, or raw value.

    Accepts what the browser actually gives you (the full Redirect URL), what
    the network tab gives you (``apisession=...``), or the bare token. Raises
    rather than guessing, so a wrong paste fails here instead of as an opaque
    401 an hour later.
    """
    raw = (text or "").strip().strip('"').strip("'")
    if not raw:
        raise BreezeSessionError("empty input: paste the redirected URL or the apisession value")

    match = _APISESSION.search(raw)
    if match:
        token = unquote(match.group(1)).strip()
    elif raw.lower().startswith("http"):
        # A URL without apisession is a login page, not a redirect back.
        raise BreezeSessionError(
            "that URL carries no 'apisession' parameter -- paste the URL you were "
            "redirected TO after logging in, not the login page"
        )
    else:
        token = raw

    if not _TOKEN_SHAPE.match(token):
        raise BreezeSessionError(
            f"that does not look like a session token (got {len(token)} chars); "
            "paste the apisession value or the full redirected URL"
        )
    return token


def save_session_token(
    token: str,
    path: str | Path = DEFAULT_SESSION_PATH,
    api_key: str | None = None,
    now: dt.datetime | None = None,
) -> Path:
    """Persist a token stamped with the IST date it was minted.

    The date stamp is the whole point: it lets :func:`load_session_token` refuse
    yesterday's token instead of letting the API return an auth error that looks
    like a market-data failure.
    """
    if not _TOKEN_SHAPE.match(token or ""):
        raise BreezeSessionError("refusing to store a token that fails the shape check")
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "date": ist_today(now).isoformat(),
        "token": token,
        "api_key_suffix": (api_key or "")[-6:] or None,
        "saved_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "note": "session token; expires at IST midnight. gitignored -- never commit.",
    }
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("breeze session token saved -> %s (valid for %s)", out, payload["date"])
    return out


def load_session_token(
    path: str | Path = DEFAULT_SESSION_PATH,
    now: dt.datetime | None = None,
) -> str:
    """Return today's token, or raise with the exact reason it is unusable."""
    p = Path(path)
    if not p.exists():
        raise BreezeSessionError(
            f"no Breeze session token at {p}. Run: "
            f".venv/bin/python scripts/breeze_token.py"
        )
    try:
        payload = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BreezeSessionError(f"{p} is unreadable ({exc}); re-run scripts/breeze_token.py") from exc

    token = str(payload.get("token") or "")
    stamped = str(payload.get("date") or "")
    if not token:
        raise BreezeSessionError(f"{p} has no token; re-run scripts/breeze_token.py")

    today = ist_today(now).isoformat()
    if stamped != today:
        raise BreezeSessionError(
            f"the Breeze session token at {p} is dated {stamped or '(undated)'} but today "
            f"is {today}; it expires at IST midnight. Re-run scripts/breeze_token.py"
        )
    return token


def session_status(
    path: str | Path = DEFAULT_SESSION_PATH,
    now: dt.datetime | None = None,
) -> dict:
    """Non-raising summary for CLIs: ``{present, date, current, age_days}``."""
    p = Path(path)
    out = {"path": str(p), "present": False, "date": None, "current": False, "age_days": None}
    if not p.exists():
        return out
    out["present"] = True
    try:
        payload = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return out
    out["date"] = payload.get("date")
    if out["date"]:
        try:
            out["age_days"] = (ist_today(now) - dt.date.fromisoformat(str(out["date"]))).days
            out["current"] = out["age_days"] == 0
        except ValueError:
            pass
    return out


__all__ = [
    "API_KEY_ENV",
    "DEFAULT_SESSION_PATH",
    "SECRET_KEY_ENV",
    "BreezeSessionError",
    "ist_today",
    "load_env_file",
    "load_session_token",
    "login_url",
    "parse_session_token",
    "save_session_token",
    "session_status",
]
