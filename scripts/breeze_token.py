"""Mint today's Breeze session token -- manual, by design.

ICICI's own guidance is that the session key is generated **every day manually,
per SEBI guidelines**, and is valid for 24 hours or until midnight. This script
therefore does not log in and stores no password: it opens the login page, you
sign in, and you paste back the URL you were redirected to (or just the
``apisession`` value). It extracts the token, stamps it with today's IST date and
writes it to a gitignored file for the price provider to read.

Why the paste, not a local callback server
------------------------------------------
Breeze requires an **https** Redirect URL registered on the app. A local
HTTP callback cannot serve https without a self-signed certificate the browser
will object to, and the registered URL may not be localhost at all. Pasting the
redirect handles every case with zero infrastructure.

Usage
-----
    export BREEZE_API_KEY="your app key"        # static; the token rotates
    .venv/bin/python scripts/breeze_token.py
    .venv/bin/python scripts/breeze_token.py --url-only     # just print the link
    .venv/bin/python scripts/breeze_token.py --check        # verify against the API

The App Key and Secret Key are read from the environment and never written to
disk. Only the session token is persisted, under ``data/`` (gitignored).
"""

from __future__ import annotations

import argparse
import os
import sys
import webbrowser

from swingml.config import configure_logging
from swingml.data.breeze import (
    API_KEY_ENV,
    DEFAULT_SESSION_PATH,
    SECRET_KEY_ENV,
    BreezeSessionError,
    load_env_file,
    login_url,
    parse_session_token,
    save_session_token,
    session_status,
)


def _check_token(token: str) -> bool:
    """Best-effort live verification via ``customer_details``.

    Optional, because it needs ``breeze-connect`` installed and the registered
    static IP: it is the one call that proves the token (and the IP) actually
    work before a full build is attempted.
    """
    try:
        from breeze_connect import BreezeConnect
    except ImportError:
        print("  --check: breeze-connect not installed (uv sync --extra breeze); skipping.")
        return False
    load_env_file()
    api_key = os.environ.get(API_KEY_ENV, "").strip()
    secret = os.environ.get(SECRET_KEY_ENV, "").strip()
    if not (api_key and secret):
        print(f"  --check: set {API_KEY_ENV} and {SECRET_KEY_ENV} to verify.")
        return False
    try:
        client = BreezeConnect(api_key=api_key)
        client.generate_session(api_secret=secret, session_token=token)
        details = client.get_customer_details(api_session=token)
    except Exception as exc:  # noqa: BLE001 - surface whatever the API says
        print(f"  --check: FAILED -- {type(exc).__name__}: {exc}")
        print("  If this mentions a static IP, register the machine's IP on the API key.")
        return False
    ok = bool(details)
    print(f"  --check: {'OK -- token accepted' if ok else 'API returned an empty payload'}")
    return ok


def run(args: argparse.Namespace) -> int:
    # A gitignored .env is the normal home for the App Key; a shell export wins.
    n_env = load_env_file()
    api_key = (args.api_key or os.environ.get(API_KEY_ENV, "")).strip()
    if not api_key:
        print("No App Key. Pass --api-key or export BREEZE_API_KEY "
              "(the App Key is static; it is the *token* that rotates).")
        return 2

    url = login_url(api_key)
    print("=" * 88)
    print("BREEZE SESSION TOKEN  --  manual, per SEBI/ICICI guidance")
    print("=" * 88)
    if n_env:
        print(f"loaded {n_env} value(s) from .env")
    print(f"1. Log in at:\n   {url}")
    print(f"2. You are redirected to your registered Redirect URL, which carries")
    print(f"   '?apisession=...' in the query string.")
    print(f"3. Paste that whole URL back here (or just the apisession value).")
    print(f"\nToken file: {args.path}   (gitignored; expires at IST midnight)")
    print()

    if args.url_only:
        return 0

    if not args.no_browser:
        try:
            webbrowser.open(url)
        except Exception:  # noqa: BLE001 - headless/CI is fine, we print the URL
            pass

    if args.token:
        pasted = args.token
    else:
        try:
            pasted = input("Redirected URL / apisession: ")
        except EOFError:
            print("no input (stdin closed). Use --token for non-interactive use.")
            return 2

    try:
        token = parse_session_token(pasted)
    except BreezeSessionError as exc:
        print(f"error: {exc}")
        return 1

    path = save_session_token(token, args.path, api_key=api_key)
    st = session_status(path)
    print(f"\nsaved -> {path}")
    print(f"  stamped {st['date']} (IST) | current: {st['current']}")
    print(f"  tail: ...{token[-6:]}")

    if args.check:
        return 0 if _check_token(token) else 1
    print("\nNext: build with the same-day source, before the 06:30 IST freeze job:")
    print("  .venv/bin/python -m swingml.cli build-dataset --universe liquidity \\")
    print("      --start 2020-01-01 --price-provider breeze \\")
    print("      --dataset-dir data/datasets_liquidity_fh")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="store today's Breeze session token (manual)")
    ap.add_argument("--api-key", default=None, help="App Key (else $BREEZE_API_KEY)")
    ap.add_argument("--token", default=None,
                    help="redirected URL or apisession value (skips the prompt)")
    ap.add_argument("--path", default=DEFAULT_SESSION_PATH,
                    help=f"where to store the token (default {DEFAULT_SESSION_PATH})")
    ap.add_argument("--url-only", action="store_true", help="print the login URL and exit")
    ap.add_argument("--no-browser", action="store_true", help="do not open a browser")
    ap.add_argument("--check", action="store_true",
                    help="verify the token with a live customer_details call "
                         "(needs breeze-connect + the registered static IP)")
    args = ap.parse_args()
    configure_logging()
    try:
        return run(args)
    except BreezeSessionError as exc:
        print(f"error: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
