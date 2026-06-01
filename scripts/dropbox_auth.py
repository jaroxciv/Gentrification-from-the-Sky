"""Stateless two-step Dropbox OAuth → saves a long-lived refresh token to .env.

Prereqs (one time, in the browser):
    1. Create a Dropbox app: https://www.dropbox.com/developers/apps
       - "Scoped access"  ->  "Full Dropbox"
       - Permissions tab: enable (READ-ONLY)
           account_info.read, files.metadata.read, files.content.read
         then click Submit.
    2. Copy .env.example to .env and fill DROPBOX_APP_KEY / DROPBOX_APP_SECRET.

Then:
    Step 1:  uv run scripts/dropbox_auth.py
             -> prints an authorize URL. Open it, click Allow, copy the code.
    Step 2:  uv run scripts/dropbox_auth.py <PASTE_CODE_HERE>
             -> exchanges the code and writes DROPBOX_REFRESH_TOKEN to .env.

Uses the classic auth-code grant (app secret, no PKCE), so the two steps
are independent processes — no shared in-memory state required. The refresh
token does not expire. Nothing here is committed (.env is git-ignored).
"""

from __future__ import annotations

import os
import sys
import urllib.parse
from pathlib import Path

import requests
from dotenv import load_dotenv, set_key

ENV_PATH = Path(__file__).resolve().parents[1] / ".env"

# Read-only scopes. Configured on the app's Permissions tab; not sent in the
# authorize URL (see print_url). Listed here for documentation only.
SCOPES = " ".join(
    [
        "account_info.read",
        "files.metadata.read",
        "files.content.read",
    ]
)


def _creds() -> tuple[str, str]:
    if not ENV_PATH.exists():
        raise SystemExit(
            f"No .env at {ENV_PATH}. Copy .env.example to .env and fill in "
            "DROPBOX_APP_KEY / DROPBOX_APP_SECRET first."
        )
    load_dotenv(ENV_PATH)
    key = os.environ.get("DROPBOX_APP_KEY", "").strip()
    secret = os.environ.get("DROPBOX_APP_SECRET", "").strip()
    if not key or not secret:
        raise SystemExit("Set DROPBOX_APP_KEY and DROPBOX_APP_SECRET in .env first.")
    return key, secret


def print_url() -> None:
    key, _ = _creds()
    params = {
        "client_id": key,
        "response_type": "code",
        "token_access_type": "offline",  # -> issues a refresh token
    }
    # NOTE: we intentionally omit the `scope` param so Dropbox uses whatever
    # scopes are enabled+Submitted on the app's Permissions tab. (Passing a
    # scope the app hasn't been granted causes a "Cannot find scope" error.)
    url = "https://www.dropbox.com/oauth2/authorize?" + urllib.parse.urlencode(
        params, quote_via=urllib.parse.quote
    )
    print("\n1. Open this URL in your browser:\n")
    print(f"   {url}\n")
    print("2. Click 'Allow' (log in first if needed).")
    print("3. Copy the authorization code, then run:\n")
    print("   uv run scripts/dropbox_auth.py <PASTE_CODE_HERE>\n")


def exchange(code: str) -> None:
    key, secret = _creds()
    resp = requests.post(
        "https://api.dropboxapi.com/oauth2/token",
        data={
            "code": code.strip(),
            "grant_type": "authorization_code",
            "client_id": key,
            "client_secret": secret,
        },
        timeout=30,
    )
    if resp.status_code != 200:
        raise SystemExit(f"Token exchange failed ({resp.status_code}): {resp.text}")
    data = resp.json()
    refresh_token = data.get("refresh_token")
    if not refresh_token:
        raise SystemExit(f"No refresh_token in response: {data}")
    set_key(str(ENV_PATH), "DROPBOX_REFRESH_TOKEN", refresh_token)
    print("\n✅ Success. DROPBOX_REFRESH_TOKEN saved to .env.")
    print("   Test it with:  uv run scripts/dropbox_ls.py")


def main() -> None:
    if len(sys.argv) > 1:
        exchange(sys.argv[1])
    else:
        print_url()


if __name__ == "__main__":
    main()
