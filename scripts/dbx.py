"""Shared Dropbox client factory. Imported by the other scripts."""

from __future__ import annotations

import os
from pathlib import Path

import dropbox
from dotenv import load_dotenv

ENV_PATH = Path(__file__).resolve().parents[1] / ".env"


def get_client() -> dropbox.Dropbox:
    """Build a Dropbox client from .env credentials (auto-refreshing token)."""
    load_dotenv(ENV_PATH)
    app_key = os.environ.get("DROPBOX_APP_KEY", "").strip()
    app_secret = os.environ.get("DROPBOX_APP_SECRET", "").strip()
    refresh_token = os.environ.get("DROPBOX_REFRESH_TOKEN", "").strip()
    if not (app_key and app_secret and refresh_token):
        raise SystemExit("Missing credentials in .env. Run `uv run scripts/dropbox_auth.py` first.")
    return dropbox.Dropbox(
        oauth2_refresh_token=refresh_token,
        app_key=app_key,
        app_secret=app_secret,
    )


def data_root() -> str:
    load_dotenv(ENV_PATH)
    return os.environ.get("DROPBOX_DATA_ROOT", "/Master/LSE/CDRC").rstrip("/")
