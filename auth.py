"""
gee_pipeline/auth.py
────────────────────────────────────────────────────────────────────────────────
Google Earth Engine authentication for the EAAF Bird Sentinel pipeline.

Supports three credential modes (tried in priority order):
  1. Service-account key file  → for CI/CD and Mac Mini M4 unattended runs
  2. Application Default Credentials → for interactive dev / Cloud Run
  3. Persistent token file     → for local dev after first `earthengine authenticate`

All modes are handled here so the rest of the pipeline never touches auth.
────────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import ee
import structlog

log = structlog.get_logger(__name__)


# ── Constants ────────────────────────────────────────────────────────────────

# GEE project ID — override via environment variable EAAF_GEE_PROJECT
DEFAULT_PROJECT = "eaaf-bird-sentinel"

# Path where a downloaded service-account JSON key can be placed
DEFAULT_SA_KEY_PATH = Path.home() / ".config" / "eaaf" / "gee_service_account.json"


# ── Public API ────────────────────────────────────────────────────────────────

def initialise(
    project:       Optional[str] = None,
    sa_key_path:   Optional[Path] = None,
    opt_url:       Optional[str]  = None,
) -> None:
    """
    Authenticate and initialise the Earth Engine API.

    Call once at process startup before any ee.* calls.

    Parameters
    ----------
    project:
        GEE Cloud project ID.  Falls back to EAAF_GEE_PROJECT env-var,
        then DEFAULT_PROJECT.
    sa_key_path:
        Path to a service-account JSON key file.  Falls back to
        EAAF_GEE_SA_KEY env-var, then DEFAULT_SA_KEY_PATH.
    opt_url:
        Optional EE REST API endpoint override (for testing).
    """
    project    = project    or os.getenv("EAAF_GEE_PROJECT",  DEFAULT_PROJECT)
    sa_key_path = Path(
        sa_key_path
        or os.getenv("EAAF_GEE_SA_KEY", str(DEFAULT_SA_KEY_PATH))
    )

    credentials = _resolve_credentials(sa_key_path)

    init_kwargs: dict = {"project": project}
    if opt_url:
        init_kwargs["opt_url"] = opt_url
    if credentials is not None:
        init_kwargs["credentials"] = credentials

    ee.Initialize(**init_kwargs)

    log.info(
        "earth_engine_initialised",
        project=project,
        credential_mode=_credential_mode(sa_key_path),
    )


# ── Internal helpers ──────────────────────────────────────────────────────────

def _resolve_credentials(sa_key_path: Path):
    """
    Return an oauth2 Credentials object, or None to let GEE use ADC.
    """
    # Mode 1: explicit service-account key file
    if sa_key_path.exists():
        import google.oauth2.service_account as sa_module  # type: ignore

        credentials = sa_module.Credentials.from_service_account_file(
            str(sa_key_path),
            scopes=["https://www.googleapis.com/auth/earthengine"],
        )
        log.info("auth_mode", mode="service_account", path=str(sa_key_path))
        return credentials

    # Mode 2: Application Default Credentials (ADC)
    # This covers Cloud Run, GCE, and `gcloud auth application-default login`
    try:
        import google.auth  # type: ignore

        credentials, _ = google.auth.default(
            scopes=["https://www.googleapis.com/auth/earthengine"]
        )
        log.info("auth_mode", mode="application_default_credentials")
        return credentials
    except Exception:
        pass

    # Mode 3: persistent token written by `earthengine authenticate`
    log.info("auth_mode", mode="persistent_token_fallback")
    return None  # ee.Initialize() will read ~/.config/earthengine/credentials


def _credential_mode(sa_key_path: Path) -> str:
    if sa_key_path.exists():
        return "service_account"
    try:
        import google.auth  # type: ignore
        google.auth.default()
        return "adc"
    except Exception:
        return "persistent_token"
