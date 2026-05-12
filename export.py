"""
gee_pipeline/export.py
────────────────────────────────────────────────────────────────────────────────
GEE export orchestration and local JSON serialisation for EAAF Bird Sentinel.

Two export strategies:

  Strategy A — Direct computation (default for ≤30 stations)
    reduceRegions() on monthly composites → getInfo() → write JSON locally.
    Fastest for small station counts; no GEE batch task overhead.

  Strategy B — GEE batch export to Cloud Storage (for 200+ stations / large
    date ranges where Strategy A may hit memory limits)
    Export table → Cloud Storage → download → write JSON locally.

Both strategies produce identical output JSON schemas.

Output schema per station (data/ecological/{station_id}.json):
    {
      "station_id":   "S01",
      "station_name": "崇明东滩",
      "habitat":      "tidal_flat",
      "updated_at":   "2025-05-01T12:00:00Z",
      "indices": {
        "NDVI": [{"date": "2023-01", "value": 0.42}, ...],
        "EVI":  [...],
        "NDWI": [...],
        "LAI":  [...],
        "HSI":  [...]
      }
    }
────────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import ee
import structlog
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)
import logging

log = structlog.get_logger(__name__)

# ── Index band names (must match indices.compute_all_indices output) ──────────
INDEX_BANDS = ["NDVI", "EVI", "NDWI", "LAI", "HSI"]

# ── Value clamping per index (physical bounds — reject sensor artefacts) ──────
INDEX_CLAMP: dict[str, tuple[float, float]] = {
    "NDVI": (-1.0,  1.0),
    "EVI":  (-1.0,  1.0),
    "NDWI": (-1.0,  1.0),
    "LAI":  ( 0.0,  8.0),
    "HSI":  ( 0.0,  1.0),
}

# ── Decimal precision per index ───────────────────────────────────────────────
INDEX_DECIMALS: dict[str, int] = {
    "NDVI": 3, "EVI": 3, "NDWI": 3, "LAI": 2, "HSI": 2,
}


# ── Strategy A: direct computation ───────────────────────────────────────────

@retry(
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=2, min=4, max=60),
    retry=retry_if_exception_type(Exception),
    before_sleep=before_sleep_log(logging.getLogger(__name__), logging.WARNING),
    reraise=True,
)
def _reduce_one_month(
    composite:  ee.Image,
    fc:         ee.FeatureCollection,
    scale:      int = 10,
) -> list[dict]:
    """
    Run reduceRegions() on one monthly composite image and return raw features.

    A single reduceRegions() call computes zonal mean for ALL stations
    simultaneously on the GEE server.  This replaces the original O(n) loop
    over stations × per-station reduceRegion().

    The @retry decorator handles transient GEE 503/429 errors with
    exponential back-off (4s → 8s → 16s → 60s cap, 4 attempts total).
    """
    reduced = composite.reduceRegions(
        collection=fc,
        reducer=ee.Reducer.mean(),
        scale=scale,           # 10m to match Sentinel-2 native resolution
        tileScale=4,           # increase memory tiles for large feature collections
    )
    # .getInfo() is the only unavoidable client-side call —
    # it happens ONCE per month, not once per station.
    return reduced.getInfo()["features"]


def extract_direct(
    monthly_composites: ee.ImageCollection,
    feature_collection: ee.FeatureCollection,
    output_dir:         Path,
    station_meta:       dict[str, dict],
    scale:              int = 10,
) -> dict[str, dict]:
    """
    Strategy A: extract index values for all stations using direct computation.

    Iterates over months client-side (one getInfo() per month) but reduces
    all stations in a single server call per month.

    Complexity: O(months) getInfo() calls  [was O(months × stations × indices)]

    Parameters
    ----------
    monthly_composites:
        ee.ImageCollection of monthly median index images.
    feature_collection:
        ee.FeatureCollection with one buffered feature per station.
    output_dir:
        Local directory for output JSON files.
    station_meta:
        Dict of station_id → station info dict.
    scale:
        GEE spatial resolution in metres.

    Returns
    -------
    Dict of station_id → result dict (also written to output_dir).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Pull the list of monthly images client-side (one getInfo for metadata only)
    n_images = monthly_composites.size().getInfo()
    log.info("extract_start", strategy="direct", n_months=n_images)

    image_list = monthly_composites.toList(n_images)

    # Accumulate results per station
    results: dict[str, dict] = {}
    for sid, meta in station_meta.items():
        results[sid] = {
            "station_id":   sid,
            "station_name": meta.get("name", ""),
            "habitat":      meta.get("habitat", ""),
            "updated_at":   "",
            "indices":      {band: [] for band in INDEX_BANDS},
        }

    # Iterate months — one getInfo() per month
    for i in range(n_images):
        composite = ee.Image(image_list.get(i))

        try:
            features = _reduce_one_month(composite, feature_collection, scale)
        except Exception as exc:
            log.error("month_extraction_failed", month_index=i, error=str(exc))
            continue  # skip this month; partial results are still useful

        for feat in features:
            props = feat.get("properties", {})
            sid   = props.get("station_id")
            date  = props.get("date")

            if not sid or sid not in results or not date:
                continue

            for band in INDEX_BANDS:
                val = props.get(band)
                if val is None:
                    continue
                lo, hi = INDEX_CLAMP[band]
                if not (lo <= float(val) <= hi):
                    continue  # clamp artefacts
                dec = INDEX_DECIMALS[band]
                results[sid]["indices"][band].append({
                    "date":  date,
                    "value": round(float(val), dec),
                })

        log.info("month_extracted", month_index=i + 1, total=n_images)

    # Sort each series by date and write to disk
    now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    for sid, rec in results.items():
        rec["updated_at"] = now_iso
        for band in INDEX_BANDS:
            rec["indices"][band].sort(key=lambda x: x["date"])

        _write_station_json(output_dir, sid, rec)

    log.info("extract_complete", stations_written=len(results))
    return results


# ── Strategy B: GEE batch export → Cloud Storage ─────────────────────────────

def export_to_cloud_storage(
    monthly_composites: ee.ImageCollection,
    feature_collection: ee.FeatureCollection,
    gcs_bucket:         str,
    gcs_prefix:         str = "eaaf/ecological",
    scale:              int = 10,
) -> list[ee.batch.Task]:
    """
    Strategy B: export all monthly reduceRegions results to Cloud Storage as CSV.

    Use this for 200+ station runs or date ranges > 3 years where Strategy A
    may hit GEE memory / request-quota limits.

    The exported CSV files are then downloaded and assembled by
    assemble_from_gcs().

    Returns a list of submitted ee.batch.Task objects for monitoring.
    """
    n_images = monthly_composites.size().getInfo()
    image_list = monthly_composites.toList(n_images)
    tasks = []

    for i in range(n_images):
        composite   = ee.Image(image_list.get(i))
        year_month  = composite.get("year_month").getInfo()

        reduced = composite.reduceRegions(
            collection=feature_collection,
            reducer=ee.Reducer.mean(),
            scale=scale,
            tileScale=4,
        )

        task = ee.batch.Export.table.toCloudStorage(
            collection=reduced,
            description=f"eaaf_eco_{year_month}",
            bucket=gcs_bucket,
            fileNamePrefix=f"{gcs_prefix}/{year_month}",
            fileFormat="CSV",
            selectors=["station_id", "date"] + INDEX_BANDS,
        )
        task.start()
        tasks.append(task)
        log.info("gcs_export_submitted", year_month=year_month, task_id=task.id)

    return tasks


def monitor_tasks(
    tasks: list[ee.batch.Task],
    poll_interval_s: int = 30,
    timeout_s:       int = 3600,
) -> dict[str, str]:
    """
    Poll GEE batch task states until all complete or timeout is reached.

    Returns a dict of task_id → final state
    ('COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT').
    """
    start = time.time()
    pending = {t.id: t for t in tasks}
    final_states: dict[str, str] = {}

    while pending and (time.time() - start) < timeout_s:
        time.sleep(poll_interval_s)
        for tid in list(pending.keys()):
            task  = pending[tid]
            state = task.status()["state"]
            if state in ("COMPLETED", "FAILED", "CANCELLED"):
                final_states[tid] = state
                del pending[tid]
                log.info("task_finished", task_id=tid, state=state)
            else:
                log.debug("task_running", task_id=tid, state=state)

    # Any still-pending tasks have timed out
    for tid in pending:
        final_states[tid] = "TIMEOUT"
        log.warning("task_timeout", task_id=tid)

    return final_states


def assemble_from_gcs(
    gcs_bucket:   str,
    gcs_prefix:   str,
    output_dir:   Path,
    station_meta: dict[str, dict],
) -> dict[str, dict]:
    """
    Download CSVs from GCS and assemble per-station JSON files.

    Requires google-cloud-storage to be installed and authenticated.
    """
    try:
        from google.cloud import storage  # type: ignore
    except ImportError:
        raise RuntimeError("google-cloud-storage is required for GCS assembly")

    client  = storage.Client()
    bucket  = client.bucket(gcs_bucket)
    blobs   = list(bucket.list_blobs(prefix=gcs_prefix))

    import csv
    import io

    results: dict[str, dict] = {
        sid: {
            "station_id":   sid,
            "station_name": meta.get("name", ""),
            "habitat":      meta.get("habitat", ""),
            "updated_at":   "",
            "indices":      {band: [] for band in INDEX_BANDS},
        }
        for sid, meta in station_meta.items()
    }

    for blob in blobs:
        if not blob.name.endswith(".csv"):
            continue
        content = blob.download_as_text()
        reader  = csv.DictReader(io.StringIO(content))
        for row in reader:
            sid  = row.get("station_id")
            date = row.get("date")
            if not sid or sid not in results or not date:
                continue
            for band in INDEX_BANDS:
                val_str = row.get(band, "")
                if not val_str:
                    continue
                try:
                    val = float(val_str)
                except ValueError:
                    continue
                lo, hi = INDEX_CLAMP[band]
                if not (lo <= val <= hi):
                    continue
                results[sid]["indices"][band].append({
                    "date":  date,
                    "value": round(val, INDEX_DECIMALS[band]),
                })

    now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    output_dir.mkdir(parents=True, exist_ok=True)

    for sid, rec in results.items():
        rec["updated_at"] = now_iso
        for band in INDEX_BANDS:
            rec["indices"][band].sort(key=lambda x: x["date"])
        _write_station_json(output_dir, sid, rec)

    log.info("gcs_assembly_complete", stations_written=len(results))
    return results


# ── JSON writer ───────────────────────────────────────────────────────────────

def _write_station_json(output_dir: Path, station_id: str, record: dict) -> None:
    """Write a station result dict to output_dir/{station_id}.json."""
    path = output_dir / f"{station_id}.json"
    with path.open("w", encoding="utf-8") as fh:
        json.dump(record, fh, indent=2, ensure_ascii=False)
    log.debug("station_json_written", path=str(path), station_id=station_id)


# ── Cache check ────────────────────────────────────────────────────────────────

def is_cache_fresh(
    output_dir:  Path,
    station_id:  str,
    max_age_h:   float = 6.0,
) -> bool:
    """
    Return True if the cached JSON file for a station is younger than max_age_h.

    Used by the pipeline to skip re-fetching data that is still fresh.
    """
    path = output_dir / f"{station_id}.json"
    if not path.exists():
        return False
    age_s = time.time() - path.stat().st_mtime
    return age_s < (max_age_h * 3600)


def load_cached_result(output_dir: Path, station_id: str) -> Optional[dict]:
    """Load and parse a cached station JSON file, or return None."""
    path = output_dir / f"{station_id}.json"
    if not path.exists():
        return None
    try:
        with path.open(encoding="utf-8") as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError):
        return None
