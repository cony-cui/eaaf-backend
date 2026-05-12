"""
gee_pipeline/pipeline.py
────────────────────────────────────────────────────────────────────────────────
Top-level pipeline orchestrator for EAAF Bird Sentinel ecological data export.

Usage:

    # Run full pipeline (20 stations, last 24 months):
    python -m gee_pipeline.pipeline

    # Run specific stations only:
    python -m gee_pipeline.pipeline --stations S01 S02 S05

    # Use GCS export strategy (for large runs):
    python -m gee_pipeline.pipeline --strategy gcs --bucket my-gcs-bucket

    # Dry run (validate auth + stations, skip GEE calls):
    python -m gee_pipeline.pipeline --dry-run

Performance notes:
  - Direct strategy:    ~2-4 min for 20 stations × 24 months on M4 Mac Mini
  - GCS strategy:       ~15-30 min (GEE batch queue dependent)
  - GEE quota usage:    1 getInfo() per month (was 1 per station×month×index)
  - Resumable:          already-exported stations are skipped if cache is fresh
────────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import structlog

from . import auth, indices, stations as st_module, export as exp_module

# ── Logging setup ─────────────────────────────────────────────────────────────
import logging
import structlog

structlog.configure(
    processors=[
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M:%S", utc=False),
        structlog.dev.ConsoleRenderer(colors=True),
    ],
    wrapper_class=structlog.make_filtering_bound_logger(logging.DEBUG),
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
)

log = structlog.get_logger(__name__)

# ── Defaults ──────────────────────────────────────────────────────────────────

DEFAULT_START      = "2023-01-01"
DEFAULT_END        = "2025-12-31"
DEFAULT_OUTPUT_DIR = Path(__file__).parent.parent / "data" / "ecological"
DEFAULT_SCALE      = 10      # metres — Sentinel-2 native 10m bands
DEFAULT_MAX_CLOUD  = 30.0    # %
DEFAULT_CACHE_H    = 6.0     # hours before a cached result is considered stale


# ── Main pipeline function ────────────────────────────────────────────────────

def run(
    station_ids:   list[str]  | None = None,
    start_date:    str               = DEFAULT_START,
    end_date:      str               = DEFAULT_END,
    output_dir:    Path              = DEFAULT_OUTPUT_DIR,
    strategy:      str               = "direct",
    gcs_bucket:    str  | None       = None,
    gcs_prefix:    str               = "eaaf/ecological",
    scale:         int               = DEFAULT_SCALE,
    max_cloud:     float             = DEFAULT_MAX_CLOUD,
    cache_hours:   float             = DEFAULT_CACHE_H,
    dry_run:       bool              = False,
    stations_path: Path | None       = None,
) -> dict[str, dict]:
    """
    Execute the full ecological index extraction pipeline.

    Parameters
    ----------
    station_ids:
        Subset of station IDs to process. None = all stations.
    start_date / end_date:
        ISO date strings defining the temporal extent.
    output_dir:
        Local directory for output JSON files.
    strategy:
        'direct'  — Strategy A (reduceRegions + getInfo per month)
        'gcs'     — Strategy B (GEE batch export to Cloud Storage)
    gcs_bucket:
        GCS bucket name (required for strategy='gcs').
    scale:
        GEE pixel scale in metres for zonal statistics.
    max_cloud:
        Maximum scene-level cloud percentage for Sentinel-2 filtering.
    cache_hours:
        If a station's output JSON is younger than this, skip re-fetching.
    dry_run:
        If True, validate configuration and print plan without running GEE calls.

    Returns
    -------
    Dict of station_id → result dict for all processed stations.
    """
    t_start = time.perf_counter()

    # ── 1. Load and validate stations ────────────────────────────────────────
    all_stations  = st_module.load_stations(stations_path)
    all_stations  = st_module.validate_stations(all_stations)

    if station_ids:
        all_stations = [s for s in all_stations if s["id"] in set(station_ids)]
        log.info("station_filter_applied", requested=station_ids, matched=len(all_stations))

    if not all_stations:
        log.error("no_valid_stations")
        sys.exit(1)

    station_meta = st_module.station_index_map(all_stations)

    # ── 2. Check cache — skip fresh stations ─────────────────────────────────
    stale_stations = [
        s for s in all_stations
        if not exp_module.is_cache_fresh(output_dir, s["id"], cache_hours)
    ]
    cached_count = len(all_stations) - len(stale_stations)
    if cached_count:
        log.info("stations_cache_fresh", count=cached_count, skipped=True)
    if not stale_stations:
        log.info("all_stations_cached_skipping_gee")
        return {
            sid: exp_module.load_cached_result(output_dir, sid)
            for sid in station_meta
        }

    log.info(
        "pipeline_start",
        stations=len(stale_stations),
        start=start_date, end=end_date,
        strategy=strategy, scale=scale,
    )

    if dry_run:
        log.info("dry_run_complete — would process stations", ids=[s["id"] for s in stale_stations])
        return {}

    # ── 3. Initialise GEE ────────────────────────────────────────────────────
    auth.initialise()

    # ── 4. Build GEE objects ─────────────────────────────────────────────────
    feature_collection = st_module.build_feature_collection(stale_stations)

    # Build a single unified FeatureCollection bbox to minimise collection size
    # (filterBounds with a FeatureCollection is more efficient than per-station filtering)
    aoi = feature_collection.geometry().bounds(maxError=1000)

    # Build Sentinel-2 collection for the full AOI, fully masked and indexed
    s2_collection = indices.build_s2_collection(
        geometry   = aoi,
        start_date = start_date,
        end_date   = end_date,
        max_cloud  = max_cloud,
    )

    # Monthly median composites (one image per month, all index bands)
    monthly = indices.build_monthly_composites(s2_collection, start_date, end_date)

    log.info("gee_collections_built")

    # ── 5. Extract data via selected strategy ────────────────────────────────
    results: dict[str, dict] = {}

    if strategy == "direct":
        # Strategy A — fastest for ≤50 stations, pulls all data in-process
        stale_meta = {s["id"]: s for s in stale_stations}
        results = exp_module.extract_direct(
            monthly_composites = monthly,
            feature_collection = feature_collection,
            output_dir         = output_dir,
            station_meta       = stale_meta,
            scale              = scale,
        )

    elif strategy == "gcs":
        # Strategy B — GEE batch export for large runs
        if not gcs_bucket:
            log.error("gcs_bucket_required_for_gcs_strategy")
            sys.exit(1)

        tasks = exp_module.export_to_cloud_storage(
            monthly_composites = monthly,
            feature_collection = feature_collection,
            gcs_bucket         = gcs_bucket,
            gcs_prefix         = gcs_prefix,
            scale              = scale,
        )
        log.info("waiting_for_gcs_export_tasks", task_count=len(tasks))
        final_states = exp_module.monitor_tasks(tasks, poll_interval_s=30, timeout_s=7200)

        failed = [tid for tid, s in final_states.items() if s != "COMPLETED"]
        if failed:
            log.warning("some_tasks_failed", task_ids=failed)

        results = exp_module.assemble_from_gcs(
            gcs_bucket   = gcs_bucket,
            gcs_prefix   = gcs_prefix,
            output_dir   = output_dir,
            station_meta = {s["id"]: s for s in stale_stations},
        )

    else:
        log.error("unknown_strategy", strategy=strategy)
        sys.exit(1)

    # ── 6. Merge cached results for stations that were skipped ───────────────
    for sid in station_meta:
        if sid not in results:
            cached = exp_module.load_cached_result(output_dir, sid)
            if cached:
                results[sid] = cached

    elapsed = time.perf_counter() - t_start
    log.info(
        "pipeline_complete",
        stations_processed=len(results),
        elapsed_s=round(elapsed, 1),
    )

    return results


# ── CLI entry point ───────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="eaaf-pipeline",
        description="EAAF Bird Sentinel ecological index extraction pipeline",
    )
    p.add_argument(
        "--stations", nargs="+", metavar="ID",
        help="Station IDs to process (default: all)",
    )
    p.add_argument("--start",    default=DEFAULT_START, help="Start date YYYY-MM-DD")
    p.add_argument("--end",      default=DEFAULT_END,   help="End   date YYYY-MM-DD")
    p.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
        help="Output directory for JSON files",
    )
    p.add_argument(
        "--strategy", choices=["direct", "gcs"], default="direct",
        help="Export strategy: 'direct' (default) or 'gcs'",
    )
    p.add_argument("--bucket",   default=None,   help="GCS bucket (required for --strategy gcs)")
    p.add_argument("--prefix",   default="eaaf/ecological", help="GCS object prefix")
    p.add_argument("--scale",    type=int,   default=DEFAULT_SCALE,     help="Pixel scale (m)")
    p.add_argument("--cloud",    type=float, default=DEFAULT_MAX_CLOUD, help="Max cloud %%")
    p.add_argument("--cache-h",  type=float, default=DEFAULT_CACHE_H,   help="Cache TTL hours")
    p.add_argument("--dry-run",  action="store_true", help="Validate config without GEE calls")
    p.add_argument(
        "--stations-file", type=Path, default=None,
        help="Path to external stations JSON (overrides built-in registry)",
    )
    return p


if __name__ == "__main__":
    args = _build_parser().parse_args()
    run(
        station_ids   = args.stations,
        start_date    = args.start,
        end_date      = args.end,
        output_dir    = args.output_dir,
        strategy      = args.strategy,
        gcs_bucket    = args.bucket,
        gcs_prefix    = args.prefix,
        scale         = args.scale,
        max_cloud     = args.cloud,
        cache_hours   = args.cache_h,
        dry_run       = args.dry_run,
        stations_path = args.stations_file,
    )
