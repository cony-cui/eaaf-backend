"""
api/main.py
────────────────────────────────────────────────────────────────────────────────
FastAPI application — EAAF Bird Sentinel ecological index API.

Endpoints consumed by the WeChat Mini Program frontend:

  GET /api/ecological-index
      ?station_id=S01&index=NDVI&months=12
      → EcologicalIndexResponse

  GET /api/stations
      → list[StationSummary]

  GET /api/ecological-index/batch
      ?station_ids=S01,S02&index=NDVI&months=12
      → list[EcologicalIndexResponse]

  POST /api/pipeline/trigger
      Body: PipelineTriggerRequest
      → PipelineStatus  (triggers background GEE extraction)

  GET /api/pipeline/status
      → PipelineStatus

Data serving strategy:
  1. Serve from pre-computed JSON files in data/ecological/ (zero GEE latency)
  2. If file missing or stale (> 6h), trigger background GEE re-fetch
  3. Return stale data immediately while refresh runs in background
────────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import structlog
import uvicorn
from fastapi import BackgroundTasks, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import ORJSONResponse
from pydantic import BaseModel, Field

log = structlog.get_logger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).parent.parent / "data" / "ecological"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# ── Valid index types ─────────────────────────────────────────────────────────
VALID_INDICES = {"NDVI", "EVI", "NDWI", "LAI", "HSI"}

# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="EAAF Bird Sentinel — Ecological Index API",
    version="2.0.0",
    default_response_class=ORJSONResponse,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # lock down to Mini Program domain in production
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# ── Pipeline background state ─────────────────────────────────────────────────
_pipeline_running = False
_pipeline_last_run: Optional[str] = None
_pipeline_error:    Optional[str] = None


# ── Pydantic schemas ──────────────────────────────────────────────────────────

class IndexDataPoint(BaseModel):
    date:  str   = Field(..., example="2024-06")
    value: float = Field(..., example=0.52)


class EcologicalIndexResponse(BaseModel):
    station_id: str
    index:      str
    dates:      list[str]
    values:     list[float]
    # Metadata
    station_name: Optional[str]  = None
    habitat:      Optional[str]  = None
    updated_at:   Optional[str]  = None
    # Trend (populated by API, not stored in JSON)
    trend_direction: Optional[str]  = None   # "up" | "down" | "stable"
    trend_text:      Optional[str]  = None


class StationSummary(BaseModel):
    id:        str
    name:      str
    lat:       float
    lon:       float
    habitat:   Optional[str]
    has_data:  bool
    updated_at: Optional[str]
    # Latest values (most recent observation)
    latest_ndvi: Optional[float] = None
    latest_hsi:  Optional[float] = None


class PipelineTriggerRequest(BaseModel):
    station_ids:  Optional[list[str]] = None
    start_date:   str = "2023-01-01"
    end_date:     str = "2025-12-31"
    force_refresh: bool = False


class PipelineStatus(BaseModel):
    running:   bool
    last_run:  Optional[str]
    error:     Optional[str]


# ── Helper: load station file ─────────────────────────────────────────────────

def _load_station_file(station_id: str) -> Optional[dict]:
    path = DATA_DIR / f"{station_id}.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        log.warning("station_file_load_error", station_id=station_id, error=str(exc))
        return None


def _slice_to_months(record: dict, index_key: str, months: int) -> tuple[list[str], list[float]]:
    """
    Return the most recent `months` observations for the given index.
    Handles missing values (None → excluded from response).
    """
    series: list[dict] = record.get("indices", {}).get(index_key, [])
    series = sorted(series, key=lambda x: x["date"])[-months:]
    dates  = [p["date"]  for p in series if p.get("value") is not None]
    values = [float(p["value"]) for p in series if p.get("value") is not None]
    return dates, values


def _simple_trend(values: list[float]) -> str:
    """Simple trend from first half vs second half mean."""
    if len(values) < 4:
        return "stable"
    mid   = len(values) // 2
    delta = sum(values[mid:]) / len(values[mid:]) - sum(values[:mid]) / len(values[:mid])
    if   delta >  0.03: return "up"
    elif delta < -0.03: return "down"
    return "stable"


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "time": datetime.now(timezone.utc).isoformat()}


@app.get("/api/ecological-index", response_model=EcologicalIndexResponse)
def get_ecological_index(
    station_id: str = Query(..., description="Station ID e.g. S01"),
    index:      str = Query(..., description="Index type: NDVI, EVI, NDWI, LAI, HSI"),
    months:     int = Query(12, ge=1, le=60, description="Number of months to return"),
    background_tasks: BackgroundTasks = BackgroundTasks(),
):
    """
    Primary endpoint consumed by the WeChat Mini Program.

    Returns time-series values for the requested index and station.
    Triggers a background GEE refresh if data is stale (> 6 hours).
    """
    index = index.upper()
    if index not in VALID_INDICES:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown index '{index}'. Valid: {sorted(VALID_INDICES)}",
        )

    record = _load_station_file(station_id)
    if not record:
        raise HTTPException(
            status_code=404,
            detail=f"No data found for station '{station_id}'. "
                   f"Run the pipeline first or check the station ID.",
        )

    dates, values = _slice_to_months(record, index, months)

    if not dates:
        raise HTTPException(
            status_code=404,
            detail=f"No {index} observations found for station '{station_id}'.",
        )

    # Trigger background refresh if data file is older than 6 hours
    path = DATA_DIR / f"{station_id}.json"
    import time
    if time.time() - path.stat().st_mtime > 21600:
        background_tasks.add_task(_background_refresh, [station_id])

    trend = _simple_trend(values)
    trend_labels = {
        "up":     "植被持续改善",
        "down":   "植被退化，建议关注",
        "stable": "生态指数平稳",
    }

    return EcologicalIndexResponse(
        station_id   = station_id,
        index        = index,
        dates        = dates,
        values       = values,
        station_name = record.get("station_name"),
        habitat      = record.get("habitat"),
        updated_at   = record.get("updated_at"),
        trend_direction = trend,
        trend_text      = trend_labels.get(trend),
    )


@app.get("/api/ecological-index/batch", response_model=list[EcologicalIndexResponse])
def get_ecological_index_batch(
    station_ids: str = Query(..., description="Comma-separated station IDs e.g. S01,S02,S05"),
    index:       str = Query(..., description="Index type"),
    months:      int = Query(12, ge=1, le=60),
):
    """
    Batch endpoint — returns data for multiple stations in one request.
    Used by the map's 'compare stations' feature.
    """
    index = index.upper()
    if index not in VALID_INDICES:
        raise HTTPException(status_code=422, detail=f"Unknown index '{index}'")

    ids      = [sid.strip() for sid in station_ids.split(",") if sid.strip()]
    results  = []

    for sid in ids:
        record = _load_station_file(sid)
        if not record:
            continue
        dates, values = _slice_to_months(record, index, months)
        if not dates:
            continue
        trend = _simple_trend(values)
        results.append(EcologicalIndexResponse(
            station_id   = sid,
            index        = index,
            dates        = dates,
            values       = values,
            station_name = record.get("station_name"),
            habitat      = record.get("habitat"),
            updated_at   = record.get("updated_at"),
            trend_direction = trend,
        ))

    return results


@app.get("/api/stations", response_model=list[StationSummary])
def list_stations():
    """
    List all stations with latest NDVI and HSI values.
    Used by the map to populate markers.
    """
    from gee_pipeline.stations import STATIONS

    summaries = []
    for s in STATIONS:
        record   = _load_station_file(s["id"])
        has_data = bool(record)

        latest_ndvi = None
        latest_hsi  = None
        updated_at  = None

        if record:
            updated_at = record.get("updated_at")
            ndvi_series = record.get("indices", {}).get("NDVI", [])
            hsi_series  = record.get("indices", {}).get("HSI",  [])
            if ndvi_series:
                latest_ndvi = ndvi_series[-1]["value"]
            if hsi_series:
                latest_hsi  = hsi_series[-1]["value"]

        summaries.append(StationSummary(
            id          = s["id"],
            name        = s["name"],
            lat         = s["lat"],
            lon         = s["lon"],
            habitat     = s.get("habitat"),
            has_data    = has_data,
            updated_at  = updated_at,
            latest_ndvi = latest_ndvi,
            latest_hsi  = latest_hsi,
        ))

    return summaries


@app.post("/api/pipeline/trigger", response_model=PipelineStatus)
def trigger_pipeline(
    req: PipelineTriggerRequest,
    background_tasks: BackgroundTasks,
):
    """
    Trigger a GEE extraction run in the background.
    Returns immediately; poll /api/pipeline/status for progress.
    """
    global _pipeline_running
    if _pipeline_running:
        return PipelineStatus(
            running=True,
            last_run=_pipeline_last_run,
            error=None,
        )

    background_tasks.add_task(
        _background_refresh,
        req.station_ids,
        req.start_date,
        req.end_date,
        req.force_refresh,
    )

    return PipelineStatus(running=True, last_run=_pipeline_last_run, error=None)


@app.get("/api/pipeline/status", response_model=PipelineStatus)
def pipeline_status():
    return PipelineStatus(
        running=_pipeline_running,
        last_run=_pipeline_last_run,
        error=_pipeline_error,
    )


# ── Background task ───────────────────────────────────────────────────────────

def _background_refresh(
    station_ids:  Optional[list[str]] = None,
    start_date:   str = "2023-01-01",
    end_date:     str = "2025-12-31",
    force_refresh: bool = False,
):
    """
    Run the GEE extraction pipeline in the background.
    Called via FastAPI BackgroundTasks — runs in a thread pool.
    """
    global _pipeline_running, _pipeline_last_run, _pipeline_error

    _pipeline_running = True
    _pipeline_error   = None
    log.info("background_pipeline_start", station_ids=station_ids)

    try:
        from gee_pipeline.pipeline import run as pipeline_run
        pipeline_run(
            station_ids  = station_ids,
            start_date   = start_date,
            end_date     = end_date,
            output_dir   = DATA_DIR,
            cache_hours  = 0.0 if force_refresh else 6.0,
        )
        _pipeline_last_run = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        log.info("background_pipeline_complete")

    except Exception as exc:
        _pipeline_error = str(exc)
        log.error("background_pipeline_error", error=str(exc))

    finally:
        _pipeline_running = False


# ── Dev server ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
