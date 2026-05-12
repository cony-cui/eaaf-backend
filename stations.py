"""
gee_pipeline/stations.py
────────────────────────────────────────────────────────────────────────────────
Station registry for the EAAF Bird Sentinel pipeline.

Responsibilities:
  - Define and validate all 20 EAAF monitoring stations
  - Build GEE geometry objects (point + buffer) for each station
  - Expose a FeatureCollection for batch reduceRegions() operations
  - Support future scaling to 200+ stations via JSON config file

Buffer strategy:
  Each station uses a circular buffer rather than a single point because:
    1. A single pixel has high noise sensitivity from sub-pixel co-registration
    2. Waterbird habitat assessment requires a landscape-scale reading
    3. 500m radius captures the functionally relevant foraging/roosting zone
       for most EAAF shorebird and waterfowl species

The buffer radius is configurable per-station to handle:
  - Small isolated wetlands (e.g. Caohai: 200m)
  - Large wetland complexes (e.g. Poyang Lake: 1000m)
────────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import ee
import structlog

log = structlog.get_logger(__name__)


# ── Station definitions ────────────────────────────────────────────────────────

# Canonical station registry — extend here or load from JSON config.
# Fields:
#   id        station identifier (matches frontend / FastAPI)
#   name      Chinese place name
#   lat/lon   GCJ-02 coordinates (backend guarantee; GEE uses WGS-84 internally
#             but the ~50m GCJ→WGS offset is negligible for 500m buffers)
#   buffer_m  spatial aggregation radius in metres
#   habitat   EAAF habitat type label (for metadata / downstream analysis)

STATIONS: list[dict] = [
    {"id": "S01", "name": "崇明东滩",       "lat": 31.5115, "lon": 121.9645, "buffer_m": 500,  "habitat": "tidal_flat"},
    {"id": "S02", "name": "米埔内后海湾",   "lat": 22.4965, "lon": 114.0168, "buffer_m": 500,  "habitat": "mangrove_mudflat"},
    {"id": "S03", "name": "双台河口",       "lat": 40.8800, "lon": 121.6600, "buffer_m": 500,  "habitat": "estuarine_wetland"},
    {"id": "S04", "name": "盐城湿地",       "lat": 33.5000, "lon": 120.5000, "buffer_m": 800,  "habitat": "coastal_wetland"},
    {"id": "S05", "name": "黄河三角洲",     "lat": 37.7667, "lon": 119.1667, "buffer_m": 1000, "habitat": "delta_wetland"},
    {"id": "S06", "name": "鄱阳湖",         "lat": 29.0000, "lon": 116.5000, "buffer_m": 1000, "habitat": "freshwater_lake"},
    {"id": "S07", "name": "兴凯湖",         "lat": 45.2500, "lon": 132.5000, "buffer_m": 800,  "habitat": "freshwater_lake"},
    {"id": "S08", "name": "三江平原",       "lat": 47.0000, "lon": 133.0000, "buffer_m": 1000, "habitat": "marsh_wetland"},
    {"id": "S09", "name": "鸭绿江口",       "lat": 39.8667, "lon": 124.2167, "buffer_m": 500,  "habitat": "estuarine_wetland"},
    {"id": "S10", "name": "达赉湖",         "lat": 48.9500, "lon": 117.5000, "buffer_m": 1000, "habitat": "saline_lake"},
    {"id": "S11", "name": "草海",           "lat": 26.8333, "lon": 104.2500, "buffer_m": 200,  "habitat": "alpine_wetland"},
    {"id": "S12", "name": "升金湖",         "lat": 30.3333, "lon": 117.0000, "buffer_m": 500,  "habitat": "freshwater_lake"},
    {"id": "S13", "name": "向海",           "lat": 44.9500, "lon": 122.8500, "buffer_m": 800,  "habitat": "reed_marsh"},
    {"id": "S14", "name": "扎龙",           "lat": 47.1667, "lon": 124.2500, "buffer_m": 1000, "habitat": "reed_marsh"},
    {"id": "S15", "name": "安庆沿江湿地",   "lat": 30.5000, "lon": 117.0000, "buffer_m": 600,  "habitat": "riverine_wetland"},
    {"id": "S16", "name": "大山包",         "lat": 27.3333, "lon": 103.2500, "buffer_m": 300,  "habitat": "alpine_meadow"},
    {"id": "S17", "name": "衡水湖",         "lat": 37.5833, "lon": 115.6667, "buffer_m": 500,  "habitat": "freshwater_lake"},
    {"id": "S18", "name": "南大港",         "lat": 38.5000, "lon": 117.4167, "buffer_m": 500,  "habitat": "coastal_wetland"},
    {"id": "S19", "name": "南矶山",         "lat": 28.9167, "lon": 116.4167, "buffer_m": 500,  "habitat": "freshwater_lake"},
    {"id": "S20", "name": "荣成天鹅湖",     "lat": 37.1667, "lon": 122.4167, "buffer_m": 400,  "habitat": "coastal_lagoon"},
]


# ── Public helpers ────────────────────────────────────────────────────────────

def load_stations(path: Optional[Path] = None) -> list[dict]:
    """
    Return station list from the embedded registry or an external JSON file.

    External JSON format:
        [{"id": "S21", "name": "...", "lat": ..., "lon": ...,
          "buffer_m": 500, "habitat": "..."}, ...]

    Parameters
    ----------
    path : optional path to an external JSON overriding the built-in registry
    """
    if path and path.exists():
        with path.open(encoding="utf-8") as fh:
            stations = json.load(fh)
        log.info("stations_loaded_from_file", path=str(path), count=len(stations))
        return stations

    log.info("stations_loaded_from_registry", count=len(STATIONS))
    return STATIONS


def validate_stations(stations: list[dict]) -> list[dict]:
    """
    Validate station records and drop invalid entries with a warning.

    Checks:
      - Required fields present
      - Latitude in [-90, 90]
      - Longitude in [-180, 180]
      - buffer_m positive integer
    """
    valid = []
    for s in stations:
        sid = s.get("id", "<unknown>")
        try:
            lat = float(s["lat"])
            lon = float(s["lon"])
            buf = int(s.get("buffer_m", 500))
            assert -90  <= lat <= 90,   f"lat {lat} out of range"
            assert -180 <= lon <= 180,  f"lon {lon} out of range"
            assert buf > 0,             f"buffer_m must be positive"
            valid.append({**s, "lat": lat, "lon": lon, "buffer_m": buf})
        except (KeyError, ValueError, AssertionError) as exc:
            log.warning("station_validation_failed", station_id=sid, reason=str(exc))

    dropped = len(stations) - len(valid)
    if dropped:
        log.warning("stations_dropped", count=dropped)
    return valid


def station_to_feature(station: dict) -> ee.Feature:
    """
    Convert a station dict to an ee.Feature with a buffered geometry.

    The buffer is applied in metres using the geodesic projection,
    which avoids area distortion at high latitudes (important for
    stations in northeast China above 45°N).
    """
    point = ee.Geometry.Point([station["lon"], station["lat"]])
    buffer = point.buffer(station["buffer_m"], maxError=1)   # 1m error tolerance

    return ee.Feature(buffer, {
        "station_id": station["id"],
        "station_name": station.get("name", ""),
        "habitat": station.get("habitat", ""),
        "buffer_m": station["buffer_m"],
        "lat": station["lat"],
        "lon": station["lon"],
    })


def build_feature_collection(stations: list[dict]) -> ee.FeatureCollection:
    """
    Build a GEE FeatureCollection from the validated station list.

    This FeatureCollection is passed to reduceRegions() which computes
    zonal statistics for all stations in a single server-side operation —
    replacing the O(n × months) per-image getInfo() loop in the original code.
    """
    features = [station_to_feature(s) for s in stations]
    fc = ee.FeatureCollection(features)
    log.info("feature_collection_built", station_count=len(features))
    return fc


def station_index_map(stations: list[dict]) -> dict[str, dict]:
    """Return a dict keyed by station_id for O(1) lookup by downstream code."""
    return {s["id"]: s for s in stations}
