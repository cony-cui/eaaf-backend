"""
tests/test_indices.py
Unit tests for gee_pipeline.indices — run without a live GEE connection
by mocking ee objects.
"""

import pytest
import sys
from unittest.mock import MagicMock, patch


# ── Mock the entire ee module so tests run without GEE credentials ────────────

class MockImage:
    def __init__(self, name="mock"):
        self._name = name
        self._bands = {}

    def select(self, band):
        img = MockImage(band)
        return img

    def divide(self, v):      return self
    def subtract(self, v):    return self
    def add(self, v):         return self
    def multiply(self, v):    return self
    def clamp(self, lo, hi):  return self
    def log(self):            return self
    def rename(self, n):      self._name = n; return self
    def addBands(self, other): return self
    def updateMask(self, m):  return self
    def copyProperties(self, *a): return self
    def set(self, *a):        return self
    def eq(self, v):          return self
    def Or(self, other):      return self
    def lt(self, v):          return self
    def date(self):           return MockDate()
    def get(self, k):         return MockString(k)


class MockDate:
    def format(self, fmt): return MockString("2024-06")
    def millis(self):      return MockNumber(1700000000000)


class MockString:
    def __init__(self, v=""):
        self._v = v
    def getInfo(self):
        return self._v


class MockNumber:
    def __init__(self, v=0):
        self._v = v
    def getInfo(self):
        return self._v


class MockReducer:
    @staticmethod
    def mean():    return "mean_reducer"
    @staticmethod
    def max():     return MockImage("max_reducer")


class MockFilter:
    @staticmethod
    def lt(prop, val): return f"filter_lt_{prop}_{val}"


class MockImageCollection:
    def __init__(self, name=""):
        self._name = name
    def filterBounds(self, g): return self
    def filterDate(self, s, e): return self
    def filter(self, f):       return self
    def map(self, fn):         return self
    def median(self):          return MockImage()
    def sort(self, k):         return self
    def size(self):            return MockNumber(10)
    def toList(self, n):       return MockList()


class MockList:
    def get(self, i): return MockImage(f"image_{i}")

    @staticmethod
    def sequence(start, end):
        return MockList()

    def map(self, fn): return self


class MockGeometry:
    def __init__(self):
        pass
    def buffer(self, r, maxError=1): return self
    def bounds(self, maxError=1):    return self


class MockFeature:
    def __init__(self, geom=None, props=None):
        self.geometry_obj = geom or MockGeometry()
        self.props = props or {}

    def geometry(self):
        return self.geometry_obj


class MockFeatureCollection:
    def __init__(self, features=None):
        self.features = features or []

    def geometry(self):
        return MockGeometry()

    def reduceColumns(self, *a): return {}


# Patch ee before importing our module
ee_mock = MagicMock()
ee_mock.Image = MockImage
ee_mock.ImageCollection = MockImageCollection
ee_mock.Geometry.Point = lambda coords: MockGeometry()
ee_mock.Feature = MockFeature
ee_mock.FeatureCollection = lambda features: MockFeatureCollection(features)
ee_mock.Reducer = MockReducer
ee_mock.Filter = MockFilter
ee_mock.List = MockList
ee_mock.Number = MockNumber
ee_mock.Date = lambda s: MockDate()

sys.modules["ee"] = ee_mock

from gee_pipeline import indices   # noqa: E402
from gee_pipeline import stations  # noqa: E402


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestComputeAllIndices:
    def test_returns_image_with_set_date(self):
        img = MockImage()
        result = indices.compute_all_indices(img)
        # Should not raise; returns a MockImage (chained)
        assert result is not None

    def test_mask_s2_clouds_does_not_raise(self):
        img = MockImage()
        result = indices.mask_s2_clouds(img)
        assert result is not None


class TestBuildMonthlyComposites:
    def test_returns_image_collection(self):
        collection = MockImageCollection()
        result = indices.build_monthly_composites(collection, "2024-01-01", "2024-03-31")
        assert result is not None


class TestStationValidation:
    def test_valid_station_passes(self):
        valid = [
            {"id": "S01", "name": "Test", "lat": 31.5, "lon": 121.9, "buffer_m": 500}
        ]
        result = stations.validate_stations(valid)
        assert len(result) == 1

    def test_invalid_lat_dropped(self):
        invalid = [
            {"id": "S01", "name": "Bad", "lat": 200, "lon": 100, "buffer_m": 500}
        ]
        result = stations.validate_stations(invalid)
        assert len(result) == 0

    def test_invalid_lon_dropped(self):
        invalid = [
            {"id": "S02", "name": "Bad", "lat": 30, "lon": -200, "buffer_m": 500}
        ]
        result = stations.validate_stations(invalid)
        assert len(result) == 0

    def test_missing_lat_dropped(self):
        invalid = [{"id": "S03", "name": "NoLat", "lon": 100, "buffer_m": 500}]
        result = stations.validate_stations(invalid)
        assert len(result) == 0

    def test_all_20_stations_pass(self):
        result = stations.validate_stations(stations.STATIONS)
        assert len(result) == 20

    def test_station_index_map(self):
        slist  = stations.validate_stations(stations.STATIONS)
        smap   = stations.station_index_map(slist)
        assert "S01" in smap
        assert smap["S01"]["name"] == "崇明东滩"

    def test_build_feature_collection_does_not_raise(self):
        slist  = stations.validate_stations(stations.STATIONS[:3])
        fc     = stations.build_feature_collection(slist)
        assert fc is not None


class TestExportCacheHelpers:
    def test_is_cache_fresh_missing_file(self, tmp_path):
        from gee_pipeline.export import is_cache_fresh
        assert is_cache_fresh(tmp_path, "S99") is False

    def test_is_cache_fresh_fresh_file(self, tmp_path):
        import time
        from gee_pipeline.export import is_cache_fresh
        p = tmp_path / "S01.json"
        p.write_text("{}")
        assert is_cache_fresh(tmp_path, "S01", max_age_h=6) is True

    def test_load_cached_result_missing(self, tmp_path):
        from gee_pipeline.export import load_cached_result
        assert load_cached_result(tmp_path, "S99") is None

    def test_load_cached_result_valid(self, tmp_path):
        import json
        from gee_pipeline.export import load_cached_result
        data = {"station_id": "S01", "indices": {}}
        (tmp_path / "S01.json").write_text(json.dumps(data))
        result = load_cached_result(tmp_path, "S01")
        assert result["station_id"] == "S01"

    def test_write_station_json(self, tmp_path):
        from gee_pipeline.export import _write_station_json
        import json
        record = {"station_id": "S05", "indices": {"NDVI": [{"date": "2024-01", "value": 0.5}]}}
        _write_station_json(tmp_path, "S05", record)
        written = json.loads((tmp_path / "S05.json").read_text())
        assert written["station_id"] == "S05"
        assert written["indices"]["NDVI"][0]["value"] == 0.5
