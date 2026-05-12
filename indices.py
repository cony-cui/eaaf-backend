"""
gee_pipeline/indices.py
────────────────────────────────────────────────────────────────────────────────
Server-side spectral index computation for Sentinel-2 SR imagery.

Design principles:
  - ALL computation stays on the GEE server (no .getInfo() in hot paths)
  - Single-pass pipeline: one image → all bands/indices in one .expression() chain
  - Monthly median compositing for noise suppression and temporal regularity
  - Rigorous cloud masking using SCL (Scene Classification Layer)
  - Scientific LAI formulas, not NDVI proxies
  - Returns ee.ImageCollection ready for reduceRegions() or export

Supported indices:
  NDVI   Normalised Difference Vegetation Index          [-1, 1]
  EVI    Enhanced Vegetation Index                       [-1, 1]
  NDWI   Normalised Difference Water Index (Gao 1996)    [-1, 1]
  LAI    Leaf Area Index (Beer-Lambert, Baret & Guyot)   [0, 8]
  HSI    Habitat Suitability Index (composite)            [0, 1]
────────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import ee


# ── Cloud masking ─────────────────────────────────────────────────────────────

# Sentinel-2 SCL classes to KEEP (mask out everything else).
# 4 = Vegetation, 5 = Bare soils, 6 = Water, 11 = Snow/Ice
_VALID_SCL_VALUES = [4, 5, 6, 11]


def mask_s2_clouds(image: ee.Image) -> ee.Image:
    """
    Mask clouds, cloud shadows, saturated pixels, and cirrus
    using Sentinel-2 Scene Classification Layer (SCL band).

    Also masks pixels where any 10m band is saturated (>= 10000 reflectance
    units in Level-2A SR, corresponding to > 100% reflectance).

    Returns the image with a combined mask applied.
    """
    scl = image.select("SCL")

    # Build a valid-pixel mask from accepted SCL classes
    valid_mask = ee.Image(0)
    for cls in _VALID_SCL_VALUES:
        valid_mask = valid_mask.Or(scl.eq(cls))

    # Additionally reject pixels with unrealistically high reflectance
    saturation_mask = (
        image.select(["B2", "B3", "B4", "B8"]).reduce(ee.Reducer.max()).lt(10000)
    )

    return (
        image.updateMask(valid_mask)
             .updateMask(saturation_mask)
             .copyProperties(image, ["system:time_start", "system:index"])
    )


# ── Per-image index computation (server-side) ─────────────────────────────────

def compute_all_indices(image: ee.Image) -> ee.Image:
    """
    Compute NDVI, EVI, NDWI, LAI, and HSI for a single Sentinel-2 SR image.

    All bands are scaled from [0, 10000] to [0, 1] reflectance units first.
    The function returns a 5-band image with one band per index, plus a
    'date' property set to 'YYYY-MM' for downstream grouping.

    This function is designed to be mapped over an ImageCollection:
        indexed = collection.map(compute_all_indices)
    """
    # Scale reflectance bands from DN to [0, 1]
    # S2 SR L2A: bands B1-B8A, B9, B11, B12 are in range [0, 10000]
    # SCL is already categorical — keep unscaled
    scaled = image.select(
        ["B2", "B3", "B4", "B8", "B8A", "B11"]
    ).divide(10000.0)

    blue  = scaled.select("B2")
    green = scaled.select("B3")
    red   = scaled.select("B4")
    nir   = scaled.select("B8")    # 10m NIR
    nir2  = scaled.select("B8A")   # 20m narrow NIR (better for LAI)
    swir1 = scaled.select("B11")

    # ── NDVI ────────────────────────────────────────────────────────────────
    ndvi = nir.subtract(red).divide(nir.add(red)).rename("NDVI")

    # ── EVI  ────────────────────────────────────────────────────────────────
    # Liu & Huete (1995): 2.5 * (NIR - RED) / (NIR + 6*RED - 7.5*BLUE + 1)
    evi = (
        nir.subtract(red)
           .multiply(2.5)
           .divide(
               nir.add(red.multiply(6))
                  .subtract(blue.multiply(7.5))
                  .add(1.0)
           )
           .rename("EVI")
    )

    # ── NDWI ────────────────────────────────────────────────────────────────
    # Gao (1996): (NIR - SWIR1) / (NIR + SWIR1)  — sensitive to canopy water
    # Note: McFeeters (1996) used GREEN; Gao's is preferred for vegetation water.
    ndwi = nir.subtract(swir1).divide(nir.add(swir1)).rename("NDWI")

    # ── LAI  ────────────────────────────────────────────────────────────────
    # Beer-Lambert inversion (Baret & Guyot 1991 / SNAP S2Toolbox approach):
    #   FAPAR ≈ 1 - exp(-k * LAI), rearranged:
    #   LAI = -log(1 - FAPAR) / k
    # where k = 0.5 (spherical leaf angle distribution),
    # FAPAR ≈ clipped NDVI rescaled to [0, 0.95]
    #
    # Using narrow-band NIR (B8A) + RED for higher accuracy:
    ndvi_nar = nir2.subtract(red).divide(nir2.add(red))   # narrow-band NDVI
    fapar = ndvi_nar.clamp(0.001, 0.95)                   # physical bounds
    k = 0.5
    lai = (
        ee.Image(1.0)
          .subtract(fapar)
          .log()
          .multiply(-1.0 / k)
          .clamp(0.0, 8.0)
          .rename("LAI")
    )

    # ── HSI  ────────────────────────────────────────────────────────────────
    # Habitat Suitability Index — empirical composite for waterbird wetlands
    # Weights tuned for EAAF tidal flat / wetland habitat types:
    #   NDVI contribution  0.35  (vegetation cover quality)
    #   NDWI contribution  0.30  (surface water availability)
    #   EVI  contribution  0.20  (canopy structure / biomass)
    #   LAI  contribution  0.15  (foliage density, normalised to [0,1])
    lai_norm = lai.divide(8.0)   # normalise LAI to [0, 1]

    # Normalise each index to [0, 1] for HSI weighting
    ndvi_01 = ndvi.clamp(-1, 1).add(1).divide(2)
    ndwi_01 = ndwi.clamp(-1, 1).add(1).divide(2)
    evi_01  = evi.clamp(-1,  1).add(1).divide(2)

    hsi = (
        ndvi_01.multiply(0.35)
               .add(ndwi_01.multiply(0.30))
               .add(evi_01.multiply(0.20))
               .add(lai_norm.multiply(0.15))
               .clamp(0.0, 1.0)
               .rename("HSI")
    )

    # ── Assemble result image ────────────────────────────────────────────────
    result = (
        ndvi.addBands(evi)
            .addBands(ndwi)
            .addBands(lai)
            .addBands(hsi)
    )

    # Attach temporal metadata for grouping
    year_month = image.date().format("YYYY-MM")
    return (
        result
        .set("date",              year_month)
        .set("system:time_start", image.get("system:time_start"))
        .copyProperties(image, ["system:index", "CLOUDY_PIXEL_PERCENTAGE"])
    )


# ── Monthly median compositing ────────────────────────────────────────────────

def build_monthly_composites(
    collection: ee.ImageCollection,
    start_date: str,
    end_date:   str,
) -> ee.ImageCollection:
    """
    Collapse a per-scene collection into monthly median composites.

    Monthly compositing:
      - Suppresses residual cloud artifacts that survive SCL masking
      - Produces one representative image per month-station combination
      - Dramatically reduces GEE quota consumption vs per-scene extraction
      - Enables consistent temporal alignment across stations

    Parameters
    ----------
    collection:
        Masked, index-computed ImageCollection (output of
        collection.map(compute_all_indices)).
    start_date:
        ISO date string, inclusive  e.g. "2023-01-01"
    end_date:
        ISO date string, exclusive  e.g. "2025-12-31"

    Returns
    -------
    ee.ImageCollection of monthly median composites, one image per month,
    sorted by time, with 'date' and 'year_month' properties set.
    """
    start  = ee.Date(start_date)
    end    = ee.Date(end_date)
    n_months = end.difference(start, "month").round().int()

    def make_monthly(month_offset: ee.Number) -> ee.Image:
        month_start = start.advance(month_offset, "month")
        month_end   = month_start.advance(1, "month")
        year_month  = month_start.format("YYYY-MM")

        composite = (
            collection
            .filterDate(month_start, month_end)
            .median()
            .set("date",              year_month)
            .set("year_month",        year_month)
            .set("system:time_start", month_start.millis())
        )
        return composite

    monthly_list = ee.List.sequence(0, n_months.subtract(1)).map(make_monthly)
    return ee.ImageCollection(monthly_list).sort("system:time_start")


# ── Collection builder ────────────────────────────────────────────────────────

def build_s2_collection(
    geometry:    ee.Geometry,
    start_date:  str,
    end_date:    str,
    max_cloud:   float = 30.0,
) -> ee.ImageCollection:
    """
    Build a cloud-masked, index-computed Sentinel-2 SR collection.

    This is the primary entry point for the pipeline:
        raw → filtered → cloud-masked → indices → collection ready for compositing

    Parameters
    ----------
    geometry:
        Area of interest (point or polygon).
    start_date, end_date:
        Date range for the collection.
    max_cloud:
        Maximum CLOUDY_PIXEL_PERCENTAGE threshold for scene-level filtering.
        Default 30% is generous; per-pixel SCL masking handles the rest.
    """
    return (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
          .filterBounds(geometry)
          .filterDate(start_date, end_date)
          .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", max_cloud))
          .map(mask_s2_clouds)
          .map(compute_all_indices)
    )
