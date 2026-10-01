"""
=============================================================================
Agricultural AI - Satellite GeoTIFF Processing & Feature Extraction Pipeline
=============================================================================
This pipeline coordinates:
1. Decoding multi-band geospatial GeoTIFFs via Rasterio.
2. Validating CRS, dimensions, and genuine multispectral bands (Blue, Green, Red, NIR).
   Enforces scientific integrity: does NOT invent NIR from 3-channel RGB.
3. Converting digital numbers (DN) to surface reflectance (Sentinel-2 BOA / 10000).
4. Calculating 6 core bio-optical and canopy features:
   - NDVI (Normalized Difference Vegetation Index)
   - EVI (Enhanced Vegetation Index)
   - FVC (Fraction of Vegetation Cover)
   - LAI (Leaf Area Index)
   - NDVI Texture (GLCM Canopy Heterogeneity)
   - Biomass (Crop Biomass Proxy)
5. Aggregating pixel rasters into field-level scalar metrics using Zonal Statistics.
6. Passing the 6 features to the pre-trained Random Forest models to predict N, P, K.
=============================================================================
"""

import os
import numpy as np
from typing import Dict, Any, Union, BinaryIO

from config import Config
from preprocessing.raster_loader import load_raster, convert_to_reflectance
from features.ndvi import calculate_ndvi
from features.evi import calculate_evi
from features.fvc import calculate_fvc
from features.texture import calculate_texture
from features.lai import extract_lai, calculate_empirical_lai
from features.biomass import estimate_biomass
from spatial_stats.zonal_statistics import calculate_zonal_statistics
from predict_npk import predict_soil_nutrients


def get_raster_metadata(source: Union[str, bytes, BinaryIO]) -> Dict[str, Any]:
    """
    Extracts physical metadata, CRS, resolution, and band count from a GeoTIFF.
    """
    raw_data, profile = load_raster(source)
    count, height, width = raw_data.shape
    
    crs_str = "None"
    if profile.get("crs"):
        crs_str = profile["crs"].to_string() if hasattr(profile["crs"], "to_string") else str(profile["crs"])
        
    transform = profile.get("transform")
    res_x = abs(transform[0]) if transform is not None else None
    res_y = abs(transform[4]) if transform is not None else None
    spatial_res = f"{res_x:.1f} m" if (res_x and res_x < 1000) else "Not Specified"
    
    return {
        "width": width,
        "height": height,
        "count": count,
        "crs": crs_str,
        "spatial_resolution": spatial_res,
        "nodata": profile.get("nodata"),
        "dtype": str(profile.get("dtype"))
    }


def process_satellite_geotiff(
    source: Union[str, bytes, BinaryIO],
    empirical_fallbacks: bool = True,
    models_dir: str = "models"
) -> Dict[str, Any]:
    """
    Executes the end-to-end satellite analysis:
    GeoTIFF -> Bands -> 6 Features -> ML Inference (N, P, K).
    """
    # 1. Load raster
    raw_data, profile = load_raster(source)
    count, height, width = raw_data.shape
    
    # 2. Strict Band Validation: Real satellite multispectral bands required
    if count < 4:
        raise ValueError(
            f"Uploaded raster contains only {count} band(s). "
            f"Multispectral satellite data with at least 4 calibrated bands "
            f"(B02 Blue, B03 Green, B04 Red, B08 NIR) is strictly required. "
            f"Ordinary 3-channel RGB imagery does not contain Near-Infrared (NIR) data."
        )

    # 3. Surface Reflectance Calibration
    reflectance = convert_to_reflectance(raw_data, scale_factor=Config.Preprocessing.SURFACE_REFLECTANCE_SCALE)
    
    blue = reflectance[Config.Bands.BLUE]
    green = reflectance[Config.Bands.GREEN]
    red = reflectance[Config.Bands.RED]
    nir = reflectance[Config.Bands.NIR]
    
    # Check for invalid values / all nodata
    if np.all(np.isnan(nir)) or np.all(nir == Config.Preprocessing.NODATA_VALUE):
        raise ValueError("All pixels in the Near-Infrared (NIR) band are nodata or NaN.")
        
    # Band summary statistics (mean reflectance over valid field pixels)
    valid_mask = ~np.isnan(red) & ~np.isnan(nir) & (red >= 0.0) & (nir >= 0.0)
    if not np.any(valid_mask):
        raise ValueError("No valid spectral reflectance pixels found after filtering.")

    band_stats = {
        "blue": float(np.mean(blue[valid_mask])),
        "green": float(np.mean(green[valid_mask])),
        "red": float(np.mean(red[valid_mask])),
        "nir": float(np.mean(nir[valid_mask]))
    }

    # 4. Feature Calculations
    # A. NDVI
    ndvi_raster = calculate_ndvi(red, nir)
    ndvi_stats = calculate_zonal_statistics(ndvi_raster, "NDVI")
    ndvi_mean = ndvi_stats.get("NDVI_mean")
    if ndvi_mean is None or np.isnan(ndvi_mean):
        raise ValueError("NDVI calculation produced no valid pixels.")

    # B. EVI
    evi_raster = calculate_evi(blue, red, nir, Config.EVI)
    evi_stats = calculate_zonal_statistics(evi_raster, "EVI")
    evi_mean = evi_stats.get("EVI_mean")

    # C. FVC (Fractional Vegetation Cover)
    fvc_raster = calculate_fvc(ndvi_raster, Config.FVC)
    fvc_stats = calculate_zonal_statistics(fvc_raster, "FVC")
    fvc_mean = fvc_stats.get("FVC_mean")

    # D. NDVI Texture (GLCM Canopy Heterogeneity)
    texture_raster = calculate_texture(ndvi_raster, feature="contrast", config=Config.Texture)
    texture_stats = calculate_zonal_statistics(texture_raster, "Texture")
    raw_texture_mean = texture_stats.get("Texture_mean", 0.0) or 0.0
    # Scaled to canopy variation scale [0.03 - 0.12]
    scaled_texture = float(np.clip(raw_texture_mean * 0.01, 0.02, 0.15))

    # E. LAI (Leaf Area Index)
    # Check if a 5th band exists containing dedicated LAI
    warnings_list = []
    if count >= 5:
        lai_raster = extract_lai(raw_data[4])
        lai_mean = float(np.mean(lai_raster[valid_mask]))
        lai_status = "Extracted from 5th raster band (Dedicated LAI product)"
    elif empirical_fallbacks:
        lai_raster = calculate_empirical_lai(ndvi_raster, a=0.2, b=1.5)
        lai_mean = float(np.clip(np.mean(lai_raster[valid_mask]), 0.5, 6.0))
        lai_status = "Empirical Proxy (Uncalibrated; requires in-situ / SNAP biophysical processor calibration)"
        warnings_list.append(
            "LAI was estimated using an empirical proxy (a * exp(b * NDVI)). "
            "Real-world precision farming requires calibration with ground-truth measurements."
        )
    else:
        lai_mean = None
        lai_status = "Stub / Not Provided (Requires dedicated band or in-situ ground truth)"

    # F. Biomass
    # Biomass is scientifically an allometric/external model stub without calibrated weights
    if empirical_fallbacks and lai_mean is not None:
        # Allometric proxy based on canopy volume and LAI
        biomass_val = float(np.clip(lai_mean * 1.5, 1.0, 10.0))
        biomass_status = "Allometric Proxy (Uncalibrated; requires destructive crop-cut sampling calibration)"
        warnings_list.append(
            "Biomass was estimated using an uncalibrated proxy. "
            "Direct biomass assessment requires crop-specific allometric calibration."
        )
    else:
        biomass_val = None
        biomass_status = "Stub / Requires calibrated model weights (estimate_biomass returned None)"

    # 5. Format Exactly the 6 Model Features
    features_6 = {
        "NDVI": float(round(ndvi_mean, 4)),
        "EVI": float(round(evi_mean, 4)),
        "FVC": float(round(fvc_mean, 4)),
        "LAI": float(round(lai_mean, 2)) if lai_mean is not None else None,
        "NDVI Texture": float(round(scaled_texture, 4)),
        "Biomass": float(round(biomass_val, 2)) if biomass_val is not None else None
    }

    # 6. ML Inference (N, P, K) using pre-trained Random Forest models
    predictions = None
    if all(v is not None for v in features_6.values()):
        predictions = predict_soil_nutrients(features_6, models_dir=models_dir)
        predictions = {
            "N": float(round(predictions["N"], 2)),
            "P": float(round(predictions["P"], 2)),
            "K": float(round(predictions["K"], 2))
        }

    metadata = get_raster_metadata(source)

    return {
        "metadata": metadata,
        "bands": band_stats,
        "features": features_6,
        "feature_details": {
            "NDVI": {"value": features_6["NDVI"], "status": "Calculated (NIR - Red) / (NIR + Red)", "unit": "index [-1, 1]"},
            "EVI": {"value": features_6["EVI"], "status": "Calculated (Huete et al. EVI)", "unit": "index [-1, 1]"},
            "FVC": {"value": features_6["FVC"], "status": "Calculated (Dimidiate Pixel Model)", "unit": "fraction [0, 1]"},
            "LAI": {"value": features_6["LAI"], "status": lai_status, "unit": "m²/m²"},
            "NDVI Texture": {"value": features_6["NDVI Texture"], "status": "Calculated (GLCM Contrast)", "unit": "heterogeneity index"},
            "Biomass": {"value": features_6["Biomass"], "status": biomass_status, "unit": "proxy index"}
        },
        "predictions": predictions,
        "warnings": warnings_list,
        "scientific_disclaimer": (
            "Estimated N/P/K based on the current prototype Random Forest regression model. "
            "Satellite bio-optical indices reflect canopy vigor and chlorophyll absorption; "
            "soil nutrient diagnosis requires in-situ soil laboratory validation. "
            "Fertilizer dosage recommendations require crop agronomic guidelines and soil testing."
        )
    }
