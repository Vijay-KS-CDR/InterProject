"""
Zonal Statistics Module.
Aggregates 2D raster pixel features into field-level summary statistics (mean, median, std, min, max).
Discards NaN and nodata values.
"""
import numpy as np
from typing import Dict, Any

def calculate_zonal_statistics(raster_array: np.ndarray, feature_name: str, nodata: float = -9999.0) -> Dict[str, Any]:
    """
    Calculate field-level summary statistics for a feature raster.
    
    Args:
        raster_array (np.ndarray): 2D array of the feature pixels.
        feature_name (str): Identifier name for feature (e.g., 'NDVI').
        nodata (float): Nodata marker to ignore.
        
    Returns:
        dict: Summary statistics including mean, median, min, max, std.
    """
    if raster_array is None:
        return {
            f"{feature_name}_mean": None,
            f"{feature_name}_median": None,
            f"{feature_name}_min": None,
            f"{feature_name}_max": None,
            f"{feature_name}_std": None,
        }
        
    flat = raster_array.flatten()
    # Mask out NaNs, Infs, and nodata
    valid_mask = ~np.isnan(flat) & ~np.isinf(flat) & (flat != nodata)
    valid_pixels = flat[valid_mask]
    
    if len(valid_pixels) == 0:
        return {
            f"{feature_name}_mean": None,
            f"{feature_name}_median": None,
            f"{feature_name}_min": None,
            f"{feature_name}_max": None,
            f"{feature_name}_std": None,
        }
        
    return {
        f"{feature_name}_mean": float(np.mean(valid_pixels)),
        f"{feature_name}_median": float(np.median(valid_pixels)),
        f"{feature_name}_min": float(np.min(valid_pixels)),
        f"{feature_name}_max": float(np.max(valid_pixels)),
        f"{feature_name}_std": float(np.std(valid_pixels)),
    }
