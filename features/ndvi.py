"""
Normalized Difference Vegetation Index (NDVI) Calculation Module.
NDVI = (NIR - Red) / (NIR + Red)
Handles division by zero, NaN values, and clamps output to [-1.0, 1.0].
"""
import numpy as np

def calculate_ndvi(red_band: np.ndarray, nir_band: np.ndarray) -> np.ndarray:
    """
    Calculate NDVI using Red (e.g. Sentinel-2 B04) and NIR (e.g. Sentinel-2 B08) bands.
    
    Args:
        red_band (np.ndarray): 2D array of Red band reflectance.
        nir_band (np.ndarray): 2D array of NIR band reflectance.
        
    Returns:
        np.ndarray: 2D array of NDVI values in [-1.0, 1.0].
    """
    red = red_band.astype(np.float32)
    nir = nir_band.astype(np.float32)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        denominator = nir + red
        # Safe division: if denominator is 0, assign 0.0
        ndvi = np.where(denominator != 0, (nir - red) / denominator, 0.0)
        
    # Standard theoretical bounds for NDVI
    ndvi = np.clip(ndvi, -1.0, 1.0)
    ndvi = np.nan_to_num(ndvi, nan=0.0)
    return ndvi
