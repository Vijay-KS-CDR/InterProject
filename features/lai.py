"""
Leaf Area Index (LAI) Module.

Scientific Note:
LAI represents total one-sided green leaf area per unit ground surface area (m2/m2).
Optical satellite bands (e.g. Sentinel-2) do not measure LAI directly. Operational
LAI retrieval requires radiative transfer model inversion (e.g., PROSAIL via the
ESA SNAP Biophysical Processor) or an empirical relationship calibrated with
in-situ field measurements.

This module preserves the friend's implementation:
1. extract_lai: Pass-through for a dedicated LAI raster product/band.
2. calculate_empirical_lai: Prototype empirical proxy (a * exp(b * NDVI)).
   NOTE: Marked explicitly as uncalibrated prototype requiring ground-truth calibration.
"""
import numpy as np

def extract_lai(lai_raster: np.ndarray) -> np.ndarray:
    """
    Extract LAI from a dedicated LAI raster band or pre-computed product.
    
    Args:
        lai_raster (np.ndarray): Pre-computed LAI raster array.
        
    Returns:
        np.ndarray or None: Validated LAI array clipped to [0.0, 10.0].
    """
    if lai_raster is None:
        return None
    return np.clip(lai_raster.astype(np.float32), 0.0, 10.0)

def calculate_empirical_lai(ndvi_raster: np.ndarray, a: float = 0.2, b: float = 1.5) -> np.ndarray:
    """
    Prototype empirical relationship: LAI = a * exp(b * NDVI).
    
    IMPORTANT SCIENTIFIC DISCLAIMER:
    Coefficients 'a' and 'b' are crop- and region-specific and MUST be calibrated
    against in-situ ground truth or SNAP biophysical processor outputs before operational use.
    """
    if ndvi_raster is None:
        return None
    with np.errstate(over='ignore', invalid='ignore'):
        empirical_lai = a * np.exp(b * np.clip(ndvi_raster, 0.0, 1.0))
    return np.clip(empirical_lai, 0.0, 10.0)
