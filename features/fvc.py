"""
Fractional Vegetation Cover (FVC) Calculation Module.
Uses the validated dimidiate pixel model:
FVC = ((NDVI - NDVI_soil) / (NDVI_veg - NDVI_soil)) ** 2
NDVI < NDVI_soil -> FVC = 0.0
NDVI > NDVI_veg -> FVC = 1.0
Clamped to [0.0, 1.0].
"""
import numpy as np
from config import Config

def calculate_fvc(ndvi_array: np.ndarray, config=Config.FVC) -> np.ndarray:
    """
    Calculate FVC from an NDVI raster array.
    
    Args:
        ndvi_array (np.ndarray): 2D array of NDVI values.
        config: FVC thresholds configuration.
        
    Returns:
        np.ndarray: 2D array of FVC values in [0.0, 1.0].
    """
    ndvi = ndvi_array.astype(np.float32)
    denominator = config.NDVI_VEG - config.NDVI_SOIL
    if denominator == 0:
        return np.zeros_like(ndvi)
        
    with np.errstate(invalid='ignore'):
        fvc = ((ndvi - config.NDVI_SOIL) / denominator) ** 2
        
    fvc = np.clip(fvc, 0.0, 1.0)
    fvc = np.where(ndvi < config.NDVI_SOIL, 0.0, fvc)
    fvc = np.where(ndvi > config.NDVI_VEG, 1.0, fvc)
    fvc = np.nan_to_num(fvc, nan=0.0)
    return fvc
