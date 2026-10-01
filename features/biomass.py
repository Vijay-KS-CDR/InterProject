"""
Biomass Estimation Module.

Scientific Note:
Above-ground crop biomass (e.g., tons/ha or g/m2) cannot be directly read from optical
satellite reflectance bands without a validated allometric model, destructive sampling calibration,
or active sensor fusion (e.g., Synthetic Aperture Radar Sentinel-1 or LiDAR GEDI).

This module preserves the friend's implementation:
Returns None unless validated model weights / calibration parameters are explicitly provided.
"""
from typing import Optional, Dict
import numpy as np

def estimate_biomass(remote_sensing_features: dict, model_weights: Optional[Dict] = None) -> Optional[np.ndarray]:
    """
    Estimate biomass using an external validated relationship.
    
    Args:
        remote_sensing_features (dict): Rasters or scalar metrics (NDVI, EVI, LAI, etc.).
        model_weights (dict, optional): Calibrated coefficients for a specific crop and phenology stage.
        
    Returns:
        np.ndarray or None: Estimated biomass raster/scalar if weights are provided; otherwise None.
    """
    if model_weights is None:
        # Strictly preserves the existing design: biomass is an uncalibrated stub
        # requiring ground-truth crop-cut calibration data.
        return None
        
    # Example interface for a calibrated allometric model if weights provided:
    # Biomass = w0 + w1 * NDVI + w2 * EVI ...
    w0 = model_weights.get('intercept', 0.0)
    w_ndvi = model_weights.get('ndvi', 0.0)
    
    if 'ndvi' in remote_sensing_features and remote_sensing_features['ndvi'] is not None:
        return np.maximum(0.0, w0 + w_ndvi * remote_sensing_features['ndvi'])
        
    return None
