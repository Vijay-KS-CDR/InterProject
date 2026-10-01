"""
Enhanced Vegetation Index (EVI) Calculation Module.
Standard equation (Huete et al., 2002):
EVI = G * (NIR - Red) / (NIR + C1 * Red - C2 * Blue + L)
Uses validated coefficients G=2.5, C1=6.0, C2=7.5, L=1.0.
"""
import numpy as np
from config import Config

def calculate_evi(blue_band: np.ndarray, red_band: np.ndarray, nir_band: np.ndarray, config=Config.EVI) -> np.ndarray:
    """
    Calculate EVI using Blue (B02), Red (B04), and NIR (B08) reflectance bands.
    
    Args:
        blue_band (np.ndarray): 2D array of Blue band reflectance.
        red_band (np.ndarray): 2D array of Red band reflectance.
        nir_band (np.ndarray): 2D array of NIR band reflectance.
        config: EVI coefficient configuration class.
        
    Returns:
        np.ndarray: 2D array of EVI values clipped to [-1.0, 1.0].
    """
    blue = blue_band.astype(np.float32)
    red = red_band.astype(np.float32)
    nir = nir_band.astype(np.float32)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        denominator = nir + (config.C1 * red) - (config.C2 * blue) + config.L
        evi = np.where(denominator != 0, config.G * (nir - red) / denominator, 0.0)
        
    evi = np.clip(evi, -1.0, 1.0)
    evi = np.nan_to_num(evi, nan=0.0)
    return evi
