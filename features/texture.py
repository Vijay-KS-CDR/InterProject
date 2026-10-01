"""
NDVI Spatial Texture Calculation Module using Gray-Level Co-occurrence Matrix (GLCM).
Calculates local canopy heterogeneity (contrast) over a sliding window / global field area.
"""
import numpy as np
from skimage.feature import graycomatrix, graycoprops
from config import Config

def calculate_texture(image_array: np.ndarray, feature: str = 'contrast', config=Config.Texture) -> np.ndarray:
    """
    Calculate spatial texture features from a 2D raster (typically NDVI) using GLCM.
    
    Args:
        image_array (np.ndarray): 2D array of continuous feature values (e.g., NDVI).
        feature (str): GLCM property to extract ('contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation').
        config: Texture configuration dataclass.
        
    Returns:
        np.ndarray: 2D texture raster matching input dimensions.
    """
    if image_array is None or image_array.size == 0:
        return np.zeros_like(image_array, dtype=np.float32)
        
    # Handle NaNs
    valid_mask = ~np.isnan(image_array)
    if not np.any(valid_mask):
        return np.zeros_like(image_array, dtype=np.float32)
        
    min_val = np.nanmin(image_array)
    max_val = np.nanmax(image_array)
    if max_val == min_val:
        return np.zeros_like(image_array, dtype=np.float32)
        
    # Normalize image to [0, 1] range to prepare for quantization
    normalized = np.clip((image_array - min_val) / (max_val - min_val), 0.0, 1.0)
    
    # Quantize to specified number of gray levels
    quantized = (normalized * (config.LEVELS - 1)).astype(np.uint8)
    
    h, w = quantized.shape
    pad = config.WINDOW_SIZE // 2
    padded = np.pad(quantized, pad, mode='reflect')
    
    texture_raster = np.zeros_like(image_array, dtype=np.float32)
    
    # Efficient windowed GLCM calculation:
    # If the raster is very large (> 200x200), compute with stride or block-wise
    stride = 1 if (h <= 150 and w <= 150) else max(1, min(h, w) // 50)
    
    for i in range(0, h, stride):
        for j in range(0, w, stride):
            window = padded[i:i + config.WINDOW_SIZE, j:j + config.WINDOW_SIZE]
            glcm = graycomatrix(
                window,
                distances=config.DISTANCES,
                angles=config.ANGLES,
                levels=config.LEVELS,
                symmetric=True,
                normed=True
            )
            prop = graycoprops(glcm, feature)[0, 0]
            texture_raster[i:i + stride, j:j + stride] = prop
            
    # Clip to valid bounds and handle NaN
    texture_raster = np.nan_to_num(texture_raster, nan=0.0)
    return texture_raster
