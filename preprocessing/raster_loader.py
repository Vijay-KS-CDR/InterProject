"""
Raster Loading and Surface Reflectance Calibration Module.
Uses rasterio to decode GeoTIFF containers, extract geospatial metadata (CRS, transform),
and scale Digital Numbers (DN) to surface reflectance [0.0 - 1.0].
"""
import rasterio
from rasterio.io import MemoryFile
import numpy as np
from typing import Tuple, Union, BinaryIO

def load_raster(source: Union[str, bytes, BinaryIO]) -> Tuple[np.ndarray, dict]:
    """
    Load a multi-band GeoTIFF raster from a file path, raw bytes, or file-like object.
    
    Args:
        source: Filepath string, bytes buffer, or open binary stream.
        
    Returns:
        tuple: (raster_data, profile)
               raster_data: 3D numpy array shaped [count, height, width]
               profile: dict with geospatial metadata (crs, transform, nodata, count, width, height)
    """
    if isinstance(source, bytes):
        with MemoryFile(source) as memfile:
            with memfile.open() as src:
                data = src.read()
                profile = src.profile.copy()
    elif hasattr(source, "read") and not isinstance(source, (str, bytes)):
        source.seek(0)
        content = source.read()
        with MemoryFile(content) as memfile:
            with memfile.open() as src:
                data = src.read()
                profile = src.profile.copy()
    else:
        with rasterio.open(source) as src:
            data = src.read()
            profile = src.profile.copy()

    if data.size == 0:
        raise ValueError("Raster contains 0 pixels or could not be decoded.")

    return data, profile

def convert_to_reflectance(data: np.ndarray, scale_factor: float = 10000.0) -> np.ndarray:
    """
    Convert Sentinel-2 / Landsat integer Digital Numbers (DN) to Top/Bottom of Atmosphere reflectance.
    Sentinel-2 L2A BOA data uses standard scale factor 10,000 (DN 10,000 = 100% reflectance).
    If data is already float in [0.0, 1.0], returns data as float32.
    """
    arr = data.astype(np.float32)
    # Check if already normalized float
    if np.nanmax(arr) <= 1.0:
        return arr
    return arr / float(scale_factor)
