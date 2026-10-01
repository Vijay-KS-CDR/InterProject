"""
Configuration parameters for satellite band extraction, vegetation indices, and texture.
Standardized for Sentinel-2 optical bands (10m resolution: B02, B03, B04, B08).
"""

class Config:
    # --- Sensor Band Indices (0-based) for standard 4-band agricultural stacked GeoTIFF ---
    class Bands:
        BLUE = 0   # Sentinel-2 B02 (~490 nm)
        GREEN = 1  # Sentinel-2 B03 (~560 nm)
        RED = 2    # Sentinel-2 B04 (~665 nm)
        NIR = 3    # Sentinel-2 B08 (~842 nm)

    # --- Validated Enhanced Vegetation Index (EVI) Constants (Huete et al.) ---
    class EVI:
        G = 2.5     # Gain factor
        C1 = 6.0    # Aerosol resistance coefficient for Red band
        C2 = 7.5    # Aerosol resistance coefficient for Blue band
        L = 1.0     # Canopy background adjustment factor

    # --- Fractional Vegetation Cover (FVC) Dimidiate Pixel Model Constants ---
    class FVC:
        NDVI_SOIL = 0.15  # NDVI of completely bare soil
        NDVI_VEG = 0.90   # NDVI of fully covered, healthy dense vegetation

    # --- Gray-Level Co-occurrence Matrix (GLCM) Texture Configuration ---
    class Texture:
        WINDOW_SIZE = 5     # 5x5 window
        LEVELS = 64         # Quantization levels
        DISTANCES = [1]     # Pixel offset
        ANGLES = [0]        # Angle (horizontal 0 deg)

    # --- Preprocessing & Nodata Defaults ---
    class Preprocessing:
        NODATA_VALUE = -9999.0
        SURFACE_REFLECTANCE_SCALE = 10000.0  # Sentinel-2 L2A standard scale
