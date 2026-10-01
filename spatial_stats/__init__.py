"""
Spatial statistics package for aggregating raster pixels to field-level scalar metrics.
"""
from spatial_stats.zonal_statistics import calculate_zonal_statistics

__all__ = ["calculate_zonal_statistics"]
