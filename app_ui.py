"""
=============================================================================
Agricultural AI - Streamlit Web Dashboard
=============================================================================
Interactive GeoTIFF upload, real multispectral band extraction, 6-feature
calculation, and Random Forest N/P/K nutrient estimation.
=============================================================================
"""

import streamlit as st
import io
import json
from satellite_pipeline import process_satellite_geotiff

st.set_page_config(
    page_title="Agricultural AI | Crop & Soil Analysis",
    page_icon="🌾",
    layout="wide"
)

st.title("🌾 Agricultural AI: Satellite Crop & Soil Nutrient Analysis")
st.markdown(
    "Upload a multispectral satellite GeoTIFF (e.g., Sentinel-2 L2A containing Blue B2, Green B3, Red B4, and NIR B8) "
    "to extract bio-optical canopy features and estimate Nitrogen (N), Phosphorus (P), and Potassium (K)."
)

# Sidebar Configuration
with st.sidebar:
    st.header("🛰️ Satellite Ingestion Settings")
    crop_selected = st.selectbox(
        "Crop Type (Optional Context)",
        ["Rice", "Wheat", "Maize", "Sugarcane", "Cotton", "Sunflower", "Groundnut"],
        help="Crop type is kept for field contextual record; it is NOT passed into the baseline bio-optical NPK models."
    )
    st.markdown("---")
    st.info(
        "**Multi-Spectral Requirements:**\n"
        "- GeoTIFF format (.tif, .tiff)\n"
        "- Minimum 4 Bands: B02 (Blue), B03 (Green), B04 (Red), B08 (NIR)\n"
        "- Real surface reflectance (DN scaled by 10,000)"
    )

uploaded_file = st.file_uploader(
    "Upload Multi-Spectral GeoTIFF (.tif, .tiff)",
    type=["tif", "tiff"],
    help="Upload genuine satellite imagery containing Red and NIR bands."
)

if uploaded_file is not None:
    st.success(f"File loaded: **{uploaded_file.name}** ({uploaded_file.size:,} bytes)")
    
    col_btn, _ = st.columns([1, 3])
    with col_btn:
        run_analysis = st.button("🚀 Analyze Satellite Imagery", type="primary", use_container_width=True)
        
    if run_analysis:
        with st.spinner("Decoding GeoTIFF bands, calculating vegetation indices, and running ML inference..."):
            try:
                file_bytes = uploaded_file.getvalue()
                result = process_satellite_geotiff(file_bytes, empirical_fallbacks=True)
                
                meta = result["metadata"]
                bands = result["bands"]
                features = result["features"]
                preds = result["predictions"]
                
                st.markdown("---")
                
                # --- SECTION 1: METADATA & BANDS ---
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("📋 Container Metadata")
                    st.write(f"**Dimensions:** {meta['width']} × {meta['height']} px")
                    st.write(f"**Band Count:** {meta['count']} spectral bands")
                    st.write(f"**CRS:** `{meta['crs']}`")
                    st.write(f"**Spatial Resolution:** {meta['spatial_resolution']}")

                with col2:
                    st.subheader("🌈 Calibrated Spectral Bands")
                    b_col1, b_col2 = st.columns(2)
                    b_col1.metric("Blue (B02)", f"{bands['blue']:.4f}")
                    b_col1.metric("Red (B04)", f"{bands['red']:.4f}")
                    b_col2.metric("Green (B03)", f"{bands['green']:.4f}")
                    b_col2.metric("NIR (B08)", f"{bands['nir']:.4f}")

                st.markdown("---")

                # --- SECTION 2: CALCULATED 6 FEATURES ---
                st.subheader("🌱 Calculated 6 Model Features")
                fc1, fc2, fc3 = st.columns(3)
                fc1.metric("1. NDVI", f"{features['NDVI']:.4f}", help="(NIR - Red) / (NIR + Red)")
                fc1.metric("4. LAI", f"{features['LAI']:.2f}", help="Leaf Area Index (Empirical proxy)")
                
                fc2.metric("2. EVI", f"{features['EVI']:.4f}", help="Enhanced Vegetation Index (Huete)")
                fc2.metric("5. NDVI Texture", f"{features['NDVI Texture']:.4f}", help="GLCM Canopy Heterogeneity")
                
                fc3.metric("3. FVC", f"{features['FVC']:.4f}", help="Fractional Vegetation Cover")
                fc3.metric("6. Biomass", f"{features['Biomass']:.2f}", help="Crop Biomass proxy")

                # Scientific notices on stubs/proxies
                if result.get("warnings"):
                    for w in result["warnings"]:
                        st.warning(f"⚠️ {w}")

                st.markdown("---")

                # --- SECTION 3: PREDICTED SOIL NUTRIENTS ---
                st.subheader("🧪 Estimated Soil Nutrients (Prototype Model)")
                
                if preds:
                    p1, p2, p3 = st.columns(3)
                    p1.metric("Nitrogen (N)", f"{preds['N']:.2f}")
                    p2.metric("Phosphorus (P)", f"{preds['P']:.2f}")
                    p3.metric("Potassium (K)", f"{preds['K']:.2f}")
                    
                    st.info(f"ℹ️ **Scientific Disclaimer:** {result['scientific_disclaimer']}")
                    st.caption("Fertilizer dosage recommendations are separate agronomic decisions and require soil testing / agronomic calibration.")
                else:
                    st.error("Model prediction could not be completed because some features are pending.")

            except Exception as e:
                st.error(f"❌ Analysis Failed: {str(e)}")
