"""
=============================================================================
Agricultural AI - FastAPI Backend Application
=============================================================================
Endpoints:
- POST /api/analyze : Upload multispectral GeoTIFF for real satellite band
                     extraction, 6-feature calculation, and NPK prediction.
- POST /api/predict : Direct feature inference with the 6 model parameters.
- GET  /            : Serves the Web Dashboard.
=============================================================================
"""

import os
import sys
from pathlib import Path
from typing import Optional, List, Dict
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from satellite_pipeline import process_satellite_geotiff
from predict_npk import predict_soil_nutrients
from groq_advisor import generate_agricultural_advice

app = FastAPI(
    title="Agricultural AI - Satellite & Soil Nutrient Analysis",
    description="Multispectral GeoTIFF processing and Random Forest N/P/K nutrient estimation.",
    version="2.0.0"
)

# Enable CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class NutrientPredictionRequest(BaseModel):
    NDVI: float = Field(..., description="Normalized Difference Vegetation Index [-1.0, 1.0]")
    EVI: float = Field(..., description="Enhanced Vegetation Index [-1.0, 1.0]")
    FVC: float = Field(..., description="Fractional Vegetation Cover [0.0, 1.0]")
    LAI: float = Field(..., description="Leaf Area Index (m²/m²)")
    NDVI_Texture: float = Field(..., alias="NDVI Texture", description="Canopy Texture Heterogeneity")
    Biomass: float = Field(..., description="Crop Biomass proxy")
    crop: Optional[str] = Field(None, description="Target crop type")
    growth_stage: Optional[str] = Field(None, description="Crop phenological growth stage")

    class Config:
        populate_by_name = True


@app.get("/", response_class=FileResponse)
def get_dashboard():
    """Serves the main application dashboard."""
    index_path = BASE_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="Dashboard index.html not found.")
    return FileResponse(index_path)


@app.get("/style.css")
def get_css():
    return FileResponse(BASE_DIR / "style.css", media_type="text/css")


@app.get("/script.js")
def get_js():
    return FileResponse(BASE_DIR / "script.js", media_type="application/javascript")


@app.get("/test_raster.tif")
def get_sample_raster():
    sample_path = BASE_DIR / "test_raster.tif"
    if not sample_path.exists():
        raise HTTPException(status_code=404, detail="test_raster.tif not found.")
    return FileResponse(sample_path, media_type="image/tiff")


@app.get("/test_raster_stressed.tif")
def get_sample_raster_stressed():
    sample_path = BASE_DIR / "test_raster_stressed.tif"
    if not sample_path.exists():
        raise HTTPException(status_code=404, detail="test_raster_stressed.tif not found.")
    return FileResponse(sample_path, media_type="image/tiff")


@app.post("/api/analyze")
async def analyze_geotiff(
    file: UploadFile = File(...),
    crop: Optional[str] = Form(None),
    growth_stage: Optional[str] = Form(None)
):
    """
    Accepts an uploaded multispectral GeoTIFF file.
    Validates bands, extracts reflectances, calculates 6 bio-optical features,
    runs N/P/K model inference, and obtains structured Groq AI agricultural advice.
    """
    filename = file.filename or "uploaded_image.tif"
    ext = filename.split(".")[-1].lower()
    
    if ext not in ["tif", "tiff"]:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format (.{ext}). Analysis requires a multispectral GeoTIFF (.tif, .tiff)."
        )

    try:
        contents = await file.read()
        if len(contents) == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty (0 bytes).")
            
        result = process_satellite_geotiff(
            contents,
            empirical_fallbacks=True,
            models_dir=str(BASE_DIR / "models")
        )
        
        # Agricultural advice via Groq LLM (fails gracefully if API unavailable)
        preds = result.get("predictions")
        feats = result.get("features")
        advisor_res = None
        if preds and feats:
            advisor_res = generate_agricultural_advice(
                predictions=preds,
                features=feats,
                crop=crop,
                growth_stage=growth_stage
            )

        result["filename"] = filename
        result["crop_selected"] = crop
        result["growth_stage_selected"] = growth_stage
        result["advisor"] = advisor_res
        result["success"] = True
        return JSONResponse(content=result)
        
    except ValueError as val_err:
        raise HTTPException(status_code=422, detail=str(val_err))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(exc)}")


@app.post("/api/predict")
def predict_nutrients(req: NutrientPredictionRequest):
    """
    Direct model inference endpoint using the 6 trained input features.
    """
    data_dict = {
        "NDVI": req.NDVI,
        "EVI": req.EVI,
        "FVC": req.FVC,
        "LAI": req.LAI,
        "NDVI Texture": req.NDVI_Texture,
        "Biomass": req.Biomass
    }
    
    try:
        predictions = predict_soil_nutrients(data_dict, models_dir=str(BASE_DIR / "models"))
        pred_dict = {
            "N": round(float(predictions["N"]), 2),
            "P": round(float(predictions["P"]), 2),
            "K": round(float(predictions["K"]), 2)
        }

        advisor_res = generate_agricultural_advice(
            predictions=pred_dict,
            features=data_dict,
            crop=req.crop,
            growth_stage=req.growth_stage
        )

        return {
            "success": True,
            "inputs": data_dict,
            "predictions": pred_dict,
            "advisor": advisor_res,
            "disclaimer": "Estimated N/P/K based on prototype model trained on synthetic data."
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
