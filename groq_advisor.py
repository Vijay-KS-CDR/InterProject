"""
=============================================================================
Agricultural AI - Groq AI Agricultural Advisor
=============================================================================
This module provides the explanation and advisory layer powered by Groq LLMs.
It receives the extracted satellite features and ML-predicted N/P/K values,
together with crop and growth stage metadata, and generates structured,
agronomically responsible advisory insights.

Key Scientific & Safety Principles:
1. N/P/K values are ML model estimates from canopy bio-optical properties,
   NOT laboratory-measured soil tests or direct satellite soil measurements.
2. No guaranteed yield or absolute yield promises.
3. No fabricated fertilizer application rates or kilograms.
4. Stress diagnosis must consider multi-factorial causes (irrigation, pests,
   pathogens, soil compaction, temperature) beyond spectral indices.
5. Soil laboratory testing is always recommended prior to fertilizer decisions.
=============================================================================
"""

import os
import json
import logging
from typing import Dict, Any, Optional
from dotenv import load_dotenv

# Load environment variables from .env file if available
load_dotenv()

logger = logging.getLogger("groq_advisor")

_RESOLVED_MODEL_CACHE: Optional[str] = None

# Prioritized list of models suitable for general text and instruction tasks
PREFERRED_TEXT_MODELS = [
    "qwen/qwen3.8-27b",
    "llama-3.3-70b-versatile",
    "llama-3.1-70b-versatile",
    "llama-3.1-8b-instant",
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
]


def resolve_active_model(client) -> str:
    """
    Dynamically identifies an active, available Groq model suitable for general
    text/instruction tasks by querying client.models.list().
    Avoids hardcoding an outdated or inaccessible model name.
    """
    global _RESOLVED_MODEL_CACHE
    if _RESOLVED_MODEL_CACHE:
        return _RESOLVED_MODEL_CACHE

    env_model = os.getenv("GROQ_MODEL", "").strip()
    try:
        model_list_resp = client.models.list()
        available_ids = set()
        for item in getattr(model_list_resp, "data", []):
            if getattr(item, "active", True):
                available_ids.add(item.id)

        # 1. If user set GROQ_MODEL in .env and it's active in the API, use it
        if env_model and env_model in available_ids:
            _RESOLVED_MODEL_CACHE = env_model
            logger.info("Using configured active Groq model: %s", _RESOLVED_MODEL_CACHE)
            return _RESOLVED_MODEL_CACHE

        # 2. Check preferred instruction/chat models in priority order
        for pref in PREFERRED_TEXT_MODELS:
            if pref in available_ids:
                _RESOLVED_MODEL_CACHE = pref
                logger.info("Dynamically selected active Groq model from API: %s", _RESOLVED_MODEL_CACHE)
                return _RESOLVED_MODEL_CACHE

        # 3. Filter for general text models (exclude audio/whisper and guard models)
        for mid in sorted(available_ids):
            mid_lower = mid.lower()
            if not any(x in mid_lower for x in ["whisper", "guard", "audio"]):
                _RESOLVED_MODEL_CACHE = mid
                logger.info("Selected fallback active Groq model: %s", _RESOLVED_MODEL_CACHE)
                return _RESOLVED_MODEL_CACHE

        if available_ids:
            _RESOLVED_MODEL_CACHE = next(iter(available_ids))
            return _RESOLVED_MODEL_CACHE

    except Exception as exc:
        logger.warning("Could not query Groq models API: %s. Using default fallback.", exc)

    _RESOLVED_MODEL_CACHE = env_model or "qwen/qwen3.8-27b"
    return _RESOLVED_MODEL_CACHE

SYSTEM_PROMPT = """You are an expert Agricultural Advisor and Remote Sensing Agronomist acting as the interpretation layer for an Agricultural AI decision support platform.

CRITICAL SCIENTIFIC SAFETY GUIDELINES:
1. The Nitrogen (N), Phosphorus (P), and Potassium (K) values provided to you are ESTIMATED by machine-learning models from canopy spectral properties. They are NOT laboratory-measured soil nutrient values.
2. NEVER claim or imply that satellite imagery directly measured soil N/P/K.
3. NEVER guarantee crop yield or say "You will get good yield."
4. ALWAYS use cautious, scientifically sound language, such as:
   - "Based on the estimated nutrient values..."
   - "Possible nutrient limitation..."
   - "Consider validating with soil testing..."
   - "Review irrigation, pests, disease, and crop growth stage..."
5. Do NOT diagnose nutrient deficiency solely from NDVI. High or low NDVI can arise from water status, soil moisture, crop stage, salinity, pests, or disease.
6. Do NOT automatically convert predicted N/P/K values into fertilizer kilograms or application rates.
7. Do NOT invent fertilizer dosage rates or arbitrary crop-specific thresholds.
8. State clearly that fertilizer rates require agronomic calibration, soil-test verification, and regional agricultural extension guidance.
9. Always mention other possible causes of crop stress (water stress, pests, disease, temperature, growth stage, soil physical conditions).
10. Explicitly suggest soil testing when fertilizer decisions require confirmation.

You must respond ONLY with a valid JSON object matching this exact structure:
{
  "summary": "Concise 2-3 sentence overview of canopy condition and estimated nutrient status.",
  "nutrient_observations": [
    "Observation regarding estimated Nitrogen and its relationship with canopy chlorophyll/vigor.",
    "Observation regarding estimated Phosphorus and vegetative/root development context.",
    "Observation regarding estimated Potassium and overall plant water regulation/stress resilience."
  ],
  "possible_concerns": [
    "Specific potential limitation or stress factor to investigate (or indicate balanced conditions if normal)."
  ],
  "actions_to_consider": [
    "Practical, safe next step for field inspection or agronomic review.",
    "Recommendation for ground-truth soil or tissue sampling."
  ],
  "things_to_monitor": [
    "Field variables to inspect such as soil moisture, pest pressure, or leaf discoloration symptoms."
  ],
  "validation_note": "These nutrient values are model estimates and should be validated with appropriate soil/field observations before fertilizer application."
}
"""


def get_groq_client():
    """
    Initializes and returns a Groq API client using the GROQ_API_KEY environment variable.
    Returns None if the key is not configured.
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key or not api_key.strip():
        logger.info("GROQ_API_KEY is not set or empty.")
        return None

    try:
        from groq import Groq
        return Groq(api_key=api_key.strip())
    except Exception as exc:
        logger.warning("Failed to initialize Groq client: %s", exc)
        return None


def generate_agricultural_advice(
    predictions: Dict[str, float],
    features: Dict[str, Any],
    crop: Optional[str] = None,
    growth_stage: Optional[str] = None,
    timeout: float = 12.0
) -> Dict[str, Any]:
    """
    Generates structured AI agricultural advice via Groq LLM.

    Args:
        predictions: Dictionary containing predicted nutrients {'N': float, 'P': float, 'K': float}.
        features: Dictionary of the 6 bio-optical features (NDVI, EVI, FVC, LAI, NDVI Texture, Biomass).
        crop: Crop type (optional, e.g. 'Rice', 'Wheat', 'Maize', etc.).
        growth_stage: Crop growth stage (optional, e.g. 'Vegetative', 'Flowering', etc.).
        timeout: Maximum seconds to wait for Groq API response.

    Returns:
        Dictionary with:
        - available: bool (True if successful, False if Groq unavailable or failed)
        - error: str or None
        - advice: dict or None
    """
    client = get_groq_client()
    if client is None:
        return {
            "available": False,
            "error": "AI agricultural advice is temporarily unavailable. Nutrient estimation was completed successfully.",
            "advice": None
        }

    # Format user input data cleanly for prompt
    crop_str = crop if crop and crop.strip() else "Not Specified"
    stage_str = growth_stage if growth_stage and growth_stage.strip() else "Not Specified"

    n_val = f"{predictions.get('N', 0.0):.2f}" if predictions else "N/A"
    p_val = f"{predictions.get('P', 0.0):.2f}" if predictions else "N/A"
    k_val = f"{predictions.get('K', 0.0):.2f}" if predictions else "N/A"

    ndvi_val = f"{features.get('NDVI', 0.0):.4f}" if features.get('NDVI') is not None else "N/A"
    evi_val = f"{features.get('EVI', 0.0):.4f}" if features.get('EVI') is not None else "N/A"
    fvc_val = f"{features.get('FVC', 0.0):.4f}" if features.get('FVC') is not None else "N/A"
    lai_val = f"{features.get('LAI', 0.0):.2f}" if features.get('LAI') is not None else "N/A"
    texture_val = f"{features.get('NDVI Texture', 0.0):.4f}" if features.get('NDVI Texture') is not None else "N/A"
    biomass_val = f"{features.get('Biomass', 0.0):.2f}" if features.get('Biomass') is not None else "N/A"

    user_content = f"""Please provide an agricultural assessment based on the following remote-sensing analysis:

FIELD & CROP METADATA:
- Crop: {crop_str}
- Growth Stage: {stage_str}

MODEL ESTIMATED NUTRIENTS (Estimated by ML prototype, NOT soil tests):
- Predicted Nitrogen: {n_val}
- Predicted Phosphorus: {p_val}
- Predicted Potassium: {k_val}

SATELLITE VEGETATION INDICES:
- NDVI: {ndvi_val}
- EVI: {evi_val}
- FVC: {fvc_val}
- LAI: {lai_val}
- NDVI Texture: {texture_val}
- Biomass: {biomass_val}

Generate the structured JSON assessment adhering strictly to the safety guidelines.
"""

    active_model = resolve_active_model(client)
    models_to_try = [active_model]
    for m in PREFERRED_TEXT_MODELS:
        if m not in models_to_try:
            models_to_try.append(m)

    last_error = None
    for model_name in models_to_try:
        try:
            chat_completion = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_content}
                ],
                temperature=0.3,
                max_tokens=1024,
                response_format={"type": "json_object"},
                timeout=timeout
            )
            raw_text = chat_completion.choices[0].message.content
            parsed_json = json.loads(raw_text)

            # Validate expected keys
            required_keys = ["summary", "nutrient_observations", "possible_concerns", "actions_to_consider", "validation_note"]
            for k in required_keys:
                if k not in parsed_json:
                    raise ValueError(f"Missing required key in advisor response: {k}")

            # Ensure validation_note has the required scientific text
            if not parsed_json.get("validation_note"):
                parsed_json["validation_note"] = (
                    "These nutrient values are model estimates and should be validated "
                    "with appropriate soil/field observations before fertilizer application."
                )

            return {
                "available": True,
                "error": None,
                "advice": parsed_json,
                "model_used": model_name
            }

        except Exception as exc:
            logger.warning("Groq advisor attempt with model '%s' failed: %s", model_name, exc)
            last_error = str(exc)

    return {
        "available": False,
        "error": "AI agricultural advice is temporarily unavailable. Nutrient estimation was completed successfully.",
        "advice": None
    }
