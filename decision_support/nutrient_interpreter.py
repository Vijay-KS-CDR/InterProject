"""
=============================================================================
Nutrient Status Interpretation Module
=============================================================================
Interprets estimated Nitrogen (N), Phosphorus (P), and Potassium (K) against
crop-specific and growth-stage-specific calibrated thresholds.

Classifications:
- Low     : Below lower boundary; potentially limiting for growth/canopy vigor.
- Adequate: Within optimal physiological range for current phenological stage.
- High    : Above upper boundary; surplus / luxury consumption (risk of lodging, delayed maturity, or leaching).

Transparent Source & Calibration:
Thresholds are stored in config/nutrient_thresholds.json and calibrated against
empirical/dataset distribution percentiles. Real field deployment requires
local agricultural university / soil testing lab calibration.
=============================================================================
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional

DEFAULT_THRESHOLDS_PATH = Path(__file__).resolve().parent.parent / "config" / "nutrient_thresholds.json"


def load_nutrient_thresholds(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """Loads the threshold configuration file."""
    path = config_path or DEFAULT_THRESHOLDS_PATH
    if not path.exists():
        raise FileNotFoundError(f"Nutrient thresholds configuration not found at {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def interpret_nutrient_status(
    crop: str,
    growth_stage: str,
    estimated_nutrients: Dict[str, float],
    config_path: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Evaluates estimated N, P, K against crop/growth-stage thresholds.
    
    Args:
        crop (str): Target crop name (Rice, Wheat, Maize, Cotton, Sugarcane).
        growth_stage (str): Phenological stage (Early, Vegetative, Flowering, Mature).
        estimated_nutrients (dict): Estimated continuous nutrient values {'N': float, 'P': float, 'K': float}.
        config_path (Path, optional): Path to nutrient_thresholds.json.
        
    Returns:
        dict: Detailed nutrient status, deficiency list, priorities, and explanations.
    """
    config_data = load_nutrient_thresholds(config_path)
    crops_cfg = config_data.get("crops", {})
    metadata = config_data.get("metadata", {})

    # Match crop or fallback to Default
    matched_crop = crop if crop in crops_cfg else "Default"
    stage_cfg = crops_cfg.get(matched_crop, {}).get(growth_stage)

    # Fallback to Vegetative if stage not found
    if not stage_cfg:
        stage_cfg = crops_cfg.get(matched_crop, {}).get("Vegetative") or crops_cfg["Default"]["Vegetative"]

    statuses = {}
    deficiencies = []
    priorities = []

    nutrient_full_names = {
        "N": "Nitrogen",
        "P": "Phosphorus",
        "K": "Potassium"
    }

    for nutrient_key in ["N", "P", "K"]:
        val = estimated_nutrients.get(nutrient_key)
        if val is None:
            continue

        thresh = stage_cfg.get(nutrient_key, {"low": 30.0, "adequate_min": 30.0, "adequate_max": 45.0, "high": 45.0})
        low_t = thresh["low"]
        high_t = thresh["high"]

        if val < low_t:
            status_label = "Low"
            deficiency_desc = "Possible deficiency (potentially limiting vegetative growth/yield)"
            deficiencies.append(nutrient_full_names[nutrient_key])
            priorities.append({
                "nutrient": nutrient_full_names[nutrient_key],
                "symbol": nutrient_key,
                "status": "Low",
                "priority_level": "High Priority",
                "reason": f"Estimated value ({val:.2f}) is below the adequate threshold ({low_t:.1f}) for {matched_crop} at {growth_stage} stage."
            })
        elif val > high_t:
            status_label = "High"
            deficiency_desc = "Surplus / Luxury consumption (no immediate limitation; monitor against over-fertilization)"
            priorities.append({
                "nutrient": nutrient_full_names[nutrient_key],
                "symbol": nutrient_key,
                "status": "High",
                "priority_level": "Surplus / Low Priority",
                "reason": f"Estimated value ({val:.2f}) exceeds optimal threshold ({high_t:.1f}); further fertilization of this nutrient is not recommended."
            })
        else:
            status_label = "Adequate"
            deficiency_desc = "Optimal physiological range for current phenological stage"
            priorities.append({
                "nutrient": nutrient_full_names[nutrient_key],
                "symbol": nutrient_key,
                "status": "Adequate",
                "priority_level": "Optimal",
                "reason": f"Estimated value ({val:.2f}) is within optimal physiological range [{thresh['adequate_min']:.1f} - {thresh['adequate_max']:.1f}]."
            })

        statuses[nutrient_key] = {
            "nutrient_name": nutrient_full_names[nutrient_key],
            "estimated_value": float(round(val, 2)),
            "status": status_label,
            "thresholds": thresh,
            "interpretation": deficiency_desc
        }

    # Generate overarching diagnostic summary
    if len(deficiencies) == 0:
        summary_text = (
            f"All three primary nutrients (N, P, K) appear within the adequate physiological range "
            f"for {matched_crop} during the {growth_stage} stage."
        )
    else:
        summary_text = (
            f"Potential nutrient limitation detected for {', '.join(deficiencies)} in {matched_crop} "
            f"at the {growth_stage} stage. Management action should focus on correcting these limiting factors."
        )

    return {
        "crop": crop,
        "matched_crop_config": matched_crop,
        "growth_stage": growth_stage,
        "statuses": statuses,
        "deficiencies": deficiencies,
        "priorities": priorities,
        "diagnostic_summary": summary_text,
        "threshold_source": metadata.get("calibration_source", "Dataset Distribution Calibration"),
        "calibration_status": metadata.get("calibration_status", "PROTOTYPE_CALIBRATION_REQUIRED"),
        "scientific_disclaimer": metadata.get("scientific_disclaimer", "")
    }
