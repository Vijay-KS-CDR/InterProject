"""
=============================================================================
Agricultural Fertilizer Advisor & Decision Support Module
=============================================================================
Translates nutrient status interpretations into agronomic management guidance:
1. Identifies primary limiting nutrients.
2. Explains the difference between nutrient requirement and fertilizer product amount.
3. Calculates fertilizer product quantities using validated commercial grades (FCO).
4. Emphasizes Liebig's Law of the Minimum: more fertilizer does NOT equal higher yield.
5. Outlines the Future Yield Model Architecture without generating unvalidated yield numbers.
=============================================================================
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional, List

DEFAULT_CATALOG_PATH = Path(__file__).resolve().parent.parent / "config" / "fertilizer_catalog.json"


def load_fertilizer_catalog(catalog_path: Optional[Path] = None) -> Dict[str, Any]:
    """Loads the commercial fertilizer catalog."""
    path = catalog_path or DEFAULT_CATALOG_PATH
    if not path.exists():
        raise FileNotFoundError(f"Fertilizer catalog not found at {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


class FutureYieldModelArchitecture:
    """
    Conceptual architecture placeholder for future yield forecasting.
    Yield prediction requires multi-season field harvest ground-truth datasets.
    """
    REQUIRED_LAYERS = [
        "Canopy Biophysical Vigor (NDVI, EVI, LAI, FVC from Sentinel-2)",
        "Canopy Spatial Heterogeneity (GLCM Texture)",
        "Estimated Nutrient Status & Soil Test Balance (N, P, K)",
        "Agro-Meteorological Records (Rainfall, Growing Degree Days, Thermal Stress)",
        "Hydrological & Irrigation Schedule (Soil Moisture, ET0)",
        "Crop Cultivar / Variety Phenology Calendar"
    ]
    STATUS = "ARCHITECTURE_STUB_ONLY"
    NOTE = "No fake yield predictions are generated. Yield estimation will be activated when a validated crop-cut harvest dataset is integrated."


def generate_fertilizer_advisory(
    nutrient_interpretation: Dict[str, Any],
    field_area_ha: float = 1.0,
    target_yield: Optional[str] = None,
    soil_test_values: Optional[Dict[str, float]] = None,
    preferred_fertilizer: Optional[str] = None,
    catalog_path: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Generates actionable, agronomic decision-support guidance based on nutrient interpretation.
    
    Args:
        nutrient_interpretation (dict): Output from interpret_nutrient_status.
        field_area_ha (float): Area of field in hectares (default 1.0).
        target_yield (str, optional): Farmer's target yield context.
        soil_test_values (dict, optional): Actual laboratory soil test measurements {'N': ..., 'P': ..., 'K': ...}.
        preferred_fertilizer (str, optional): User-selected product ('Urea', 'DAP', 'MOP', etc.).
        catalog_path (Path, optional): Path to fertilizer_catalog.json.
        
    Returns:
        dict: Complete structured decision support advice, product amounts, and scientific cautions.
    """
    catalog_data = load_fertilizer_catalog(catalog_path)
    products = catalog_data.get("fertilizer_products", {})
    principles = catalog_data.get("agronomic_principles", {})

    crop = nutrient_interpretation.get("crop", "Rice")
    growth_stage = nutrient_interpretation.get("growth_stage", "Vegetative")
    deficiencies = nutrient_interpretation.get("deficiencies", [])
    statuses = nutrient_interpretation.get("statuses", {})

    has_soil_test = bool(soil_test_values and any(v is not None for v in soil_test_values.values()))

    # Determine nutrient focus
    if deficiencies:
        nutrient_focus = " + ".join(deficiencies)
        urgency = "Moderate to High — nutrient limitation observed"
    else:
        nutrient_focus = "Maintenance / Balanced Nutrition"
        urgency = "Low — canopy shows adequate primary nutrient balance"

    # Select recommended fertilizer products based on detected limitation
    recommendations_list = []
    
    # 1. Nitrogen limitation handling
    if "Nitrogen" in deficiencies:
        prod_key = "Urea" if preferred_fertilizer in [None, "Urea"] else preferred_fertilizer
        prod_info = products.get("Urea", {})
        
        # Prototype agronomic corrective rate: ~15-25 kg N/ha for mild/moderate deficiency
        est_deficit_kg_n = 20.0
        kg_urea_per_ha = est_deficit_kg_n * prod_info["kg_product_per_kg_nutrient"]["N"]
        total_field_urea = kg_urea_per_ha * field_area_ha

        recommendations_list.append({
            "target_nutrient": "Nitrogen (N)",
            "deficiency_status": statuses.get("N", {}).get("status", "Low"),
            "product_name": prod_info.get("name", "Urea"),
            "product_grade": prod_info.get("grade", "46-0-0"),
            "nutrient_vs_product_explanation": (
                "Urea contains 46% elemental Nitrogen. To supply 1 kg of active Nitrogen, "
                "2.17 kg of commercial Urea fertilizer must be applied."
            ),
            "estimated_nutrient_correction_rate": f"{est_deficit_kg_n:.1f} kg N/ha (Benchmark guidance)",
            "calculated_product_per_ha": f"{kg_urea_per_ha:.1f} kg {prod_info.get('name', 'Urea')} / ha",
            "total_product_for_field": f"{total_field_urea:.1f} kg for {field_area_ha:.2f} ha",
            "timing_and_management": prod_info.get("management_note", "")
        })

    # 2. Phosphorus limitation handling
    if "Phosphorus" in deficiencies:
        prod_info = products.get("DAP", {})
        est_deficit_kg_p = 15.0
        kg_dap_per_ha = est_deficit_kg_p * prod_info["kg_product_per_kg_nutrient"]["P2O5"]
        total_field_dap = kg_dap_per_ha * field_area_ha

        recommendations_list.append({
            "target_nutrient": "Phosphorus (P)",
            "deficiency_status": statuses.get("P", {}).get("status", "Low"),
            "product_name": prod_info.get("name", "DAP"),
            "product_grade": prod_info.get("grade", "18-46-0"),
            "nutrient_vs_product_explanation": (
                "DAP contains 46% available P2O5 and 18% Nitrogen. Applying 2.17 kg of DAP "
                "supplies 1 kg of P2O5 and simultaneously provides ~0.39 kg of Nitrogen."
            ),
            "estimated_nutrient_correction_rate": f"{est_deficit_kg_p:.1f} kg P2O5/ha (Benchmark guidance)",
            "calculated_product_per_ha": f"{kg_dap_per_ha:.1f} kg {prod_info.get('name', 'DAP')} / ha",
            "total_product_for_field": f"{total_field_dap:.1f} kg for {field_area_ha:.2f} ha",
            "timing_and_management": prod_info.get("management_note", "")
        })

    # 3. Potassium limitation handling
    if "Potassium" in deficiencies:
        prod_info = products.get("MOP", {})
        est_deficit_kg_k = 20.0
        kg_mop_per_ha = est_deficit_kg_k * prod_info["kg_product_per_kg_nutrient"]["K2O"]
        total_field_mop = kg_mop_per_ha * field_area_ha

        recommendations_list.append({
            "target_nutrient": "Potassium (K)",
            "deficiency_status": statuses.get("K", {}).get("status", "Low"),
            "product_name": prod_info.get("name", "MOP"),
            "product_grade": prod_info.get("grade", "0-0-60"),
            "nutrient_vs_product_explanation": (
                "Muriate of Potash (MOP) contains 60% soluble K2O. Supplying 1 kg of active K2O "
                "requires 1.67 kg of commercial MOP."
            ),
            "estimated_nutrient_correction_rate": f"{est_deficit_kg_k:.1f} kg K2O/ha (Benchmark guidance)",
            "calculated_product_per_ha": f"{kg_mop_per_ha:.1f} kg {prod_info.get('name', 'MOP')} / ha",
            "total_product_for_field": f"{total_field_mop:.1f} kg for {field_area_ha:.2f} ha",
            "timing_and_management": prod_info.get("management_note", "")
        })

    # If no deficiencies detected
    if not deficiencies:
        recommendations_list.append({
            "target_nutrient": "Balanced Maintenance",
            "deficiency_status": "Adequate across N, P, K",
            "product_name": "No immediate corrective fertilizer needed",
            "product_grade": "N/A",
            "nutrient_vs_product_explanation": (
                "Remote-sensing vegetation indices indicate adequate canopy vigor and greenness. "
                "Excessive fertilizer application can cause nutrient leaching, increased pest vulnerability, and lodging."
            ),
            "estimated_nutrient_correction_rate": "0.0 kg/ha",
            "calculated_product_per_ha": "0.0 kg/ha",
            "total_product_for_field": "0.0 kg",
            "timing_and_management": "Maintain standard irrigation and scout for early weed or pest pressure."
        })

    # Agronomic Action Rationale
    if deficiencies:
        action_rationale = (
            f"Based on remote-sensing bio-optical features, {crop} canopy signals indicate potential "
            f"limitation in {', '.join(deficiencies)} during the {growth_stage} stage. "
            f"Corrective application should prioritize the limiting nutrient(s) using balanced fertilizer placement."
        )
    else:
        action_rationale = (
            f"Canopy reflectance features show healthy vegetative development for {crop} at {growth_stage} stage. "
            f"Current priority is sustaining soil moisture and crop protection rather than adding extra fertilizer."
        )

    # Soil test verification message
    if has_soil_test:
        soil_test_notice = (
            "User provided soil-test measurements. Note: Laboratory chemical soil tests reflect root-zone nutrient reserves "
            "and take precedence over indirect bio-optical canopy observations."
        )
    else:
        soil_test_notice = (
            "Recommendation is based on remote-sensing / prototype model estimates and MUST be validated with laboratory soil testing. "
            "Field soil test records were not provided."
        )

    return {
        "crop": crop,
        "growth_stage": growth_stage,
        "field_area_ha": field_area_ha,
        "target_yield": target_yield or "Standard regional benchmark",
        "nutrient_focus": nutrient_focus,
        "urgency_level": urgency,
        "recommendations": recommendations_list,
        "action_rationale": action_rationale,
        "soil_test_notice": soil_test_notice,
        "soil_test_provided": has_soil_test,
        "principles": {
            "yield_warning": principles.get("yield_warning", ""),
            "multi_factors": principles.get("yield_multi_factors", [])
        },
        "future_yield_architecture": {
            "status": FutureYieldModelArchitecture.STATUS,
            "required_layers": FutureYieldModelArchitecture.REQUIRED_LAYERS,
            "note": FutureYieldModelArchitecture.NOTE
        },
        "scientific_disclaimer": (
            "This decision support guidance is an educational / research prototype. "
            "It does not guarantee crop yield increases. Actual fertilizer rates must be validated with "
            "local agricultural extension officers, in-situ soil tests, and state fertilizer recommendations."
        )
    }
