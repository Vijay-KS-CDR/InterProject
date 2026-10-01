"""
Unit tests for the Agricultural Decision Support layer:
- Nutrient status interpretation (Low, Adequate, High)
- Deficiency identification & priorities
- Fertilizer product composition conversion (Urea, DAP, MOP)
- Verification of soil test notices & multi-factor yield architecture
"""

import sys
import unittest
from decision_support.nutrient_interpreter import interpret_nutrient_status
from decision_support.fertilizer_advisor import generate_fertilizer_advisory, FutureYieldModelArchitecture


def test_nutrient_interpretation_low():
    # Vegetative stage for Rice: N < 38 is Low, P < 17 is Low, K < 28 is Low
    est_nutrients = {"N": 30.0, "P": 12.0, "K": 22.0}
    res = interpret_nutrient_status("Rice", "Vegetative", est_nutrients)
    
    assert res["statuses"]["N"]["status"] == "Low"
    assert res["statuses"]["P"]["status"] == "Low"
    assert res["statuses"]["K"]["status"] == "Low"
    assert "Nitrogen" in res["deficiencies"]
    assert "Phosphorus" in res["deficiencies"]
    assert "Potassium" in res["deficiencies"]


def test_nutrient_interpretation_adequate():
    # Vegetative stage for Rice: N 38-45 is Adequate, P 17-22 is Adequate, K 28-36 is Adequate
    est_nutrients = {"N": 42.0, "P": 19.5, "K": 32.0}
    res = interpret_nutrient_status("Rice", "Vegetative", est_nutrients)
    
    assert res["statuses"]["N"]["status"] == "Adequate"
    assert res["statuses"]["P"]["status"] == "Adequate"
    assert res["statuses"]["K"]["status"] == "Adequate"
    assert len(res["deficiencies"]) == 0


def test_nutrient_interpretation_high():
    # Vegetative stage for Rice: N > 45 is High
    est_nutrients = {"N": 52.0, "P": 20.0, "K": 30.0}
    res = interpret_nutrient_status("Rice", "Vegetative", est_nutrients)
    
    assert res["statuses"]["N"]["status"] == "High"
    assert res["statuses"]["P"]["status"] == "Adequate"
    assert res["statuses"]["K"]["status"] == "Adequate"
    assert "Nitrogen" not in res["deficiencies"]


def test_fertilizer_advisory_nitrogen_deficiency():
    interpretation = {
        "crop": "Rice",
        "growth_stage": "Vegetative",
        "deficiencies": ["Nitrogen"],
        "statuses": {"N": {"status": "Low"}}
    }
    
    advisory = generate_fertilizer_advisory(interpretation, field_area_ha=2.0)
    
    assert len(advisory["recommendations"]) == 1
    rec = advisory["recommendations"][0]
    assert rec["target_nutrient"] == "Nitrogen (N)"
    assert "Urea" in rec["product_name"]
    # 20 kg N deficit * 2.174 kg Urea/kg N = 43.5 kg/ha; for 2 ha = 87.0 kg
    assert "43.5 kg" in rec["calculated_product_per_ha"]
    assert "87.0 kg" in rec["total_product_for_field"]
    assert advisory["soil_test_provided"] is False
    assert "MUST be validated with laboratory soil testing" in advisory["soil_test_notice"]


def test_fertilizer_advisory_with_soil_tests():
    interpretation = {
        "crop": "Wheat",
        "growth_stage": "Early",
        "deficiencies": ["Phosphorus"],
        "statuses": {"P": {"status": "Low"}}
    }
    soil_tests = {"N": 280.0, "P": 9.5, "K": 180.0}
    advisory = generate_fertilizer_advisory(interpretation, soil_test_values=soil_tests)
    
    assert advisory["soil_test_provided"] is True
    assert "Laboratory chemical soil tests reflect root-zone nutrient reserves" in advisory["soil_test_notice"]


def test_future_yield_architecture():
    # Ensures that yield architecture is a documented schema and does NOT return fake numbers
    assert FutureYieldModelArchitecture.STATUS == "ARCHITECTURE_STUB_ONLY"
    assert len(FutureYieldModelArchitecture.REQUIRED_LAYERS) >= 5
    assert "No fake yield predictions" in FutureYieldModelArchitecture.NOTE


if __name__ == "__main__":
    test_nutrient_interpretation_low()
    test_nutrient_interpretation_adequate()
    test_nutrient_interpretation_high()
    test_fertilizer_advisory_nitrogen_deficiency()
    test_fertilizer_advisory_with_soil_tests()
    test_future_yield_architecture()
    print("ALL 6 DECISION SUPPORT TESTS PASSED SUCCESSFULLY!")
