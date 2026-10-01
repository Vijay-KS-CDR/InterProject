"""
Agricultural Decision Support Package.
Provides nutrient status interpretation, deficiency identification,
fertilizer product advisory, and agronomic management guidance.
"""
from decision_support.nutrient_interpreter import interpret_nutrient_status
from decision_support.fertilizer_advisor import generate_fertilizer_advisory, FutureYieldModelArchitecture

__all__ = [
    "interpret_nutrient_status",
    "generate_fertilizer_advisory",
    "FutureYieldModelArchitecture"
]
