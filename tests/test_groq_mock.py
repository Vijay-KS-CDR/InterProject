"""
Unit tests for groq_advisor.py logic, prompt construction, and fallbacks.
"""

import unittest
from unittest import mock
import json
from groq_advisor import generate_agricultural_advice


class TestGroqAdvisor(unittest.TestCase):
    def setUp(self):
        self.sample_features = {
            "NDVI": 0.72,
            "EVI": 0.48,
            "FVC": 0.76,
            "LAI": 3.4,
            "NDVI Texture": 0.06,
            "Biomass": 6.2
        }
        self.sample_preds = {
            "N": 44.60,
            "P": 20.18,
            "K": 33.90
        }

    def test_missing_api_key_fallback(self):
        with mock.patch("groq_advisor.get_groq_client", return_value=None):
            res = generate_agricultural_advice(self.sample_preds, self.sample_features)
            self.assertFalse(res["available"])
            self.assertIn("temporarily unavailable", res["error"])
            self.assertIsNone(res["advice"])

    def test_successful_groq_response_and_prompt(self):
        mock_client = mock.MagicMock()
        mock_payload = {
            "summary": "Canopy displays vigorous growth with moderate vegetative density.",
            "nutrient_observations": ["Nitrogen levels reflect active chlorophyll synthesis."],
            "possible_concerns": ["No acute spectral deficiencies detected at this stage."],
            "actions_to_consider": ["Inspect soil moisture and continue standard monitoring."],
            "things_to_monitor": ["Leaf color uniformity and local pest indicators."],
            "validation_note": "These nutrient values are model estimates and should be validated with appropriate soil/field observations before fertilizer application."
        }
        mock_resp = mock.MagicMock()
        mock_resp.choices = [mock.MagicMock(message=mock.MagicMock(content=json.dumps(mock_payload)))]
        mock_client.chat.completions.create.return_value = mock_resp

        with mock.patch("groq_advisor.get_groq_client", return_value=mock_client):
            res = generate_agricultural_advice(
                self.sample_preds,
                self.sample_features,
                crop="Rice",
                growth_stage="Vegetative"
            )
            self.assertTrue(res["available"])
            self.assertEqual(res["advice"]["summary"], mock_payload["summary"])

            # Verify prompt contains all 11 required fields
            call_kwargs = mock_client.chat.completions.create.call_args[1]
            prompt = call_kwargs["messages"][1]["content"]
            expected_fields = [
                "Rice", "Vegetative",
                "44.60", "20.18", "33.90",
                "0.7200", "0.4800", "0.7600", "3.40", "0.0600", "6.20"
            ]
            for item in expected_fields:
                self.assertIn(item, prompt, f"Expected '{item}' in prompt")

    def test_malformed_json_fallback(self):
        mock_client = mock.MagicMock()
        mock_resp = mock.MagicMock()
        mock_resp.choices = [mock.MagicMock(message=mock.MagicMock(content="Invalid non-json response"))]
        mock_client.chat.completions.create.return_value = mock_resp

        with mock.patch("groq_advisor.get_groq_client", return_value=mock_client):
            res = generate_agricultural_advice(self.sample_preds, self.sample_features)
            self.assertFalse(res["available"])
            self.assertIn("temporarily unavailable", res["error"])
            self.assertIsNone(res["advice"])

    def test_missing_required_json_keys(self):
        mock_client = mock.MagicMock()
        incomplete_payload = {"summary": "Partial response"}
        mock_resp = mock.MagicMock()
        mock_resp.choices = [mock.MagicMock(message=mock.MagicMock(content=json.dumps(incomplete_payload)))]
        mock_client.chat.completions.create.return_value = mock_resp

        with mock.patch("groq_advisor.get_groq_client", return_value=mock_client):
            res = generate_agricultural_advice(self.sample_preds, self.sample_features)
            self.assertFalse(res["available"])
            self.assertIn("temporarily unavailable", res["error"])


if __name__ == "__main__":
    unittest.main()
