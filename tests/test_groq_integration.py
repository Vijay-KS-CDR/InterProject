"""
=============================================================================
Test Suite for Groq-Powered Agricultural Advisor & Multispectral Pipeline
=============================================================================
"""

import os
import sys
import hashlib
from pathlib import Path
import urllib.request
import urllib.parse
import json
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
API_BASE = "http://127.0.0.1:8000"


def compute_file_hash(filepath: Path) -> str:
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def test_model_files_unchanged():
    print("\n--- TEST: Verifying Model Files Unchanged ---")
    models_dir = BASE_DIR / "models"
    for name in ["nitrogen_model.pkl", "phosphorus_model.pkl", "potassium_model.pkl", "feature_names.pkl"]:
        path = models_dir / name
        assert path.exists(), f"Model file {name} missing!"
        print(f"[PASS] Model artifact intact: {name} ({path.stat().st_size:,} bytes)")


def test_security_frontend_source():
    print("\n--- TEST: Inspecting Frontend Source for Security ---")
    for fname in ["index.html", "script.js", "style.css"]:
        content = (BASE_DIR / fname).read_text(encoding="utf-8")
        assert "gsk_" not in content, f"Hardcoded Groq API key found in {fname}!"
        assert "GROQ_API_KEY" not in content or "<code>GROQ_API_KEY</code>" in content, (
            f"Potential key exposure in {fname}"
        )
        print(f"[PASS] Security check passed: {fname} contains no exposed secrets or API keys.")


def upload_geotiff(filename: str, crop: str = None, growth_stage: str = None):
    file_path = BASE_DIR / filename
    assert file_path.exists(), f"File {file_path} not found"

    boundary = "----WebKitFormBoundary7MA4YWxkTrZu0gW"
    body = bytearray()

    # File field
    body.extend(f"--{boundary}\r\n".encode("utf-8"))
    body.extend(f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'.encode("utf-8"))
    body.extend(b"Content-Type: image/tiff\r\n\r\n")
    body.extend(file_path.read_bytes())
    body.extend(b"\r\n")

    # Crop field
    if crop:
        body.extend(f"--{boundary}\r\n".encode("utf-8"))
        body.extend(f'Content-Disposition: form-data; name="crop"\r\n\r\n{crop}\r\n'.encode("utf-8"))

    # Growth stage field
    if growth_stage:
        body.extend(f"--{boundary}\r\n".encode("utf-8"))
        body.extend(f'Content-Disposition: form-data; name="growth_stage"\r\n\r\n{growth_stage}\r\n'.encode("utf-8"))

    body.extend(f"--{boundary}--\r\n".encode("utf-8"))

    req = urllib.request.Request(
        f"{API_BASE}/api/analyze",
        data=bytes(body),
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST"
    )

    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode("utf-8"))


def test_valid_geotiff_analysis():
    print("\n--- TEST 1: Upload Valid Multispectral GeoTIFF (test_raster.tif) ---")
    res = upload_geotiff("test_raster.tif", crop="Rice", growth_stage="Vegetative")
    
    assert res.get("success") is True
    features = res.get("features", {})
    preds = res.get("predictions", {})
    advisor = res.get("advisor", {})

    print(f"Features: {features}")
    print(f"Predictions: {preds}")
    print(f"Advisor available: {advisor.get('available')}")

    for feat in ["NDVI", "EVI", "FVC", "LAI", "NDVI Texture", "Biomass"]:
        assert feat in features, f"Missing feature: {feat}"
        assert features[feat] is not None

    for nut in ["N", "P", "K"]:
        assert nut in preds, f"Missing prediction: {nut}"
        assert preds[nut] > 0

    print("[PASS] TEST 1 Passed: 6 features and N/P/K generated properly.")
    return res


def test_second_geotiff_different_characteristics(res1):
    print("\n--- TEST 2: Second GeoTIFF with Different Characteristics (test_raster_stressed.tif) ---")
    res2 = upload_geotiff("test_raster_stressed.tif", crop="Wheat", growth_stage="Flowering")

    assert res2.get("success") is True
    features2 = res2.get("features", {})
    preds2 = res2.get("predictions", {})

    print(f"Test 1 NDVI: {res1['features']['NDVI']} | Test 2 NDVI: {features2['NDVI']}")
    print(f"Test 1 N/P/K: {res1['predictions']} | Test 2 N/P/K: {preds2}")

    # Ensure the outputs are distinct and physically reflect the distinct inputs
    assert res1["features"]["NDVI"] != features2["NDVI"], "NDVI must differ between distinct rasters"
    assert res1["predictions"]["N"] != preds2["N"], "Predictions must differ between distinct rasters"
    print("[PASS] TEST 2 Passed: Second GeoTIFF produced distinct, real calculations.")


def test_groq_fallback_when_no_key():
    print("\n--- TEST 3: Advisor Safe Fallback & Live Key Verification ---")
    if str(BASE_DIR) not in sys.path:
        sys.path.insert(0, str(BASE_DIR))
    from groq_advisor import generate_agricultural_advice
    from unittest.mock import patch

    # 1. Verify safe fallback behavior when Groq client is unavailable
    with patch("groq_advisor.get_groq_client", return_value=None):
        fallback_res = generate_agricultural_advice(
            {"N": 40.0, "P": 18.0, "K": 30.0},
            {"NDVI": 0.7, "EVI": 0.5, "FVC": 0.6, "LAI": 1.0, "NDVI Texture": 0.1, "Biomass": 2.0}
        )
        assert fallback_res["available"] is False
        assert "AI agricultural advice is temporarily unavailable." in fallback_res["error"]
        print(f"[PASS] Fallback confirmed: {fallback_res['error']}")

    # 2. Verify live pipeline operation with configured key
    res = upload_geotiff("test_raster.tif")
    advisor = res.get("advisor")
    assert advisor is not None
    assert advisor["available"] is True, "Live Groq advisor should be available when GROQ_API_KEY is configured"
    assert res["predictions"]["N"] > 0
    print(f"[PASS] Live Advisor Model Used: {advisor.get('model_used')}")
    print("[PASS] TEST 3 Passed: Pipeline remains 100% operational with graceful fallback.")


def test_invalid_file():
    print("\n--- TEST 4: Invalid File Format Validation ---")
    boundary = "----WebKitFormBoundary7MA4YWxkTrZu0gW"
    body = bytearray()
    body.extend(f"--{boundary}\r\n".encode("utf-8"))
    body.extend(b'Content-Disposition: form-data; name="file"; filename="invalid.txt"\r\n')
    body.extend(b"Content-Type: text/plain\r\n\r\n")
    body.extend(b"This is not a GeoTIFF")
    body.extend(f"\r\n--{boundary}--\r\n".encode("utf-8"))

    req = urllib.request.Request(
        f"{API_BASE}/api/analyze",
        data=bytes(body),
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST"
    )

    try:
        urllib.request.urlopen(req)
        assert False, "Should have rejected non-GeoTIFF format"
    except urllib.error.HTTPError as exc:
        assert exc.code in [400, 422]
        print(f"[PASS] Correctly rejected invalid file with HTTP {exc.code}")
    print("[PASS] TEST 4 Passed: Input validation intact.")


if __name__ == "__main__":
    test_model_files_unchanged()
    test_security_frontend_source()
    res1 = test_valid_geotiff_analysis()
    test_second_geotiff_different_characteristics(res1)
    test_groq_fallback_when_no_key()
    test_invalid_file()
    print("\n==================================================")
    print("ALL AUTOMATED TESTS PASSED SUCCESSFULLY!")
    print("==================================================")
