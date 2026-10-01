"""
=============================================================================
Soil Nutrient Prediction Inference Script (N, P, K)
=============================================================================
This script loads the three pre-trained Random Forest models and the stored
feature column order from the 'models/' directory to predict Nitrogen (N),
Phosphorus (P), and Potassium (K) for new remote-sensing input observations.

Educational Walkthrough:
------------------------
1. joblib.load() restores the trained Random Forest estimators and feature list.
2. The input observation is formatted as a pandas DataFrame matching the exact
   feature names and sequence used during training.
3. Each model outputs a continuous regression estimate for its respective nutrient.
4. Note: This is an AI/ML prototype trained on a synthetic dataset for pipeline
   demonstration and does not provide fertilizer dosage recommendations.
=============================================================================
"""

import os
import joblib
import pandas as pd

_MODEL_CACHE = {}

def load_models_and_features(models_dir="models"):
    """
    Loads the trained Random Forest models and feature name list using joblib.
    Caches in memory to avoid repeated disk reads.
    """
    if models_dir in _MODEL_CACHE:
        return _MODEL_CACHE[models_dir]

    nitrogen_model_path = os.path.join(models_dir, "nitrogen_model.pkl")
    phosphorus_model_path = os.path.join(models_dir, "phosphorus_model.pkl")
    potassium_model_path = os.path.join(models_dir, "potassium_model.pkl")
    features_path = os.path.join(models_dir, "feature_names.pkl")

    # Check that model files exist
    for p in [nitrogen_model_path, phosphorus_model_path, potassium_model_path, features_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"Required model artifact '{p}' not found! "
                "Please run 'python train_npk_models.py' first to train and save the models."
            )

    print(f"Loading trained models from '{models_dir}/'...")
    model_N = joblib.load(nitrogen_model_path)
    model_P = joblib.load(phosphorus_model_path)
    model_K = joblib.load(potassium_model_path)
    feature_names = joblib.load(features_path)

    _MODEL_CACHE[models_dir] = (model_N, model_P, model_K, feature_names)
    return model_N, model_P, model_K, feature_names


def predict_soil_nutrients(input_data: dict, models_dir="models"):
    """
    Predicts Nitrogen, Phosphorus, and Potassium for a given dictionary of features.
    """
    # 1. Load models and exact feature column order
    model_N, model_P, model_K, feature_names = load_models_and_features(models_dir)

    # 2. Convert dictionary into a pandas DataFrame matching feature_names
    input_df = pd.DataFrame([input_data])[feature_names]

    # 3. Generate predictions using each specialized Random Forest model
    pred_N = model_N.predict(input_df)[0]
    pred_P = model_P.predict(input_df)[0]
    pred_K = model_K.predict(input_df)[0]

    return {
        "N": pred_N,
        "P": pred_P,
        "K": pred_K
    }


def main():
    ndvi = float(input("Enter NDVI: "))
    evi = float(input("Enter EVI: "))
    fvc = float(input("Enter FVC: "))
    lai = float(input("Enter LAI: "))
    ndvi_texture = float(input("Enter NDVI Texture: "))
    biomass = float(input("Enter Biomass: "))

    input_data = {
        "NDVI": ndvi,
        "EVI": evi,
        "FVC": fvc,
        "LAI": lai,
        "NDVI Texture": ndvi_texture,
        "Biomass": biomass
    }

    predictions = predict_soil_nutrients(input_data)

    print("\n--------------------------------")
    print("INPUT FEATURES")
    print("--------------------------------")
    print(f"NDVI: {input_data['NDVI']}")
    print(f"EVI: {input_data['EVI']}")
    print(f"FVC: {input_data['FVC']}")
    print(f"LAI: {input_data['LAI']}")
    print(f"NDVI Texture: {input_data['NDVI Texture']}")
    print(f"Biomass: {input_data['Biomass']}")
    print()
    print("--------------------------------")
    print("PREDICTED NUTRIENTS")
    print("--------------------------------")
    print(f"Nitrogen: {predictions['N']:.2f}")
    print(f"Phosphorus: {predictions['P']:.2f}")
    print(f"Potassium: {predictions['K']:.2f}")
    print("--------------------------------")


if __name__ == "__main__":
    main()
