"""
=============================================================================
Crop Soil Nutrient Estimation (N, P, K) Using Random Forest Regressors
=============================================================================
This script trains three separate Random Forest Regression models to estimate:
  1. Nitrogen (N)
  2. Phosphorus (P)
  3. Potassium (K)

from six satellite-derived vegetation and canopy features:
  - NDVI (Normalized Difference Vegetation Index)
  - EVI (Enhanced Vegetation Index)
  - FVC (Fraction of Vegetation Cover)
  - LAI (Leaf Area Index)
  - NDVI Texture (Canopy Heterogeneity)
  - Biomass (Crop Biomass Proxy)

Educational / Beginner-Friendly Walkthrough:
--------------------------------------------
1. Loading the CSV with pandas.
2. Separating Features (X) and Targets (y) without target leakage.
3. Why separate models are trained for N, P, and K.
4. Splitting into 80% train and 20% test with random_state=42.
5. How Random Forest Regression works and what n_estimators=200 means.
6. Evaluating predictions using MAE, RMSE, and R-squared.
7. Inspecting Actual vs. Predicted values on test samples.
8. Visualizing feature importances and diagnostic scatter plots.
9. Persisting trained models and feature order with joblib.
=============================================================================
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def main():
    print("=" * 75)
    print("  AGRICULTURAL AI/ML: NPK SOIL NUTRIENT REGRESSION TRAINING")
    print("=" * 75)

    # -------------------------------------------------------------------------
    # 1. LOAD THE DATASET
    # -------------------------------------------------------------------------
    # We use pandas read_csv() to load our tabular dataset into a DataFrame.
    # The file contains 5,000 observations of crop remote sensing features and
    # corresponding laboratory-style soil nutrient values.
    csv_file = "synthetic_crop_data.csv"
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"Dataset file '{csv_file}' not found in working directory!")

    print(f"\n[Step 1] Loading dataset from '{csv_file}'...")
    df = pd.read_csv(csv_file)
    print(f"Dataset loaded successfully. Shape: {df.shape[0]:,} rows and {df.shape[1]} columns.")

    # -------------------------------------------------------------------------
    # 2. SELECT INPUT FEATURES (X) AND TARGETS (y)
    # -------------------------------------------------------------------------
    # Rule against Target Leakage:
    # Target leakage occurs when information from the target variable is
    # mistakenly included in the training features. N, P, and K are our target
    # variables, so they must NEVER be in X.
    #
    # We also exclude Date, Location, Crop, Growth Stage, and NDVI Zone for this
    # first baseline model to evaluate pure bio-optical remote-sensing signals.
    feature_columns = [
        "NDVI",
        "EVI",
        "FVC",
        "LAI",
        "NDVI Texture",
        "Biomass"
    ]
    target_columns = ["N", "P", "K"]

    print(f"\n[Step 2] Defining Input Features (X) and Targets (y):")
    print(f" - Input Features ({len(feature_columns)}): {', '.join(feature_columns)}")
    print(f" - Target Nutrients ({len(target_columns)}): {', '.join(target_columns)}")
    print(" - Excluded (to prevent leakage / keep baseline purely spectral):")
    print("   N, P, K (Targets), Date, Location, Crop, Growth Stage, NDVI Zone.")

    X = df[feature_columns].copy()
    y = df[target_columns].copy()

    # -------------------------------------------------------------------------
    # 3. TRAIN / TEST SPLIT (80% Train, 20% Test)
    # -------------------------------------------------------------------------
    # Why split the data?
    # If we evaluate a model on the same data it trained on, it might simply
    # memorize the rows (overfitting). Splitting guarantees that test data
    # remains completely unseen during training, measuring true generalizability.
    #
    # Using random_state=42 ensures exact reproducibility.
    # We split X and y together so that all three nutrient models (N, P, K)
    # are trained and evaluated on the exact same rows.
    print(f"\n[Step 3] Splitting dataset into 80% Training and 20% Testing (random_state=42)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    print(f" - Training samples: {X_train.shape[0]:,} rows")
    print(f" - Testing samples : {X_test.shape[0]:,} rows")

    # -------------------------------------------------------------------------
    # 4. RANDOM FOREST REGRESSION OVERVIEW & CONFIGURATION
    # -------------------------------------------------------------------------
    # How Random Forest Regression works:
    # - A Decision Tree splits data based on feature thresholds to make a prediction.
    # - A single decision tree can be noisy and prone to overfitting.
    # - A Random Forest combines an ensemble (forest) of many decision trees.
    #
    # What n_estimators=200 means:
    # - n_estimators=200 means the algorithm constructs 200 distinct decision trees.
    # - Each tree learns slightly different patterns because Random Forest uses:
    #     a) Bootstrap sampling (each tree gets a random subset of training rows).
    #     b) Feature subsampling (at each decision split, a random subset of features is evaluated).
    # - When making a prediction for a new sample, all 200 trees cast their prediction.
    # - The forest averages the 200 individual predictions to produce the final,
    #   smooth, and robust estimate.
    #
    # Why three separate models?
    # - Nitrogen, Phosphorus, and Potassium have different biochemical behaviors
    #   and distinct functional relationships with canopy chlorophyll, leaf structure,
    #   and biomass accumulation. Dedicated models allow the trees to specialize
    #   in the specific patterns of each nutrient.
    rf_config = {
        "n_estimators": 200,   # 200 decision trees
        "random_state": 42,    # Reproducible bootstrap sampling
        "n_jobs": -1           # Utilize all available CPU cores for parallel training
    }

    models = {
        "Nitrogen": RandomForestRegressor(**rf_config),
        "Phosphorus": RandomForestRegressor(**rf_config),
        "Potassium": RandomForestRegressor(**rf_config)
    }

    # -------------------------------------------------------------------------
    # 5. TRAIN THE THREE MODELS
    # -------------------------------------------------------------------------
    print("\n[Step 5] Training Random Forest Regressors (200 trees each)...")
    predictions = {}
    metrics = {}

    target_mapping = {
        "Nitrogen": "N",
        "Phosphorus": "P",
        "Potassium": "K"
    }

    for nutrient_name, col_key in target_mapping.items():
        print(f" -> Training {nutrient_name} Model (Target: {col_key})...")
        models[nutrient_name].fit(X_train, y_train[col_key])
        
        # Predict on the unseen 20% test set
        pred = models[nutrient_name].predict(X_test)
        predictions[col_key] = pred

        # ---------------------------------------------------------------------
        # 6. EVALUATION METRICS
        # ---------------------------------------------------------------------
        # MAE (Mean Absolute Error):
        # Average absolute difference between the actual and predicted nutrient values.
        # Expressed in the exact same units as the target (kg/ha).
        #
        # RMSE (Root Mean Squared Error):
        # Square root of the average squared errors. Penalizes larger errors more heavily
        # than MAE. Very useful for flagging occasional large mistakes.
        #
        # R² (Coefficient of Determination):
        # Measures the proportion of variance in the target that is predictable
        # from the remote-sensing features (1.0 = perfect prediction, 0.0 = baseline mean).
        actual = y_test[col_key]
        mae = mean_absolute_error(actual, pred)
        rmse = np.sqrt(mean_squared_error(actual, pred))
        r2 = r2_score(actual, pred)

        metrics[nutrient_name] = {
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2
        }

    # Print Evaluation Summary
    print("\n" + "=" * 75)
    print("                      MODEL EVALUATION RESULTS (TEST SET)")
    print("=" * 75)
    for nutrient_name in ["Nitrogen", "Phosphorus", "Potassium"]:
        m = metrics[nutrient_name]
        print(f"{nutrient_name} Model")
        print(f"MAE  : {m['MAE']:.3f} kg/ha")
        print(f"RMSE : {m['RMSE']:.3f} kg/ha")
        print(f"R²   : {m['R2']:.4f}\n")

    print("-" * 75)
    print("Metric Explanations:")
    print(" * MAE  (Mean Absolute Error): Average absolute difference between actual & predicted.")
    print(" * RMSE (Root Mean Squared Error): Error with heavier penalty on large discrepancies.")
    print(" * R²   (R-squared): Fraction of variance explained by remote-sensing features.")
    print("\n[NOTE]: This model is trained on a synthetic dataset for demonstration purposes.")
    print("It demonstrates the ML pipeline, but real-world agricultural validation requires")
    print("calibrated satellite imagery and verified soil lab tests.")
    print("=" * 75)

    # -------------------------------------------------------------------------
    # 7. DISPLAY 10 ACTUAL VS PREDICTED TEST SAMPLES
    # -------------------------------------------------------------------------
    print("\n[Step 7] Comparison: 10 Unseen Test Samples (Actual vs. Predicted):")
    print("-" * 75)
    print(f"{'Sample':<8} | {'Actual N':<10} {'Pred N':<10} | {'Actual P':<10} {'Pred P':<10} | {'Actual K':<10} {'Pred K':<10}")
    print("-" * 75)

    # Convert to numpy arrays for clean indexed printing
    act_N = y_test["N"].values
    pr_N = predictions["N"]
    act_P = y_test["P"].values
    pr_P = predictions["P"]
    act_K = y_test["K"].values
    pr_K = predictions["K"]

    for i in range(10):
        print(f"#{i+1:<7} | {act_N[i]:<10.1f} {pr_N[i]:<10.2f} | {act_P[i]:<10.1f} {pr_P[i]:<10.2f} | {act_K[i]:<10.1f} {pr_K[i]:<10.2f}")
    print("-" * 75)

    # -------------------------------------------------------------------------
    # 8 & 9. GENERATE AND SAVE PLOTS (RESULTS FOLDER)
    # -------------------------------------------------------------------------
    os.makedirs("results", exist_ok=True)
    print("\n[Step 8 & 9] Generating Diagnostic Visualizations in 'results/'...")

    color_scheme = {
        "Nitrogen": {"bar": "#10b981", "scatter": "#059669", "key": "N"},
        "Phosphorus": {"bar": "#3b82f6", "scatter": "#2563eb", "key": "P"},
        "Potassium": {"bar": "#f59e0b", "scatter": "#d97706", "key": "K"}
    }

    # 8. Feature Importance Bar Charts
    for nutrient_name, info in color_scheme.items():
        col_key = info["key"]
        importances = models[nutrient_name].feature_importances_
        
        plt.figure(figsize=(9, 5))
        bars = plt.bar(feature_columns, importances, color=info["bar"], edgecolor="black", alpha=0.85)
        plt.title(f"{nutrient_name} (Target: {col_key}) - Random Forest Feature Importance", fontsize=12, fontweight="bold", pad=12)
        plt.xlabel("Input Remote-Sensing Features", fontsize=10, fontweight="bold")
        plt.ylabel("Importance Score (Gini / Variance Reduction)", fontsize=10, fontweight="bold")
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.ylim(0, max(importances) * 1.25)

        # Annotate bar values
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2.0, height + 0.01, f"{height:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

        plt.tight_layout()
        chart_path = os.path.join("results", f"{nutrient_name.lower()}_feature_importance.png")
        plt.savefig(chart_path, dpi=300)
        plt.close()
        print(f" -> Saved feature importance chart: {chart_path}")

    # 9. Actual vs Predicted Scatter Plots
    for nutrient_name, info in color_scheme.items():
        col_key = info["key"]
        act = y_test[col_key].values
        prd = predictions[col_key]

        plt.figure(figsize=(7, 7))
        plt.scatter(act, prd, alpha=0.45, color=info["scatter"], edgecolors="none", s=32, label="Test Observations")
        
        # 45-degree reference line representing Perfect Prediction (Actual == Predicted)
        min_val = min(act.min(), prd.min()) - 1
        max_val = max(act.max(), prd.max()) + 1
        plt.plot([min_val, max_val], [min_val, max_val], color="red", linestyle="--", linewidth=2, label="Ideal Line (Predicted = Actual)")
        
        m = metrics[nutrient_name]
        plt.title(f"{nutrient_name} (Target: {col_key}): Actual vs. Predicted Values\nMAE: {m['MAE']:.2f} | RMSE: {m['RMSE']:.2f} | R²: {m['R2']:.3f}", fontsize=11, fontweight="bold", pad=10)
        plt.xlabel(f"Actual {nutrient_name} ({col_key}) [kg/ha]", fontsize=10, fontweight="bold")
        plt.ylabel(f"Predicted {nutrient_name} ({col_key}) [kg/ha]", fontsize=10, fontweight="bold")
        plt.xlim(min_val, max_val)
        plt.ylim(min_val, max_val)
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.legend(loc="upper left")
        
        plt.tight_layout()
        scatter_path = os.path.join("results", f"{nutrient_name.lower()}_actual_vs_predicted.png")
        plt.savefig(scatter_path, dpi=300)
        plt.close()
        print(f" -> Saved actual vs predicted plot : {scatter_path}")

    # -------------------------------------------------------------------------
    # 10. SAVE TRAINED MODELS AND FEATURE NAMES LIST
    # -------------------------------------------------------------------------
    os.makedirs("models", exist_ok=True)
    print("\n[Step 10] Saving Trained Models & Feature List to 'models/'...")

    joblib.dump(models["Nitrogen"], os.path.join("models", "nitrogen_model.pkl"))
    joblib.dump(models["Phosphorus"], os.path.join("models", "phosphorus_model.pkl"))
    joblib.dump(models["Potassium"], os.path.join("models", "potassium_model.pkl"))
    joblib.dump(feature_columns, os.path.join("models", "feature_names.pkl"))

    print(" -> Saved models/nitrogen_model.pkl")
    print(" -> Saved models/phosphorus_model.pkl")
    print(" -> Saved models/potassium_model.pkl")
    print(" -> Saved models/feature_names.pkl")

    print("\n" + "=" * 75)
    print("TRAINING SCRIPT FINISHED SUCCESSFULLY!")
    print("=" * 75)

if __name__ == "__main__":
    main()
