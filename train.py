"""
Train crop water requirement model and save pipeline + config for the API.
Implements: target cap, 15-zone -> 4-climate mapping, median aggregation,
stratified split, RF tuning, CV and crop-level evaluation.
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# Paths
DATA_PATH = Path(__file__).resolve().parent / "DATASET_15_agro_zones.csv"
MODEL_PATH = Path(__file__).resolve().parent / "model.joblib"
CONFIG_PATH = Path(__file__).resolve().parent / "config.json"

CATEGORICAL_COLS = ["CROP TYPE", "SOIL TYPE", "REGION", "WEATHER CONDITION"]
NUMERIC_FEATURES = ["temp_mid", "temp_mid_sq"]
TARGET_COL = "WATER REQUIREMENT"

# Cap target at agronomic max (mm/day) to reduce outlier impact
TARGET_CAP_MM = 20.0

# 15 agro-climatic zones -> 4 climates for training (fewer, clearer region signal)
ZONE_TO_CLIMATE = {
    "Western Himalayan Region": "SEMI HUMID",
    "Eastern Himalayan Region": "HUMID",
    "Lower Gangetic Plain Region": "HUMID",
    "Middle Gangetic Plain Region": "HUMID",
    "Upper Gangetic Plain Region": "SEMI ARID",
    "Trans-Gangetic Plain Region": "SEMI ARID",
    "Eastern Plateau & Hills Region": "SEMI HUMID",
    "Central Plateau & Hills Region": "SEMI ARID",
    "Western Plateau & Hills Region": "SEMI ARID",
    "Southern Plateau & Hills Region": "SEMI HUMID",
    "East Coast Plains & Hills Region": "HUMID",
    "West Coast Plains & Ghats Region": "HUMID",
    "Gujarat Plains & Hills Region": "SEMI ARID",
    "Western Dry Region": "DESERT",
    "Island Region": "HUMID",
}

# Crop-wise physical minimum (mm/day). Applied only at inference.
CROP_MIN_MM = {
    "RICE": 4.0,
    "WHEAT": 2.0,
    "MAIZE": 3.0,
}


def parse_temperature_midpoint(temp_str: str) -> float:
    """Parse '10-20' -> 15, '20-30' -> 25, etc."""
    low, high = temp_str.strip().split("-")
    return (int(low) + int(high)) / 2.0


def load_and_prepare_data() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, list]:
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=[TARGET_COL])

    # Cap target to agronomic max to reduce outlier impact
    df[TARGET_COL] = df[TARGET_COL].clip(upper=TARGET_CAP_MM)

    # Save 15 zone names for API config before mapping to climate
    region_list_15_zones = sorted(df["REGION"].dropna().unique().tolist())

    # Map 15 zones -> 4 climates for training
    df = df.copy()
    df["REGION"] = df["REGION"].map(ZONE_TO_CLIMATE)
    df = df.dropna(subset=["REGION"])

    # Aggregate duplicates: one row per (crop, soil, region, temp, weather), median target
    key = CATEGORICAL_COLS + ["TEMPERATURE"]
    df = df.groupby(key, as_index=False)[TARGET_COL].median()

    df["TEMPERATURE"] = df["TEMPERATURE"].astype(str)
    X = df[CATEGORICAL_COLS + ["TEMPERATURE"]].copy()
    X["temp_mid"] = X["TEMPERATURE"].map(parse_temperature_midpoint)
    X["temp_mid_sq"] = X["temp_mid"] ** 2
    X_feat = X[CATEGORICAL_COLS + NUMERIC_FEATURES]
    y = df[TARGET_COL]
    return X_feat, y, X, region_list_15_zones


def print_target_diagnostics(df: pd.DataFrame, y: pd.Series) -> None:
    """Print target distribution overall and for RICE."""
    print("\n--- Target diagnostics (after cap & aggregate) ---")
    print("Overall target distribution:")
    print(y.describe().to_string())
    rice_mask = df["CROP TYPE"] == "RICE"
    if rice_mask.any():
        y_rice = y.loc[rice_mask]
        print("\nTarget distribution for CROP TYPE == 'RICE':")
        print(y_rice.describe().to_string())
    else:
        print("\nNo rows with CROP TYPE == 'RICE' in dataset.")
    print("---\n")


def get_allowed_categories(X_with_temp: pd.DataFrame, region_list_15: list) -> dict:
    return {
        "crop_type": sorted(X_with_temp["CROP TYPE"].dropna().unique().tolist()),
        "soil_type": sorted(X_with_temp["SOIL TYPE"].dropna().unique().tolist()),
        "region": region_list_15,
        "weather_condition": sorted(X_with_temp["WEATHER CONDITION"].dropna().unique().tolist()),
        "temperature": sorted(X_with_temp["TEMPERATURE"].dropna().unique().tolist()),
    }


def print_top_feature_importances(pipe: Pipeline, top_k: int = 10) -> None:
    preprocessor = pipe.named_steps["preprocessor"]
    regressor = pipe.named_steps["regressor"]
    names = preprocessor.get_feature_names_out()
    imp = regressor.feature_importances_
    pairs = sorted(zip(names, imp), key=lambda x: -x[1])
    print(f"\n--- Top {top_k} feature importances ---")
    for name, importance in pairs[:top_k]:
        print(f"  {name}: {importance:.4f}")
    print("---\n")


def print_crop_level_r2(pipe: Pipeline, X_val: pd.DataFrame, y_val: pd.Series) -> None:
    """Print R² per crop on validation set."""
    crops = X_val["CROP TYPE"].unique()
    print("\n--- R² by crop (validation) ---")
    for crop in sorted(crops):
        mask = X_val["CROP TYPE"] == crop
        if mask.sum() < 2:
            continue
        X_c = X_val.loc[mask]
        y_c = y_val.loc[mask]
        r2 = pipe.score(X_c, y_c)
        print(f"  {crop}: R² = {r2:.4f} (n={mask.sum()})")
    print("---\n")


def main():
    X, y, X_with_temp, region_list_15_zones = load_and_prepare_data()

    print_target_diagnostics(X_with_temp, y)

    # Stratified split by crop type
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=X["CROP TYPE"]
    )

    preprocessor = ColumnTransformer(
        [
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_COLS,
            ),
            ("num", "passthrough", NUMERIC_FEATURES),
        ],
        remainder="drop",
    )

    base_pipe = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("regressor", RandomForestRegressor(random_state=42, n_jobs=-1)),
        ]
    )

    param_dist = {
        "regressor__n_estimators": [200, 300, 400],
        "regressor__max_depth": [6, 8, 10, 12],
        "regressor__min_samples_leaf": [3, 5, 8, 12],
    }

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    search = RandomizedSearchCV(
        base_pipe,
        param_distributions=param_dist,
        n_iter=16,
        cv=skf.split(X_train, X_train["CROP TYPE"]),
        scoring="r2",
        random_state=42,
        n_jobs=1,
    )
    search.fit(X_train, y_train)

    print(f"Best CV R² (mean): {search.best_score_:.4f}")
    print(f"Best params: {search.best_params_}")

    # 5-fold CV on full train+val for stability report
    cv_scores = cross_val_score(
        search.best_estimator_,
        X,
        y,
        cv=skf.split(X, X["CROP TYPE"]),
        scoring="r2",
        n_jobs=1,
    )
    print(f"5-fold CV R²: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    model = search.best_estimator_
    val_score = model.score(X_val, y_val)
    print(f"Validation R²: {val_score:.4f}")

    # Refit on full data for production
    model.fit(X, y)
    train_score = model.score(X, y)
    print(f"Train R² (full data): {train_score:.4f}")

    print_top_feature_importances(model, top_k=10)
    print_crop_level_r2(model, X_val, y_val)

    joblib.dump(model, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

    config = get_allowed_categories(X_with_temp, region_list_15_zones)
    config["target_col"] = TARGET_COL
    config["crop_min_mm"] = CROP_MIN_MM
    config["zone_to_climate"] = ZONE_TO_CLIMATE
    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Config saved to {CONFIG_PATH}")


if __name__ == "__main__":
    main()
