import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = Path(__file__).resolve().parent / "model.joblib"
CONFIG_PATH = Path(__file__).resolve().parent / "config.json"

MM_PER_DAY_TO_LITRE_PER_ACRE = 4046.86


def load_artifacts():
    with open(CONFIG_PATH) as f:
        config = json.load(f)
    model = joblib.load(MODEL_PATH)
    return config, model


def parse_temp(s: str) -> float:
    low, high = s.strip().split("-")
    return (int(low) + int(high)) / 2.0


def predict(crop_type, soil_type, region, temperature, weather_condition, config, model):
    temp_mid = parse_temp(temperature)
    zone_to_climate = config.get("zone_to_climate") or {}
    region_climate = zone_to_climate.get((region or "").strip(), (region or "").strip())
    row = pd.DataFrame(
        [
            {
                "CROP TYPE": crop_type,
                "SOIL TYPE": soil_type,
                "REGION": region_climate,
                "WEATHER CONDITION": weather_condition,
                "temp_mid": temp_mid,
                "temp_mid_sq": temp_mid * temp_mid,
            }
        ],
        columns=["CROP TYPE", "SOIL TYPE", "REGION", "WEATHER CONDITION", "temp_mid", "temp_mid_sq"],
    )
    pred = model.predict(row)[0]
    pred_mm = round(float(pred), 4)
    crop_min = config.get("crop_min_mm") or {}
    pred_mm = max(pred_mm, crop_min.get((crop_type or "").strip().upper(), 0))
    litre_per_acre = round(pred_mm * MM_PER_DAY_TO_LITRE_PER_ACRE, 2)
    return pred_mm, litre_per_acre


def main():
    st.set_page_config(page_title="Crop Water Requirement", layout="centered")
    st.title("Crop Water Requirement Prediction")
    st.write(
        "Predict crop water requirement in mm/day and L/acre/day from crop, soil, Indian agro-climatic zone, temperature range, and weather condition."
    )

    if not MODEL_PATH.exists() or not CONFIG_PATH.exists():
        st.error("Model artifacts not found. Run `python train.py` first.")
        return

    config, model = load_artifacts()

    crop_type = st.selectbox("Crop type", config["crop_type"])
    soil_type = st.selectbox("Soil type", config["soil_type"])
    region = st.selectbox("Agro-climatic zone", config["region"])
    temperature = st.selectbox("Temperature range", config["temperature"])
    weather_condition = st.selectbox("Weather condition", config["weather_condition"])

    if st.button("Predict"):
        pred_mm, litre_per_acre = predict(
            crop_type,
            soil_type,
            region,
            temperature,
            weather_condition,
            config,
            model,
        )
        st.subheader("Prediction")
        st.write(f"**Water requirement:** {pred_mm} mm/day")
        st.write(f"**Water requirement:** {litre_per_acre} L/acre/day")


if __name__ == "__main__":
    main()
