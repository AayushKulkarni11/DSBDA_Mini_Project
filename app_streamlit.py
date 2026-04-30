import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = Path(__file__).resolve().parent / "model.joblib"
CONFIG_PATH = Path(__file__).resolve().parent / "config.json"

MM_PER_DAY_TO_LITRE_PER_ACRE = 4046.86


@st.cache_data
def load_config():
    with open(CONFIG_PATH) as f:
        return json.load(f)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


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


def show_landing_page():
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; padding: 2rem 0;'>
            <h1 style='color: #2e7d32; font-size: 3rem; margin-bottom: 0.5rem;'>🌾 Crop Water Advisor</h1>
            <h3 style='color: #555; font-weight: 400;'>AI-Powered Irrigation Planning for Indian Agriculture</h3>
            <p style='color: #777; max-width: 700px; margin: 1rem auto; line-height: 1.6;'>
                Optimize water usage and boost crop yields with machine learning. 
                Get instant, location-specific irrigation recommendations based on crop type, soil conditions, 
                local climate, and weather patterns across India's 15 agro-climatic zones.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            """
            <div style='background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%); padding: 1.5rem; border-radius: 15px; text-align: center; box-shadow: 0 4px 6px rgba(0,0,0,0.1);'>
                <h2 style='color: #2e7d32; margin: 0;'>15</h2>
                <p style='color: #555; margin: 0.5rem 0 0 0; font-weight: 500;'>Agro-Climatic Zones</p>
                <p style='color: #777; font-size: 0.9rem; margin-top: 0.5rem;'>Covering all major regions of India</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            """
            <div style='background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%); padding: 1.5rem; border-radius: 15px; text-align: center; box-shadow: 0 4px 6px rgba(0,0,0,0.1);'>
                <h2 style='color: #1976d2; margin: 0;'>15</h2>
                <p style='color: #555; margin: 0.5rem 0 0 0; font-weight: 500;'>Crop Varieties</p>
                <p style='color: #777; font-size: 0.9rem; margin-top: 0.5rem;'>From rice to sugarcane</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            """
            <div style='background: linear-gradient(135deg, #fff3e0 0%, #ffe0b2 100%); padding: 1.5rem; border-radius: 15px; text-align: center; box-shadow: 0 4px 6px rgba(0,0,0,0.1);'>
                <h2 style='color: #f57c00; margin: 0;'>~10K</h2>
                <p style='color: #555; margin: 0.5rem 0 0 0; font-weight: 500;'>Training Samples</p>
                <p style='color: #777; font-size: 0.9rem; margin-top: 0.5rem;'>Powered by Random Forest</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown("## 🎯 How It Works")
    st.markdown("Follow these simple steps to get accurate water requirement predictions:")

    steps_col1, steps_col2, steps_col3, steps_col4 = st.columns(4)
    with steps_col1:
        st.markdown(
            """
            <div style='text-align: center; padding: 1rem;'>
                <div style='font-size: 3rem;'>🌱</div>
                <h4 style='margin-top: 0.5rem; color: #333;'>1. Select Crop</h4>
                <p style='color: #666; font-size: 0.9rem;'>Choose from 15 major Indian crops</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with steps_col2:
        st.markdown(
            """
            <div style='text-align: center; padding: 1rem;'>
                <div style='font-size: 3rem;'>🌍</div>
                <h4 style='margin-top: 0.5rem; color: #333;'>2. Choose Zone</h4>
                <p style='color: #666; font-size: 0.9rem;'>Select your agro-climatic region</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with steps_col3:
        st.markdown(
            """
            <div style='text-align: center; padding: 1rem;'>
                <div style='font-size: 3rem;'>🌡️</div>
                <h4 style='margin-top: 0.5rem; color: #333;'>3. Set Conditions</h4>
                <p style='color: #666; font-size: 0.9rem;'>Soil type, temperature, weather</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with steps_col4:
        st.markdown(
            """
            <div style='text-align: center; padding: 1rem;'>
                <div style='font-size: 3rem;'>💧</div>
                <h4 style='margin-top: 0.5rem; color: #333;'>4. Get Results</h4>
                <p style='color: #666; font-size: 0.9rem;'>Instant mm/day and L/acre values</p>
            </div>
            """,
            unsafe_allow_html=True,
        )




def show_model_info(config):
    st.markdown("## 🤖 Model Information")
    st.markdown(
        """
        **Algorithm:** Random Forest Regressor  
        **Preprocessing:** One-hot encoding for categorical features  
        **Features:** Crop, Soil, Climate (zone-mapped), Weather, Temperature
        """
    )

    st.markdown("### 🌿 Supported Crops (15)")
    crops = config["crop_type"]
    crop_cols = st.columns(5)
    for i, crop in enumerate(crops):
        with crop_cols[i % 5]:
            st.markdown(f"- {crop.title()}")

    st.markdown("### 🗺️ Agro-Climatic Zones (15)")
    zones = config["region"]
    zone_data = []
    for zone in zones:
        climate = config["zone_to_climate"].get(zone, "N/A")
        zone_data.append({"Zone": zone, "Climate Type": climate})

    st.dataframe(pd.DataFrame(zone_data), width='stretch', hide_index=True)


def show_guidelines():
    st.markdown("## 📋 Using This Tool")
    st.markdown(
        """
        Follow these steps to get accurate predictions for your field conditions:
        """
    )

    with st.expander("🔹 Step 1: Identify Your Agro-Climatic Zone", expanded=True):
        st.markdown(
            """
            India is divided into 15 agro-climatic zones for agricultural planning. 
            Examples:
            - **Western Himalayan Region** – Jammu & Kashmir, Himachal Pradesh, Uttarakhand
            - **Trans-Gangetic Plain Region** – Punjab, Haryana, Delhi, parts of Rajasthan
            - **Lower Gangetic Plain Region** – West Bengal, Bihar plains
            - **Southern Plateau & Hills Region** – Karnataka, Andhra Pradesh, Tamil Nadu plateaus
            
            Select the zone that matches your location. If unsure, consult your local agriculture office.
            """
        )

    with st.expander("🔹 Step 2: Know Your Soil Type"):
        st.markdown(
            """
            - **DRY** – Low moisture retention, sandy or rocky soil
            - **HUMID** – Moderate moisture, loamy or mixed texture
            - **WET** – High water retention, clay-rich or flooded conditions
            """
        )

    with st.expander("🔹 Step 3: Check Weather Conditions"):
        st.markdown(
            """
            - **NORMAL** – Typical weather without extremes
            - **RAINY** – Active rainfall or recent heavy precipitation
            - **SUNNY** – Clear, hot, high evaporation conditions
            - **WINDY** – Strong winds increasing transpiration
            """
        )

    with st.expander("🔹 Step 4: Temperature Range"):
        st.markdown(
            """
            Choose the range that includes your current daytime temperature:
            - **10–20 °C** – Cool winter days
            - **20–30 °C** – Moderate climate
            - **30–40 °C** – Hot summer conditions
            - **40–50 °C** – Extreme heat (rare, desert areas)
            """
        )

    with st.expander("🔹 Understanding the Output"):
        st.markdown(
            """
            - **mm/day** – Millimeters of water per day (depth applied over field)
            - **L/acre/day** – Liters per acre per day (1 mm/day = 4046.86 L/acre/day)
            
            The model applies minimum thresholds for water-intensive crops (Rice, Wheat, Maize) 
            to ensure physiologically plausible recommendations.
            """
        )


def show_prediction_page(config, model):
    st.markdown("## 💧 Water Requirement Calculator")
    st.markdown("Select your field conditions below to get an instant irrigation recommendation.")

    col1, col2 = st.columns(2)
    with col1:
        crop_type = st.selectbox(
            "🌾 Crop Type",
            config["crop_type"],
            help="Select the crop you are planning to irrigate",
        )
        soil_type = st.selectbox(
            "🪨 Soil Type",
            config["soil_type"],
            help="DRY: sandy/rocky | HUMID: loamy | WET: clay-rich/flooded",
        )
        region = st.selectbox(
            "🗺️ Agro-Climatic Zone",
            config["region"],
            help="Select your region from India's 15 agro-climatic zones",
        )

    with col2:
        temperature = st.selectbox(
            "🌡️ Temperature Range (°C)",
            config["temperature"],
            help="Daytime air temperature range",
        )
        weather_condition = st.selectbox(
            "☀️ Weather Condition",
            config["weather_condition"],
            help="Current or expected weather pattern",
        )

    st.markdown("---")
    predict_col, info_col = st.columns([2, 3])
    
    with predict_col:
        if st.button("🔮 Predict Water Requirement", type="primary", width='stretch'):
            with st.spinner("Calculating..."):
                pred_mm, litre_per_acre = predict(
                    crop_type, soil_type, region, temperature, weather_condition, config, model
                )
            st.session_state["prediction"] = (pred_mm, litre_per_acre, crop_type, soil_type, region, temperature, weather_condition)

    if "prediction" in st.session_state:
        pred_mm, litre_per_acre, ct, st_, rg, temp, wc = st.session_state["prediction"]
        st.markdown(
            f"""
            <div style='background: linear-gradient(135deg, #e3f2fd 0%, #f3e5f5 100%); padding: 2rem; border-radius: 15px; text-align: center; margin-top: 1rem; box-shadow: 0 4px 12px rgba(0,0,0,0.1);'>
                <h3 style='color: #1976d2; margin-bottom: 1.5rem;'>💧 Irrigation Recommendation</h3>
                <div style='display: flex; justify-content: center; gap: 3rem;'>
                    <div>
                        <div style='font-size: 0.9rem; color: #555; text-transform: uppercase; letter-spacing: 1px;'>Water Requirement</div>
                        <div style='font-size: 2.5rem; color: #1976d2; font-weight: bold; margin: 0.5rem 0;'>{pred_mm} <span style='font-size: 1.2rem; color: #777;'>mm/day</span></div>
                    </div>
                    <div>
                        <div style='font-size: 0.9rem; color: #555; text-transform: uppercase; letter-spacing: 1px;'>Flow Volume</div>
                        <div style='font-size: 2.5rem; color: #388e3c; font-weight: bold; margin: 0.5rem 0;'>{litre_per_acre} <span style='font-size: 1.2rem; color: #777;'>L/acre/day</span></div>
                    </div>
                </div>
                <div style='margin-top: 1rem; padding: 1rem; background: rgba(255,255,255,0.7); border-radius: 8px; font-size: 0.9rem; color: #444;'>
                    <strong>Input:</strong> {ct.title()} | {st_.title()} soil | {rg} | {temp}°C | {wc.title()}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with info_col:
        st.markdown("### 📝 Quick Reference")
        st.markdown(
            f"""
            **Your Selection Summary:**
            - **Crop:** {crop_type.title()}
            - **Soil Type:** {soil_type.title()}
            - **Zone:** {region}
            - **Temperature:** {temperature} °C
            - **Weather:** {weather_condition.title()}
            
            💡 **Tip:** For flood irrigation, apply the recommended daily amount. 
            For drip systems, adjust based on system efficiency (typically 85-95%).
            """
        )


def main():
    st.set_page_config(
        page_title="Crop Water Advisor",
        page_icon="🌾",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    with st.sidebar:
        st.title("🌾 Crop Water Advisor")
        st.markdown("---")
        app_mode = st.radio(
            "Navigate",
            ["🏠 Home", "🤖 Model Details", "📋 How to Use", "💧 Predict"],
            label_visibility="collapsed",
        )
        st.markdown("---")
        st.markdown(
            """
            <div style='font-size: 0.85rem; color: #888;'>
            <strong>Version:</strong> 1.0.0<br>
            <strong>Model:</strong> Random Forest<br>
            <strong>Data:</strong> 10,801 samples<br>
            <strong>Powered by:</strong> scikit-learn + Streamlit
            </div>
            """,
            unsafe_allow_html=True,
        )

    if not MODEL_PATH.exists() or not CONFIG_PATH.exists():
        st.error("❌ Model artifacts not found. Run `python train.py` first.")
        st.info("📝 Run these commands in your terminal:")
        st.code("python expand_dataset_agro_zones.py\npython train.py")
        return

    config, model = load_config(), load_model()

    if app_mode == "🏠 Home":
        show_landing_page()
    elif app_mode == "🤖 Model Details":
        show_model_info(config)
    elif app_mode == "📋 How to Use":
        show_guidelines()
    elif app_mode == "💧 Predict":
        show_prediction_page(config, model)


if __name__ == "__main__":
    main()
