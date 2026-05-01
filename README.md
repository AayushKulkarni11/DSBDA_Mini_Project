# Crop Water Requirement Model

A machine learning-powered web application that predicts daily crop water requirements for Indian agricultural conditions. Provides recommendations in **mm/day** and **L/acre/day** based on crop type, soil type, agro-climatic zone, temperature range, and weather conditions.

![Python](https://img.shields.io/badge/Python-3.x-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.6.x-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.0%2B-red)

---

## 📋 Table of Contents

- [Features](#-features)
- [Quick Start](#-quick-start)
- [Project Structure](#-project-structure)
- [Data](#-data)
- [Model Details](#-model-details)
- [Configuration](#-configuration)
- [Training Pipeline](#-training-pipeline)
- [Evaluation Metrics](#-evaluation-metrics)
- [API Reference](#-api-reference)
- [Troubleshooting](#-troubleshooting)
- [Technical Notes](#-technical-notes)
- [Future Enhancements](#-future-enhancements)
- [License](#-license)

---

## ✨ Features

- **15 crops supported:** Banana, Bean, Cabbage, Citrus, Cotton, Maize, Melon, Mustard, Onion, Potato, Rice, Soyabean, Sugarcane, Tomato, Wheat
- **3 soil types:** Dry, Humid, Wet
- **15 Indian agro-climatic zones:** Including Western Himalayan, Gangetic Plains, Deccan Plateau, Coastal regions, etc.
- **4 temperature ranges:** 10–20 °C, 20–30 °C, 30–40 °C, 40–50 °C
- **4 weather conditions:** Normal, Rainy, Sunny, Windy
- **Dual output units:** mm/day and L/acre/day
- **Interactive UI:** Built with Streamlit for easy testing
- **Physically plausible:** Minimum water constraints for rice, wheat, maize

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository** (if applicable) and navigate to the project folder:

```bash
cd Crop_Water_Model
```

2. **Create a virtual environment** (recommended):

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. **Install dependencies**:

```bash
pip install -r requirements.txt
```

**⚠️ Important:** `scikit-learn` must be `>=1.6.0,<1.7`. Version 1.7+ will fail to load the saved model due to internal API changes.

### Run the Application

```bash
streamlit run app_streamlit.py
```

The app will open in your browser at `http://localhost:8501`.

---

## 📁 Project Structure

```
Crop_Water_Model/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── config.json                        # Allowed categories & zone-climate mapping (generated)
├── model.joblib                       # Trained Random Forest pipeline (generated)
├── train.py                          # Model training script
├── app_streamlit.py                  # Streamlit web interface
├── expand_dataset_agro_zones.py      # Dataset expansion utility
├── DATASET - Sheet1.csv              # Original dataset (4 climate regions, 2,880 rows)
├── DATASET_15_agro_zones.csv         # Expanded dataset (15 zones, 10,801 rows)
├── RS_Session_256_AS_110.csv         # India Govt. irrigation reference data
├── DATASET - Sheet1 2.csv            # Duplicate of original dataset
├── .gitignore                        # Git exclusions
├── .venv/                            # Virtual environment (local)
└── __pycache__/                      # Python bytecode cache (local)
```

---

## 📊 Data

### Source Datasets

#### 1. Original Dataset (`DATASET - Sheet1.csv`)
- **Rows:** 2,880
- **Columns:**
  - `CROP TYPE` – Crop variety
  - `SOIL TYPE` – DRY, HUMID, WET
  - `REGION` – 4 climate types: DESERT, SEMI ARID, SEMI HUMID, HUMID
  - `TEMPERATURE` – Range buckets: 10–20, 20–30, 30–40, 40–50 °C
  - `WEATHER CONDITION` – NORMAL, RAINY, SUNNY, WINDY
  - `WATER REQUIREMENT` – Target in mm/day (continuous)

#### 2. Expanded Dataset (`DATASET_15_agro_zones.csv`)
- **Rows:** 10,801
- **Coverage:** All 15 Indian agro-climatic zones
- **Creation:** Mapped each zone to one of the 4 original climates to inherit appropriate water requirements

#### 3. Reference Data (`RS_Session_256_AS_110.csv`)
Government-published irrigation water requirements by crop and method (Surface vs. Drip). Used for validation and cross-checking predictions.

---

## 🤖 Model Details

### Architecture

**Pipeline:**
1. **Preprocessor**
   - OneHotEncoder for categorical features (crop, soil, region, weather)
   - Passthrough for numeric features (temperature midpoint, squared)
2. **Regressor**
   - RandomForestRegressor with tuned hyperparameters

### Features

| Feature | Type | Encoding |
|---------|------|----------|
| CROP TYPE | Categorical | One-Hot |
| SOIL TYPE | Categorical | One-Hot |
| REGION (mapped to climate) | Categorical | One-Hot |
| WEATHER CONDITION | Categorical | One-Hot |
| temp_mid | Numeric | Float (midpoint of range) |
| temp_mid_sq | Numeric | Float (quadratic term) |

**Total post-encoding features:** 28 (15 crops + 3 soils + 4 climates + 4 weather + 2 numeric)

### Training

- **Algorithm:** Random Forest
- **Hyperparameter Tuning:** RandomizedSearchCV (5-fold stratified CV)
- **Search Space:**
  - `n_estimators`: [200, 300, 400]
  - `max_depth`: [6, 8, 10, 12]
  - `min_samples_leaf`: [3, 5, 8, 12]
- **Stratification:** By crop type to ensure representation across splits
- **Target Processing:** Capped at 20 mm/day to reduce outlier impact

---

## ⚙️ Configuration

The `config.json` file defines valid input categories and mappings:

### Allowed Values

| Field | Options |
|-------|---------|
| crop_type | BANANA, BEAN, CABBAGE, CITRUS, COTTON, MAIZE, MELON, MUSTARD, ONION, POTATO, RICE, SOYABEAN, SUGARCANE, TOMATO, WHEAT |
| soil_type | DRY, HUMID, WET |
| region | 15 Indian agro-climatic zones (see below) |
| temperature | 10-20, 20-30, 30-40, 40-50 |
| weather_condition | NORMAL, RAINY, SUNNY, WINDY |

### 15 Agro-Climatic Zones

1. Western Himalayan Region
2. Eastern Himalayan Region
3. Lower Gangetic Plain Region
4. Middle Gangetic Plain Region
5. Upper Gangetic Plain Region
6. Trans-Gangetic Plain Region
7. Eastern Plateau & Hills Region
8. Central Plateau & Hills Region
9. Western Plateau & Hills Region
10. Southern Plateau & Hills Region
11. East Coast Plains & Hills Region
12. West Coast Plains & Ghats Region
13. Gujarat Plains & Hills Region
14. Western Dry Region
15. Island Region

### Zone → Climate Mapping

| Zone | Climate |
|------|---------|
| Western Himalayan Region | SEMI HUMID |
| Eastern Himalayan Region | HUMID |
| Lower/Middle/Upper Gangetic Plain | HUMID / HUMID / SEMI ARID |
| Trans-Gangetic Plain | SEMI ARID |
| Eastern Plateau & Hills | SEMI HUMID |
| Central Plateau & Hills | SEMI ARID |
| Western Plateau & Hills | SEMI ARID |
| Southern Plateau & Hills | SEMI HUMID |
| East/West Coast Plains & Ghats | HUMID |
| Gujarat Plains & Hills | SEMI ARID |
| Western Dry Region | DESERT |
| Island Region | HUMID |

### Crop Minimum Water Requirements

Enforced at inference to prevent physiologically impossible low predictions:

| Crop | Minimum (mm/day) |
|------|------------------|
| RICE | 4.0 |
| WHEAT | 2.0 |
| MAIZE | 3.0 |

---

## 🔄 Training Pipeline

### Step 1: Expand Dataset (one-time)

```bash
python expand_dataset_agro_zones.py
```

Expands the original 4-region dataset to all 15 zones by mapping each zone to its climate proxy.

### Step 2: Train Model

```bash
python train.py
```

**Process:**
1. Loads expanded dataset
2. Caps WATER REQUIREMENT at 20 mm/day
3. Maps 15 zones → 4 climates
4. Aggregates duplicate rows via median
5. Engineers temperature features (`temp_mid`, `temp_mid_sq`)
6. Performs 80/20 stratified split (by crop)
7. Runs randomized search CV (16 iterations, 5-fold)
8. Evaluates on validation set
9. Refits on full data
10. Saves pipeline to `model.joblib`
11. Saves config to `config.json`

**Output includes:**
- Best CV R² score
- Validation R²
- 5-fold CV stability metrics
- Top 10 feature importances
- Per-crop R² breakdown

---

## 📈 Evaluation Metrics

The training script reports:

- **Best CV R²** – Cross-validation performance during hyperparameter search
- **5-fold CV R²** – Stability across full dataset (mean ± std)
- **Validation R²** – Holdout set performances
- **Train R²** – In-sample performance after refitting
- **Per-Crop R²** – Model performance for each crop individually
- **Feature Importances** – Most influential predictors

---

## 🌐 API Reference

### Using the Model Directly

```python
import json
import joblib
import pandas as pd
from pathlib import Path

# Load artifacts
config = json.load(open("config.json"))
model = joblib.load("model.joblib")

# Prepare input
def predict(crop, soil, region, temp_range, weather):
    temp_mid = sum(map(int, temp_range.split("-"))) / 2
    zone_to_climate = config["zone_to_climate"]
    climate = zone_to_climate.get(region, region)

    X = pd.DataFrame([{
        "CROP TYPE": crop,
        "SOIL TYPE": soil,
        "REGION": climate,
        "WEATHER CONDITION": weather,
        "temp_mid": temp_mid,
        "temp_mid_sq": temp_mid ** 2,
    }])

    pred_mm = model.predict(X)[0]
    # Apply crop minimum
    crop_min = config.get("crop_min_mm", {}).get(crop.upper(), 0)
    pred_mm = max(pred_mm, crop_min)
    # Convert to L/acre/day
    pred_l_acre = pred_mm * 4046.86
    return round(pred_mm, 4), round(pred_l_acre, 2)
```

---

## ❓ Troubleshooting

| Issue | Solution |
|-------|----------|
| **ModelNotFoundError** | Run `python train.py` first to generate `model.joblib` and `config.json` |
| **All predictions identical** | Retrain the model; dataset may not have loaded correctly |
| ** sklearn version error** | Ensure environment uses `scikit-learn>=1.6.0,<1.7` |
| **Unknown category error** | Check `config.json` for allowed values; contact maintainers if missing |
| **Streamlit command not found** | Install with `pip install streamlit` |

---

## ⚠️ Technical Notes

### Version Compatibility

- **scikit-learn** is pinned to 1.6.x due to internal changes in 1.7 that break deserialization (`_RemainderColsList`).
- Train and serve in the **same Python environment** to avoid version mismatches.

### Artifact Regeneration

`model.joblib` and `config.json` are in `.gitignore`. They can always be regenerated:

```bash
rm model.joblib config.json
python expand_dataset_agro_zones.py
python train.py
```

### Climate Proxy Strategy

The model learns from **4 climate types**, not the 15 zones directly. At inference, user-selected zones are internally mapped to their climate, so predictions are climate-specific, not zone-specific. The zone dropdown preserves UX familiarity with India's agricultural planning divisions.

---

## 🔮 Future Enhancements

Potential improvements for production deployment:

- **Alternative models:** XGBoost, Gradient Boosting, Neural Networks
- **Additional features:** Soil pH, elevation, irrigation method, season (Kharif/Rabi), rainfall history
- **Real data integration:** Replace simulated data with field measurements from government sources
- **Advanced metrics:** MAE, RMSE, prediction intervals via quantile regression
- **Deployment:** FastAPI backend + React frontend for multi-user access
- **Monitoring:** Prediction logging, model drift detection, periodic retraining
- **Mobile app:** Lightweight interface for offline field use

---

## 📜 License

This project is provided as-is for educational and research purposes. Please adhere to data source licenses when redistributing or extending.

---

**Last updated:** April 2026  
**Maintainer:** [Project Owner]  
**Location:** `/Users/aayushkulkarni/Desktop/Crop_Water_Model`
