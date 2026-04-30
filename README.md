# Crop Water Requirement Model

Predicts crop water requirement from crop type, soil type, **15 India agro-climatic zones**, temperature range, and weather condition. Outputs in **mm/day** and **L/acre/day**. Uses a Random Forest model trained on the expanded dataset and exposed via a Streamlit UI.

## Setup

```bash
pip install -r requirements.txt
```

## Data and training

The model uses **15 agro-climatic zones of India** (e.g. Western Himalayan Region, Trans-Gangetic Plain Region, Western Dry Region). The training data is built from the original 4-region dataset:

1. **Expand dataset** (run once to create `DATASET_15_agro_zones.csv`):

```bash
python expand_dataset_agro_zones.py
```

2. **Train the model** to produce `model.joblib` and `config.json`:

```bash
python train.py
```

Use the **same Python environment** for training and running the Streamlit app so the saved model loads correctly.

If predictions look the same for every input, **retrain** so the model learns from the dataset (which has different water requirements per crop/soil/region/weather/temp).

## Run the Streamlit app

```bash
streamlit run app_streamlit.py
```

Open the local URL shown by Streamlit. Pick crop, soil, **agro-climatic zone**, temperature, weather and click **Predict**. Results show both **mm/day** and **L/acre/day**.

## Project layout

```bash
streamlit run app_streamlit.py
```

Open the local URL shown by Streamlit. Pick crop, soil, **agro-climatic zone**, temperature, weather and click **Predict**. Results show both **mm/day** and **L/acre/day**.

## Project layout

- `expand_dataset_agro_zones.py` – Build 15-zone dataset from original 4-region data
- `train.py` – Load data, preprocess, train Random Forest, save pipeline and config
- `app_streamlit.py` – Streamlit UI for testing predictions
- `model.joblib` – Trained pipeline (created by `train.py`)
- `config.json` – Allowed categories including 15 regions (created by `train.py`)
- `DATASET - Sheet1.csv` – Original training data (4 regions)
- `DATASET_15_agro_zones.csv` – Expanded training data (15 agro-climatic zones)
