"""
Expand DATASET - Sheet1.csv from 4 climate regions to 15 India agro-climatic zones.
Each zone is mapped to one of the 4 original regions (DESERT, SEMI ARID, SEMI HUMID, HUMID)
so water requirements are inherited from the closest climate type.
Output: DATASET_15_agro_zones.csv (used by train.py).
"""
from pathlib import Path

import pandas as pd

DATA_PATH = Path(__file__).resolve().parent / "DATASET - Sheet1.csv"
OUT_PATH = Path(__file__).resolve().parent / "DATASET_15_agro_zones.csv"

# 15 Agro-Climatic Zones of India -> original 4 regions (climate proxy)
AGRO_ZONE_TO_REGION = {
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


def main():
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=["WATER REQUIREMENT"])
    parts = []
    for zone_name, orig_region in AGRO_ZONE_TO_REGION.items():
        subset = df[df["REGION"] == orig_region].copy()
        subset["REGION"] = zone_name
        parts.append(subset)
    out = pd.concat(parts, ignore_index=True)
    out.to_csv(OUT_PATH, index=False)
    print(f"Expanded dataset saved to {OUT_PATH}")
    print(f"Rows: {len(out)}, Regions: {out['REGION'].nunique()}")


if __name__ == "__main__":
    main()
