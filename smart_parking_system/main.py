# ============================================================
#  Smart Parking System - Main Entry Point
#  Runs the full pipeline:
#    1. Generate dataset
#    2. Preprocess data
#    3. Engineer features
#    4. EDA and visualisations
#    5. Train and evaluate ML models
# ============================================================

import sys
import io
import os

# Force UTF-8 output on Windows terminals
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# Add src/ and data/ to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "data"))

from generate_dataset    import generate_dataset
from data_preprocessing  import preprocess
from feature_engineering import engineer_features
from eda_analysis        import run_eda
from ml_model            import train

BANNER = """
==============================================================
   SMART PARKING SYSTEM - AI + Ultrasonic Sensors
   ECE Capstone Project - Full Pipeline
==============================================================
"""


def main():
    print(BANNER)

    print("=" * 60)
    print("STEP 1 / 5 : Generating synthetic sensor dataset ...")
    print("=" * 60)
    generate_dataset()

    print("\n" + "=" * 60)
    print("STEP 2 / 5 : Preprocessing data ...")
    print("=" * 60)
    preprocess(save=True)

    print("\n" + "=" * 60)
    print("STEP 3 / 5 : Engineering features ...")
    print("=" * 60)
    engineer_features(save=True)

    print("\n" + "=" * 60)
    print("STEP 4 / 5 : Running EDA and generating plots ...")
    print("=" * 60)
    run_eda()

    print("\n" + "=" * 60)
    print("STEP 5 / 5 : Training and evaluating ML models ...")
    print("=" * 60)
    trained, results = train(save_best=True)

    print("\n" + "=" * 60)
    print("  [OK]  Pipeline complete!")
    print("  [>]   Dataset   -> data/parking_data.csv")
    print("  [>]   Features  -> data/parking_features.csv")
    print("  [>]   Reports   -> reports/*.png")
    print("  [>]   Models    -> models/*.pkl")
    print()
    print("  [>>]  Launch the dashboard:")
    print("        streamlit run dashboard/app.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
