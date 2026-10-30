# Smart Parking System — Using AI and Ultrasonic Sensors

> **ECE Capstone Project – II**  
> Detects parking slot occupancy via simulated ultrasonic sensors, trains ML models, and displays real-time status on an interactive dashboard.

---

## 📁 Project Structure

```
smart_parking_system/
├── data/
│   ├── generate_dataset.py      # Synthetic sensor data generator
│   ├── parking_data.csv         # Raw dataset (auto-generated)
│   ├── parking_data_clean.csv   # Preprocessed dataset
│   └── parking_features.csv     # Feature-engineered dataset
│
├── src/
│   ├── data_preprocessing.py    # Data cleaning & validation
│   ├── feature_engineering.py   # Feature extraction
│   ├── eda_analysis.py          # EDA plots (8 charts)
│   ├── ml_model.py              # Model training & evaluation
│   └── sensor_simulator.py      # Live sensor simulation
│
├── models/
│   ├── decision_tree.pkl
│   ├── random_forest.pkl
│   ├── logistic_regression.pkl
│   └── best_model.pkl           # Auto-selected best model
│
├── dashboard/
│   └── app.py                   # Streamlit web dashboard
│
├── reports/                     # All generated plots & metrics
├── main.py                      # Full pipeline runner
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup & Installation

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the full pipeline (dataset → EDA → models)
python main.py

# 3. Launch the dashboard
streamlit run dashboard/app.py
```

---

## 🧠 AI / ML Pipeline

| Step | Description |
|------|-------------|
| **Data Generation** | Simulates ultrasonic readings for 10 slots over 30 days |
| **Preprocessing** | Removes duplicates, handles missing values, validates readings |
| **Feature Engineering** | Cyclic time encoding, rolling occupancy rates, lot-level features |
| **EDA** | 8 plots: distribution, heatmap, trends, correlation |
| **Model Training** | Decision Tree, Random Forest, Logistic Regression |
| **Evaluation** | Accuracy, Precision, Recall, F1-Score, ROC-AUC + confusion matrices |
| **Dashboard** | Live slot grid, gauge chart, historical analysis tabs |

---

## 📊 Features Used for ML

- `hour`, `minute`, `day_of_week`, `is_weekend`
- `hour_sin`, `hour_cos` (cyclic time encoding)
- `is_peak_hour`, `time_of_day`
- `distance_cm`, `distance_norm`, `sensor_confidence`
- `prev_occupied`, `rolling_1h_rate`, `rolling_3h_rate`
- `lot_occupancy_rate`, `lot_available_slots`

---

## 📈 Dashboard Features

- 🟢🔴 **Real-time slot grid** (Occupied / Vacant)
- 📊 **Occupancy gauge** with threshold alerts
- 📉 **Live trend chart** (rolling history)
- 📅 **Historical tabs**: Hourly / Day-of-week / Per-slot
- 🤖 **ML prediction toggle** (sensor vs AI mode)
- ⏱️ **Auto-refresh** every 5 seconds

---

## 🧪 Model Evaluation Metrics

After training you will see a table like:

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|
| Decision Tree | ~94% | ~93% | ~95% | ~94% | ~97% |
| Random Forest | ~96% | ~96% | ~97% | ~96% | ~99% |
| Logistic Regression | ~88% | ~87% | ~90% | ~88% | ~94% |

---

## 📦 Dependencies

| Package | Purpose |
|---|---|
| `pandas`, `numpy` | Data manipulation |
| `scikit-learn` | ML models & evaluation |
| `matplotlib`, `seaborn` | Static EDA plots |
| `plotly` | Interactive dashboard charts |
| `streamlit` | Web dashboard |
| `joblib` | Model persistence |
