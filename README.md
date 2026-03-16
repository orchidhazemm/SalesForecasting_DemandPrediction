# DEPI Graduation Project

A **Sales Forecasting & On-Demand Predictions** system for product demand and revenue planning. Built as a graduation project with a full data pipeline, ML models, and an interactive Streamlit app.

## What This Project Does

- **Sales forecasting** — Predicts sales from marketing budget and dates (SARIMAX-based).
- **Demand by product** — Forecasts demand for product categories (e.g. towels, sheets, pillows) with confidence intervals.
- **Dashboards** — General plots (bar, line, pie, box, histogram, scatter) and analyses (top products, order size/value, seasonal patterns).
- **Model retraining** — Upload new data and marketing CSV to merge, preprocess, and retrain sales and demand models.

## Repository Structure

| Path | Description |
|------|-------------|
| `DEPI_PROJECT/` | Main application and pipeline |
| `DEPI_PROJECT/app.py` | Streamlit app (Forecasting Sales, Demand Product, Visualizations, Retraining) |
| `DEPI_PROJECT/main.py` | Data pipeline: load → clean → group → feature engineering → time-series features |
| `DEPI_PROJECT/utils/` | Preprocessing, model helpers (SARIMAX, etc.), EDA/plotting |
| `DEPI_PROJECT/data/` | Input/output CSVs (depi_*, marketing_*) |
| `DEPI_PROJECT/Main_Project.ipynb` | Exploratory analysis and preprocessing workflow |

## Quick Start

1. **Setup** (from repo root):
   ```bash
   cd DEPI_PROJECT
   pip install streamlit pandas numpy scikit-learn statsmodels matplotlib seaborn joblib
   ```

2. **Run data pipeline** (optional; builds `depi_ungrouped`, `depi_grouped`, `depi_time_series` from `data/depi_v0.csv`):
   ```bash
   cd DEPI_PROJECT && python main.py
   ```

3. **Run the app**:
   ```bash
   cd DEPI_PROJECT && streamlit run app.py
   ```

For detailed features, usage, and file structure, see **[DEPI_PROJECT/README.md](DEPI_PROJECT/README.md)**.

## Tech Stack

- **Python** — Pipeline and app
- **Streamlit** — Web UI
- **Pandas / NumPy** — Data handling
- **statsmodels (SARIMAX)** — Time-series forecasting
- **scikit-learn** — Preprocessing and utilities
- **Matplotlib / Seaborn** — Visualizations
- **Joblib** — Model persistence
- **MLflow** — Experiment tracking (model training)

## Data Flow

1. Raw data (`depi_v0.csv`) → `main.py` → cleaned/grouped/time-series CSVs.
2. App uses these CSVs and trained models for forecasting and dashboards.
3. Retraining: upload new data + marketing CSV → merge → preprocess → train SARIMAX (sales and demand) → save models.

---

*DEPI Graduation Project — Sales Forecasting & On-Demand Predictions*
