# CLIMATEPRED AI
An Explainable Machine Learning Framework for Urban Heat Prediction, Climate Trend Analysis and Climate Risk Intelligence

**21CEO310T – Global Warming and Climate Change · LLT / CLA-2 working model**
Submitted by: Balasudhan R (RA2411003012164) — SRM IST, Kattankulathur

## Links
- 🌐 **Live app:** https://climatepred-ai.onrender.com (free tier — first load after idle takes ~1 min)
- 📓 **Colab notebook:** https://colab.research.google.com/drive/1W0NntVHDVVoxveGa4qT9vbZwPZe5ce0W?usp=sharing

## Contents
| Path | What it is |
|---|---|
| `ClimatePred_AI.ipynb` | Complete, **already executed** notebook (data → EDA → features → 6 regressors → metrics → heat-risk classifiers → SHAP → simulator → Gradio UI) |
| `app.py` | Web application (10 sections: overview, prediction, dataset explorer, EDA, model lab, evaluation, SHAP, simulator, climate science, documentation) — loads saved models, no training needed |
| `data/raw/*.csv` | Real NASA POWER daily data, 6 Tamil Nadu cities, 2015-01-01 → 2025-10-31 |
| `data/processed/climatepred_features.csv` | Engineered modelling table |
| `models/` | Trained XGBoost regressor, Random Forest, heat-risk classifier, metadata |
| `reports/ClimatePred_AI_Technical_Report.pdf` | Technical report (also `.docx`) |
| `reports/model_comparison.csv`, `classification_comparison.csv` | Actual evaluation results |
| `reports/figures/` | All 15 figures |
| `pipeline.py` | Same code as the notebook, as a plain script |

## Run the demo (no training)
```bash
pip install -r requirements.txt
python app.py            # opens http://127.0.0.1:7860
python app.py --share    # public link
```

## Re-run everything (optional, ~3 min)
Open `ClimatePred_AI.ipynb` in Colab / Jupyter and *Run all*. It downloads the data from NASA POWER if `data/raw/` is missing, so the whole pipeline is reproducible from scratch (seed 42).

## Key results (chronological test period Mar 2024 – Oct 2025)
| Model | MAE °C | MSE | RMSE °C | R² |
|---|---|---|---|---|
| Persistence baseline | 1.137 | 2.389 | 1.546 | 0.827 |
| Linear Regression | 1.064 | 1.919 | 1.385 | 0.861 |
| SVR (RBF) | 1.009 | 1.757 | 1.326 | 0.873 |
| Decision Tree | 1.078 | 2.002 | 1.415 | 0.855 |
| Random Forest | 1.014 | 1.773 | 1.332 | 0.872 |
| **XGBoost (tuned)** | **1.002** | **1.720** | **1.312** | **0.876** |

Heat-risk classification: XGBoost accuracy 0.949, ROC-AUC 0.980. Unseen-city test: beats persistence on all six held-out cities.

Data: NASA POWER Project (https://power.larc.nasa.gov/). Predictions are research outputs, not official forecasts.
