"""
CLIMATEPRED AI – standalone demo app.
Loads the models trained in ClimatePred_AI.ipynb (no training needed) and serves a Gradio UI.

    pip install -r requirements.txt
    python app.py
"""
import json, os, sys
import numpy as np, pandas as pd, joblib, shap
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import gradio as gr

HERE = os.path.dirname(os.path.abspath(__file__))
meta = json.load(open(os.path.join(HERE, "models/metadata.json")))
FEATURES, TARGET = meta["features"], meta["target"]
M = meta["final_test_metrics"]
reg = joblib.load(os.path.join(HERE, "models/xgb_tmax_next_day.joblib"))
clf = joblib.load(os.path.join(HERE, "models/heat_risk_classifier.joblib"))
data = pd.read_csv(os.path.join(HERE, "data/processed/climatepred_features.csv"), parse_dates=["date"])
explainer = shap.TreeExplainer(reg)
RISK = ["Normal", "Elevated", "Extreme"]
CITIES = sorted(data.city.unique())
fmin, fmax = data[FEATURES].min(), data[FEATURES].max()
cmp_tbl = pd.read_csv(os.path.join(HERE, "reports/model_comparison.csv"))

def predict(city, date, solar_pct, cloud_add, rh_add, wind_add):
    sel = data[(data.city == city) & (data.date == pd.Timestamp(date))]
    if sel.empty:
        return "No data for that date. Use a date between 2015-02-01 and 2025-10-30.", None, None
    row = sel.iloc[0]
    X = row[FEATURES].to_frame().T.astype(float)
    t = float(reg.predict(X)[0]); risk = RISK[int(clf.predict(X)[0])]
    sv = pd.Series(explainer.shap_values(X)[0], index=FEATURES).sort_values(key=np.abs, ascending=False).head(6)

    md = (f"### {city} — observations of {date}\n"
          f"**Predicted next-day maximum temperature: {t:.1f} °C** &nbsp;|&nbsp; **Heat risk: {risk}** "
          f"&nbsp;|&nbsp; actual next day (verification): {row[TARGET]:.1f} °C\n\n"
          f"Final model: tuned XGBoost — test MAE {M['MAE']:.2f} °C · RMSE {M['RMSE']:.2f} °C · R² {M['R2']:.3f}")

    mod = X.copy(); changes = {}
    if solar_pct != 100: mod["ALLSKY_SFC_SW_DWN"] *= solar_pct / 100; changes["ALLSKY_SFC_SW_DWN"] = 1
    if cloud_add: mod["CLOUD_AMT"] += cloud_add; changes["CLOUD_AMT"] = 1
    if rh_add: mod["RH2M"] += rh_add; changes["RH2M"] = 1
    if wind_add: mod["WS2M"] += wind_add; changes["WS2M"] = 1
    if changes:
        ts = float(reg.predict(mod)[0]); rs = RISK[int(clf.predict(mod)[0])]
        ok = all(fmin[f] <= float(mod[f].iloc[0]) <= fmax[f] for f in changes)
        md += (f"\n\n### Climate Intervention Simulator\nScenario next-day Tmax **{ts:.1f} °C** (Δ {ts - t:+.2f} °C); "
               f"risk {risk} → **{rs}**. Inputs within training range: {ok}.\n\n"
               "_Hypothetical model scenario based on learned associations — not causal proof._")

    fig, ax = plt.subplots(figsize=(6, 3.4))
    ax.barh(sv.index[::-1], sv.values[::-1], color=["#d62728" if v > 0 else "#1f77b4" for v in sv.values[::-1]])
    ax.axvline(0, color="k", lw=0.8); ax.set_title("Top SHAP drivers of this prediction"); ax.set_xlabel("contribution (°C)")
    plt.tight_layout()
    hist = data[(data.city == city) & (data.date <= pd.Timestamp(date))].tail(60)
    fig2, ax2 = plt.subplots(figsize=(6, 3.4))
    ax2.plot(hist.date, hist.T2M_MAX, label="observed Tmax, last 60 days")
    ax2.scatter([pd.Timestamp(date) + pd.Timedelta(days=1)], [t], color="red", zorder=5, label="predicted tomorrow")
    ax2.legend(); ax2.set_title("Recent history and prediction"); ax2.tick_params(axis="x", rotation=30); plt.tight_layout()
    return md, fig, fig2

with gr.Blocks(title="CLIMATEPRED AI") as demo:
    gr.Markdown("# 🌡️ CLIMATEPRED AI — Explainable Urban Heat Prediction (Tamil Nadu)\n"
                "Trained on NASA POWER daily climate data, 2015–2025, for six cities. Choose a city and date: the model predicts "
                "the **next day's maximum temperature** and its heat-risk category, and explains why.")
    with gr.Row():
        city = gr.Dropdown(CITIES, value="Chennai", label="City")
        date = gr.Textbox(value="2025-05-20", label="Date of today's observations (YYYY-MM-DD)")
    gr.Markdown("### Climate Intervention Simulator (what-if)")
    with gr.Row():
        solar = gr.Slider(60, 110, value=100, step=5, label="Solar radiation (% of actual) — shade / albedo")
        cloud = gr.Slider(0, 40, value=0, step=5, label="Extra cloud cover (pp)")
        rh = gr.Slider(0, 20, value=0, step=2, label="Extra humidity (pp) — greening / water bodies")
        wind = gr.Slider(0, 3, value=0, step=0.5, label="Extra wind (m/s) — ventilation corridors")
    btn = gr.Button("Predict", variant="primary")
    out = gr.Markdown()
    with gr.Row():
        f1 = gr.Plot(); f2 = gr.Plot()
    btn.click(predict, [city, date, solar, cloud, rh, wind], [out, f1, f2])
    gr.Markdown("### Model comparison (chronological test period 2024-03 → 2025-10)")
    gr.Dataframe(cmp_tbl[["Model", "Test MAE", "Test MSE", "Test RMSE", "Test R2"]].round(3), interactive=False)
    gr.Markdown("Predictions are research outputs, not official forecasts. Team: Balasudhan R, Deepshika — 21CEO310T.")

if __name__ == "__main__":
    demo.launch(share="--share" in sys.argv)
