"""
CLIMATEPRED AI – web application.
Loads the models trained in ClimatePred_AI.ipynb (no training needed) and serves a multi-section Gradio UI.

    pip install -r requirements.txt
    python app.py
"""
import io, json, os, sys
import numpy as np, pandas as pd, joblib, shap
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import gradio as gr

HERE = os.path.dirname(os.path.abspath(__file__))
P = lambda *a: os.path.join(HERE, *a)
FIG = lambda name: P("reports", "figures", name)

# ----------------------------------------------------------------------------- load artefacts
meta = json.load(open(P("models/metadata.json")))
FEATURES, TARGET = meta["features"], meta["target"]
M = meta["final_test_metrics"]
split = json.load(open(P("reports/split_info.json")))
reg = joblib.load(P("models/xgb_tmax_next_day.joblib"))
clf = joblib.load(P("models/heat_risk_classifier.joblib"))
data = pd.read_csv(P("data/processed/climatepred_features.csv"), parse_dates=["date"])
explainer = shap.TreeExplainer(reg)
RISK = ["Normal", "Elevated", "Extreme"]
CITIES = sorted(data.city.unique())
fmin, fmax = data[FEATURES].min(), data[FEATURES].max()
cmp_tbl = pd.read_csv(P("reports/model_comparison.csv"))
clf_tbl = pd.read_csv(P("reports/classification_comparison.csv"))
DATE_MIN, DATE_MAX = data.date.min().date(), data.date.max().date()
COLAB_URL = "https://colab.research.google.com/drive/1W0NntVHDVVoxveGa4qT9vbZwPZe5ce0W?usp=sharing"
GITHUB_URL = "https://github.com/BALASUDHAN1807/CLIMATEPRED-X"

# raw data (for dataset explorer / overview)
WEATHER = ["T2M", "T2M_MAX", "T2M_MIN", "RH2M", "PRECTOTCORR", "WS2M", "ALLSKY_SFC_SW_DWN", "CLOUD_AMT", "PS"]
def _load_raw(city):
    lines = open(P("data/raw", f"{city.lower()}.csv")).readlines()
    s = next(i for i, l in enumerate(lines) if l.startswith("-END HEADER-")) + 1
    d = pd.read_csv(io.StringIO("".join(lines[s:])))
    d["date"] = pd.to_datetime(d.YEAR.astype(str) + "-" + d.DOY.astype(str), format="%Y-%j")
    d["city"] = city
    return d.drop(columns=["YEAR", "DOY"])
raw = pd.concat([_load_raw(c) for c in CITIES], ignore_index=True).replace(-999, np.nan)
raw = raw[["city", "date"] + WEATHER]

PARAM_DESC = pd.DataFrame([
    ["date", "Observation date (daily, local solar time)", "—"], ["city", "Tamil Nadu city (6)", "—"],
    ["T2M", "Mean air temperature at 2 m (MERRA-2)", "°C"], ["T2M_MAX", "Maximum air temperature at 2 m", "°C"],
    ["T2M_MIN", "Minimum air temperature at 2 m", "°C"], ["RH2M", "Relative humidity at 2 m", "%"],
    ["PRECTOTCORR", "Bias-corrected precipitation", "mm/day"], ["WS2M", "Wind speed at 2 m", "m/s"],
    ["ALLSKY_SFC_SW_DWN", "All-sky surface shortwave downward irradiance (CERES)", "MJ/m²/day"],
    ["CLOUD_AMT", "Cloud amount (CERES)", "%"], ["PS", "Surface pressure", "kPa"]], columns=["Column", "Description", "Unit"])

def fig_to_img(fig):
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=110, bbox_inches="tight"); plt.close(fig); buf.seek(0)
    from PIL import Image; return Image.open(buf)

# ----------------------------------------------------------------------------- prediction + simulator
def predict(city, date, solar_pct, cloud_add, rh_add, wind_add, history):
    history = history or []
    try:
        ts = pd.Timestamp(date)
    except Exception:
        return f"⚠️ Invalid date. Use YYYY-MM-DD between {DATE_MIN} and {DATE_MAX}.", None, None, None, history, pd.DataFrame(history), None
    sel = data[(data.city == city) & (data.date == ts)]
    if sel.empty:
        return (f"⚠️ No observations for {city} on {date}. The dataset covers **{DATE_MIN} → {DATE_MAX}** "
                f"(the model predicts the day after the chosen date; future dates need live weather inputs).", None, None, None, history, pd.DataFrame(history), None)
    row = sel.iloc[0]
    X = row[FEATURES].to_frame().T.astype(float)
    t = float(reg.predict(X)[0]); risk = RISK[int(clf.predict(X)[0])]
    sv = pd.Series(explainer.shap_values(X)[0], index=FEATURES).sort_values(key=np.abs, ascending=False)
    base_val = float(explainer.expected_value)

    md = (f"## {city} — observations of {date}\n"
          f"### Predicted maximum temperature for {(ts + pd.Timedelta(days=1)).date()}: **{t:.1f} °C** — heat risk **{risk}**\n"
          f"Actual recorded value (for verification): {row[TARGET]:.1f} °C &nbsp;·&nbsp; error {abs(row[TARGET]-t):.1f} °C\n\n"
          f"**Why this prediction?** The model's average prediction is {base_val:.1f} °C. Today's inputs move it by "
          + ", ".join(f"{f} {v:+.2f} °C" for f, v in sv.head(4).items())
          + f". Heat-risk classes are per-city percentiles of the training years (Elevated ≥ P90, Extreme ≥ P97.5).\n\n"
          f"*Model: tuned XGBoost — test MAE {M['MAE']:.2f} °C · RMSE {M['RMSE']:.2f} °C · R² {M['R2']:.3f}. SHAP explains model behaviour, not physical causation.*")

    # scenario
    mod = X.copy(); changes = {}
    if solar_pct != 100: mod["ALLSKY_SFC_SW_DWN"] *= solar_pct / 100; changes["Solar radiation"] = f"{solar_pct}% of actual"
    if cloud_add: mod["CLOUD_AMT"] += cloud_add; changes["Cloud amount"] = f"+{cloud_add} pp"
    if rh_add: mod["RH2M"] += rh_add; changes["Humidity"] = f"+{rh_add} pp"
    if wind_add: mod["WS2M"] += wind_add; changes["Wind speed"] = f"+{wind_add} m/s"
    sim_img = None
    if changes:
        tsim = float(reg.predict(mod)[0]); rs = RISK[int(clf.predict(mod)[0])]
        ok = all(fmin[f] <= float(mod[f].iloc[0]) <= fmax[f] for f in ["ALLSKY_SFC_SW_DWN", "CLOUD_AMT", "RH2M", "WS2M"])
        md += (f"\n\n## Climate Intervention Simulator (hypothetical)\n"
               f"| | Baseline | Scenario |\n|---|---|---|\n| Predicted Tmax | {t:.2f} °C | **{tsim:.2f} °C** |\n| Heat risk | {risk} | **{rs}** |\n| Δ temperature | | **{tsim - t:+.2f} °C** |\n\n"
               f"Changes applied: " + "; ".join(f"{k}: {v}" for k, v in changes.items())
               + f". Inputs within the model's training range: {'yes' if ok else '⚠️ no — extrapolation, treat with caution'}.\n\n"
               "_This is the trained model's learned sensitivity to the modified inputs — an association, not a causal estimate of a real intervention._")
        fig, ax = plt.subplots(figsize=(5, 3.4))
        bars = ax.bar(["Baseline", "Scenario"], [t, tsim], color=["#9E9E9E", "#2E75B6"])
        for b, v in zip(bars, [t, tsim]): ax.text(b.get_x() + b.get_width()/2, v + 0.05, f"{v:.2f} °C", ha="center", fontsize=10)
        lo = min(t, tsim) - 2; ax.set_ylim(lo, max(t, tsim) + 1); ax.set_ylabel("Predicted next-day Tmax (°C)"); ax.set_title("Baseline vs scenario")
        sim_img = fig_to_img(fig)
        history = history + [{"city": city, "date": str(date), "solar_%": solar_pct, "cloud_+pp": cloud_add, "humidity_+pp": rh_add,
                              "wind_+m/s": wind_add, "baseline_tmax": round(t, 2), "scenario_tmax": round(tsim, 2), "delta_C": round(tsim - t, 2),
                              "baseline_risk": risk, "scenario_risk": rs, "within_training_range": ok}]

    top = sv.head(8)
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.barh(top.index[::-1], top.values[::-1], color=["#d62728" if v > 0 else "#1f77b4" for v in top.values[::-1]])
    ax.axvline(0, color="k", lw=0.8); ax.set_title("Top SHAP drivers of this prediction (°C)"); ax.set_xlabel("contribution to predicted Tmax")
    hist = data[(data.city == city) & (data.date <= ts)].tail(90)
    fig2, ax2 = plt.subplots(figsize=(6, 3.6))
    ax2.plot(hist.date, hist.T2M_MAX, label="observed Tmax (last 90 days)", color="#444")
    ax2.scatter([ts + pd.Timedelta(days=1)], [t], color="red", s=60, zorder=5, label="predicted tomorrow")
    ax2.scatter([ts + pd.Timedelta(days=1)], [row[TARGET]], facecolors="none", edgecolors="green", s=80, zorder=5, label="actual tomorrow")
    ax2.legend(fontsize=8); ax2.set_title("Recent history and prediction"); ax2.tick_params(axis="x", rotation=30); ax2.set_ylabel("°C")
    hist_df = pd.DataFrame(history)
    hist_csv = None
    if len(hist_df):
        hist_csv = P("data", "processed", "scenario_history.csv"); hist_df.to_csv(hist_csv, index=False)
    return md, fig_to_img(fig), fig_to_img(fig2), sim_img, history, hist_df, hist_csv

# ----------------------------------------------------------------------------- overview
def overview(city):
    g = raw[raw.city == city]
    yr = g[g.date.dt.year <= 2024].groupby(g.date.dt.year).agg(mean_T=("T2M", "mean"), mean_Tmax=("T2M_MAX", "mean"),
            abs_max=("T2M_MAX", "max"), hot_days=("T2M_MAX", lambda s: int((s >= 38).sum())), rain=("PRECTOTCORR", "sum")).round(2)
    base = yr.loc[2015:2019, "mean_T"].mean()
    yr["anomaly_vs_2015_19"] = (yr.mean_T - base).round(2)
    slope = np.polyfit(yr.index, yr.mean_T, 1)[0] * 10; slope_hot = np.polyfit(yr.index, yr.hot_days, 1)[0] * 10
    kpi = (f"### {city}\n| KPI | Value |\n|---|---|\n| Records | {len(g)} days ({g.date.min().date()} → {g.date.max().date()}) |\n"
           f"| Mean temperature (2015–2024) | {g.T2M.mean():.1f} °C |\n| Mean daily maximum | {g.T2M_MAX.mean():.1f} °C |\n"
           f"| Hottest day on record | {g.T2M_MAX.max():.1f} °C on {g.loc[g.T2M_MAX.idxmax(), 'date'].date()} |\n"
           f"| Avg. hot days / year (Tmax ≥ 38 °C) | {yr.hot_days.mean():.1f} |\n| Trend of annual mean (2015–2024) | {slope:+.2f} °C / decade |\n"
           f"| Trend of hot days | {slope_hot:+.1f} days / decade |\n| Mean annual rainfall | {yr.rain.mean():.0f} mm |\n\n"
           f"Anomaly baseline = this city's own 2015–2019 mean ({base:.2f} °C). Ten years is a short window: these trends show the recent tendency, not a 30-year climate normal. "
           f"Data: NASA POWER (MERRA-2 / CERES), retrieved 13 Nov 2026.")
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
    axes[0].plot(yr.index, yr.mean_T, marker="o", color="#C0504D"); axes[0].plot(yr.index, np.polyval(np.polyfit(yr.index, yr.mean_T, 1), yr.index), "--", color="gray")
    axes[0].set_title("Annual mean temperature (°C)")
    axes[1].bar(yr.index, yr.anomaly_vs_2015_19, color=np.where(yr.anomaly_vs_2015_19 >= 0, "#E53935", "#1E88E5")); axes[1].axhline(0, color="k", lw=.8); axes[1].set_title("Anomaly vs 2015–19 baseline (°C)")
    axes[2].bar(yr.index, yr.hot_days, color="#FB8C00"); axes[2].set_title("Hot days per year (Tmax ≥ 38 °C)")
    for a in axes: a.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    monthly = g.groupby(g.date.dt.month).T2M_MAX.mean()
    fig2, ax = plt.subplots(figsize=(7, 3.4)); ax.plot(monthly.index, monthly.values, marker="o", color="#2E75B6"); ax.set_xticks(range(1, 13)); ax.set_title(f"Seasonal cycle of daily maximum temperature — {city}"); ax.set_ylabel("°C"); plt.tight_layout()
    return kpi, fig_to_img(fig), fig_to_img(fig2), yr.reset_index().rename(columns={"date": "year"})

six = raw.groupby("city").agg(mean_T=("T2M", "mean"), mean_Tmax=("T2M_MAX", "mean"), max_Tmax=("T2M_MAX", "max"), mean_RH=("RH2M", "mean"),
                              rain_per_year=("PRECTOTCORR", lambda s: s.sum() / (len(s) / 365.25)), solar=("ALLSKY_SFC_SW_DWN", "mean")).round(2).reset_index()

# ----------------------------------------------------------------------------- dataset explorer
def iqr_table(g):
    rows = []
    for col in WEATHER:
        q1, q3 = g[col].quantile([0.25, 0.75]); iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n = int(((g[col] < lo) | (g[col] > hi)).sum())
        rows.append([col, round(lo, 2), round(hi, 2), n, round(100 * n / len(g), 2)])
    return pd.DataFrame(rows, columns=["Variable", "IQR lower fence", "IQR upper fence", "Outliers", "Outliers %"])

PAGE = 25
def explore(city, query, page):
    g = raw if city == "All cities" else raw[raw.city == city]
    if query and query.strip():
        q = query.strip()
        try:   # date / year search, else free-text match on any column
            mask = g.astype(str).apply(lambda col: col.str.contains(q, case=False, regex=False)).any(axis=1)
            g = g[mask]
        except Exception:
            pass
    n_pages = max(1, int(np.ceil(len(g) / PAGE)))
    page = int(min(max(1, page), n_pages))
    view = g.iloc[(page - 1) * PAGE: page * PAGE].reset_index(drop=True)
    quality = pd.DataFrame({"Column": g.columns, "dtype": g.dtypes.astype(str).values, "Missing": g.isna().sum().values,
                            "Missing %": (g.isna().mean() * 100).round(2).values, "Unique": g.nunique().values})
    info = (f"**Rows:** {len(g):,} &nbsp;·&nbsp; **Columns:** {g.shape[1]} &nbsp;·&nbsp; **Page {page} / {n_pages}** &nbsp;·&nbsp; "
            f"**Duplicate (city,date) rows:** {int(g.duplicated(['city','date']).sum())} &nbsp;·&nbsp; **Missing cells:** {int(g.isna().sum().sum())} "
            f"&nbsp;·&nbsp; **Period:** {g.date.min().date() if len(g) else '—'} → {g.date.max().date() if len(g) else '—'}\n\n"
            "Provenance: NASA POWER Daily API (MERRA-2 reanalysis + CERES SYN1deg), community = AG, retrieved 13 Nov 2026; sentinel −999 = missing.")
    stats = g[WEATHER].describe().T.reset_index().rename(columns={"index": "feature"}).round(2) if len(g) else pd.DataFrame()
    if len(g): stats.insert(2, "median", g[WEATHER].median().values.round(2))
    out_csv = P("data", "processed", "export_filtered.csv"); g.to_csv(out_csv, index=False)
    return info, view, quality, stats, (iqr_table(g) if len(g) else pd.DataFrame()), out_csv, gr.update(maximum=n_pages, value=page)

# ----------------------------------------------------------------------------- static text
SCIENCE = f"""
## The greenhouse effect
Sunlight (shortwave radiation) warms the Earth's surface; the surface re-emits energy as infrared (longwave) radiation. Greenhouse gases — water vapour, **CO₂**, **CH₄**, **N₂O**, ozone — absorb part of this outgoing infrared and re-radiate it, keeping the surface ≈ 33 °C warmer than it would otherwise be. Human activity has raised atmospheric CO₂ from ≈ 280 ppm (pre-industrial) to > 420 ppm, CH₄ has more than doubled, and N₂O has risen ≈ 25 %; this added radiative forcing is the cause of the ≈ 1.1–1.3 °C of global warming observed since 1850–1900 (IPCC AR6).

| Gas | Main anthropogenic sources | Global-warming potential (100-yr) | Atmospheric lifetime |
|---|---|---|---|
| CO₂ | Fossil fuels, cement, land-use change | 1 | centuries |
| CH₄ | Livestock, rice paddies, landfills, gas leaks | ≈ 28–30 | ≈ 12 yr |
| N₂O | Fertilisers, industry, combustion | ≈ 273 | ≈ 110 yr |

## Climate vs weather
Weather is the state of the atmosphere over hours to days; climate is its statistics over decades. This project **predicts weather** (tomorrow's maximum temperature) and **analyses climate** (how annual means and the count of hot days have shifted over 2015–2024). A short record cannot prove or disprove long-term warming; the IPCC uses ≥ 30-year normals.

## Urban heat island (UHI)
Cities replace vegetation and soil with concrete, asphalt and roofs. These surfaces have low albedo (absorb more sunlight), high heat capacity (release heat at night), block ventilation, and reduce evapotranspiration (less latent cooling). Result: cities are typically 1–5 °C warmer than their rural surroundings, most strongly at night. Global warming raises the baseline on which UHI sits, so heat-wave days multiply faster than the mean temperature rises — which is why this project tracks **extremes** (Tmax ≥ 38 °C, per-city 90th/97.5th percentiles) and not only averages.

## How the model connects to this physics
The learned relationships in Section *Explainable AI* mirror the surface energy balance: incoming **solar radiation** and clear skies push temperature up; **humidity, rainfall and cloud** push it down (latent cooling, reduced insolation); the strong weight on **recent temperature history** reflects the thermal inertia of land and air masses.

## Adaptation and mitigation for Tamil Nadu cities
- **Mitigation** (reduce the forcing): renewable energy, efficient cooling, transport electrification, methane capture.
- **Adaptation** (reduce the harm): cool/reflective roofs and pavements (albedo), urban tree canopy and parks (shade + evapotranspiration), restored water bodies, ventilation corridors in city planning, heat-action plans with early warnings — exactly the levers the Climate Simulator lets you test.

*Data note: greenhouse-gas concentrations are not part of the NASA POWER weather dataset used for the ML model; the values above are from IPCC AR6 / NOAA GML public reports and are given for scientific context.*
"""

MODEL_LAB = f"""
## Experimental protocol
| Item | Setting |
|---|---|
| Task | Predict **next-day maximum temperature** (°C) from observations up to today |
| Training period | {split['train'][0]} → {split['train'][1]} — {split['train'][2]:,} rows (70 %) |
| Validation period | {split['val'][0]} → {split['val'][1]} — {split['val'][2]:,} rows (15 %) — model selection & tuning only |
| Test period | {split['test'][0]} → {split['test'][1]} — {split['test'][2]:,} rows (15 %) — touched once |
| Features | {len(FEATURES)} (today's 9 observations, Tmax lags 1/2/3/7, rolling mean 7/30, rolling std 7, yesterday's Tmin, 3-day rain, diurnal range, cyclic day-of-year, month, lat, lon) |
| Leakage control | No future information in inputs; chronological split; per-city rolling windows |
| Extra checks | 5-fold expanding-window TimeSeriesSplit; leave-one-city-out generalisation |
| Reproducibility | seed 42, saved joblib models, printed library versions, requirements.txt |

## Algorithms and why they suit this task
| Model | Hyper-parameters | Why it is here |
|---|---|---|
| Persistence | tomorrow = today | The naïve forecast every model must beat |
| Linear Regression | standardised inputs | Baseline; temperature is partly linear in its lags |
| Ridge | α = 1.0 | Regularised linear model (checks collinearity among lags) |
| **SVR (RBF)** | C = 10, ε = 0.1, γ = scale | Kernel method capturing smooth non-linear interactions (e.g. humidity × radiation) |
| Decision Tree | depth 8, min leaf 20 | Interpretable non-linear splits; shows over-fitting of a single tree |
| **Random Forest** | 300 trees, depth 14, min leaf 3 | Bagged ensemble — robust to outliers, low variance, handles mixed feature scales |
| **XGBoost** | 600 rounds, depth 5, lr 0.02 (tuned on validation from a 3×3 grid), subsample 0.8 | Gradient boosting — best accuracy, fast, native SHAP support |
| LSTM | not used | Dataset (≈ 24 k rows) too small to justify deep sequence models |

## Cross-validation (5-fold TimeSeriesSplit, MAE °C)
| Model | mean ± sd | folds |
|---|---|---|
| Linear Regression | 0.997 ± 0.041 | 1.032, 0.969, 0.932, 1.038, 1.015 |
| Random Forest | 0.965 ± 0.043 | 1.007, 0.944, 0.895, 1.013, 0.964 |
| XGBoost | 0.957 ± 0.042 | 0.996, 0.936, 0.891, 1.007, 0.954 |

## Over-fitting check (MAE °C)
| Model | Train | Validation | Test |
|---|---|---|---|
| Linear Regression | 1.006 | 0.953 | 1.064 |
| SVR | 0.795 | 0.923 | 1.009 |
| Decision Tree | 0.927 | 0.988 | 1.078 |
| Random Forest | 0.546 | 0.921 | 1.014 |
| XGBoost | 0.764 | 0.922 | 1.016 |

**Best model** (selected on validation MAE): tuned XGBoost — test MAE {M['MAE']:.3f}, MSE {M['MSE']:.3f}, RMSE {M['RMSE']:.3f}, R² {M['R2']:.3f}.
"""

EVAL_TEXT = """
**Reading the metrics** — MAE: average absolute error in °C (lower is better). MSE: average squared error in °C² (penalises large misses). RMSE: √MSE, back in °C. R²: share of the test-set variance explained (1 = perfect, 0 = predicting the mean).

**Interpretation** — Every learned model beats persistence; the non-linear models (SVR, Random Forest, XGBoost) beat the linear ones, and tuned XGBoost is best. The gain over persistence is modest (~12 % lower MAE) because daily temperature is strongly autocorrelated — an honest, expected result for this task. Errors are largest in the pre-monsoon transition (April–May) and in inland cities (Tiruchirappalli, Madurai) where day-to-day swings are bigger.

**Unseen-city test (train on 5 cities, test on the 6th, XGBoost MAE vs persistence MAE)** — Chennai 0.83 vs 0.87 · Coimbatore 0.90 vs 1.03 · Madurai 1.21 vs 1.31 · Tiruchirappalli 1.17 vs 1.29 · Salem 1.10 vs 1.21 · Tirunelveli 1.02 vs 1.12. The model transfers to locations it never saw.
"""

XAI_TEXT = """
**What SHAP does** — SHAP (SHapley Additive exPlanations) splits every prediction into additive contributions of each input feature, relative to the model's average prediction. Red = feature pushes temperature up, blue = pushes it down. *SHAP explains the model's behaviour, not necessarily physical causation.*

**Global findings** — Today's Tmax and the 7-/30-day rolling means dominate (thermal persistence). Next come rainfall and humidity (latent cooling → lower predictions), seasonality (day-of-year), solar radiation and cloud amount (more sun → hotter, more cloud → cooler). Location (lon/lat) matters only slightly: the model learned climate relationships, not city identity.

**Individual explanation** — the waterfall below is for the hottest test day: Salem, 17 Apr 2024 (actual 41.3 °C, predicted 41.7 °C). Use the *Temperature Prediction* tab to get this explanation for any city and date.
"""

DOCS = f"""
## Research problem
Cities face rising heat from urbanisation, reduced vegetation and a warming climate. Standard forecasts give a number but no explanation, no risk categorisation and no way to test mitigation ideas.

## Objectives
1. Predict next-day maximum temperature for six Tamil Nadu cities from public climate data.
2. Identify periods and locations of elevated/extreme heat with documented thresholds.
3. Quantify how humidity, radiation, cloud, wind and rainfall relate to temperature.
4. Compare Linear Regression, SVR, Decision Tree, Random Forest and XGBoost with MAE, MSE, RMSE and R².
5. Explain every prediction (SHAP) and evaluate hypothetical cooling interventions.

## Methodology
NASA POWER daily data → cleaning & EDA (missing values, duplicates, IQR outliers, physical sanity checks) → leakage-free feature engineering → chronological 70/15/15 split → model training & validation-based tuning → test evaluation, TimeSeriesSplit CV, leave-one-city-out → heat-risk classification → SHAP → intervention simulator → this web application.

## Research contribution
The integration — multi-city, leakage-free prediction + documented heat-risk classes + per-prediction SHAP explanations + unseen-city generalisation test + what-if simulator — in one reproducible, open pipeline. Individual algorithms are standard; the framework and its evaluation design are ours.

## Limitations
Reanalysis (~50 km grid) is not station data and smooths neighbourhood-scale heat islands · satellite land-surface temperature ≠ air temperature · a 10-year window cannot establish a climate normal · risk thresholds are statistical, not health-based · simulator outputs are associational · predictions are research outputs, not official IMD forecasts.

## Links
- 📓 **Google Colab notebook** (complete pipeline, executed): {COLAB_URL}
- 💻 **GitHub source code**: {GITHUB_URL}
- 📊 **Dataset**: NASA POWER Daily API — https://power.larc.nasa.gov/ (MERRA-2 + CERES; retrieved 13 Nov 2026)

## References
1. NASA POWER Project (2026). Prediction Of Worldwide Energy Resources, Daily API. https://power.larc.nasa.gov/
2. Lundberg & Lee (2017). A Unified Approach to Interpreting Model Predictions. NeurIPS.
3. Chen & Guestrin (2016). XGBoost: A Scalable Tree Boosting System. KDD.
4. Oke, T. R. (1982). The energetic basis of the urban heat island. QJRMS.
5. IPCC (2021). AR6 WG I, Ch. 11 — Weather and Climate Extreme Events in a Changing Climate.
6. India Meteorological Department — heat-wave criteria. https://mausam.imd.gov.in/

**Submitted by:** Balasudhan R (RA2411003012164) — 21CEO310T Global Warming and Climate Change, SRM IST Kattankulathur.
"""

# ----------------------------------------------------------------------------- UI
DARK_JS = "() => { document.documentElement.classList.add('dark'); document.body.classList.add('dark'); }"
with gr.Blocks(title="CLIMATEPRED AI", theme=gr.themes.Soft(primary_hue="orange", neutral_hue="slate"), js=DARK_JS) as demo:
    gr.Markdown("# 🌡️ CLIMATEPRED AI\n### An Explainable Machine Learning Framework for Urban Heat Prediction, Climate Trend Analysis and Climate Risk Intelligence\n"
                f"Real NASA POWER data · 6 Tamil Nadu cities · {DATE_MIN} → {DATE_MAX} · models trained in the Colab notebook and loaded here.")
    with gr.Tabs():
        with gr.Tab("01 · Climate Overview"):
            ov_city = gr.Dropdown(CITIES, value="Chennai", label="City")
            ov_kpi = gr.Markdown()
            ov_fig = gr.Image(label="Climate trends 2015–2024", show_label=True)
            ov_fig2 = gr.Image(label="Seasonal cycle")
            ov_tbl = gr.Dataframe(label="Annual statistics", interactive=False)
            gr.Markdown("### Six-city comparison (2015–2025 means)")
            gr.Dataframe(six, interactive=False)
            gr.Image(FIG("02_tmax_timeseries.png"), label="Daily maximum temperature, all cities (30-day rolling mean)")
            ov_city.change(overview, ov_city, [ov_kpi, ov_fig, ov_fig2, ov_tbl])
            demo.load(overview, ov_city, [ov_kpi, ov_fig, ov_fig2, ov_tbl])

        with gr.Tab("02 · Temperature Prediction"):
            gr.Markdown(f"Choose a city and the date of **today's observations**; the model predicts the **next day's maximum temperature**, its heat-risk class, and explains the prediction. "
                        f"Valid dates: **{DATE_MIN} → {DATE_MAX}** (latest available observation {DATE_MAX}; later dates would need live weather inputs).")
            with gr.Row():
                city = gr.Dropdown(CITIES, value="Chennai", label="City")
                date = gr.Textbox(value=str(DATE_MAX), label="Date of today's observations (YYYY-MM-DD)")
            gr.Markdown("#### Climate Intervention Simulator — leave at defaults for a plain prediction")
            with gr.Row():
                solar = gr.Slider(60, 110, value=100, step=5, label="Solar radiation (% of actual) — shade / albedo")
                cloud = gr.Slider(0, 40, value=0, step=5, label="Extra cloud cover (pp)")
                rh = gr.Slider(0, 20, value=0, step=2, label="Extra humidity (pp) — greening / water bodies")
                wind = gr.Slider(0, 3, value=0, step=0.5, label="Extra wind (m/s) — ventilation corridors")
            btn = gr.Button("Predict", variant="primary")
            out = gr.Markdown()
            with gr.Row():
                f1 = gr.Image(label="SHAP explanation of this prediction"); f2 = gr.Image(label="Recent history vs prediction")
            f3 = gr.Image(label="Scenario comparison (appears when a slider is changed)")
            gr.Markdown("#### Scenario history (this session)")
            hist_state = gr.State([])
            hist_tbl = gr.Dataframe(interactive=False)
            hist_csv = gr.File(label="⬇️ Export scenario history (CSV)")
            btn.click(predict, [city, date, solar, cloud, rh, wind, hist_state], [out, f1, f2, f3, hist_state, hist_tbl, hist_csv])
            demo.load(predict, [city, date, solar, cloud, rh, wind, hist_state], [out, f1, f2, f3, hist_state, hist_tbl, hist_csv])

        with gr.Tab("03 · Dataset Explorer"):
            with gr.Row():
                ex_city = gr.Dropdown(["All cities"] + CITIES, value="All cities", label="City")
                ex_q = gr.Textbox(label="Search (e.g. 2024-05, Chennai, 41.3)", placeholder="type and press Enter")
                ex_page = gr.Slider(1, 950, value=1, step=1, label="Page (25 rows per page)")
            ex_info = gr.Markdown()
            ex_head = gr.Dataframe(label="Dataset table", interactive=False)
            ex_csv = gr.File(label="⬇️ Export current selection as CSV")
            gr.Markdown("### Column dictionary (NASA POWER parameters)")
            gr.Dataframe(PARAM_DESC, interactive=False)
            ex_qual = gr.Dataframe(label="Data quality: types, missing values, unique counts", interactive=False)
            ex_out = gr.Dataframe(label="Outlier detection (IQR rule, 1.5 × IQR fences) — outliers are kept: they are real extreme-weather days", interactive=False)
            ex_s = gr.Dataframe(label="Statistical summary (mean, median, std, min, quartiles, max)", interactive=False)
            gr.Markdown(f"Full raw per-city CSVs and the processed modelling table: {GITHUB_URL}/tree/main/data")
            ex_outs = [ex_info, ex_head, ex_qual, ex_s, ex_out, ex_csv, ex_page]
            ex_city.change(explore, [ex_city, ex_q, ex_page], ex_outs)
            ex_q.submit(explore, [ex_city, ex_q, ex_page], ex_outs)
            ex_page.release(explore, [ex_city, ex_q, ex_page], ex_outs)
            demo.load(explore, [ex_city, ex_q, ex_page], ex_outs)

        with gr.Tab("04 · Preprocessing & EDA"):
            gr.Markdown(f"""
### A. Data cleaning (before → after)
| Step | Finding | Treatment |
|---|---|---|
| Header parsing | NASA CSV has a metadata block + YEAR/DOY columns | Parsed; YEAR + day-of-year → proper date |
| Missing values | Sentinel −999 → NaN; **0 missing** after download ({len(raw):,} rows × 9 variables) | Time-interpolation per city kept in pipeline for robustness |
| Duplicates | **0** duplicate (city, date) rows | Drop-duplicates step kept |
| Physical validity | Tmax ≥ Tmean ≥ Tmin checked — **0 violations** | Rows violating it would be dropped |
| Outliers (IQR) | Detected per city/variable, mostly rainfall (monsoon days) and extreme Tmax | **Kept** — extreme days are the real events we study |
| Date validation | Continuous daily record {DATE_MIN} → {DATE_MAX}, 3 957 days per city | ✔ |

### B. Statistical analysis — see the *Dataset Explorer* tab for mean, median, std, min, max per variable.

### C. Visualisation""")
            gr.Image(FIG("01_boxplots.png"), label="Outlier analysis — box plots by city")
            gr.Image(FIG("04_distributions.png"), label="Distributions of the climate variables")
            gr.Image(FIG("05_correlation.png"), label="Correlation heatmap — Tmax rises with solar radiation (+0.46) and falls with humidity (−0.60), cloud (−0.43) and rainfall (−0.39)")
            gr.Image(FIG("03_seasonal_cycle.png"), label="Monthly variation of maximum temperature")
            gr.Image(FIG("06_warming_trend.png"), label="Annual temperature trend and hot days per year, 2015–2024")
            gr.Markdown("""
### D. Feature engineering (24 features, no future information)
| Group | Features |
|---|---|
| Today's observations | T2M, T2M_MAX, T2M_MIN, RH2M, PRECTOTCORR, WS2M, ALLSKY_SFC_SW_DWN, CLOUD_AMT, PS |
| Temperature history | tmax_lag1, lag2, lag3, lag7 · tmax_roll7, tmax_roll30 (rolling means) · tmax_std7 · tmin_lag1 |
| Derived | diurnal_range (Tmax − Tmin) · rain_roll3 (3-day rainfall) |
| Calendar / season | month · doy_sin, doy_cos (cyclic day-of-year) |
| Location | lat, lon |
| **Target** | **T2M_MAX of the next day** |""")
            gr.Image(FIG("18_split_timeline.png"), label="Chronological train / validation / test split")

        with gr.Tab("05 · ML Model Laboratory"):
            gr.Markdown(MODEL_LAB)
            gr.Dataframe(cmp_tbl.round(3), label="Full comparison table (validation + test, training time)", interactive=False)

        with gr.Tab("06 · Model Evaluation"):
            gr.Markdown("### Regression — next-day maximum temperature (test period Mar 2024 → Oct 2025)")
            gr.Dataframe(cmp_tbl[["Model", "Test MAE", "Test MSE", "Test RMSE", "Test R2"]].round(3), interactive=False)
            gr.Markdown(EVAL_TEXT)
            gr.Image(FIG("07_model_comparison.png"), label="Model MAE / RMSE / R² comparison")
            gr.Image(FIG("08_pred_vs_actual.png"), label="Actual vs predicted · residual distribution · 120-day forecast strip (Chennai)")
            with gr.Row():
                gr.Image(FIG("17_error_by_city.png"), label="Error by city"); gr.Image(FIG("16_error_by_month.png"), label="Error by month")
            gr.Image(FIG("09_leave_one_city_out.png"), label="Unseen-city generalisation")
            gr.Markdown("### Heat-risk classification (Normal / Elevated / Extreme)")
            gr.Dataframe(clf_tbl.round(3), interactive=False)
            gr.Markdown("Accuracy ≈ 95 % and ROC-AUC ≈ 0.98, but macro-recall shows the rare *Extreme* class is hardest to catch; for public-health use, recall on Extreme would be the metric to optimise.")
            gr.Image(FIG("10_confusion_matrices.png"), label="Confusion matrices")

        with gr.Tab("07 · Explainable AI"):
            gr.Markdown(XAI_TEXT)
            with gr.Row():
                gr.Image(FIG("12_shap_bar.png"), label="A. Global feature importance (mean |SHAP|)")
                gr.Image(FIG("11_shap_summary.png"), label="B. SHAP summary plot")
            gr.Image(FIG("13_shap_dependence.png"), label="Dependence: humidity and solar radiation, coloured by cloud amount")
            gr.Image(FIG("14_shap_waterfall.png"), label="C. Individual prediction explanation — Salem, 17 Apr 2024")

        with gr.Tab("08 · Climate Simulator"):
            gr.Markdown("""
The simulator (sliders in the *Temperature Prediction* tab) takes a real day, modifies environmental inputs that proxy urban cooling interventions, and re-runs the trained model:

| Slider | Urban intervention it proxies | Physical mechanism |
|---|---|---|
| Solar radiation ↓ | Shade trees, cool roofs, high-albedo pavements | Less shortwave absorbed by surfaces |
| Cloud amount ↑ | (natural) reduced insolation | Less incoming radiation |
| Humidity ↑ | Urban greening, restored lakes / water bodies | More latent (evaporative) cooling |
| Wind ↑ | Ventilation corridors in city planning | Heat removed by advection |

**Results on the hottest Chennai test day (30 Apr 2024, baseline 38.9 °C):** −15 % solar → −0.16 °C · +20 pp cloud → −0.02 °C · +10 pp humidity & −10 % solar → −0.31 °C · +1.5 m/s wind → −0.15 °C · combined → −0.32 °C. Averaged over the whole test period the greening scenario lowers predicted Tmax by 0.33 °C.

**How to read this** — effects are small because the inputs are grid-scale (~50 km) weather, so neighbourhood interventions are only proxied, and because the model leans heavily on thermal persistence. The outputs are the model's *learned sensitivity*, clearly labelled hypothetical; they are not causal estimates. Inputs are checked against the training-data range before prediction.""")
            gr.Image(FIG("15_scenario_sweep.png"), label="Sensitivity of predicted Tmax to solar radiation (Chennai)")

        with gr.Tab("09 · Climate Science"):
            gr.Markdown(SCIENCE)

        with gr.Tab("10 · Research & Documentation"):
            gr.Markdown(DOCS)
            gr.Image(FIG("arch.png"), label="System architecture")
            gr.File(P("reports/ClimatePred_AI_Technical_Report.pdf"), label="📄 Technical report (PDF)")

    gr.Markdown("Predictions are research outputs, not official forecasts. © 2026 Balasudhan R — SRM IST.")

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)), share="--share" in sys.argv)
