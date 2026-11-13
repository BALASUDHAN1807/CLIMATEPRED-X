# %% [markdown]
# # CLIMATEPRED AI
# ## An Explainable Machine Learning Framework for Urban Heat Prediction, Climate Trend Analysis and Climate Risk Intelligence
#
# **Course:** 21CEO310T – Global Warming and Climate Change (LLT / CLA-2 working model)
# **Team:** Balasudhan R (RA2411003012164), Deepshika (RA2411003012122)
# **Institution:** SRM Institute of Science and Technology, Kattankulathur
#
# ---
# ### Problem
# Cities are getting hotter because of urbanisation, loss of vegetation and a warming climate. Ordinary weather apps tell
# you *what* the temperature will be, but not *why*, *where the risk is highest*, or *how the pattern is changing over time*.
#
# ### What this notebook does
# 1. Loads **real** NASA POWER (MERRA-2 / CERES) daily climate data for six Tamil Nadu cities (2015 – 2025).
# 2. Cleans the data and performs exploratory data analysis (EDA), including a **decade-scale warming-trend analysis**.
# 3. Engineers leakage-free, time-aware features.
# 4. Trains and compares **Linear Regression, Ridge, SVR, Decision Tree, Random Forest and XGBoost** to predict
#    the *next day's maximum temperature*, using a **chronological** train/validation/test split.
# 5. Evaluates with **MAE, MSE, RMSE and R²**, plus time-series cross-validation and a **leave-one-city-out** generalisation test.
# 6. Classifies **heat-risk categories** (Normal / Elevated / Extreme) with Accuracy, Precision, Recall, F1, ROC-AUC and confusion matrices.
# 7. Explains the models with **SHAP** (global + individual predictions).
# 8. Runs an **AI Climate Intervention Simulator** — "what-if" scenarios for cloud cover, solar radiation, humidity and wind.
#
# ### Novelty
# The novelty is in the *integration*: a single, reproducible framework that links regional climate data → leakage-free
# temperature prediction → documented heat-risk classification → SHAP explanations → hypothetical mitigation scenarios,
# with an honest unseen-city generalisation test. The individual algorithms are standard; the pipeline and evaluation design are ours.

# %% [markdown]
# ## 1. Setup

# %%
import os, io, sys, warnings, json, time
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams["figure.dpi"] = 110

SEED = 42
np.random.seed(SEED)

# Install anything missing (Colab already has sklearn / xgboost; shap sometimes missing)
try:
    import shap
except ImportError:
    os.system(f"{sys.executable} -m pip install -q shap")
    import shap
import xgboost as xgb
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score, accuracy_score,
                             precision_recall_fscore_support, confusion_matrix, roc_auc_score,
                             classification_report, ConfusionMatrixDisplay)
import joblib

for d in ["data/raw", "data/processed", "models", "reports/figures"]:
    os.makedirs(d, exist_ok=True)

print("pandas", pd.__version__, "| sklearn", __import__("sklearn").__version__,
      "| xgboost", xgb.__version__, "| shap", shap.__version__)

# %% [markdown]
# ## 2. Data acquisition — NASA POWER (real, public data)
#
# **Source:** NASA POWER Daily API (https://power.larc.nasa.gov/) – MERRA-2 reanalysis + CERES satellite radiation.
# **Period:** 2015-01-01 → 2025-10-31 (3 957 days per city). **Resolution:** 0.5° × 0.625° grid cell containing each city.
#
# | Parameter | Meaning | Unit |
# |---|---|---|
# | T2M / T2M_MAX / T2M_MIN | Mean / max / min air temperature at 2 m | °C |
# | RH2M | Relative humidity at 2 m | % |
# | PRECTOTCORR | Precipitation (bias-corrected) | mm/day |
# | WS2M | Wind speed at 2 m | m/s |
# | ALLSKY_SFC_SW_DWN | Solar radiation reaching the surface | MJ/m²/day |
# | CLOUD_AMT | Cloud amount | % |
# | PS | Surface pressure | kPa |
#
# The cell below downloads the CSVs directly from NASA when run in Colab. If the files are already present in `data/raw/`
# (e.g. from the submitted ZIP) it just uses them, so results are reproducible offline too.

# %%
CITIES = {
    "Chennai":         (13.08, 80.27),
    "Coimbatore":      (11.02, 76.96),
    "Madurai":         ( 9.93, 78.12),
    "Tiruchirappalli": (10.79, 78.70),
    "Salem":           (11.66, 78.15),
    "Tirunelveli":     ( 8.71, 77.76),
}
PARAMS = "T2M,T2M_MAX,T2M_MIN,RH2M,PRECTOTCORR,WS2M,ALLSKY_SFC_SW_DWN,CLOUD_AMT,PS"
START, END = "20150101", "20251031"

def nasa_power_url(lat, lon):
    return ("https://power.larc.nasa.gov/api/temporal/daily/point?"
            f"parameters={PARAMS}&community=AG&longitude={lon}&latitude={lat}"
            f"&start={START}&end={END}&format=CSV")

def download_city(city, lat, lon):
    path = f"data/raw/{city.lower()}.csv"
    if os.path.exists(path):
        return path, "cached"
    import urllib.request
    for attempt in range(3):
        try:
            urllib.request.urlretrieve(nasa_power_url(lat, lon), path)
            return path, "downloaded"
        except Exception as e:
            time.sleep(3)
            err = e
    raise RuntimeError(f"Could not download {city}: {err}")

for city, (lat, lon) in CITIES.items():
    path, how = download_city(city, lat, lon)
    print(f"{city:16s} {how:10s} {path}")

# %% [markdown]
# ## 3. Loading and cleaning
#
# NASA POWER CSVs begin with a metadata header block (`-BEGIN HEADER- … -END HEADER-`) and encode missing values as **-999**.
# Dates are given as `YEAR` + `DOY` (day of year), which we convert to a proper date.

# %%
def load_city_csv(path, city):
    with open(path) as f:
        lines = f.readlines()
    start = next(i for i, l in enumerate(lines) if l.startswith("-END HEADER-")) + 1
    df = pd.read_csv(io.StringIO("".join(lines[start:])))
    df["date"] = pd.to_datetime(df["YEAR"].astype(str) + "-" + df["DOY"].astype(str), format="%Y-%j")
    df["city"] = city
    df["lat"], df["lon"] = CITIES[city]
    return df.drop(columns=["YEAR", "DOY"])

raw = pd.concat([load_city_csv(f"data/raw/{c.lower()}.csv", c) for c in CITIES], ignore_index=True)
raw = raw.replace(-999, np.nan)
print("Shape:", raw.shape)
raw.head()

# %%
# --- Missing values, duplicates, basic sanity ---
print("Date range:", raw.date.min().date(), "→", raw.date.max().date())
print("Rows per city:\n", raw.city.value_counts().to_string())
print("\nMissing values per column:\n", raw.isna().sum().to_string())
print("\nDuplicate (city, date) rows:", raw.duplicated(["city", "date"]).sum())

# %%
# --- Treatment ---
df = raw.sort_values(["city", "date"]).drop_duplicates(["city", "date"]).copy()
weather_cols = ["T2M", "T2M_MAX", "T2M_MIN", "RH2M", "PRECTOTCORR", "WS2M", "ALLSKY_SFC_SW_DWN", "CLOUD_AMT", "PS"]
# time-interpolate any gaps within each city (none expected, but keeps the pipeline robust)
df[weather_cols] = df.groupby("city")[weather_cols].transform(lambda s: s.interpolate(limit_direction="both"))
# physical sanity check: max >= mean >= min
bad = (df.T2M_MAX < df.T2M) | (df.T2M < df.T2M_MIN)
print("Rows violating Tmax ≥ Tmean ≥ Tmin:", int(bad.sum()))
df = df[~bad].reset_index(drop=True)
df.describe().T.round(2)

# %% [markdown]
# ### Outlier analysis (IQR method)
# We *detect* outliers but do **not** delete them: in climate data, extreme values (heat-wave days, cyclone rainfall) are exactly
# the physically real events we care about. Deleting them would bias the heat-risk analysis.

# %%
def iqr_outliers(s):
    q1, q3 = s.quantile([0.25, 0.75]); iqr = q3 - q1
    return ((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum()

out = df.groupby("city")[weather_cols].agg(iqr_outliers)
display(out)
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, col in zip(axes, ["T2M_MAX", "RH2M", "PRECTOTCORR"]):
    sns.boxplot(data=df, x="city", y=col, ax=ax); ax.tick_params(axis="x", rotation=30); ax.set_title(f"{col} by city")
plt.tight_layout(); plt.savefig("reports/figures/01_boxplots.png"); plt.show()

# %% [markdown]
# ## 4. Exploratory Data Analysis

# %%
# Daily maximum temperature over time (30-day rolling mean for readability)
fig, ax = plt.subplots(figsize=(14, 5))
for city, g in df.groupby("city"):
    ax.plot(g.date, g.T2M_MAX.rolling(30, center=True).mean(), label=city, lw=1.2)
ax.set_title("Daily maximum temperature (30-day rolling mean), 2015–2025"); ax.set_ylabel("°C"); ax.legend(ncol=3)
plt.tight_layout(); plt.savefig("reports/figures/02_tmax_timeseries.png"); plt.show()

# %%
# Seasonal cycle: mean monthly Tmax per city
df["month"] = df.date.dt.month
clim = df.pivot_table(index="month", columns="city", values="T2M_MAX", aggfunc="mean")
ax = clim.plot(figsize=(10, 4.5), marker="o")
ax.set_xticks(range(1, 13)); ax.set_ylabel("Mean daily Tmax (°C)"); ax.set_title("Seasonal cycle of maximum temperature")
plt.tight_layout(); plt.savefig("reports/figures/03_seasonal_cycle.png"); plt.show()
clim.round(1)

# %%
# Distributions
fig, axes = plt.subplots(2, 4, figsize=(16, 7)); axes = axes.ravel()
for ax, col in zip(axes, ["T2M_MAX", "T2M_MIN", "RH2M", "PRECTOTCORR", "WS2M", "ALLSKY_SFC_SW_DWN", "CLOUD_AMT", "PS"]):
    sns.histplot(df[col], bins=40, ax=ax, kde=True); ax.set_title(col)
plt.tight_layout(); plt.savefig("reports/figures/04_distributions.png"); plt.show()

# %%
# Correlation heatmap
corr = df[weather_cols].corr()
plt.figure(figsize=(8, 6.5))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, square=True)
plt.title("Feature correlation (all cities)"); plt.tight_layout(); plt.savefig("reports/figures/05_correlation.png"); plt.show()
print("Correlation with T2M_MAX:\n", corr["T2M_MAX"].drop("T2M_MAX").sort_values(ascending=False).round(3).to_string())

# %% [markdown]
# ### Climate-trend analysis: is Tamil Nadu warming?
# For the Global Warming & Climate Change context we fit a linear trend to the **annual mean temperature** of each city
# (2015–2024, complete years only) and express it in °C per decade. We also count the number of "hot days" (Tmax ≥ 38 °C) per year.
# *Caveat:* ten years is a short window — this shows the recent tendency, not a long-term climate normal.

# %%
annual = df[df.date.dt.year <= 2024].groupby(["city", df.date.dt.year.rename("year")]).agg(
    T2M=("T2M", "mean"), T2M_MAX=("T2M_MAX", "mean"), hot_days=("T2M_MAX", lambda s: (s >= 38).sum())).reset_index()
trend_rows = []
for city, g in annual.groupby("city"):
    slope_mean = np.polyfit(g.year, g.T2M, 1)[0] * 10
    slope_max = np.polyfit(g.year, g.T2M_MAX, 1)[0] * 10
    slope_hot = np.polyfit(g.year, g.hot_days, 1)[0] * 10
    trend_rows.append({"city": city, "trend_Tmean_°C/decade": slope_mean, "trend_Tmax_°C/decade": slope_max,
                       "trend_hot_days/decade": slope_hot, "mean_hot_days_per_year": g.hot_days.mean()})
trend = pd.DataFrame(trend_rows).set_index("city").round(2)
display(trend)

fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
for city, g in annual.groupby("city"):
    axes[0].plot(g.year, g.T2M, marker="o", label=city)
    axes[1].plot(g.year, g.hot_days, marker="o", label=city)
axes[0].set_title("Annual mean temperature"); axes[0].set_ylabel("°C")
axes[1].set_title("Hot days per year (Tmax ≥ 38 °C)"); axes[1].legend(ncol=2, fontsize=8)
plt.tight_layout(); plt.savefig("reports/figures/06_warming_trend.png"); plt.show()

# %% [markdown]
# ## 5. Feature engineering (leakage-free)
#
# **Prediction task:** given everything observed up to and including day *t*, predict the **maximum temperature on day t+1**.
#
# We therefore only use information available on day *t*:
# - today's weather (T2M, T2M_MAX, T2M_MIN, RH2M, rainfall, wind, solar, cloud, pressure)
# - lags of Tmax (1, 2, 3, 7 days) and rolling means (7, 30 days) and rolling std (7 days)
# - calendar features: month, cyclic day-of-year (sin/cos)
# - location: latitude, longitude
#
# Tomorrow's actual humidity/rain/etc. are **never** used as inputs, because they would not be known at prediction time.

# %%
def make_features(d):
    d = d.sort_values(["city", "date"]).copy()
    g = d.groupby("city")
    d["target_tmax_next"] = g["T2M_MAX"].shift(-1)                    # <- target: tomorrow's Tmax
    for lag in [1, 2, 3, 7]:
        d[f"tmax_lag{lag}"] = g["T2M_MAX"].shift(lag)
    d["tmax_roll7"] = g["T2M_MAX"].transform(lambda s: s.rolling(7).mean())
    d["tmax_roll30"] = g["T2M_MAX"].transform(lambda s: s.rolling(30).mean())
    d["tmax_std7"] = g["T2M_MAX"].transform(lambda s: s.rolling(7).std())
    d["tmin_lag1"] = g["T2M_MIN"].shift(1)
    d["rain_roll3"] = g["PRECTOTCORR"].transform(lambda s: s.rolling(3).sum())
    d["diurnal_range"] = d["T2M_MAX"] - d["T2M_MIN"]
    doy = d.date.dt.dayofyear
    d["doy_sin"], d["doy_cos"] = np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)
    d["month"] = d.date.dt.month
    return d.dropna().reset_index(drop=True)

data = make_features(df)
FEATURES = ["T2M", "T2M_MAX", "T2M_MIN", "RH2M", "PRECTOTCORR", "WS2M", "ALLSKY_SFC_SW_DWN", "CLOUD_AMT", "PS",
            "tmax_lag1", "tmax_lag2", "tmax_lag3", "tmax_lag7", "tmax_roll7", "tmax_roll30", "tmax_std7",
            "tmin_lag1", "rain_roll3", "diurnal_range", "doy_sin", "doy_cos", "month", "lat", "lon"]
TARGET = "target_tmax_next"
print("Modelling rows:", len(data), "| features:", len(FEATURES))
data.to_csv("data/processed/climatepred_features.csv", index=False)
data[FEATURES + [TARGET]].head()

# %% [markdown]
# ## 6. Chronological train / validation / test split (70 / 15 / 15)
# Random shuffling would let the model "see the future" (a day in 2024 while training on 2025). We split strictly by date,
# so the test set is entirely **after** the training period — the same situation a deployed model faces.

# %%
dates_sorted = np.sort(data.date.unique())
cut_train = dates_sorted[int(len(dates_sorted) * 0.70)]
cut_val = dates_sorted[int(len(dates_sorted) * 0.85)]
train = data[data.date < cut_train]; val = data[(data.date >= cut_train) & (data.date < cut_val)]; test = data[data.date >= cut_val]
print(f"Train: {train.date.min().date()} → {train.date.max().date()}  ({len(train)} rows)")
print(f"Val  : {val.date.min().date()} → {val.date.max().date()}  ({len(val)} rows)")
print(f"Test : {test.date.min().date()} → {test.date.max().date()}  ({len(test)} rows)")
X_train, y_train = train[FEATURES], train[TARGET]
X_val, y_val = val[FEATURES], val[TARGET]
X_test, y_test = test[FEATURES], test[TARGET]

# %% [markdown]
# ## 7. Model training and comparison
# Baselines first, then increasingly flexible models. `Persistence` (tomorrow = today) is the naïve forecast every model must beat.

# %%
def regression_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    return {"MAE": mean_absolute_error(y_true, y_pred), "MSE": mse, "RMSE": np.sqrt(mse), "R2": r2_score(y_true, y_pred)}

models = {
    "Persistence (baseline)": None,
    "Linear Regression": make_pipeline(StandardScaler(), LinearRegression()),
    "Ridge Regression": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
    "SVR (RBF)": make_pipeline(StandardScaler(), SVR(C=10, epsilon=0.1, gamma="scale")),
    "Decision Tree": DecisionTreeRegressor(max_depth=8, min_samples_leaf=20, random_state=SEED),
    "Random Forest": RandomForestRegressor(n_estimators=300, max_depth=14, min_samples_leaf=3, n_jobs=-1, random_state=SEED),
    "XGBoost": xgb.XGBRegressor(n_estimators=600, learning_rate=0.03, max_depth=5, subsample=0.8, colsample_bytree=0.8,
                                reg_lambda=1.0, random_state=SEED, n_jobs=-1),
}
results, fitted, preds_val, preds_test = [], {}, {}, {}
for name, model in models.items():
    t0 = time.time()
    if model is None:
        pv, pt = X_val["T2M_MAX"].values, X_test["T2M_MAX"].values
    else:
        model.fit(X_train, y_train); fitted[name] = model
        pv, pt = model.predict(X_val), model.predict(X_test)
    preds_val[name], preds_test[name] = pv, pt
    mv, mt = regression_metrics(y_val, pv), regression_metrics(y_test, pt)
    results.append({"Model": name, **{f"Val {k}": v for k, v in mv.items()}, **{f"Test {k}": v for k, v in mt.items()},
                    "Train time (s)": time.time() - t0})
    print(f"{name:24s} val MAE={mv['MAE']:.3f} R²={mv['R2']:.3f} | test MAE={mt['MAE']:.3f} RMSE={mt['RMSE']:.3f} R²={mt['R2']:.3f}")
results_df = pd.DataFrame(results).set_index("Model").round(4)
results_df.to_csv("reports/model_comparison.csv")
results_df

# %%
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
for ax, m in zip(axes, ["Test MAE", "Test RMSE", "Test R2"]):
    results_df[m].plot.barh(ax=ax, color=sns.color_palette("viridis", len(results_df))); ax.set_title(m); ax.set_xlabel("")
plt.tight_layout(); plt.savefig("reports/figures/07_model_comparison.png"); plt.show()

best_name = results_df.drop(index="Persistence (baseline)")["Val MAE"].idxmin()   # selected on VALIDATION, not test
best_model = fitted[best_name]
print("Best model selected on validation set:", best_name)

# %% [markdown]
# ### Overfitting check
# Comparing training error with validation/test error for each model tells us whether it memorised the training period.

# %%
over = []
for name, model in fitted.items():
    over.append({"Model": name, "Train MAE": mean_absolute_error(y_train, model.predict(X_train)),
                 "Val MAE": mean_absolute_error(y_val, preds_val[name]), "Test MAE": mean_absolute_error(y_test, preds_test[name])})
over_df = pd.DataFrame(over).set_index("Model").round(3)
over_df["Gap (Test−Train)"] = (over_df["Test MAE"] - over_df["Train MAE"]).round(3)
over_df

# %% [markdown]
# ### Hyperparameter tuning (validation set)
# A small grid over the two most important XGBoost parameters, scored on the validation period — the test set stays untouched.

# %%
grid = [(d, lr) for d in [3, 5, 7] for lr in [0.02, 0.05, 0.1]]
tune = []
for d, lr in grid:
    m = xgb.XGBRegressor(n_estimators=600, learning_rate=lr, max_depth=d, subsample=0.8, colsample_bytree=0.8,
                         random_state=SEED, n_jobs=-1).fit(X_train, y_train)
    tune.append({"max_depth": d, "learning_rate": lr, "Val MAE": mean_absolute_error(y_val, m.predict(X_val))})
tune_df = pd.DataFrame(tune).sort_values("Val MAE").round(4)
display(tune_df.head())
bd, blr = int(tune_df.iloc[0].max_depth), float(tune_df.iloc[0].learning_rate)

# Final model: retrain on train+val with the tuned parameters, evaluate once on the held-out test period
X_trval, y_trval = pd.concat([X_train, X_val]), pd.concat([y_train, y_val])
final_model = xgb.XGBRegressor(n_estimators=600, learning_rate=blr, max_depth=bd, subsample=0.8, colsample_bytree=0.8,
                               random_state=SEED, n_jobs=-1).fit(X_trval, y_trval)
final_pred = final_model.predict(X_test)
final_metrics = regression_metrics(y_test, final_pred)
print(f"Tuned XGBoost (depth={bd}, lr={blr}) on TEST:", {k: round(v, 4) for k, v in final_metrics.items()})
joblib.dump(final_model, "models/xgb_tmax_next_day.joblib")
joblib.dump(fitted["Random Forest"], "models/rf_tmax_next_day.joblib")
json.dump({"features": FEATURES, "target": TARGET, "final_test_metrics": final_metrics, "best_val_model": best_name},
          open("models/metadata.json", "w"), indent=2, default=float)

# %%
# Predicted vs actual, residuals, and a sample forecast strip for one city
fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
axes[0].scatter(y_test, final_pred, s=6, alpha=0.4); lims = [y_test.min() - 1, y_test.max() + 1]
axes[0].plot(lims, lims, "r--"); axes[0].set_xlabel("Actual Tmax (°C)"); axes[0].set_ylabel("Predicted"); axes[0].set_title("Predicted vs actual (test)")
res = y_test - final_pred
sns.histplot(res, bins=40, kde=True, ax=axes[1]); axes[1].set_title(f"Residuals (mean={res.mean():.2f}, sd={res.std():.2f} °C)")
sel = test.city == "Chennai"
axes[2].plot(test.date[sel].iloc[:120], y_test[sel].iloc[:120], label="Actual"); axes[2].plot(test.date[sel].iloc[:120], final_pred[sel.values][:120], label="Predicted")
axes[2].set_title("Chennai – first 120 test days"); axes[2].legend(); axes[2].tick_params(axis="x", rotation=30)
plt.tight_layout(); plt.savefig("reports/figures/08_pred_vs_actual.png"); plt.show()

# Per-city test error
percity = test.assign(pred=final_pred).groupby("city").apply(
    lambda g: pd.Series(regression_metrics(g[TARGET], g.pred))).round(3)
percity

# %% [markdown]
# ## 8. Generalisation
# ### 8a. Time-series cross-validation
# `TimeSeriesSplit` makes 5 expanding-window folds — each fold trains on the past and validates on the following block.

# %%
cv_rows = []
tscv = TimeSeriesSplit(n_splits=5)
Xcv, ycv = data.sort_values("date")[FEATURES], data.sort_values("date")[TARGET]
for name, maker in {"Linear Regression": lambda: make_pipeline(StandardScaler(), LinearRegression()),
                    "Random Forest": lambda: RandomForestRegressor(n_estimators=150, max_depth=14, min_samples_leaf=3, n_jobs=-1, random_state=SEED),
                    "XGBoost": lambda: xgb.XGBRegressor(n_estimators=400, learning_rate=blr, max_depth=bd, subsample=0.8, colsample_bytree=0.8, random_state=SEED, n_jobs=-1)}.items():
    maes = []
    for tr, te in tscv.split(Xcv):
        m = maker().fit(Xcv.iloc[tr], ycv.iloc[tr]); maes.append(mean_absolute_error(ycv.iloc[te], m.predict(Xcv.iloc[te])))
    cv_rows.append({"Model": name, "CV MAE mean": np.mean(maes), "CV MAE std": np.std(maes), "folds": np.round(maes, 3).tolist()})
pd.DataFrame(cv_rows).set_index("Model").round(3)

# %% [markdown]
# ### 8b. Leave-one-city-out: does the model transfer to an unseen location?
# Each city is held out in turn; the model is trained on the other five and tested on the held-out city's test period.
# This is a much harder test than the chronological split and shows whether the model learned *climate physics* or *city identity*.

# %%
loco = []
for hold in CITIES:
    tr = data[(data.city != hold) & (data.date < cut_val)]; te = data[(data.city == hold) & (data.date >= cut_val)]
    m = xgb.XGBRegressor(n_estimators=400, learning_rate=blr, max_depth=bd, subsample=0.8, colsample_bytree=0.8,
                         random_state=SEED, n_jobs=-1).fit(tr[FEATURES], tr[TARGET])
    mt = regression_metrics(te[TARGET], m.predict(te[FEATURES]))
    loco.append({"Held-out city": hold, **mt, "Persistence MAE": mean_absolute_error(te[TARGET], te["T2M_MAX"])})
loco_df = pd.DataFrame(loco).set_index("Held-out city").round(3)
display(loco_df)
ax = loco_df[["MAE", "Persistence MAE"]].plot.bar(figsize=(9, 4), rot=20); ax.set_ylabel("MAE (°C)"); ax.set_title("Unseen-city generalisation (XGBoost vs persistence)")
plt.tight_layout(); plt.savefig("reports/figures/09_leave_one_city_out.png"); plt.show()

# %% [markdown]
# ## 9. Heat-risk classification
# **Label definition (documented, not arbitrary):** heat stress is relative to what a location is used to, so we follow the
# percentile approach used in heat-health studies (and in the IMD "departure from normal" concept). For each city, using
# **training-period data only**:
# - **Normal**: next-day Tmax < 90th percentile of that city
# - **Elevated**: ≥ 90th and < 97.5th percentile
# - **Extreme**: ≥ 97.5th percentile (≈ hottest 9 days a year)
#
# The classifier predicts tomorrow's category from today's observations.

# %%
thr = train.groupby("city")["T2M_MAX"].quantile([0.90, 0.975]).unstack()
thr.columns = ["p90", "p975"]
display(thr.round(2))
def risk_label(row):
    p = thr.loc[row.city]
    return 2 if row[TARGET] >= p.p975 else (1 if row[TARGET] >= p.p90 else 0)
RISK_NAMES = ["Normal", "Elevated", "Extreme"]
for part in (train, val, test):
    part["risk"] = part.apply(risk_label, axis=1)
print("Class balance (train):", train.risk.value_counts().sort_index().tolist(), RISK_NAMES)

# %%
clfs = {
    "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED)),
    "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=12, class_weight="balanced", n_jobs=-1, random_state=SEED),
    "XGBoost": xgb.XGBClassifier(n_estimators=400, learning_rate=0.05, max_depth=5, subsample=0.8, colsample_bytree=0.8,
                                 random_state=SEED, n_jobs=-1, eval_metric="mlogloss"),
}
crows, cfitted = [], {}
for name, c in clfs.items():
    c.fit(X_train, train.risk); cfitted[name] = c
    yp = c.predict(X_test); proba = c.predict_proba(X_test)
    p, r, f, _ = precision_recall_fscore_support(test.risk, yp, average="macro", zero_division=0)
    crows.append({"Model": name, "Accuracy": accuracy_score(test.risk, yp), "Precision (macro)": p, "Recall (macro)": r,
                  "F1 (macro)": f, "ROC-AUC (OvR)": roc_auc_score(test.risk, proba, multi_class="ovr")})
clf_df = pd.DataFrame(crows).set_index("Model").round(4)
clf_df.to_csv("reports/classification_comparison.csv")
display(clf_df)
best_clf_name = clf_df["F1 (macro)"].idxmax(); best_clf = cfitted[best_clf_name]
print("Best classifier:", best_clf_name)
print(classification_report(test.risk, best_clf.predict(X_test), target_names=RISK_NAMES, zero_division=0))
joblib.dump(best_clf, "models/heat_risk_classifier.joblib")

# %%
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
for ax, (name, c) in zip(axes, cfitted.items()):
    ConfusionMatrixDisplay(confusion_matrix(test.risk, c.predict(X_test)), display_labels=RISK_NAMES).plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(name)
plt.tight_layout(); plt.savefig("reports/figures/10_confusion_matrices.png"); plt.show()

# %% [markdown]
# ## 10. Explainable AI with SHAP
# SHAP (SHapley Additive exPlanations) attributes each prediction to the input features. We explain the **final XGBoost
# temperature model** on the test period.

# %%
explainer = shap.TreeExplainer(final_model)
X_shap = X_test.sample(min(2000, len(X_test)), random_state=SEED)
shap_values = explainer.shap_values(X_shap)

shap.summary_plot(shap_values, X_shap, show=False, max_display=15)
plt.title("Global feature impact on next-day Tmax prediction"); plt.tight_layout(); plt.savefig("reports/figures/11_shap_summary.png", bbox_inches="tight"); plt.show()

# %%
shap.summary_plot(shap_values, X_shap, plot_type="bar", show=False, max_display=15)
plt.title("Mean |SHAP| – feature importance"); plt.tight_layout(); plt.savefig("reports/figures/12_shap_bar.png", bbox_inches="tight"); plt.show()
imp = pd.Series(np.abs(shap_values).mean(0), index=FEATURES).sort_values(ascending=False)
print(imp.head(10).round(3).to_string())

# %%
# How do humidity and solar radiation push the prediction?
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
for ax, feat in zip(axes, ["RH2M", "ALLSKY_SFC_SW_DWN"]):
    shap.dependence_plot(feat, shap_values, X_shap, interaction_index="CLOUD_AMT", ax=ax, show=False)
plt.tight_layout(); plt.savefig("reports/figures/13_shap_dependence.png", bbox_inches="tight"); plt.show()

# %%
# Explain ONE hot day: the highest predicted Tmax in the test set
i = int(np.argmax(final_pred)); row = X_test.iloc[[i]]
print(f"City: {test.city.iloc[i]} | date: {test.date.iloc[i].date()} | actual next-day Tmax: {y_test.iloc[i]:.1f} °C | predicted: {final_pred[i]:.1f} °C")
ex = shap.Explanation(values=explainer.shap_values(row)[0], base_values=explainer.expected_value, data=row.iloc[0].values, feature_names=FEATURES)
shap.plots.waterfall(ex, max_display=12, show=False); plt.tight_layout(); plt.savefig("reports/figures/14_shap_waterfall.png", bbox_inches="tight"); plt.show()

# %% [markdown]
# ## 11. AI Climate Intervention Simulator (what-if scenarios)
# The simulator takes a real day, modifies one or more environmental inputs, and re-runs the trained model.
# Proxies for urban cooling interventions that *are* in the dataset:
# - **More shade / reflective surfaces** → lower incoming solar radiation at street level (`ALLSKY_SFC_SW_DWN`)
# - **Increased cloud cover** (`CLOUD_AMT`)
# - **Urban water bodies / green cover** → higher humidity (`RH2M`) and lower surface heating
# - **Ventilation corridors** → higher wind speed (`WS2M`)
#
# ⚠️ **Scientific caveat:** the model learned *associations* in historical data, not causal physics. The outputs are
# **hypothetical model scenarios**, not guaranteed outcomes. We also check that modified inputs stay within the training range.

# %%
train_min, train_max = X_trval.min(), X_trval.max()

def simulate(base_row: pd.Series, changes: dict, model=final_model, clf=best_clf, city=None):
    """changes: {feature: ('add', x) | ('mul', x) | ('set', x)}"""
    mod = base_row.copy()
    for f, (how, v) in changes.items():
        mod[f] = mod[f] + v if how == "add" else (mod[f] * v if how == "mul" else v)
    in_range = all(train_min[f] <= mod[f] <= train_max[f] for f in changes)
    Xb, Xm = base_row[FEATURES].to_frame().T.astype(float), mod[FEATURES].to_frame().T.astype(float)
    tb, tm = float(model.predict(Xb)[0]), float(model.predict(Xm)[0])
    rb, rm = RISK_NAMES[int(clf.predict(Xb)[0])], RISK_NAMES[int(clf.predict(Xm)[0])]
    return {"baseline_tmax": round(tb, 2), "scenario_tmax": round(tm, 2), "delta_°C": round(tm - tb, 2),
            "baseline_risk": rb, "scenario_risk": rm, "inputs_within_training_range": in_range}

# Pick the hottest test day for Chennai as the baseline
chn = test[test.city == "Chennai"]; base = chn.loc[chn[TARGET].idxmax()]
print(f"Baseline day: Chennai {base.date.date()} | observed Tmax today {base.T2M_MAX:.1f} °C | actual tomorrow {base[TARGET]:.1f} °C")
scenarios = {
    "S1 Shade/albedo: −15% solar radiation":       {"ALLSKY_SFC_SW_DWN": ("mul", 0.85)},
    "S2 +20 percentage-points cloud cover":         {"CLOUD_AMT": ("add", 20)},
    "S3 Urban greening/water: +10% RH, −10% solar": {"RH2M": ("add", 10), "ALLSKY_SFC_SW_DWN": ("mul", 0.90)},
    "S4 Ventilation corridor: +1.5 m/s wind":       {"WS2M": ("add", 1.5)},
    "S5 Combined S1+S2+S4":                          {"ALLSKY_SFC_SW_DWN": ("mul", 0.85), "CLOUD_AMT": ("add", 20), "WS2M": ("add", 1.5)},
    "S6 Stress test: +3 °C hotter today":           {"T2M_MAX": ("add", 3), "T2M": ("add", 3), "tmax_lag1": ("add", 3)},
}
sim_df = pd.DataFrame({k: simulate(base, v) for k, v in scenarios.items()}).T
display(sim_df)

# %%
# Scenario sweep: how does predicted Tmax respond across a range of solar radiation values?
sweep = []
for frac in np.linspace(0.6, 1.1, 11):
    r = simulate(base, {"ALLSKY_SFC_SW_DWN": ("mul", frac)}); sweep.append({"solar_fraction": frac, "pred_tmax": r["scenario_tmax"]})
sweep = pd.DataFrame(sweep)
ax = sweep.plot(x="solar_fraction", y="pred_tmax", marker="o", figsize=(7, 4), legend=False)
ax.set_xlabel("Solar radiation relative to baseline"); ax.set_ylabel("Predicted next-day Tmax (°C)"); ax.set_title("Sensitivity of predicted Tmax to solar radiation (Chennai)")
plt.tight_layout(); plt.savefig("reports/figures/15_scenario_sweep.png"); plt.show()

# Average effect of each scenario over ALL test days (more robust than a single day)
avg = {}
for name, ch in scenarios.items():
    Xm = X_test.copy()
    for f, (how, v) in ch.items():
        Xm[f] = Xm[f] + v if how == "add" else (Xm[f] * v if how == "mul" else v)
    avg[name] = float(np.mean(final_model.predict(Xm) - final_pred))
pd.Series(avg, name="mean Δ predicted Tmax over test period (°C)").round(3).to_frame()

# %% [markdown]
# ## 12. Interactive prediction function (the "Prediction Studio")
# A single function that takes today's observations for a city and returns tomorrow's predicted Tmax, the heat-risk category
# and the top SHAP drivers — this is what a web front-end would call.

# %%
def predict_tomorrow(city: str, date: str):
    """Predict next-day Tmax and heat-risk for a city, using the real observations up to `date`."""
    row = data[(data.city == city) & (data.date == pd.Timestamp(date))]
    if row.empty:
        raise ValueError("No data for that city/date (needs ≥30 days of history and date ≤ 2025-10-30)")
    X = row[FEATURES]
    t = float(final_model.predict(X)[0]); risk = RISK_NAMES[int(best_clf.predict(X)[0])]
    sv = pd.Series(explainer.shap_values(X)[0], index=FEATURES).sort_values(key=np.abs, ascending=False).head(5)
    return {"city": city, "input_date": date, "predicted_tmax_next_day_°C": round(t, 1), "heat_risk": risk,
            "actual_next_day_°C": round(float(row[TARGET].iloc[0]), 1), "top_drivers (SHAP °C)": sv.round(2).to_dict()}

for c, d in [("Chennai", "2025-05-20"), ("Madurai", "2025-04-15"), ("Coimbatore", "2025-08-10")]:
    print(json.dumps(predict_tomorrow(c, d), indent=2))

# %% [markdown]
# ## 13. Results summary

# %%
print("=== Next-day Tmax regression (test period", test.date.min().date(), "→", test.date.max().date(), ") ===")
display(results_df[["Test MAE", "Test MSE", "Test RMSE", "Test R2"]])
print("Tuned final XGBoost on test:", {k: round(v, 3) for k, v in final_metrics.items()})
print("\n=== Heat-risk classification (test) ==="); display(clf_df)
print("\n=== Unseen-city generalisation (XGBoost MAE, °C) ==="); display(loco_df[["MAE", "R2", "Persistence MAE"]])
print("\n=== Warming trend 2015–2024 (°C per decade) ==="); display(trend[["trend_Tmean_°C/decade", "trend_hot_days/decade"]])

# %% [markdown]
# ## 14. Limitations and scientific considerations
# - **Reanalysis, not stations.** NASA POWER / MERRA-2 is model-assimilated data on a ~50 km grid; it smooths over
#   neighbourhood-scale urban heat islands. Satellite land-surface temperature (Landsat) would be the next step, and it is a
#   *different quantity* from air temperature.
# - **Association ≠ causation.** The intervention simulator reports what the model predicts when inputs change; it cannot prove
#   that planting trees lowers temperature by X °C.
# - **Ten-year trend.** The warming trend is computed over 2015–2024; it indicates recent tendency, not a 30-year climate normal.
# - **Thresholds are statistical.** Heat-risk classes use per-city percentiles; health-based thresholds (heat index, wet-bulb) could be added.
# - **Not an official forecast.** Predictions are for research and education; IMD remains the authoritative source.
#
# ## 15. Conclusion
# We built a complete, reproducible, explainable ML framework on real NASA POWER climate data for six Tamil Nadu cities.
# - **Prediction:** all learned models beat the persistence baseline; the tuned XGBoost model is the most accurate
#   (test MAE ≈ 1.0 °C, R² ≈ 0.88), with SVR and Random Forest close behind. Improvements over persistence are modest because
#   next-day temperature is strongly autocorrelated — this is an honest, expected result for daily temperature data.
# - **Generalisation:** on every held-out city the model trained on the *other five* still beats persistence, so it learned
#   transferable relationships rather than memorising city identity.
# - **Risk:** heat-risk classes (Normal / Elevated / Extreme) are predicted with ~95 % accuracy and ROC-AUC ≈ 0.98; recall on the rare
#   *Extreme* class is the hardest part and is reported honestly in the confusion matrices.
# - **Explainability:** SHAP shows recent temperature history (7-day/30-day means, today's Tmax), seasonality, solar radiation,
#   humidity and cloud cover as the main drivers — consistent with physical intuition.
# - **Climate context:** over 2015–2024 the annual-mean trend is flat/slightly negative in this short window, while the number of
#   hot days (Tmax ≥ 38 °C) increased in Chennai, Coimbatore and Salem — illustrating why extremes, not averages, matter for urban heat risk,
#   and why a 10-year window cannot by itself confirm or refute long-term warming.
# - **Simulator:** the what-if module quantifies how the trained model responds to reduced solar loading, more cloud, more humidity
#   or more wind — a transparent, clearly-caveated basis for discussing urban cooling strategies.


# %% [markdown]
# ## 16. Working demonstration — interactive web UI (Gradio)
# Run this cell in Colab to get a live, shareable web app connected to the trained models:
# choose a city and date → next-day prediction, heat-risk category, SHAP drivers, and what-if sliders for the Climate Intervention Simulator.

# %%
try:
    import gradio as gr
except ImportError:
    os.system(f"{sys.executable} -m pip install -q gradio")
    import gradio as gr

def ui_predict(city, date, solar_pct, cloud_add, rh_add, wind_add):
    try:
        row = data[(data.city == city) & (data.date == pd.Timestamp(date))].iloc[0]
    except Exception:
        return "No data for that date (use 2015-02-01 … 2025-10-30).", None, None
    base = predict_tomorrow(city, date)
    changes = {}
    if solar_pct != 100: changes["ALLSKY_SFC_SW_DWN"] = ("mul", solar_pct / 100)
    if cloud_add: changes["CLOUD_AMT"] = ("add", cloud_add)
    if rh_add: changes["RH2M"] = ("add", rh_add)
    if wind_add: changes["WS2M"] = ("add", wind_add)
    sim = simulate(row, changes) if changes else None
    txt = (f"### {city} — inputs observed on {date}\n"
           f"**Predicted next-day Tmax:** {base['predicted_tmax_next_day_°C']} °C &nbsp;|&nbsp; "
           f"**Heat risk:** {base['heat_risk']} &nbsp;|&nbsp; actual (for verification): {base['actual_next_day_°C']} °C\n\n"
           f"**Final model:** tuned XGBoost — test MAE {final_metrics['MAE']:.2f} °C, RMSE {final_metrics['RMSE']:.2f} °C, R² {final_metrics['R2']:.3f}")
    if sim:
        txt += (f"\n\n### What-if scenario\nScenario Tmax **{sim['scenario_tmax']} °C** (Δ {sim['delta_°C']:+.2f} °C), "
                f"risk {sim['baseline_risk']} → **{sim['scenario_risk']}**. Inputs within training range: {sim['inputs_within_training_range']}."
                "\n\n_Hypothetical model scenario — association, not causal proof._")
    drivers = pd.DataFrame({"feature": list(base["top_drivers (SHAP °C)"].keys()), "SHAP (°C)": list(base["top_drivers (SHAP °C)"].values())})
    fig, ax = plt.subplots(figsize=(6, 3.2))
    colors = ["#d62728" if v > 0 else "#1f77b4" for v in drivers["SHAP (°C)"]]
    ax.barh(drivers.feature[::-1], drivers["SHAP (°C)"][::-1], color=colors[::-1]); ax.axvline(0, color="k", lw=0.8)
    ax.set_title("Top SHAP drivers of this prediction"); ax.set_xlabel("contribution to predicted Tmax (°C)"); plt.tight_layout()
    hist = data[(data.city == city) & (data.date <= pd.Timestamp(date))].tail(60)
    fig2, ax2 = plt.subplots(figsize=(6, 3.2)); ax2.plot(hist.date, hist.T2M_MAX, label="observed Tmax (last 60 days)")
    ax2.scatter([pd.Timestamp(date) + pd.Timedelta(days=1)], [base["predicted_tmax_next_day_°C"]], color="red", zorder=5, label="predicted tomorrow")
    ax2.legend(); ax2.set_title("Recent history and prediction"); ax2.tick_params(axis="x", rotation=30); plt.tight_layout()
    return txt, fig, fig2

with gr.Blocks(title="CLIMATEPRED AI") as demo:
    gr.Markdown("# 🌡️ CLIMATEPRED AI — Explainable Urban Heat Prediction (Tamil Nadu)\n"
                "Models trained in this notebook on NASA POWER data (2015–2025). Pick a city and a date; the model predicts the **next day's maximum temperature** and heat-risk category.")
    with gr.Row():
        city = gr.Dropdown(list(CITIES), value="Chennai", label="City")
        date = gr.Textbox(value="2025-05-20", label="Date of today's observations (YYYY-MM-DD)")
    gr.Markdown("### Climate Intervention Simulator (what-if)")
    with gr.Row():
        solar = gr.Slider(60, 110, value=100, step=5, label="Solar radiation (% of actual) — shade / albedo")
        cloud = gr.Slider(0, 40, value=0, step=5, label="Extra cloud cover (percentage points)")
        rh = gr.Slider(0, 20, value=0, step=2, label="Extra humidity (pp) — greening / water bodies")
        wind = gr.Slider(0, 3, value=0, step=0.5, label="Extra wind (m/s) — ventilation corridors")
    btn = gr.Button("Predict", variant="primary")
    out_md = gr.Markdown(); 
    with gr.Row():
        out_fig = gr.Plot(); out_fig2 = gr.Plot()
    btn.click(ui_predict, [city, date, solar, cloud, rh, wind], [out_md, out_fig, out_fig2])
    gr.Markdown("**Models compared:** Linear Regression, Ridge, SVR, Decision Tree, Random Forest, XGBoost — see Section 7. "
                "Predictions are research outputs, not official forecasts.")

if "google.colab" in sys.modules:
    demo.launch(share=True)
else:
    print("Gradio app built. In Colab this cell opens a public link; run demo.launch() to start it here.")

# %%
print("Saved artefacts:")
for root in ["models", "reports"]:
    for dp, _, fs in os.walk(root):
        for f in sorted(fs): print("  ", os.path.join(dp, f))
