"""Live prediction: fetch recent real observations from Open-Meteo, build the training-time features, predict next-day Tmax.

Open-Meteo (https://open-meteo.com/) is free, needs no key, and serves ERA5/ICON-based hourly + daily data with ~1 h latency.
Units are converted to match the NASA POWER training data:
  wind 10 m km/h -> 2 m m/s  (÷3.6, ×0.75 log-profile factor)   pressure hPa -> kPa   radiation W/m² hourly -> MJ/m²/day
"""
import json, urllib.request, urllib.parse, logging, traceback
log = logging.getLogger("climatepred.live")
from datetime import date, timedelta
import numpy as np, pandas as pd

CITIES = {"Chennai": (13.08, 80.27), "Coimbatore": (11.02, 76.96), "Madurai": (9.93, 78.12),
          "Tiruchirappalli": (10.79, 78.70), "Salem": (11.66, 78.15), "Tirunelveli": (8.71, 77.76)}

def fetch_open_meteo(lat, lon, past_days=45, timeout=20):
    params = {
        "latitude": lat, "longitude": lon, "timezone": "Asia/Kolkata", "past_days": past_days, "forecast_days": 3,
        "daily": "temperature_2m_max,temperature_2m_min,temperature_2m_mean,precipitation_sum",
        "hourly": "relative_humidity_2m,wind_speed_10m,cloud_cover,surface_pressure,shortwave_radiation",
    }
    url = "https://api.open-meteo.com/v1/forecast?" + urllib.parse.urlencode(params)
    log.info("live fetch: %s", url)
    req = urllib.request.Request(url, headers={"User-Agent": "CLIMATEPRED-AI/1.0 (student project)"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        js = json.load(r)
    if "error" in js and js.get("error"):
        raise RuntimeError(f"Open-Meteo error: {js.get('reason')}")
    return js

def fetch_nasa_power_recent(lat, lon, city, days=70, timeout=40):
    """Fallback: NASA POWER daily API (same product as training). Lags real time by a few days."""
    end = date.today(); start = end - timedelta(days=days)
    params = {"parameters": "T2M,T2M_MAX,T2M_MIN,RH2M,PRECTOTCORR,WS2M,ALLSKY_SFC_SW_DWN,CLOUD_AMT,PS", "community": "AG",
              "longitude": lon, "latitude": lat, "start": start.strftime("%Y%m%d"), "end": end.strftime("%Y%m%d"), "format": "JSON"}
    url = "https://power.larc.nasa.gov/api/temporal/daily/point?" + urllib.parse.urlencode(params)
    log.info("fallback fetch: %s", url)
    req = urllib.request.Request(url, headers={"User-Agent": "CLIMATEPRED-AI/1.0 (student project)"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        js = json.load(r)
    p = js["properties"]["parameter"]
    df = pd.DataFrame(p); df.index = pd.to_datetime(df.index, format="%Y%m%d"); df = df.replace(-999, np.nan).dropna()
    df = df.reset_index().rename(columns={"index": "date"})
    df["city"] = city; df["lat"], df["lon"] = CITIES[city]; df["complete"] = True
    return df

def to_daily_frame(js, city):
    """Convert an Open-Meteo response to the NASA-POWER-style daily table used in training."""
    d = pd.DataFrame(js["daily"]); d["date"] = pd.to_datetime(d["time"])
    h = pd.DataFrame(js["hourly"]); h["date"] = pd.to_datetime(h["time"]).dt.normalize()
    hd = h.groupby("date").agg(RH2M=("relative_humidity_2m", "mean"), WS10=("wind_speed_10m", "mean"),
                               CLOUD_AMT=("cloud_cover", "mean"), PS=("surface_pressure", "mean"),
                               SW=("shortwave_radiation", "sum"), n=("relative_humidity_2m", "count")).reset_index()
    out = d.merge(hd, on="date", how="left")
    out = out.rename(columns={"temperature_2m_max": "T2M_MAX", "temperature_2m_min": "T2M_MIN",
                              "temperature_2m_mean": "T2M", "precipitation_sum": "PRECTOTCORR"})
    out["WS2M"] = out["WS10"] / 3.6 * 0.75
    out["PS"] = out["PS"] / 10.0
    out["ALLSKY_SFC_SW_DWN"] = out["SW"] * 3600 / 1e6
    out["city"] = city; out["lat"], out["lon"] = CITIES[city]
    out["complete"] = out["n"] >= 24
    cols = ["date", "city", "lat", "lon", "T2M", "T2M_MAX", "T2M_MIN", "RH2M", "PRECTOTCORR", "WS2M", "ALLSKY_SFC_SW_DWN", "CLOUD_AMT", "PS", "complete"]
    return out[cols]

def build_features(daily):
    """Identical feature recipe to the training pipeline (make_features), for a single city."""
    d = daily.sort_values("date").copy()
    for lag in [1, 2, 3, 7]:
        d[f"tmax_lag{lag}"] = d["T2M_MAX"].shift(lag)
    d["tmax_roll7"] = d["T2M_MAX"].rolling(7).mean()
    d["tmax_roll30"] = d["T2M_MAX"].rolling(30).mean()
    d["tmax_std7"] = d["T2M_MAX"].rolling(7).std()
    d["tmin_lag1"] = d["T2M_MIN"].shift(1)
    d["rain_roll3"] = d["PRECTOTCORR"].rolling(3).sum()
    d["diurnal_range"] = d["T2M_MAX"] - d["T2M_MIN"]
    doy = d.date.dt.dayofyear
    d["doy_sin"], d["doy_cos"] = np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)
    d["month"] = d.date.dt.month
    return d

def live_predict(city, reg, clf, features, risk_names, explainer=None, today=None):
    """Returns dict with the prediction for the day after the latest complete observation day, plus Open-Meteo's forecast."""
    lat, lon = CITIES[city]
    today = today or date.today()
    source = "Open-Meteo forecast API (ERA5/ICON), timezone Asia/Kolkata"
    try:
        js = fetch_open_meteo(lat, lon)
        daily = to_daily_frame(js, city)
    except Exception as e:
        log.error("Open-Meteo failed: %s\n%s", e, traceback.format_exc())
        daily = fetch_nasa_power_recent(lat, lon, city)
        source = f"NASA POWER daily API (fallback — Open-Meteo unavailable: {type(e).__name__}); NASA data lags real time by a few days"
    obs = daily[(daily.date.dt.date < today) & daily.complete]          # complete observed days only
    if len(obs) < 31:
        raise RuntimeError(f"Only {len(obs)} complete days returned; need ≥ 31 for the 30-day rolling features.")
    feats = build_features(obs).dropna(subset=features)
    row = feats.iloc[-1]
    X = row[features].to_frame().T.astype(float)
    t = float(reg.predict(X)[0]); risk = risk_names[int(clf.predict(X)[0])]
    target_day = (row.date + pd.Timedelta(days=1)).date()
    om = daily[daily.date.dt.date == target_day]
    om_tmax = float(om.T2M_MAX.iloc[0]) if len(om) and pd.notna(om.T2M_MAX.iloc[0]) else None
    drivers = None
    if explainer is not None:
        sv = pd.Series(explainer.shap_values(X)[0], index=features).sort_values(key=np.abs, ascending=False).head(6)
        drivers = sv.round(2).to_dict()
    return {"city": city, "latest_observation_day": str(row.date.date()), "prediction_for": str(target_day),
            "predicted_tmax": round(t, 1), "heat_risk": risk, "open_meteo_value_for_that_day": om_tmax,
            "today_inputs": {k: round(float(row[k]), 2) for k in ["T2M", "T2M_MAX", "T2M_MIN", "RH2M", "PRECTOTCORR", "WS2M", "ALLSKY_SFC_SW_DWN", "CLOUD_AMT", "PS"]},
            "drivers": drivers, "recent": obs[["date", "T2M_MAX"]].tail(30), "fetched_days": int(len(obs)),
            "source": source}
