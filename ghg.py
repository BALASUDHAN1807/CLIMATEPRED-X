"""Greenhouse-gas and global temperature-anomaly context module (NOAA GML + NASA GISTEMP public data).
Loaded by app.py; degrades gracefully if the data files are absent."""
import os, io
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
EXT = os.path.join(HERE, "data", "external")

def _noaa(name):
    path = os.path.join(EXT, name)
    if not os.path.exists(path): return None
    lines = [l for l in open(path, encoding="utf-8", errors="ignore") if not l.startswith("#")]
    df = pd.read_csv(io.StringIO("".join(lines)))
    df.columns = [c.strip() for c in df.columns]
    return df[["year", "mean"]].astype(float)

def _gistemp():
    path = os.path.join(EXT, "GLB.Ts+dSST.csv")
    if not os.path.exists(path): return None
    lines = open(path, encoding="utf-8", errors="ignore").readlines()
    start = next(i for i, l in enumerate(lines) if l.startswith("Year"))
    df = pd.read_csv(io.StringIO("".join(lines[start:])), na_values=["***", "****"])
    df = df[["Year", "J-D"]].rename(columns={"Year": "year", "J-D": "anomaly"})
    df["year"] = df.year.astype(int); df["anomaly"] = pd.to_numeric(df.anomaly, errors="coerce")
    return df.dropna()

def load():
    return {"co2": _noaa("co2_annmean_mlo.csv"), "ch4": _noaa("ch4_annmean_gl.csv"),
            "n2o": _noaa("n2o_annmean_gl.csv"), "temp": _gistemp()}

def figure(d):
    series = [(d["co2"], "CO₂ — Mauna Loa annual mean (ppm)", "#C0504D"), (d["ch4"], "CH₄ — global annual mean (ppb)", "#4F81BD"),
              (d["n2o"], "N₂O — global annual mean (ppb)", "#9BBB59"), (d["temp"], "Global temperature anomaly vs 1951–80 (°C) — NASA GISTEMP v4", "#F79646")]
    fig, axes = plt.subplots(2, 2, figsize=(13, 7.5))
    for ax, (df, title, col) in zip(axes.ravel(), series):
        ax.set_title(title, fontsize=10)
        if df is None: ax.text(0.5, 0.5, "data file not available", ha="center", transform=ax.transAxes); continue
        y = "mean" if "mean" in df.columns else "anomaly"
        ax.plot(df.year, df[y], color=col, lw=1.6)
        if y == "anomaly": ax.axhline(0, color="k", lw=.6); ax.fill_between(df.year, 0, df[y], where=df[y] > 0, color=col, alpha=.25)
    plt.tight_layout(); return fig

def summary(d):
    out = []
    for key, label, unit in [("co2", "CO₂", "ppm"), ("ch4", "CH₄", "ppb"), ("n2o", "N₂O", "ppb")]:
        df = d[key]
        if df is None: out.append(f"- {label}: file not available"); continue
        first, last = df.iloc[0], df.iloc[-1]
        out.append(f"- **{label}**: {first['mean']:.1f} {unit} in {int(first.year)} → **{last['mean']:.1f} {unit} in {int(last.year)}** (+{100*(last['mean']/first['mean']-1):.0f} %)")
    t = d["temp"]
    if t is not None:
        last = t.iloc[-1]; dec = t[t.year >= 2015].anomaly.mean()
        out.append(f"- **Global temperature anomaly** (NASA GISTEMP, vs 1951–1980): **{last.anomaly:+.2f} °C in {int(last.year)}**; 2015–{int(last.year)} average {dec:+.2f} °C")
    else:
        out.append("- Global temperature anomaly: file not available")
    return "\n".join(out)

def correlation(d):
    """CO₂ vs temperature anomaly on overlapping years — a relationship, explicitly not proof of causation by itself."""
    if d["co2"] is None or d["temp"] is None: return None, "—"
    m = d["co2"].merge(d["temp"], on="year")
    r = np.corrcoef(m["mean"], m["anomaly"])[0, 1]
    fig, ax = plt.subplots(figsize=(6, 4))
    sc = ax.scatter(m["mean"], m["anomaly"], c=m.year, cmap="viridis", s=18); plt.colorbar(sc, ax=ax, label="year")
    ax.set_xlabel("CO₂ (ppm, Mauna Loa)"); ax.set_ylabel("Global temperature anomaly (°C)"); ax.set_title(f"CO₂ vs global temperature anomaly, {int(m.year.min())}–{int(m.year.max())} (r = {r:.2f})")
    plt.tight_layout(); return fig, f"{r:.2f}"

DATA_NOTE = ("**Sources:** NOAA Global Monitoring Laboratory — Mauna Loa CO₂ annual means (Lan, Tans & Thoning), global CH₄ and N₂O annual means "
             "(https://gml.noaa.gov/ccgg/trends/); NASA GISS Surface Temperature Analysis (GISTEMP v4), global annual mean anomaly vs 1951–1980 "
             "(https://data.giss.nasa.gov/gistemp/). These are atmospheric **concentrations** and **anomalies** — distinct from emissions and from the "
             "absolute local temperatures used by the ML model, and they are shown for scientific context only; the urban-heat model is not trained on them.")
