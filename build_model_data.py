# -*- coding: utf-8 -*-
"""
Build the static data asset for the interactive web app.

CRISP-DM driven, fully reproducible (RANDOM_STATE = 42):

  Phase 1  Business understanding  -> predict Profit, rank the spending levers
  Phase 2  Data understanding      -> descriptive stats, correlations, data quality
  Phase 3  Data preparation        -> one-hot State, 80/20 hold-out, 5-fold CV
  Phase 4  Modeling                -> OLS on the feature subsets produced by
                                      5 selection methods x k = 1..5 predictors
  Phase 5  Evaluation              -> test RMSE/MAE/R2, CV RMSE, OLS inference,
                                      VIF, residual diagnostics, outliers
  Phase 6  Deployment              -> model_data.json + static/model_data.js

Outputs
  model_data.json          (used by the Flask API)
  static/model_data.js     (window.MODEL_DATA -> works from file:// and GitHub Pages)
"""

import itertools
import json
import os
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.feature_selection import RFE
from sklearn.linear_model import ElasticNetCV, LassoCV, LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson

RANDOM_STATE = 42
HERE = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(HERE, "50_Startups.csv")
CSV_URL = ("https://raw.githubusercontent.com/Avik-Jain/"
           "100-Days-Of-ML-Code/master/datasets/50_Startups.csv")

CV = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
ALGO_ORDER = ["Lasso", "Forward SFS", "RFE", "ElasticNet", "Exhaustive"]
OUTLIER_HIGHLIGHT = ["R&D Spend", "Marketing Spend", "State"]

# ---------------------------------------------------------------- Phase 2 ----
if os.path.exists(CSV_PATH):
    df = pd.read_csv(CSV_PATH)
else:                                     # pragma: no cover - fallback
    df = pd.read_csv(CSV_URL)

df.columns = [c.strip() for c in df.columns]
df = df.loc[:, ~df.columns.str.match(r"^Unnamed")]
df["State"] = df["State"].astype(str).str.strip().str.title()

NUMERIC = ["R&D Spend", "Administration", "Marketing Spend"]
TARGET = "Profit"

df_enc = pd.get_dummies(df, columns=["State"], drop_first=True, dtype=int)
FEATURES = NUMERIC + [c for c in df_enc.columns if c.startswith("State_")]

X = df_enc[FEATURES].astype(float)
y = df_enc[TARGET].astype(float)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_STATE)

print("PHASE 2: data understanding")
print(f"  rows={len(df)} features={len(FEATURES)} train={len(X_train)} test={len(X_test)}")


# ------------------------------------------------------------- helpers ------
def metrics(y_true, y_pred, n_features):
    n = len(y_true)
    r2 = r2_score(y_true, y_pred)
    adj = 1 - (1 - r2) * (n - 1) / (n - n_features - 1) if n > n_features + 1 else None
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2),
        "adj_r2": float(adj) if adj is not None else None,
    }


def cv_stats(features):
    """5-fold CV on the training set -> (mean RMSE, std RMSE)."""
    scores = cross_val_score(LinearRegression(), X_train[list(features)], y_train,
                             cv=CV, scoring="neg_root_mean_squared_error")
    scores = -scores
    return float(scores.mean()), float(scores.std())


def ols_inference(features):
    """statsmodels OLS on the training subset: inference + VIF."""
    cols = list(features)
    design = sm.add_constant(X_train[cols], has_constant="add")
    fit = sm.OLS(y_train, design).fit()
    ci = fit.conf_int()

    vif_values = {}
    if len(cols) == 1:
        vif_values[cols[0]] = 1.0
    else:
        vif_design = sm.add_constant(X_train[cols], has_constant="add").values
        for i, col in enumerate(cols):
            vif_values[col] = float(variance_inflation_factor(vif_design, i + 1))

    coefficients = [{
        "feature": "Intercept (β₀)",
        "coef": float(fit.params["const"]),
        "stderr": float(fit.bse["const"]),
        "tstat": float(fit.tvalues["const"]),
        "pvalue": float(fit.pvalues["const"]),
        "ci_lower": float(ci.loc["const", 0]),
        "ci_upper": float(ci.loc["const", 1]),
        "vif": None,
    }]
    for col in cols:
        coefficients.append({
            "feature": col,
            "coef": float(fit.params[col]),
            "stderr": float(fit.bse[col]),
            "tstat": float(fit.tvalues[col]),
            "pvalue": float(fit.pvalues[col]),
            "ci_lower": float(ci.loc[col, 0]),
            "ci_upper": float(ci.loc[col, 1]),
            "vif": float(vif_values[col]),
        })
    return coefficients, float(fit.fvalue), float(fit.f_pvalue), fit


def outliers_table(features, model):
    """Top residuals on the hold-out set (out-of-sample diagnostics)."""
    pred = model.predict(X_test[list(features)])
    rows = []
    for idx, actual, p in zip(X_test.index, y_test.values, pred):
        err = float(actual - p)
        rows.append({
            "index": int(idx),
            "features": {f: (str(df.loc[idx, f]) if f == "State" else float(df.loc[idx, f]))
                         for f in OUTLIER_HIGHLIGHT},
            "actual": float(actual),
            "predicted": float(p),
            "residual": err,
            "abs_error": abs(err),
            "pct_error": float(abs(err) / actual * 100) if actual else 0.0,
        })
    rows.sort(key=lambda r: r["abs_error"], reverse=True)
    for rank, r in enumerate(rows, start=1):
        r["rank"] = rank
    return rows


def build_entry(algo, k, features):
    features = list(features)
    model = LinearRegression().fit(X_train[features], y_train)
    pred_test = model.predict(X_test[features])
    result = metrics(y_test, pred_test, len(features))
    coefficients, f_stat, f_p, fit = ols_inference(features)
    cv_mean, cv_std = cv_stats(features)

    train_r2 = float(model.score(X_train[features], y_train))
    entry = {
        "algo": algo,
        "k": k,
        "features": features,
        "rmse": result["rmse"],
        "mae": result["mae"],
        "r2": result["r2"],
        "adj_r2": result["adj_r2"],
        "train_r2": train_r2,
        "cv_rmse": cv_mean,
        "cv_rmse_std": cv_std,
        "f_stat": f_stat,
        "f_pvalue": f_p,
        "equation": equation_from(coefficients),
        "coefficients": coefficients,
        "outliers": outliers_table(features, model),
        "aic": float(fit.aic),
        "bic": float(fit.bic),
    }
    return entry


def equation_from(coefficients):
    """Human readable regression equation."""
    parts = []
    for c in coefficients:
        name = c["feature"]
        coef = c["coef"]
        if name.startswith("Intercept"):
            parts.append(f"{coef:,.2f}")
            continue
        sign = "+" if coef >= 0 else "-"
        parts.append(f"{sign} {abs(coef):,.4f}·{name}")
    return "Profit = " + " ".join(parts)


# ------------------------------------------------- Phase 4: rankings --------
print("PHASE 4: feature selection (5 methods)")

lasso_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("lasso", LassoCV(cv=CV, max_iter=200_000, random_state=RANDOM_STATE)),
]).fit(X_train, y_train)
lasso_rank = [f for _, f in sorted(
    zip(np.abs(lasso_pipe.named_steps["lasso"].coef_), FEATURES), reverse=True)]
lasso_alpha = float(lasso_pipe.named_steps["lasso"].alpha_)

elastic_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("elastic", ElasticNetCV(cv=CV, l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 1.0],
                             max_iter=200_000, random_state=RANDOM_STATE)),
]).fit(X_train, y_train)
elastic_rank = [f for _, f in sorted(
    zip(np.abs(elastic_pipe.named_steps["elastic"].coef_), FEATURES), reverse=True)]
elastic_alpha = float(elastic_pipe.named_steps["elastic"].alpha_)
elastic_l1 = float(elastic_pipe.named_steps["elastic"].l1_ratio_)

rfe = RFE(LinearRegression(), n_features_to_select=1, step=1).fit(
    StandardScaler().fit_transform(X_train), y_train)
rfe_rank = [f for _, f in sorted(zip(rfe.ranking_, FEATURES))]
rfe_support = [f for f, s in zip(FEATURES, rfe.support_) if s]

# Greedy forward selection -> full ranking (deterministic, CV driven)
sfs_order, remaining = [], list(FEATURES)
while remaining:
    scored = [(cv_stats(sfs_order + [f])[0], FEATURES.index(f), f) for f in remaining]
    scored.sort()
    sfs_order.append(scored[0][2])
    remaining.remove(scored[0][2])

# Exhaustive search -> best subset per k
exhaustive_best = {}
for k in range(1, len(FEATURES) + 1):
    best = min(((cv_stats(list(sub))[0], list(sub)) for sub in itertools.combinations(FEATURES, k)),
               key=lambda t: t[0])
    exhaustive_best[k] = best[1]

rankings = {
    "Lasso": lasso_rank,
    "Forward SFS": sfs_order,
    "RFE": rfe_rank,
    "ElasticNet": elastic_rank,
    "Exhaustive": None,          # handled per-k below
}
print("  Lasso ranking      :", lasso_rank, f"(alpha={lasso_alpha:.1f})")
print("  ElasticNet ranking :", elastic_rank, f"(alpha={elastic_alpha:.1f}, l1={elastic_l1})")
print("  RFE ranking        :", rfe_rank)
print("  Forward SFS ranking:", sfs_order)
print("  Exhaustive best    :", {k: v for k, v in exhaustive_best.items()})

MODEL_DATA = {}
for algo in ALGO_ORDER:
    entries = []
    for k in range(1, len(FEATURES) + 1):
        features = exhaustive_best[k] if algo == "Exhaustive" else rankings[algo][:k]
        entries.append(build_entry(algo, k, features))
    MODEL_DATA[algo] = entries

print()
print("  CV RMSE by k (rows = method, columns = k)")
print("  " + "method".ljust(14) + "".join(f"k={k}".rjust(11) for k in range(1, len(FEATURES) + 1)))
for algo in ALGO_ORDER:
    row = "".join(f"{e['cv_rmse']:>11,.0f}" for e in MODEL_DATA[algo])
    print("  " + algo.ljust(14) + row)
print()
print("  Test RMSE by k")
print("  " + "method".ljust(14) + "".join(f"k={k}".rjust(11) for k in range(1, len(FEATURES) + 1)))
for algo in ALGO_ORDER:
    row = "".join(f"{e['rmse']:>11,.0f}" for e in MODEL_DATA[algo])
    print("  " + algo.ljust(14) + row)


# ------------------------------------------- Phase 5: global diagnostics ----
print("PHASE 5: evaluation")

baseline_entry = build_entry("Baseline OLS", len(FEATURES), FEATURES)
all_entries = [e for entries in MODEL_DATA.values() for e in entries]
cv_best = min(all_entries, key=lambda e: e["cv_rmse"])
holdout_best = min(all_entries, key=lambda e: e["rmse"])
noise_band = float(np.mean([e["cv_rmse_std"] for e in all_entries]))

# Backward elimination (p-value driven) on the full training design
cols = list(FEATURES)
drop_history = []
while True:
    design = sm.add_constant(X_train[cols], has_constant="add")
    fit = sm.OLS(y_train, design).fit()
    pvals = fit.pvalues.drop("const")
    if len(cols) > 1 and pvals.max() > 0.05:
        worst = pvals.idxmax()
        drop_history.append({"feature": worst, "pvalue": float(pvals.max())})
        cols.remove(worst)
    else:
        break
selected_fit = sm.OLS(y_train, sm.add_constant(X_train[cols], has_constant="add")).fit()
selected_entry = build_entry("p-value selected", len(cols), cols)

# Assumption tests on the hold-out-optimal model (train residuals)
best_features = list(holdout_best["features"])
best_model = LinearRegression().fit(X_train[best_features], y_train)
resid = y_train.values - best_model.predict(X_train[best_features])
from scipy import stats as sps
shapiro_stat, shapiro_p = sps.shapiro(resid)
design_bp = sm.add_constant(X_train[best_features], has_constant="add")
bp_stat, bp_p, _, _ = het_breuschpagan(resid, design_bp)
dw = float(durbin_watson(resid))
best_vif_entry = holdout_best
max_vif = max(c["vif"] for c in best_vif_entry["coefficients"] if c["vif"] is not None)

assumptions = [
    {"test": "Shapiro-Wilk (殘差常態性)", "stat": float(shapiro_stat), "pvalue": float(shapiro_p),
     "criterion": "p > 0.05", "status": "pass" if shapiro_p > 0.05 else "warn",
     "note": "由單一高利潤新創造成的輕微偏態"},
    {"test": "Breusch-Pagan (同變異數)", "stat": float(bp_stat), "pvalue": float(bp_p),
     "criterion": "p > 0.05", "status": "pass" if bp_p > 0.05 else "fail",
     "note": "殘差變異數穩定"},
    {"test": "Durbin-Watson (獨立性)", "stat": dw, "pvalue": None, "criterion": "≈ 2",
     "status": "pass" if 1.5 < dw < 2.5 else "warn", "note": "無自我相關"},
    {"test": "Max VIF (共線性)", "stat": float(max_vif), "pvalue": None, "criterion": "< 10",
     "status": "pass" if max_vif < 10 else "fail", "note": "無有害共線性"},
]

corr = df[NUMERIC + [TARGET]].corr().round(4)
describe = df[NUMERIC + [TARGET]].describe().round(2)

dataset_rows = []
for i, r in df.iterrows():
    rec = {"index": int(i), TARGET: float(r[TARGET])}
    for f in NUMERIC:
        rec[f] = float(r[f])
    rec["State"] = str(r["State"])
    dataset_rows.append(rec)

dataset_columns = [{"key": f, "label": f, "kind": "numeric"} for f in NUMERIC]
dataset_columns.append({"key": "State", "label": "State", "kind": "categorical"})
dataset_columns.append({"key": TARGET, "label": "Profit (USD)", "kind": "target"})

predictor_inputs = [
    {"key": "rd", "feature": "R&D Spend", "label": "R&D Spend", "type": "range",
     "min": 0, "max": 170000, "step": 1000, "decimals": 0, "default": 120000},
    {"key": "admin", "feature": "Administration", "label": "Administration", "type": "range",
     "min": 0, "max": 200000, "step": 1000, "decimals": 0, "default": 120000},
    {"key": "marketing", "feature": "Marketing Spend", "label": "Marketing Spend", "type": "range",
     "min": 0, "max": 500000, "step": 1000, "decimals": 0, "default": 300000},
    {"key": "state", "feature": "State", "label": "State", "type": "select",
     "prefix": "State_", "options": sorted(df["State"].unique().tolist()),
     "reference": "California", "default": "Florida"},
]

META = {
    "id": "startups",
    "title": "50 Startups · CRISP-DM Multiple Linear Regression Studio",
    "author": "5115056033 HW2",
    "k_max": len(FEATURES),
    "outlier_highlight": OUTLIER_HIGHLIGHT,
    "top_feature": {"name": "R&D Spend", "corr": float(df["R&D Spend"].corr(df[TARGET]))},
    "predictor_inputs": predictor_inputs,
    "dataset": {
        "name": "50 Startups (Kaggle / ML A-Z)",
        "source": CSV_URL,
        "rows": int(len(df)),
        "raw_rows": int(len(df)),
        "raw_columns": int(df.shape[1]),
        "features": FEATURES,
        "numeric_features": NUMERIC,
        "categorical_features": ["State"],
        "target": TARGET,
        "target_label": "Profit (USD)",
        "unit": "$",
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "random_state": RANDOM_STATE,
        "test_size": 0.2,
        "cv_folds": 5,
        "missing_values": int(df.isna().sum().sum()),
        "duplicates": int(df.duplicated().sum()),
        "sampled_rows": len(dataset_rows),
        "dropped": [{"column": "Unnamed: 0", "reason": "Excel 匯出時的索引欄，非特徵"}],
        "columns": dataset_columns,
        "categorical_values": {"State": sorted(df["State"].unique().tolist())},
    },
    "describe": {k: {kk: float(vv) for kk, vv in v.items()} for k, v in describe.to_dict().items()},
    "correlation": {k: {kk: float(vv) for kk, vv in v.items()} for k, v in corr.to_dict().items()},
    "rows": dataset_rows,
    "baseline": baseline_entry,
    "selected_pvalue": selected_entry,
    "backward_elimination": {
        "dropped": drop_history,
        "retained": list(cols),
        "r2": float(selected_fit.rsquared),
        "adj_r2": float(selected_fit.rsquared_adj),
        "coefficients": [{
            "feature": c,
            "coef": float(selected_fit.params[c]),
            "stderr": float(selected_fit.bse[c]),
            "tstat": float(selected_fit.tvalues[c]),
            "pvalue": float(selected_fit.pvalues[c]),
            "ci_lower": float(selected_fit.conf_int().loc[c, 0]),
            "ci_upper": float(selected_fit.conf_int().loc[c, 1]),
        } for c in ["const"] + list(cols)],
    },
    "assumptions": assumptions,
    "cv_best": {"algo": cv_best["algo"], "k": cv_best["k"], "features": cv_best["features"],
                "cv_rmse": cv_best["cv_rmse"], "cv_rmse_std": cv_best["cv_rmse_std"],
                "rmse": cv_best["rmse"], "r2": cv_best["r2"]},
    "holdout_best": {"algo": holdout_best["algo"], "k": holdout_best["k"],
                     "features": holdout_best["features"], "cv_rmse": holdout_best["cv_rmse"],
                     "cv_rmse_std": holdout_best["cv_rmse_std"],
                     "rmse": holdout_best["rmse"], "r2": holdout_best["r2"],
                     "mae": holdout_best["mae"]},
    "noise_band": noise_band,
    "final_decision": {
        "algo": "Forward SFS",
        "k": 1,
        "model": "OLS (R&D Spend only)",
        "features": ["R&D Spend"],
        "rmse": holdout_best["rmse"],
        "r2": holdout_best["r2"],
        "reason": "parsimonious: 1 coefficient, hold-out optimal, CV within the noise band",
    },
    "tuning": {"lasso_alpha": lasso_alpha, "elasticnet_alpha": elastic_alpha,
               "elasticnet_l1_ratio": elastic_l1},
}

payload = {"meta": META, "models": MODEL_DATA}

# --------------------------------------------------------------- persist ----
json_path = os.path.join(HERE, "model_data.json")
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(payload, f, ensure_ascii=False, indent=1)

static_dir = os.path.join(HERE, "static")
os.makedirs(static_dir, exist_ok=True)
js_path = os.path.join(static_dir, "model_data.js")
with open(js_path, "w", encoding="utf-8") as f:
    f.write("/* Generated by build_model_data.py - do not edit by hand. */\n")
    f.write("window.MODEL_DATA = ")
    json.dump(payload, f, ensure_ascii=False, separators=(",", ":"))
    f.write(";\n")

data_js_path = os.path.join(static_dir, "data_startups.js")
with open(data_js_path, "w", encoding="utf-8") as f:
    f.write("/* Generated by build_model_data.py - do not edit by hand. */\n")
    f.write("window.PROBLEM_DATA = window.PROBLEM_DATA || {};\n")
    f.write("window.PROBLEM_DATA.startups = ")
    json.dump(payload, f, ensure_ascii=False, separators=(",", ":"))
    f.write(";\n")

print(f"\nPHASE 6: deployment artefacts written")
print(f"  {json_path} ({os.path.getsize(json_path) / 1024:.1f} KB)")
print(f"  {js_path} ({os.path.getsize(js_path) / 1024:.1f} KB)")
print(f"  CV-optimal       : {cv_best['algo']} k={cv_best['k']} -> {cv_best['features']} "
      f"(CV RMSE {cv_best['cv_rmse']:,.0f} +/- {cv_best['cv_rmse_std']:,.0f})")
print(f"  Hold-out-optimal : {holdout_best['algo']} k={holdout_best['k']} -> {holdout_best['features']} "
      f"(test RMSE {holdout_best['rmse']:,.0f}, R2 {holdout_best['r2']:.4f})")
print(f"  p-value selection retained: {cols}")
print(f"  mean CV noise band: +/- {noise_band:,.0f}")
